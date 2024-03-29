"""
Generalized assigment problem: solve constrained multiclass optimal binning
problem. Mixed-Integer programming implementation.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

from ortools.linear_solver import pywraplp

from .mip import BinningMIP
from .model_data import multiclass_model_data


class MulticlassBinningMIP(BinningMIP):
    def __init__(self, monotonic_trend, min_n_bins, max_n_bins, min_bin_size,
                 max_bin_size, min_event_rate_diff, max_pvalue,
                 max_pvalue_policy, mip_solver, user_splits_fixed, time_limit):

        self.monotonic_trend = monotonic_trend

        self.min_n_bins = min_n_bins
        self.max_n_bins = max_n_bins
        self.min_bin_size = min_bin_size
        self.max_bin_size = max_bin_size

        self.min_event_rate_diff = min_event_rate_diff
        self.max_pvalue = max_pvalue
        self.max_pvalue_policy = max_pvalue_policy

        self.mip_solver = mip_solver
        self.user_splits_fixed = user_splits_fixed
        self.time_limit = time_limit

        self.solver_ = None

        self._n = None
        self._x = None

    def build_model(self, n_nonevent, n_event, trend_changes):
        # Parameters
        (D, V, pvalue_violation_indices,
         min_diff_violation_indices) = multiclass_model_data(
            n_nonevent, n_event, self.max_pvalue, self.max_pvalue_policy,
            self.min_event_rate_diff)

        n = len(n_nonevent)
        n_records = n_nonevent + n_event
        n_classes = len(self.monotonic_trend)

        # Initialize solver
        if self.mip_solver == "bop":
            solver = pywraplp.Solver(
                'BinningMIP', pywraplp.Solver.BOP_INTEGER_PROGRAMMING)
        elif self.mip_solver == "cbc":
            solver = pywraplp.Solver(
                'BinningMIP', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

        # Decision variables
        x, y, t, d, u, bin_size_diff = self.decision_variables(
            solver, n, n_classes)

        # Objective function
        solver.Maximize(solver.Sum([solver.Sum([(V[c][i][i] * x[i, i]) +
                        solver.Sum([(V[c][i][j] - V[c][i][j+1]) * x[i, j]
                                    for j in range(i)]) for i in range(n)])
                        for c in range(n_classes)]))

        # Constraint: unique assignment
        self.add_constraint_unique_assignment(solver, n, x)

        # Constraint: continuity
        self.add_constraint_continuity(solver, n, x)

        # Constraint: min / max bins
        self.add_constraint_min_max_bins(solver, n, x, d)

        # Constraint: min / max bin size
        self.add_constraint_min_max_bin_size(solver, n, x, u, n_records,
                                             bin_size_diff)

        # Constraints: monotonicity
        for c in range(n_classes):
            if self.monotonic_trend[c] == "ascending":
                self.add_constraint_monotonic_ascending(solver, n, D[c], x)

            elif self.monotonic_trend[c] == "descending":
                self.add_constraint_monotonic_descending(solver, n, D[c], x)

            elif self.monotonic_trend[c] in ("peak", "valley"):
                for i in range(n):
                    solver.Add(t[c] >= i - n * (1 - y[c, i]))
                    solver.Add(t[c] <= i + n * y[c, i])

                if self.monotonic_trend[c] == "peak":
                    self.add_constraint_monotonic_peak(
                        solver, n, D[c], x, c, y)
                else:
                    self.add_constraint_monotonic_valley(
                        solver, n, D[c], x, c, y)

            elif self.monotonic_trend == "peak_heuristic":
                self.add_constraint_monotonic_peak_heuristic(
                    solver, n, D[c], x, trend_changes[c])

            elif self.monotonic_trend == "valley_heuristic":
                self.add_constraint_monotonic_valley_heuristic(
                    solver, n, D[c], x, trend_changes[c])

        # Constraint: max-pvalue
        for c in range(n_classes):
            self.add_constraint_violation(solver, x,
                                          pvalue_violation_indices[c])

        # Constraint: min diff
        for c in range(n_classes):
            self.add_constraint_violation(solver, x,
                                          min_diff_violation_indices[c])

        # Constraint: fixed splits
        self.add_constraint_fixed_splits(solver, n, x)

        self.solver_ = solver
        self._n = n
        self._x = x

    def decision_variables(self, solver, n, n_classes):
        x = {}
        for i in range(n):
            for j in range(i + 1):
                x[i, j] = solver.BoolVar("x[{}, {}]".format(i, j))

        y = None
        t = None
        d = None
        u = None
        bin_size_diff = None

        if "peak" in self.monotonic_trend or "valley" in self.monotonic_trend:
            # Auxiliary binary variables
            y = {}
            t = {}
            for c in range(n_classes):
                if self.monotonic_trend[c] in ("peak", "valley"):
                    for i in range(n):
                        y[c, i] = solver.BoolVar("y[{}]".format(i))

                    # Change points
                    t[c] = solver.IntVar(0, n, "t[{}]".format(c))

        if self.min_n_bins is not None and self.max_n_bins is not None:
            n_bin_diff = self.max_n_bins - self.min_n_bins

            # Range constraints auxiliary variables
            d = solver.IntVar(0, n_bin_diff, "n_bin_diff")

        if self.min_bin_size is not None and self.max_bin_size is not None:
            bin_size_diff = self.max_bin_size - self.min_bin_size

            # Range constraints auxiliary variables
            u = {}
            for i in range(n):
                u[i] = solver.IntVar(0, bin_size_diff, "u[{}]".format(i))

        return x, y, t, d, u, bin_size_diff

    def add_constraint_monotonic_peak(self, solver, n, D, x, c, y):
        for i in range(1, n):
            for z in range(i):
                solver.Add(
                    n * (y[c, i] + y[c, z]) + 1 + (D[z][z] - 1) * x[z, z] +
                    solver.Sum([(D[z][j] - D[z][j+1]) * x[z, j]
                                for j in range(z)]) -
                    solver.Sum([(D[i][j] - D[i][j + 1]) * x[i, j]
                                for j in range(i)]) -
                    D[i][i] * x[i, i] >= 0)

                solver.Add(
                    n * (2 - y[c, i] - y[c, z]) + 1 + (D[i][i] - 1) * x[i, i] +
                    solver.Sum([(D[i][j] - D[i][j + 1]) * x[i, j]
                                for j in range(i)]) -
                    solver.Sum([(D[z][j] - D[z][j+1]) * x[z, j]
                                for j in range(z)]) -
                    D[z][z] * x[z, z] >= 0)

    def add_constraint_monotonic_valley(self, solver, n, D, x, c, y):
        for i in range(1, n):
            for z in range(i):
                solver.Add(
                    n * (y[c, i] + y[c, z]) + 1 + (D[i][i] - 1) * x[i, i] +
                    solver.Sum([(D[i][j] - D[i][j + 1]) * x[i, j]
                                for j in range(i)]) -
                    solver.Sum([(D[z][j] - D[z][j+1]) * x[z, j]
                                for j in range(z)]) -
                    D[z][z] * x[z, z] >= 0)

                solver.Add(
                    n * (2 - y[c, i] - y[c, z]) + 1 + (D[z][z] - 1) * x[z, z] +
                    solver.Sum([(D[z][j] - D[z][j+1]) * x[z, j]
                                for j in range(z)]) -
                    solver.Sum([(D[i][j] - D[i][j + 1]) * x[i, j]
                                for j in range(i)]) -
                    D[i][i] * x[i, i] >= 0)
