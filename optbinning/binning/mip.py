"""
Generalized assigment problem: solve constrained optimal binning problem.
Mixed-Integer programming implementation.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import numpy as np

from ortools.linear_solver import pywraplp

from .model_data import model_data


class BinningMIP:
    def __init__(self, monotonic_trend, min_n_bins, max_n_bins, min_bin_size,
                 max_bin_size, min_bin_n_event, max_bin_n_event,
                 min_bin_n_nonevent, max_bin_n_nonevent, min_event_rate_diff,
                 max_pvalue, max_pvalue_policy, gamma, user_splits_fixed,
                 mip_solver, time_limit):

        self.monotonic_trend = monotonic_trend

        self.min_n_bins = min_n_bins
        self.max_n_bins = max_n_bins
        self.min_bin_size = min_bin_size
        self.max_bin_size = max_bin_size
        self.min_bin_n_event = min_bin_n_event
        self.max_bin_n_event = max_bin_n_event
        self.min_bin_n_nonevent = min_bin_n_nonevent
        self.max_bin_n_nonevent = max_bin_n_nonevent

        self.min_event_rate_diff = min_event_rate_diff
        self.max_pvalue = max_pvalue
        self.max_pvalue_policy = max_pvalue_policy
        self.gamma = gamma
        self.user_splits_fixed = user_splits_fixed

        self.mip_solver = mip_solver
        self.time_limit = time_limit

        self.solver_ = None

        self._n = None
        self._x = None

    def build_model(self, divergence, n_nonevent, n_event, trend_change):
        # Parameters
        D, V, pvalue_violation_indices = model_data(divergence, n_nonevent,
                                                    n_event, self.max_pvalue,
                                                    self.max_pvalue_policy)
        n = len(n_nonevent)
        n_records = n_nonevent + n_event

        # Initialize solver
        if self.mip_solver == "bop":
            solver = pywraplp.Solver(
                'BinningMIP', pywraplp.Solver.BOP_INTEGER_PROGRAMMING)
        elif self.mip_solver == "cbc":
            solver = pywraplp.Solver(
                'BinningMIP', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

        # Decision variables
        x, y, t, d, u, bin_size_diff = self.decision_variables(solver, n)

        # Objective function
        if self.gamma:
            total_records = int(n_records.sum())
            regularization = self.gamma / total_records
            pmax = solver.IntVar(0, total_records, "pmax")
            pmin = solver.IntVar(0, total_records, "pmin")

            solver.Maximize(solver.Sum([(V[i][i] * x[i, i]) +
                            solver.Sum([(V[i][j] - V[i][j+1]) * x[i, j]
                                        for j in range(i)])
                                        for i in range(n)]) -
                            regularization * (pmax - pmin))
        else:
            solver.Maximize(solver.Sum([(V[i][i] * x[i, i]) +
                            solver.Sum([(V[i][j] - V[i][j+1]) * x[i, j]
                                        for j in range(i)])
                                        for i in range(n)]))

        # Constraint: unique assignment
        self.add_constraint_unique_assignment(solver, n, x)

        # Constraint: continuity
        self.add_constraint_continuity(solver, n, x)

        # Constraint: min / max bins
        self.add_constraint_min_max_bins(solver, n, x, d)

        # Constraint: min / max bin size
        self.add_constraint_min_max_bin_size(solver, n, x, u, n_records,
                                             bin_size_diff)

        # Constraint: min / max n_nonevent per bin
        if (self.min_bin_n_nonevent is not None or
                self.max_bin_n_nonevent is not None):
            for i in range(n):
                bin_ne_size = solver.Sum([n_nonevent[j] * x[i, j]
                                          for j in range(i + 1)])

                if self.min_bin_n_nonevent is not None:
                    solver.Add(bin_ne_size >= self.min_bin_n_nonevent*x[i, i])

                if self.max_bin_n_nonevent is not None:
                    solver.Add(bin_ne_size <= self.max_bin_n_nonevent*x[i, i])

        # Constraint: min / max n_event per bin
        if (self.min_bin_n_event is not None or
                self.max_bin_n_event is not None):
            for i in range(n):
                bin_e_size = solver.Sum([n_event[j] * x[i, j]
                                         for j in range(i + 1)])

                if self.min_bin_n_event is not None:
                    solver.Add(bin_e_size >= self.min_bin_n_event * x[i, i])

                if self.max_bin_n_event is not None:
                    solver.Add(bin_e_size <= self.max_bin_n_event * x[i, i])

        # Constraints: monotonicity
        if self.monotonic_trend == "ascending":
            self.add_constraint_monotonic_ascending(solver, n, D, x)

        elif self.monotonic_trend == "descending":
            self.add_constraint_monotonic_descending(solver, n, D, x)

        elif self.monotonic_trend == "concave":
            self.add_constraint_monotonic_concave(solver, n, D, x)

        elif self.monotonic_trend == "convex":
            self.add_constraint_monotonic_convex(solver, n, D, x)

        elif self.monotonic_trend in ("peak", "valley"):
            for i in range(n):
                solver.Add(t >= i - n * (1 - y[i]))
                solver.Add(t <= i + n * y[i])

            if self.monotonic_trend == "peak":
                self.add_constraint_monotonic_peak(solver, n, D, x, y)
            else:
                self.add_constraint_monotonic_valley(solver, n, D, x, y)

        elif self.monotonic_trend == "peak_heuristic":
            self.add_constraint_monotonic_peak_heuristic(
                solver, n, D, x, trend_change)

        elif self.monotonic_trend == "valley_heuristic":
            self.add_constraint_monotonic_valley_heuristic(
                solver, n, D, x, trend_change)

        # Constraint: reduction of dominating bins
        if self.gamma:
            for i in range(n):
                bin_size = solver.Sum([n_records[j] * x[i, j]
                                       for j in range(i + 1)])

                solver.Add(pmin <= total_records * (1 - x[i, i]) + bin_size)
                solver.Add(pmax >= bin_size)
            solver.Add(pmin <= pmax)

        # Constraint: max-pvalue
        self.add_max_pvalue_constraint(solver, x, pvalue_violation_indices)

        # Constraint: fixed splits
        self.add_constraint_fixed_splits(solver, n, x)

        self.solver_ = solver
        self._n = n
        self._x = x

    def solve(self):
        self.solver_.SetTimeLimit(self.time_limit * 1000)
        status = self.solver_.Solve()

        if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            if status == pywraplp.Solver.OPTIMAL:
                status_name = "OPTIMAL"
            else:
                status_name = "FEASIBLE"

            solution = np.array([self._x[i, i].solution_value()
                                for i in range(self._n)]).astype(bool)
        else:
            if status == pywraplp.Solver.ABNORMAL:
                status_name = "ABNORMAL"
            elif status == pywraplp.Solver.INFEASIBLE:
                status_name = "INFEASIBLE"
            elif status == pywraplp.Solver.UNBOUNDED:
                status_name = "UNBOUNDED"
            else:
                status_name = "UNKNOWN"

            solution = np.zeros(self._n).astype(bool)
            solution[-1] = True

        return status_name, solution

    def decision_variables(self, solver, n):
        x = {}
        for i in range(n):
            for j in range(i + 1):
                x[i, j] = solver.BoolVar("x[{}, {}]".format(i, j))

        y = None
        t = None
        d = None
        u = None
        bin_size_diff = None

        if self.monotonic_trend in ("peak", "valley"):
            # Auxiliary binary variables
            y = {}
            for i in range(n):
                y[i] = solver.BoolVar("y[{}]".format(i))

            # Change point
            t = solver.IntVar(0, n, "t")

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

    def add_constraint_unique_assignment(self, solver, n, x):
        for j in range(n):
            solver.Add(solver.Sum([x[i, j] for i in range(j, n)]) == 1)

    def add_constraint_continuity(self, solver, n, x):
        for i in range(n):
            for j in range(i):
                solver.Add(x[i, j] - x[i, j+1] <= 0)

    def add_constraint_min_max_bins(self, solver, n, x, d):
        if self.min_n_bins is not None or self.max_n_bins is not None:
            trace = solver.Sum([x[i, i] for i in range(n)])

            if self.min_n_bins is not None and self.max_n_bins is not None:
                solver.Add(d + trace - self.max_n_bins == 0)
            elif self.min_n_bins is not None:
                solver.Add(trace >= self.min_n_bins)
            elif self.max_n_bins is not None:
                solver.Add(trace <= self.max_n_bins)

    def add_constraint_min_max_bin_size(self, solver, n, x, u, n_records,
                                        bin_size_diff):
        if self.min_bin_size is not None or self.max_bin_size is not None:
            for i in range(n):
                bin_size = solver.Sum([n_records[j] * x[i, j]
                                       for j in range(i + 1)])

                if (self.min_bin_size is not None and
                        self.max_bin_size is not None):
                    solver.Add(u[i] + bin_size -
                               self.max_bin_size * x[i, i] == 0)
                    solver.Add(u[i] <= bin_size_diff * x[i, i])
                elif self.min_bin_size is not None:
                    solver.Add(bin_size >= self.min_bin_size * x[i, i])
                elif self.max_bin_size is not None:
                    solver.Add(bin_size <= self.max_bin_size * x[i, i])

    def add_constraint_monotonic_ascending(self, solver, n, D, x):
        for i in range(1, n):
            for z in range(i):
                solver.Add(
                    solver.Sum([(D[z][j] - D[z][j+1]) * x[z, j]
                                for j in range(z)]) +
                    D[z][z] * x[z, z] - 1 - (D[i][i] - 1) * x[i, i] -
                    solver.Sum([(D[i][j] - D[i][j + 1]) * x[i, j]
                                for j in range(i)]) +
                    self.min_event_rate_diff * (x[i, i] + x[z, z] - 1) <= 0)

        # Preprocessing
        if self.min_event_rate_diff == 0:
            for i in range(n - 1):
                if D[i+1][i] - D[i+1][i+1] > 0:
                    solver.Add(x[i, i] == 0)
                    for j in range(n - i - 1):
                        if D[i+1+j][i] - D[i+1+j][i+1+j] > 0:
                            solver.Add(x[i+j, i+j] == 0)

    def add_constraint_monotonic_descending(self, solver, n, D, x):
        for i in range(1, n):
            for z in range(i):
                solver.Add(
                    solver.Sum([(D[i][j] - D[i][j + 1]) * x[i, j]
                                for j in range(i)]) + D[i][i] * x[i, i] -
                    1 - (D[z][z] - 1) * x[z, z] -
                    solver.Sum([(D[z][j] - D[z][j+1]) * x[z, j]
                                for j in range(z)]) +
                    self.min_event_rate_diff * (x[i, i] + x[z, z] - 1) <= 0)

        # Preprocessing
        if self.min_event_rate_diff == 0:
            for i in range(n - 1):
                if D[i+1][i] - D[i+1][i+1] < 0:
                    solver.Add(x[i, i] == 0)
                    for j in range(n - i - 1):
                        if D[i+1+j][i] - D[i+1+j][i+1+j] < 0:
                            solver.Add(x[i+j, i+j] == 0)

    def add_constraint_monotonic_concave(self, solver, n, D, x):
        for i in range(2, n):
            for j in range(1, i):
                for k in range(j):
                    solver.Add(
                        -(solver.Sum([(D[i][z] - D[i][z+1]) * x[i, z]
                                      for z in range(i)]) + D[i][i]*x[i, i]) +
                        2 * (solver.Sum([(D[j][z] - D[j][z+1]) * x[j, z]
                                         for z in range(j)])
                             + D[j][j] * x[j, j]) -
                        (solver.Sum([(D[k][z] - D[k][z+1]) * x[k, z]
                                     for z in range(k)]) +
                         D[k][k] * x[k, k]) >= (
                         x[i, i] + x[j, j] + x[k, k] - 3))

    def add_constraint_monotonic_convex(self, solver, n, D, x):
        for i in range(2, n):
            for j in range(1, i):
                for k in range(j):
                    solver.Add(
                        (solver.Sum([(D[i][z] - D[i][z+1]) * x[i, z]
                                     for z in range(i)]) + D[i][i] * x[i, i]) -
                        2 * (solver.Sum([(D[j][z] - D[j][z+1]) * x[j, z]
                                         for z in range(j)]) +
                             D[j][j] * x[j, j]) +
                        (solver.Sum([(D[k][z] - D[k][z+1]) * x[k, z]
                                     for z in range(k)]) +
                         D[k][k] * x[k, k]) >= (
                         x[i, i] + x[j, j] + x[k, k] - 3))

    def add_constraint_monotonic_peak(self, solver, n, D, x, y):
        for i in range(1, n):
            for z in range(i):
                solver.Add(
                    y[i] + y[z] + 1 + (D[z][z] - 1) * x[z, z] +
                    solver.Sum([(D[z][j] - D[z][j+1]) * x[z, j]
                                for j in range(z)]) -
                    solver.Sum([(D[i][j] - D[i][j + 1]) * x[i, j]
                                for j in range(i)]) -
                    D[i][i] * x[i, i] >= 0)

                solver.Add(
                    2 - y[i] - y[z] + 1 + (D[i][i] - 1) * x[i, i] +
                    solver.Sum([(D[i][j] - D[i][j + 1]) * x[i, j]
                                for j in range(i)]) -
                    solver.Sum([(D[z][j] - D[z][j+1]) * x[z, j]
                                for j in range(z)]) -
                    D[z][z] * x[z, z] >= 0)

    def add_constraint_monotonic_valley(self, solver, n, D, x, y):
        for i in range(1, n):
            for z in range(i):
                solver.Add(
                    y[i] + y[z] + 1 + (D[i][i] - 1) * x[i, i] +
                    solver.Sum([(D[i][j] - D[i][j + 1]) * x[i, j]
                                for j in range(i)]) -
                    solver.Sum([(D[z][j] - D[z][j+1]) * x[z, j]
                                for j in range(z)]) -
                    D[z][z] * x[z, z] >= 0)

                solver.Add(
                    2 - y[i] - y[z] + 1 + (D[z][z] - 1) * x[z, z] +
                    solver.Sum([(D[z][j] - D[z][j+1]) * x[z, j]
                                for j in range(z)]) -
                    solver.Sum([(D[i][j] - D[i][j + 1]) * x[i, j]
                                for j in range(i)]) -
                    D[i][i] * x[i, i] >= 0)

    def add_constraint_monotonic_peak_heuristic(self, solver, n, D, x, tc):
        for i in range(1, tc):
            for z in range(i):
                solver.Add(
                    solver.Sum([(D[z][j] - D[z][j+1]) * x[z, j]
                                for j in range(z)]) +
                    D[z][z] * x[z, z] - 1 - (D[i][i] - 1) * x[i, i] -
                    solver.Sum([(D[i][j] - D[i][j + 1]) * x[i, j]
                                for j in range(i)]) +
                    self.min_event_rate_diff * (x[i, i] + x[z, z] - 1) <= 0)

        # Preprocessing
        if self.min_event_rate_diff == 0:
            for i in range(tc - 1):
                if D[i+1][i] - D[i+1][i+1] > 0:
                    solver.Add(x[i, i] == 0)
                    for j in range(tc - i - 1):
                        if D[i+1+j][i] - D[i+1+j][i+1+j] > 0:
                            solver.Add(x[i+j, i+j] == 0)

        for i in range(tc, n):
            for z in range(tc, i):
                solver.Add(
                    solver.Sum([(D[i][j] - D[i][j + 1]) * x[i, j]
                                for j in range(i)]) + D[i][i] * x[i, i] -
                    1 - (D[z][z] - 1) * x[z, z] -
                    solver.Sum([(D[z][j] - D[z][j+1]) * x[z, j]
                                for j in range(z)]) +
                    self.min_event_rate_diff * (x[i, i] + x[z, z] - 1) <= 0)

        # Preprocessing
        if self.min_event_rate_diff == 0:
            for i in range(tc, n - 1):
                if D[i+1][i] - D[i+1][i+1] < 0:
                    solver.Add(x[i, i] == 0)
                    for j in range(tc, n - i - 1):
                        if D[i+1+j][i] - D[i+1+j][i+1+j] < 0:
                            solver.Add(x[i+j, i+j] == 0)

    def add_constraint_monotonic_valley_heuristic(self, solver, n, D, x, tc):
        for i in range(1, tc):
            for z in range(i):
                solver.Add(
                    solver.Sum([(D[i][j] - D[i][j + 1]) * x[i, j]
                                for j in range(i)]) + D[i][i] * x[i, i] -
                    1 - (D[z][z] - 1) * x[z, z] -
                    solver.Sum([(D[z][j] - D[z][j+1]) * x[z, j]
                                for j in range(z)]) +
                    self.min_event_rate_diff * (x[i, i] + x[z, z] - 1) <= 0)

        # Preprocessing
        if self.min_event_rate_diff == 0:
            for i in range(tc - 1):
                if D[i+1][i] - D[i+1][i+1] < 0:
                    solver.Add(x[i, i] == 0)
                    for j in range(tc - i - 1):
                        if D[i+1+j][i] - D[i+1+j][i+1+j] < 0:
                            solver.Add(x[i+j, i+j] == 0)

        for i in range(tc, n):
            for z in range(tc, i):
                solver.Add(
                    solver.Sum([(D[z][j] - D[z][j+1]) * x[z, j]
                                for j in range(z)]) +
                    D[z][z] * x[z, z] - 1 - (D[i][i] - 1) * x[i, i] -
                    solver.Sum([(D[i][j] - D[i][j + 1]) * x[i, j]
                                for j in range(i)]) +
                    self.min_event_rate_diff * (x[i, i] + x[z, z] - 1) <= 0)

        # Preprocessing
        if self.min_event_rate_diff == 0:
            for i in range(tc, n - 1):
                if D[i+1][i] - D[i+1][i+1] > 0:
                    solver.Add(x[i, i] == 0)
                    for j in range(tc, n - i - 1):
                        if D[i+1+j][i] - D[i+1+j][i+1+j] > 0:
                            solver.Add(x[i+j, i+j] == 0)

    def add_max_pvalue_constraint(self, solver, x, pvalue_violation_indices):
        for ind1, ind2 in pvalue_violation_indices:
            solver.Add(x[ind1[0], ind1[1]] + x[ind2[0], ind2[1]] <= 1)

    def add_constraint_fixed_splits(self, solver, n, x):
        if self.user_splits_fixed is not None:
            for i in range(n - 1):
                if self.user_splits_fixed[i]:
                    solver.Add(x[i, i] == 1)
