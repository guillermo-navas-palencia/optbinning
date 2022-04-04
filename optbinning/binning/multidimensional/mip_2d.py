"""
Generalized assigment problem: solve constrained optimal 2D binning problem.
Mixed-Integer programming implementation.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2021

import numpy as np

from ortools.linear_solver import pywraplp


class Binning2DMIP:
    def __init__(self, monotonic_trend_x, monotonic_trend_y, min_n_bins,
                 max_n_bins, min_diff_x, min_diff_y, gamma, n_jobs,
                 time_limit):

        self.monotonic_trend_x = monotonic_trend_x
        self.monotonic_trend_y = monotonic_trend_y
        self.min_n_bins = min_n_bins
        self.max_n_bins = max_n_bins
        self.min_diff_x = min_diff_x
        self.min_diff_y = min_diff_y
        self.gamma = gamma

        self.n_jobs = n_jobs
        self.time_limit = time_limit

        self.solver_ = None
        self.event_rate_ = None
        self.iv_ = None

        self._model = None
        self._x = None
        self._n_rectangles = None

    def build_model(self, n_grid, n_rectangles, cols, c, d_connected_x,
                    d_connected_y, er, n_records):
        # Initialize solver
        solver = pywraplp.Solver(
            'BinningMIP', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

        # Decision variables
        x, d = self.decision_variables(solver, n_rectangles)

        # Objective function
        if self.gamma:
            total_records = int(n_records.sum())
            regularization = self.gamma / total_records
            pmax = solver.NumVar(0, total_records, "pmax")
            pmin = solver.NumVar(0, total_records, "pmin")

            solver.Maximize(
                solver.Sum([c[i] * x[i] for i in range(n_rectangles)]) -
                regularization * (pmax - pmin))
        else:
            solver.Maximize(
                solver.Sum([c[i] * x[i] for i in range(n_rectangles)]))

        # Constraint: unique assignment
        self.add_constraint_unique_assignment(solver, x, n_grid, cols)

        # Constraint: min / max bins
        self.add_constraint_min_max_bins(solver, n_rectangles, x, d)

        # Constraint: monotonicity
        self.add_constraint_monotonic(
            solver, n_rectangles, x, er, d_connected_x, d_connected_y,
            self.min_diff_x, self.min_diff_y)

        # Constraint: reduction of dominating bins
        if self.gamma:
            for i in range(n_rectangles):
                bin_size = n_records[i] * x[i]

                solver.Add(pmin <= total_records * (1 - x[i]) + bin_size)
                solver.Add(pmax >= bin_size)
            solver.Add(pmin <= pmax)

        # Save data for post-processing
        self.solver_ = solver
        self._x = x
        self._n_rectangles = n_rectangles

    def solve(self):
        # Solve
        self.solver_.SetTimeLimit(self.time_limit * 1000)
        self.solver_.SetNumThreads(self.n_jobs)
        status = self.solver_.Solve()

        if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            if status == pywraplp.Solver.OPTIMAL:
                status_name = "OPTIMAL"
            else:
                status_name = "FEASIBLE"

            solution = np.array([self._x[i].solution_value()
                                 for i in range(self._n_rectangles)])

            solution = solution.astype(bool)
        else:
            if status == pywraplp.Solver.ABNORMAL:
                status_name = "ABNORMAL"
            elif status == pywraplp.Solver.INFEASIBLE:
                status_name = "INFEASIBLE"
            elif status == pywraplp.Solver.UNBOUNDED:
                status_name = "UNBOUNDED"
            else:
                status_name = "UNKNOWN"

            solution = np.zeros(self._n_rectangles).astype(bool)

        return status_name, solution

    def decision_variables(self, solver, n_rectangles):
        x = {}

        for i in range(n_rectangles):
            x[i] = solver.BoolVar("x[{}]".format(i))

        d = None

        if self.min_n_bins is not None and self.max_n_bins is not None:
            n_bin_diff = self.max_n_bins - self.min_n_bins

            # Range constraints auxiliary variables
            d = solver.NumVar(0, n_bin_diff, "n_bin_diff")

        return x, d

    def add_constraint_unique_assignment(self, solver, x, n_grid, cols):
        for j in range(n_grid):
            solver.Add(solver.Sum([x[i] for i in cols[j]]) == 1)

    def add_constraint_min_max_bins(self, solver, n_rectangles, x, d):
        if self.min_n_bins is not None or self.max_n_bins is not None:
            n_bins = solver.Sum([x[i] for i in range(n_rectangles)])

            if self.min_n_bins is not None and self.max_n_bins is not None:
                solver.Add(d + n_bins - self.max_n_bins == 0)
            elif self.min_n_bins is not None:
                solver.Add(n_bins >= self.min_n_bins)
            elif self.max_n_bins is not None:
                solver.Add(n_bins <= self.max_n_bins)

    def add_constraint_monotonic(self, solver, n_rectangles, x, er,
                                 d_connected_x, d_connected_y, min_diff_x,
                                 min_diff_y):

        if (self.monotonic_trend_x is not None and
                self.monotonic_trend_y is not None):
            for i in range(n_rectangles):
                ind_x = []
                ind_y = []
                for j in d_connected_x[i]:
                    if self.monotonic_trend_x == "ascending":
                        if er[i] + min_diff_x >= er[j]:
                            ind_x.append(j)
                    elif self.monotonic_trend_x == "descending":
                        if er[i] <= er[j] + min_diff_x:
                            ind_x.append(j)

                if ind_x:
                    solver.Add(solver.Sum([x[j] for j in ind_x]) <=
                               len(ind_x) * (1 - x[i]))

                for j in d_connected_y[i]:
                    if self.monotonic_trend_y == "ascending":
                        if er[i] + min_diff_y >= er[j]:
                            ind_y.append(j)
                    elif self.monotonic_trend_y == "descending":
                        if er[i] <= er[j] + min_diff_y:
                            ind_y.append(j)

                if ind_y:
                    solver.Add(solver.Sum([x[j] for j in ind_y]) <=
                               len(ind_y) * (1 - x[i]))

        elif self.monotonic_trend_x is not None:
            for i in range(n_rectangles):
                ind_x = []
                for j in d_connected_x[i]:
                    if self.monotonic_trend_x == "ascending":
                        if er[i] + min_diff_x >= er[j]:
                            ind_x.append(j)
                    elif self.monotonic_trend_x == "descending":
                        if er[i] <= er[j] + min_diff_x:
                            ind_x.append(j)

                if ind_x:
                    solver.Add(solver.Sum([x[j] for j in ind_x]) <=
                               len(ind_x) * (1 - x[i]))

        elif self.monotonic_trend_y is not None:
            for i in range(n_rectangles):
                ind_y = []
                for j in d_connected_y[i]:
                    if self.monotonic_trend_y == "ascending":
                        if er[i] + min_diff_y >= er[j]:
                            ind_y.append(j)
                    elif self.monotonic_trend_y == "descending":
                        if er[i] <= er[j] + min_diff_y:
                            ind_y.append(j)

                if ind_y:
                    solver.Add(solver.Sum([x[j] for j in ind_y]) <=
                               len(ind_y) * (1 - x[i]))
