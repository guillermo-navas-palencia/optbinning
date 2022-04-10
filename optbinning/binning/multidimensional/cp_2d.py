"""
Generalized assigment problem: solve constrained optimal 2D binning problem.
Constraint programming implementation.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2021

import numpy as np

from ortools.sat.python import cp_model


class Binning2DCP:
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
        # Parameters
        scale = int(1e6)

        # Initialize model
        model = cp_model.CpModel()

        # Decision variables
        x, d = self.decision_variables(model, n_rectangles)

        # Objective function
        if self.gamma:
            total_records = int(n_records.sum())
            regularization = int(np.ceil(scale * self.gamma / total_records))
            pmax = model.NewIntVar(0, total_records, "pmax")
            pmin = model.NewIntVar(0, total_records, "pmin")

            model.Maximize(sum([c[i] * x[i] for i in range(n_rectangles)]) -
                           regularization * (pmax - pmin))
        else:
            model.Maximize(sum([c[i] * x[i] for i in range(n_rectangles)]))

        # Constraint: unique assignment
        self.add_constraint_unique_assignment(model, x, n_grid, cols)

        # Constraint: min / max bins
        self.add_constraint_min_max_bins(model, n_rectangles, x, d)

        # Constraint: monotonicity
        self.add_constraint_monotonic(
            model, n_rectangles, x, er, d_connected_x, d_connected_y,
            self.min_diff_x, self.min_diff_y)

        # Constraint: reduction of dominating bins
        if self.gamma:
            for i in range(n_rectangles):
                bin_size = n_records[i] * x[i]

                model.Add(pmin <= total_records * (1 - x[i]) + bin_size)
                model.Add(pmax >= bin_size)
            model.Add(pmin <= pmax)

        # Save data for post-processing
        self._model = model
        self._x = x
        self._n_rectangles = n_rectangles

    def solve(self):
        # Solve
        self.solver_ = cp_model.CpSolver()
        if self.n_jobs > 1:
            self.solver_.parameters.num_search_workers = self.n_jobs
        else:
            self.solver_.parameters.linearization_level = 2

        self.solver_.parameters.max_time_in_seconds = self.time_limit

        status = self.solver_.Solve(self._model)
        status_name = self.solver_.StatusName(status)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            solution = np.array([self.solver_.BooleanValue(self._x[i])
                                 for i in range(self._n_rectangles)])
        else:
            solution = np.zeros(self._n_rectangles).astype(np.bool)

        return status_name, solution

    def decision_variables(self, model, n_rectangles):
        x = {}
        for i in range(n_rectangles):
            x[i] = model.NewBoolVar("x[{}]".format(i))

        d = None

        if self.min_n_bins is not None and self.max_n_bins is not None:
            n_bin_diff = self.max_n_bins - self.min_n_bins

            # Range constraints auxiliary variables
            d = model.NewIntVar(0, n_bin_diff, "n_bin_diff")

        return x, d

    def add_constraint_unique_assignment(self, model, x, n_grid, cols):
        for j in range(n_grid):
            model.Add(sum([x[i] for i in cols[j]]) == 1)

    def add_constraint_min_max_bins(self, model, n_rectangles, x, d):
        if self.min_n_bins is not None or self.max_n_bins is not None:
            n_bins = sum([x[i] for i in range(n_rectangles)])

            if self.min_n_bins is not None and self.max_n_bins is not None:
                model.Add(d + n_bins - self.max_n_bins == 0)
            elif self.min_n_bins is not None:
                model.Add(n_bins >= self.min_n_bins)
            elif self.max_n_bins is not None:
                model.Add(n_bins <= self.max_n_bins)

    def add_constraint_monotonic(self, model, n_rectangles, x,
                                 er, d_connected_x, d_connected_y, min_diff_x,
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
                    model.Add(sum([x[j] for j in ind_x]) <=
                              len(ind_x) * (1 - x[i]))

                for j in d_connected_y[i]:
                    if self.monotonic_trend_y == "ascending":
                        if er[i] + min_diff_y >= er[j]:
                            ind_y.append(j)
                    elif self.monotonic_trend_y == "descending":
                        if er[i] <= er[j] + min_diff_y:
                            ind_y.append(j)

                if ind_y:
                    model.Add(sum([x[j] for j in ind_y]) <=
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
                    model.Add(sum([x[j] for j in ind_x]) <=
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
                    model.Add(sum([x[j] for j in ind_y]) <=
                              len(ind_y) * (1 - x[i]))
