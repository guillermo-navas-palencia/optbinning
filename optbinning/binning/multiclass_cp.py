"""
Generalized assigment problem: solve constrained multiclass optimal binning
problem. Constraint programming implementation.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

from ortools.sat.python import cp_model

from .cp import BinningCP
from .model_data import multiclass_model_data


class MulticlassBinningCP(BinningCP):
    def __init__(self, monotonic_trend, min_n_bins, max_n_bins, min_bin_size,
                 max_bin_size, max_pvalue, max_pvalue_policy,
                 user_splits_fixed, time_limit):

        self.monotonic_trend = monotonic_trend

        self.min_n_bins = min_n_bins
        self.max_n_bins = max_n_bins
        self.min_bin_size = min_bin_size
        self.max_bin_size = max_bin_size

        self.max_pvalue = max_pvalue
        self.max_pvalue_policy = max_pvalue_policy
        self.user_splits_fixed = user_splits_fixed
        self.time_limit = time_limit

        self.min_event_rate_diff = 0

        self.solver_ = None

        self._model = None
        self._n = None
        self._x = None

    def build_model(self, n_nonevent, n_event, trend_changes):
        # Parameters
        M = int(1e6)
        D, V, pvalue_violation_indices = multiclass_model_data(
            n_nonevent, n_event, self.max_pvalue, self.max_pvalue_policy, M)

        n = len(n_nonevent)
        n_records = n_nonevent + n_event
        n_classes = len(self.monotonic_trend)

        # Initialize model
        model = cp_model.CpModel()

        # Decision variables
        x, y, t, d, u, bin_size_diff = self.decision_variables(
            model, n, n_classes)

        # Objective function
        model.Maximize(sum([sum([(V[c][i][i] * x[i, i]) +
                            sum([(V[c][i][j] - V[c][i][j+1]) * x[i, j]
                                 for j in range(i)]) for i in range(n)])
                            for c in range(n_classes)]))

        # Constraint: unique assignment
        self.add_constraint_unique_assignment(model, n, x)

        # Constraint: continuity
        self.add_constraint_continuity(model, n, x)

        # Constraint: min / max bins
        self.add_constraint_min_max_bins(model, n, x, d)

        # Constraint: min / max bin size
        self.add_constraint_min_max_bin_size(model, n, x, u, n_records,
                                             bin_size_diff)

        # Constraints: monotonicity
        for c in range(n_classes):
            if self.monotonic_trend[c] == "ascending":
                self.add_constraint_monotonic_ascending(model, n, D[c], x, M)

            if self.monotonic_trend[c] == "descending":
                self.add_constraint_monotonic_descending(model, n, D[c], x, M)

            elif self.monotonic_trend[c] in ("peak", "valley"):
                for i in range(n):
                    model.Add(t[c] >= i - n * (1 - y[c, i]))
                    model.Add(t[c] <= i + n * y[c, i])

                if self.monotonic_trend[c] == "peak":
                    self.add_constraint_monotonic_peak(
                        model, n, D[c], x, c, y, M)
                else:
                    self.add_constraint_monotonic_valley(
                        model, n, D[c], x, c, y, M)

            elif self.monotonic_trend == "peak_heuristic":
                self.add_constraint_monotonic_peak_heuristic(
                    model, n, D[c], x, trend_changes[c], M)

            elif self.monotonic_trend == "valley_heuristic":
                self.add_constraint_monotonic_valley_heuristic(
                    model, n, D[c], x, trend_changes[c], M)

        # constraint: max-pvalue
        for c in range(n_classes):
            self.add_max_pvalue_constraint(model, x,
                                           pvalue_violation_indices[c])

        # Constraint: fixed splits
        self.add_constraint_fixed_splits(model, n, x)

        self._model = model
        self._x = x
        self._n = n

    def decision_variables(self, model, n, n_classes):
        x = {}
        for i in range(n):
            for j in range(i + 1):
                x[i, j] = model.NewBoolVar("x[{}, {}]".format(i, j))

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
                        y[c, i] = model.NewBoolVar("y[{}]".format(i))

                    # Change points
                    t[c] = model.NewIntVar(0, n, "t[{}]".format(c))

        if self.min_n_bins is not None and self.max_n_bins is not None:
            n_bin_diff = self.max_n_bins - self.min_n_bins

            # Range constraints auxiliary variables
            d = model.NewIntVar(0, n_bin_diff, "n_bin_diff")

        if self.min_bin_size is not None and self.max_bin_size is not None:
            bin_size_diff = self.max_bin_size - self.min_bin_size

            # Range constraints auxiliary variables
            u = {}
            for i in range(n):
                u[i] = model.NewIntVar(0, bin_size_diff, "u[{}]".format(i))

        return x, y, t, d, u, bin_size_diff

    def add_constraint_monotonic_peak(self, model, n, D, x, c, y, M):
        for i in range(1, n):
            for z in range(i):
                model.Add(
                    M * (y[c, i] + y[c, z]) + M + (D[z][z] - M) * x[z, z] +
                    sum([(D[z][j] - D[z][j+1]) * x[z, j]
                         for j in range(z)]) -
                    sum([(D[i][j] - D[i][j + 1]) * x[i, j]
                         for j in range(i)]) -
                    D[i][i] * x[i, i] >= 0)

                model.Add(
                    M * (2 - y[c, i] - y[c, z]) + M + (D[i][i] - M) * x[i, i] +
                    sum([(D[i][j] - D[i][j + 1]) * x[i, j]
                         for j in range(i)]) -
                    sum([(D[z][j] - D[z][j+1]) * x[z, j]
                         for j in range(z)]) -
                    D[z][z] * x[z, z] >= 0)

    def add_constraint_monotonic_valley(self, model, n, D, x, c, y, M):
        for i in range(1, n):
            for z in range(i):
                model.Add(
                    M * (y[c, i] + y[c, z]) + M + (D[i][i] - M) * x[i, i] +
                    sum([(D[i][j] - D[i][j + 1]) * x[i, j]
                         for j in range(i)]) -
                    sum([(D[z][j] - D[z][j+1]) * x[z, j]
                         for j in range(z)]) -
                    D[z][z] * x[z, z] >= 0)

                model.Add(
                    M * (2 - y[c, i] - y[c, z]) + M + (D[z][z] - M) * x[z, z] +
                    sum([(D[z][j] - D[z][j+1]) * x[z, j]
                         for j in range(z)]) -
                    sum([(D[i][j] - D[i][j + 1]) * x[i, j]
                         for j in range(i)]) -
                    D[i][i] * x[i, i] >= 0)
