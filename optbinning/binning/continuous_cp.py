"""
Generalized assigment problem: solve constrained continuous optimal binning
problem. Constraint programming implementation.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import numpy as np

from ortools.sat.python import cp_model

from .cp import BinningCP
from .model_data import continuous_model_data


class ContinuousBinningCP(BinningCP):
    def __init__(self, monotonic_trend, min_n_bins, max_n_bins, min_bin_size,
                 max_bin_size, min_mean_diff, max_pvalue, max_pvalue_policy,
                 gamma, user_splits_fixed, time_limit):

        self.monotonic_trend = monotonic_trend

        self.min_n_bins = min_n_bins
        self.max_n_bins = max_n_bins
        self.min_bin_size = min_bin_size
        self.max_bin_size = max_bin_size

        self.min_mean_diff = min_mean_diff
        self.max_pvalue = max_pvalue
        self.max_pvalue_policy = max_pvalue_policy
        self.gamma = gamma
        self.user_splits_fixed = user_splits_fixed
        self.time_limit = time_limit

        self.solver_ = None

        self._model = None
        self._n = None
        self._x = None

    def build_model(self, n_records, sums, ssums, trend_change):
        # Parameters
        M = int(1e6)
        U, V, pvalue_violation_indices = continuous_model_data(
            n_records, sums, ssums, self.max_pvalue, self.max_pvalue_policy, M)

        n = len(n_records)

        # Initialize model
        model = cp_model.CpModel()

        # Decision variables
        x, y, t, d, u, bin_size_diff = self.decision_variables(model, n)

        if self.gamma:
            total_records = int(n_records.sum())
            regularization = int(np.ceil(M * self.gamma / total_records))
            pmax = model.NewIntVar(0, total_records, "pmax")
            pmin = model.NewIntVar(0, total_records, "pmin")

            model.Maximize(sum([(V[i][i] * x[i, i]) +
                           sum([(V[i][j] - V[i][j+1]) * x[i, j]
                                for j in range(i)]) for i in range(n)]) -
                           regularization * (pmax - pmin))
        else:
            model.Maximize(sum([(V[i][i] * x[i, i]) +
                           sum([(V[i][j] - V[i][j+1]) * x[i, j]
                                for j in range(i)]) for i in range(n)]))

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
        if self.monotonic_trend == "ascending":
            self.add_constraint_monotonic_ascending(model, n, U, x, M)

        if self.monotonic_trend == "descending":
            self.add_constraint_monotonic_descending(model, n, U, x, M)

        elif self.monotonic_trend == "concave":
            self.add_constraint_monotonic_concave(model, n, U, x)

        elif self.monotonic_trend == "convex":
            self.add_constraint_monotonic_convex(model, n, U, x)

        elif self.monotonic_trend in ("valley", "peak"):
            for i in range(n):
                model.Add(t >= i - n * (1 - y[i]))
                model.Add(t <= i + n * y[i])

            if self.monotonic_trend == "peak":
                self.add_constraint_monotonic_peak(model, n, U, x, y)
            else:
                self.add_constraint_monotonic_valley(model, n, U, x, y)

        elif self.monotonic_trend == "peak_heuristic":
            self.add_constraint_monotonic_peak_heuristic(
                model, n, U, x, trend_change, M)

        elif self.monotonic_trend == "valley_heuristic":
            self.add_constraint_monotonic_valley_heuristic(
                model, n, U, x, trend_change, M)

        # Constraint: max-pvalue
        self.add_max_pvalue_constraint(model, x, pvalue_violation_indices)

        # Constraint: fixed splits
        self.add_constraint_fixed_splits(model, n, x)

        self._model = model
        self._x = x
        self._n = n

    def add_constraint_monotonic_ascending(self, model, n, U, x, M):
        min_mean_diff = int(M * self.min_mean_diff)
        for i in range(1, n):
            for z in range(i):
                model.Add(
                    sum([(U[z][j] - U[z][j+1]) * x[z, j]
                         for j in range(z)]) +
                    U[z][z] * x[z, z] - U[i][i] * x[i, i] -
                    sum([(U[i][j] - U[i][j + 1]) * x[i, j]
                         for j in range(i)]) + min_mean_diff <= 0
                    ).OnlyEnforceIf([x[z, z], x[i, i]])

    def add_constraint_monotonic_descending(self, model, n, U, x, M):
        min_mean_diff = int(M * self.min_mean_diff)
        for i in range(1, n):
            for z in range(i):
                model.Add(
                    sum([(U[i][j] - U[i][j + 1]) * x[i, j]
                         for j in range(i)]) + U[i][i] * x[i, i] -
                    U[z][z] * x[z, z] -
                    sum([(U[z][j] - U[z][j+1]) * x[z, j]
                         for j in range(z)]) + min_mean_diff <= 0
                    ).OnlyEnforceIf([x[z, z], x[i, i]])

    def add_constraint_monotonic_concave(self, model, n, U, x):
        for i in range(2, n):
            for j in range(1, i):
                for k in range(j):
                    model.Add(
                        -(sum([(U[i][z] - U[i][z+1]) * x[i, z]
                               for z in range(i)]) + U[i][i] * x[i, i]) +
                        2 * (sum([(U[j][z] - U[j][z+1]) * x[j, z]
                                  for z in range(j)]) + U[j][j] * x[j, j]) -
                        (sum([(U[k][z] - U[k][z+1]) * x[k, z]
                              for z in range(k)]) + U[k][k] * x[k, k]) >= 0
                        ).OnlyEnforceIf([x[k, k], x[j, j], x[i, i]])

    def add_constraint_monotonic_convex(self, model, n, U, x):
        for i in range(2, n):
            for j in range(1, i):
                for k in range(j):
                    model.Add(
                        (sum([(U[i][z] - U[i][z+1]) * x[i, z]
                              for z in range(i)]) + U[i][i] * x[i, i]) -
                        2 * (sum([(U[j][z] - U[j][z+1]) * x[j, z]
                                  for z in range(j)]) + U[j][j] * x[j, j]) +
                        (sum([(U[k][z] - U[k][z+1]) * x[k, z]
                              for z in range(k)]) + U[k][k] * x[k, k]) >= 0
                        ).OnlyEnforceIf([x[k, k], x[j, j], x[i, i]])

    def add_constraint_monotonic_peak(self, model, n, U, x, y):
        M = max([abs(U[i][j]) for i in range(n) for j in range(i + 1)])
        for i in range(1, n):
            for z in range(i):
                model.Add(
                    M * (y[i] + y[z]) + U[z][z] * x[z, z] +
                    sum([(U[z][j] - U[z][j+1]) * x[z, j]
                         for j in range(z)]) -
                    sum([(U[i][j] - U[i][j + 1]) * x[i, j]
                         for j in range(i)]) -
                    U[i][i] * x[i, i] >= 0).OnlyEnforceIf([x[z, z], x[i, i]])

                model.Add(
                    M * (2 - y[i] - y[z]) + U[i][i] * x[i, i] +
                    sum([(U[i][j] - U[i][j + 1]) * x[i, j]
                         for j in range(i)]) -
                    sum([(U[z][j] - U[z][j+1]) * x[z, j]
                         for j in range(z)]) -
                    U[z][z] * x[z, z] >= 0).OnlyEnforceIf([x[z, z], x[i, i]])

    def add_constraint_monotonic_valley(self, model, n, U, x, y):
        M = max([abs(U[i][j]) for i in range(n) for j in range(i + 1)])
        for i in range(1, n):
            for z in range(i):
                model.Add(
                    M * (y[i] + y[z]) + U[i][i] * x[i, i] +
                    sum([(U[i][j] - U[i][j + 1]) * x[i, j]
                         for j in range(i)]) -
                    sum([(U[z][j] - U[z][j+1]) * x[z, j]
                         for j in range(z)]) -
                    U[z][z] * x[z, z] >= 0).OnlyEnforceIf([x[z, z], x[i, i]])

                model.Add(
                    M * (2 - y[i] - y[z]) + U[z][z] * x[z, z] +
                    sum([(U[z][j] - U[z][j+1]) * x[z, j]
                         for j in range(z)]) -
                    sum([(U[i][j] - U[i][j + 1]) * x[i, j]
                         for j in range(i)]) -
                    U[i][i] * x[i, i] >= 0).OnlyEnforceIf([x[z, z], x[i, i]])

    def add_constraint_monotonic_peak_heuristic(self, model, n, U, x, tc, M):
        min_mean_diff = int(M * self.min_mean_diff)
        for i in range(1, tc):
            for z in range(i):
                model.Add(
                    sum([(U[z][j] - U[z][j+1]) * x[z, j]
                         for j in range(z)]) +
                    U[z][z] * x[z, z] - U[i][i] * x[i, i] -
                    sum([(U[i][j] - U[i][j + 1]) * x[i, j]
                         for j in range(i)]) + min_mean_diff <= 0
                    ).OnlyEnforceIf([x[z, z], x[i, i]])

        for i in range(tc, n):
            for z in range(tc, i):
                model.Add(
                    sum([(U[i][j] - U[i][j + 1]) * x[i, j]
                         for j in range(i)]) + U[i][i] * x[i, i] -
                    U[z][z] * x[z, z] -
                    sum([(U[z][j] - U[z][j+1]) * x[z, j]
                         for j in range(z)]) + min_mean_diff <= 0
                    ).OnlyEnforceIf([x[z, z], x[i, i]])

    def add_constraint_monotonic_valley_heuristic(self, model, n, U, x, tc, M):
        min_mean_diff = int(M * self.min_mean_diff)
        for i in range(1, tc):
            for z in range(i):
                model.Add(
                    sum([(U[i][j] - U[i][j + 1]) * x[i, j]
                         for j in range(i)]) + U[i][i] * x[i, i] -
                    U[z][z] * x[z, z] -
                    sum([(U[z][j] - U[z][j+1]) * x[z, j]
                         for j in range(z)]) + min_mean_diff <= 0
                    ).OnlyEnforceIf([x[z, z], x[i, i]])

        for i in range(tc, n):
            for z in range(tc, i):
                model.Add(
                    sum([(U[z][j] - U[z][j+1]) * x[z, j]
                         for j in range(z)]) +
                    U[z][z] * x[z, z] - U[i][i] * x[i, i] -
                    sum([(U[i][j] - U[i][j + 1]) * x[i, j]
                         for j in range(i)]) + min_mean_diff <= 0
                    ).OnlyEnforceIf([x[z, z], x[i, i]])
