"""
Generalized assigment problem: solve constrained optimal binning problem.
Constraint programming implementation.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import numpy as np

from ortools.sat.python import cp_model

from .model_data import model_data
from .model_data import multiclass_model_data


class BinningCP:
    def __init__(self, monotonic_trend, min_n_bins, max_n_bins, min_bin_size,
                 max_bin_size, min_bin_n_event, max_bin_n_event,
                 min_bin_n_nonevent, max_bin_n_nonevent, min_event_rate_diff,
                 max_pvalue, max_pvalue_policy, gamma, user_splits_fixed,
                 time_limit):

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

        self.time_limit = time_limit

        self.solver_ = None

        self._model = None
        self._n = None
        self._x = None

    def build_model(self, divergence, n_nonevent, n_event, trend_change):
        # Parameters
        M = int(1e6)
        D, V, pvalue_violation_indices = model_data(divergence, n_nonevent,
                                                    n_event, self.max_pvalue,
                                                    self.max_pvalue_policy, M)

        n = len(n_nonevent)
        n_records = n_nonevent + n_event

        # Initialize model
        model = cp_model.CpModel()

        # Decision variables
        x, y, t, d, u, bin_size_diff = self.decision_variables(model, n)

        # Objective function
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

        # Constraint: min / max n_nonevent per bin
        if (self.min_bin_n_nonevent is not None or
                self.max_bin_n_nonevent is not None):
            for i in range(n):
                bin_ne_size = sum([n_nonevent[j] * x[i, j]
                                   for j in range(i + 1)])

                if self.min_bin_n_nonevent is not None:
                    model.Add(bin_ne_size >= self.min_bin_n_nonevent * x[i, i])

                if self.max_bin_n_nonevent is not None:
                    model.Add(bin_ne_size <= self.max_bin_n_nonevent * x[i, i])

        # Constraint: min / max n_event per bin
        if (self.min_bin_n_event is not None or
                self.max_bin_n_event is not None):
            for i in range(n):
                bin_e_size = sum([n_event[j] * x[i, j] for j in range(i + 1)])

                if self.min_bin_n_event is not None:
                    model.Add(bin_e_size >= self.min_bin_n_event * x[i, i])

                if self.max_bin_n_event is not None:
                    model.Add(bin_e_size <= self.max_bin_n_event * x[i, i])

        # Constraints: monotonicity
        if self.monotonic_trend == "ascending":
            self.add_constraint_monotonic_ascending(model, n, D, x, M)

        elif self.monotonic_trend == "descending":
            self.add_constraint_monotonic_descending(model, n, D, x, M)

        elif self.monotonic_trend == "concave":
            self.add_constraint_monotonic_concave(model, n, D, x, M)

        elif self.monotonic_trend == "convex":
            self.add_constraint_monotonic_convex(model, n, D, x, M)

        elif self.monotonic_trend in ("valley", "peak"):
            for i in range(n):
                model.Add(t >= i - n * (1 - y[i]))
                model.Add(t <= i + n * y[i])

            if self.monotonic_trend == "peak":
                self.add_constraint_monotonic_peak(model, n, D, x, y, M)
            else:
                self.add_constraint_monotonic_valley(model, n, D, x, y, M)

        elif self.monotonic_trend == "peak_heuristic":
            self.add_constraint_monotonic_peak_heuristic(
                model, n, D, x, trend_change, M)

        elif self.monotonic_trend == "valley_heuristic":
            self.add_constraint_monotonic_valley_heuristic(
                model, n, D, x, trend_change, M)

        # Constraint: reduction of dominating bins
        if self.gamma:
            for i in range(n):
                bin_size = sum([n_records[j] * x[i, j] for j in range(i + 1)])

                model.Add(pmin <= total_records * (1 - x[i, i]) + bin_size)
                model.Add(pmax >= bin_size)
            model.Add(pmin <= pmax)

        # Constraint: max-pvalue
        self.add_max_pvalue_constraint(model, x, pvalue_violation_indices)

        # Constraint: fixed splits
        self.add_constraint_fixed_splits(model, n, x)

        self._model = model
        self._x = x
        self._n = n

    def build_model_scenarios(self, n_nonevent, n_event, w):
        # Parameters
        M = int(1e6)
        D, V, pvalue_violation_indices = multiclass_model_data(
            n_nonevent, n_event, self.max_pvalue, self.max_pvalue_policy, M)

        n = len(n_nonevent)
        n_records = n_nonevent + n_event
        n_scenarios = len(w)

        if w is not None:
            sw = 10 ** np.abs(np.log10(np.min(w)))
            w = [np.int64(w[s] * sw) for s in range(n_scenarios)]

        # Initialize model
        model = cp_model.CpModel()

        # Decision variables
        x, y, t, d = self.decision_variables_scenarios(model, n)

        # Objective function
        model.Maximize(sum([w[s] * sum([(V[s][i][i] * x[i, i]) +
                            sum([(V[s][i][j] - V[s][i][j+1]) * x[i, j]
                                 for j in range(i)]) for i in range(n)])
                            for s in range(n_scenarios)]))

        # Constraint: unique assignment
        self.add_constraint_unique_assignment(model, n, x)

        # Constraint: continuity
        self.add_constraint_continuity(model, n, x)

        # Constraint: min / max bins
        self.add_constraint_min_max_bins(model, n, x, d)

        # Constraint: min / max bin size
        self.add_constraint_min_max_bin_size_scenarios(model, n, x, n_records)

        # Constraints: monotonicity
        if self.monotonic_trend == "ascending":
            for s in range(n_scenarios):
                self.add_constraint_monotonic_ascending(model, n, D[s], x, M)

        elif self.monotonic_trend == "descending":
            for s in range(n_scenarios):
                self.add_constraint_monotonic_descending(model, n, D[s], x, M)

        elif self.monotonic_trend == "concave":
            for s in range(n_scenarios):
                self.add_constraint_monotonic_concave(model, n, D[s], x, M)

        elif self.monotonic_trend == "convex":
            for s in range(n_scenarios):
                self.add_constraint_monotonic_convex(model, n, D[s], x, M)

        elif self.monotonic_trend in ("peak", "valley"):
            for i in range(n):
                model.Add(t >= i - n * (1 - y[i]))
                model.Add(t <= i + n * y[i])

            if self.monotonic_trend == "peak":
                for s in range(n_scenarios):
                    self.add_constraint_monotonic_peak(
                        model, n, D[s], x, y, M)
            else:
                for s in range(n_scenarios):
                    self.add_constraint_monotonic_valley(
                        model, n, D[s], x, y, M)

        # Constraint: max-pvalue
        for s in range(n_scenarios):
            self.add_max_pvalue_constraint(model, x,
                                           pvalue_violation_indices[s])

        # Constraint: fixed splits
        self.add_constraint_fixed_splits(model, n, x)

        self._model = model
        self._x = x
        self._n = n

    def solve(self):
        self.solver_ = cp_model.CpSolver()
        self.solver_.parameters.max_time_in_seconds = self.time_limit

        status = self.solver_.Solve(self._model)
        status_name = self.solver_.StatusName(status)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            solution = np.array([self.solver_.BooleanValue(self._x[i, i])
                                 for i in range(self._n)]).astype(bool)
        else:
            solution = np.zeros(self._n).astype(bool)
            solution[-1] = True

        return status_name, solution

    def decision_variables(self, model, n):
        x = {}
        for i in range(n):
            for j in range(i + 1):
                x[i, j] = model.NewBoolVar("x[{}, {}]".format(i, j))

        y = None
        t = None
        d = None
        u = None
        bin_size_diff = None

        if self.monotonic_trend in ("peak", "valley"):
            # Auxiliary binary variables
            y = {}
            for i in range(n):
                y[i] = model.NewBoolVar("y[{}]".format(i))

            # Change point
            t = model.NewIntVar(0, n, "t")

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

    def decision_variables_scenarios(self, model, n):
        x = {}
        for i in range(n):
            for j in range(i + 1):
                x[i, j] = model.NewBoolVar("x[{}, {}]".format(i, j))

        y = None
        t = None
        d = None

        if self.monotonic_trend in ("peak", "valley"):
            # Auxiliary binary variables
            y = {}
            for i in range(n):
                y[i] = model.NewBoolVar("y[{}]".format(i))

            # Change point
            t = model.NewIntVar(0, n, "t")

        if self.min_n_bins is not None and self.max_n_bins is not None:
            n_bin_diff = self.max_n_bins - self.min_n_bins

            # Range constraints auxiliary variables
            d = model.NewIntVar(0, n_bin_diff, "n_bin_diff")

        return x, y, t, d

    def add_constraint_unique_assignment(self, model, n, x):
        for j in range(n):
            model.Add(sum([x[i, j] for i in range(j, n)]) == 1)

    def add_constraint_continuity(self, model, n, x):
        for i in range(n):
            for j in range(i):
                model.Add(x[i, j] - x[i, j+1] <= 0)

    def add_constraint_min_max_bins(self, model, n, x, d):
        if self.min_n_bins is not None or self.max_n_bins is not None:
            trace = sum([x[i, i] for i in range(n)])

            if self.min_n_bins is not None and self.max_n_bins is not None:
                model.Add(d + trace - self.max_n_bins == 0)
            elif self.min_n_bins is not None:
                model.Add(trace >= self.min_n_bins)
            elif self.max_n_bins is not None:
                model.Add(trace <= self.max_n_bins)

    def add_constraint_min_max_bin_size(self, model, n, x, u, n_records,
                                        bin_size_diff):
        if self.min_bin_size is not None or self.max_bin_size is not None:
            for i in range(n):
                bin_size = sum([n_records[j] * x[i, j] for j in range(i + 1)])

                if (self.min_bin_size is not None and
                        self.max_bin_size is not None):
                    model.Add(u[i] + bin_size -
                              self.max_bin_size * x[i, i] == 0)
                    model.Add(u[i] <= bin_size_diff * x[i, i])
                elif self.min_bin_size is not None:
                    model.Add(bin_size >= self.min_bin_size * x[i, i])
                elif self.max_bin_size is not None:
                    model.Add(bin_size <= self.max_bin_size * x[i, i])

    def add_constraint_min_max_bin_size_scenarios(self, model, n, x,
                                                  n_records):
        if self.min_bin_size is not None or self.max_bin_size is not None:
            n_scenarios = n_records.shape[1]
            for s in range(n_scenarios):
                for i in range(n):
                    bin_size = sum([n_records[j, s] * x[i, j]
                                    for j in range(i + 1)])

                    if self.min_bin_size is not None:
                        model.Add(bin_size >= self.min_bin_size[s] * x[i, i])
                    if self.max_bin_size is not None:
                        model.Add(bin_size <= self.max_bin_size[s] * x[i, i])

    def add_constraint_monotonic_ascending(self, model, n, D, x, M):
        min_event_rate_diff = int(M * self.min_event_rate_diff)
        for i in range(1, n):
            for z in range(i):
                model.Add(
                    sum([(D[z][j] - D[z][j+1]) * x[z, j]
                         for j in range(z)]) +
                    D[z][z] * x[z, z] - M - (D[i][i] - M) * x[i, i] -
                    sum([(D[i][j] - D[i][j + 1]) * x[i, j]
                         for j in range(i)]) +
                    min_event_rate_diff * (x[i, i] + x[z, z] - 1) <= 0)

        # Preprocessing
        if self.min_event_rate_diff == 0:
            for i in range(n - 1):
                if D[i+1][i] - D[i+1][i+1] > 0:
                    model.Add(x[i, i] == 0)
                    for j in range(n - i - 1):
                        if D[i+1+j][i] - D[i+1+j][i+1+j] > 0:
                            model.Add(x[i+j, i+j] == 0)

    def add_constraint_monotonic_descending(self, model, n, D, x, M):
        min_event_rate_diff = int(M * self.min_event_rate_diff)
        for i in range(1, n):
            for z in range(i):
                model.Add(
                    sum([(D[i][j] - D[i][j + 1]) * x[i, j]
                         for j in range(i)]) + D[i][i] * x[i, i] -
                    M - (D[z][z] - M) * x[z, z] -
                    sum([(D[z][j] - D[z][j+1]) * x[z, j]
                         for j in range(z)]) +
                    min_event_rate_diff * (x[i, i] + x[z, z] - 1) <= 0)

        # Preprocessing
        if self.min_event_rate_diff == 0:
            for i in range(n - 1):
                if D[i+1][i] - D[i+1][i+1] < 0:
                    model.Add(x[i, i] == 0)
                    for j in range(n - i - 1):
                        if D[i+1+j][i] - D[i+1+j][i+1+j] < 0:
                            model.Add(x[i+j, i+j] == 0)

    def add_constraint_monotonic_concave(self, model, n, D, x, M):
        for i in range(2, n):
            for j in range(1, i):
                for k in range(j):
                    model.Add(
                        -(sum([(D[i][z] - D[i][z+1]) * x[i, z]
                               for z in range(i)]) + D[i][i] * x[i, i]) +
                        2 * (sum([(D[j][z] - D[j][z+1]) * x[j, z]
                                  for z in range(j)]) + D[j][j] * x[j, j]) -
                        (sum([(D[k][z] - D[k][z+1]) * x[k, z]
                              for z in range(k)]) + D[k][k] * x[k, k]) >=
                        M * (x[i, i] + x[j, j] + x[k, k] - 3))

    def add_constraint_monotonic_convex(self, model, n, D, x, M):
        for i in range(2, n):
            for j in range(1, i):
                for k in range(j):
                    model.Add(
                        (sum([(D[i][z] - D[i][z+1]) * x[i, z]
                              for z in range(i)]) + D[i][i] * x[i, i]) -
                        2 * (sum([(D[j][z] - D[j][z+1]) * x[j, z]
                                  for z in range(j)]) + D[j][j] * x[j, j]) +
                        (sum([(D[k][z] - D[k][z+1]) * x[k, z]
                              for z in range(k)]) + D[k][k] * x[k, k]) >=
                        M * (x[i, i] + x[j, j] + x[k, k] - 3))

    def add_constraint_monotonic_peak(self, model, n, D, x, y, M):
        for i in range(1, n):
            for z in range(i):
                model.Add(
                    M * (y[i] + y[z]) + M + (D[z][z] - M) * x[z, z] +
                    sum([(D[z][j] - D[z][j+1]) * x[z, j]
                         for j in range(z)]) -
                    sum([(D[i][j] - D[i][j + 1]) * x[i, j]
                         for j in range(i)]) -
                    D[i][i] * x[i, i] >= 0)

                model.Add(
                    M * (2 - y[i] - y[z]) + M + (D[i][i] - M) * x[i, i] +
                    sum([(D[i][j] - D[i][j + 1]) * x[i, j]
                         for j in range(i)]) -
                    sum([(D[z][j] - D[z][j+1]) * x[z, j]
                         for j in range(z)]) -
                    D[z][z] * x[z, z] >= 0)

    def add_constraint_monotonic_valley(self, model, n, D, x, y, M):
        for i in range(1, n):
            for z in range(i):
                model.Add(
                    M * (y[i] + y[z]) + M + (D[i][i] - M) * x[i, i] +
                    sum([(D[i][j] - D[i][j + 1]) * x[i, j]
                         for j in range(i)]) -
                    sum([(D[z][j] - D[z][j+1]) * x[z, j]
                         for j in range(z)]) -
                    D[z][z] * x[z, z] >= 0)

                model.Add(
                    M * (2 - y[i] - y[z]) + M + (D[z][z] - M) * x[z, z] +
                    sum([(D[z][j] - D[z][j+1]) * x[z, j]
                         for j in range(z)]) -
                    sum([(D[i][j] - D[i][j + 1]) * x[i, j]
                         for j in range(i)]) -
                    D[i][i] * x[i, i] >= 0)

    def add_constraint_monotonic_peak_heuristic(self, model, n, D, x, tc, M):
        min_event_rate_diff = int(M * self.min_event_rate_diff)
        for i in range(1, tc):
            for z in range(i):
                model.Add(
                    sum([(D[z][j] - D[z][j+1]) * x[z, j]
                         for j in range(z)]) +
                    D[z][z] * x[z, z] - M - (D[i][i] - M) * x[i, i] -
                    sum([(D[i][j] - D[i][j + 1]) * x[i, j]
                         for j in range(i)]) +
                    min_event_rate_diff * (x[i, i] + x[z, z] - 1) <= 0)

        # Preprocessing
        if self.min_event_rate_diff == 0:
            for i in range(tc - 1):
                if D[i+1][i] - D[i+1][i+1] > 0:
                    model.Add(x[i, i] == 0)
                    for j in range(tc - i - 1):
                        if D[i+1+j][i] - D[i+1+j][i+1+j] > 0:
                            model.Add(x[i+j, i+j] == 0)

        for i in range(tc, n):
            for z in range(tc, i):
                model.Add(
                    sum([(D[i][j] - D[i][j + 1]) * x[i, j]
                         for j in range(i)]) + D[i][i] * x[i, i] -
                    M - (D[z][z] - M) * x[z, z] -
                    sum([(D[z][j] - D[z][j+1]) * x[z, j]
                         for j in range(z)]) +
                    min_event_rate_diff * (x[i, i] + x[z, z] - 1) <= 0)

        # Preprocessing
        if self.min_event_rate_diff == 0:
            for i in range(tc, n - 1):
                if D[i+1][i] - D[i+1][i+1] < 0:
                    model.Add(x[i, i] == 0)
                    for j in range(tc, n - i - 1):
                        if D[i+1+j][i] - D[i+1+j][i+1+j] < 0:
                            model.Add(x[i+j, i+j] == 0)

    def add_constraint_monotonic_valley_heuristic(self, model, n, D, x, tc, M):
        min_event_rate_diff = int(M * self.min_event_rate_diff)
        for i in range(1, tc):
            for z in range(i):
                model.Add(
                    sum([(D[i][j] - D[i][j + 1]) * x[i, j]
                         for j in range(i)]) + D[i][i] * x[i, i] -
                    M - (D[z][z] - M) * x[z, z] -
                    sum([(D[z][j] - D[z][j+1]) * x[z, j]
                         for j in range(z)]) +
                    min_event_rate_diff * (x[i, i] + x[z, z] - 1) <= 0)

        # Preprocessing
        if self.min_event_rate_diff == 0:
            for i in range(tc - 1):
                if D[i+1][i] - D[i+1][i+1] < 0:
                    model.Add(x[i, i] == 0)
                    for j in range(tc - i - 1):
                        if D[i+1+j][i] - D[i+1+j][i+1+j] < 0:
                            model.Add(x[i+j, i+j] == 0)

        for i in range(tc, n):
            for z in range(tc, i):
                model.Add(
                    sum([(D[z][j] - D[z][j+1]) * x[z, j]
                         for j in range(z)]) +
                    D[z][z] * x[z, z] - M - (D[i][i] - M) * x[i, i] -
                    sum([(D[i][j] - D[i][j + 1]) * x[i, j]
                         for j in range(i)]) +
                    min_event_rate_diff * (x[i, i] + x[z, z] - 1) <= 0)

        # Preprocessing
        if self.min_event_rate_diff == 0:
            for i in range(tc, n - 1):
                if D[i+1][i] - D[i+1][i+1] > 0:
                    model.Add(x[i, i] == 0)
                    for j in range(tc, n - i - 1):
                        if D[i+1+j][i] - D[i+1+j][i+1+j] > 0:
                            model.Add(x[i+j, i+j] == 0)

    def add_max_pvalue_constraint(self, model, x, pvalue_violation_indices):
        for ind1, ind2 in pvalue_violation_indices:
            model.AddImplication(x[ind1[0], ind1[1]],
                                 x[ind2[0], ind2[1]].Not())

    def add_constraint_fixed_splits(self, model, n, x):
        if self.user_splits_fixed is not None:
            for i in range(n - 1):
                if self.user_splits_fixed[i]:
                    model.Add(x[i, i] == 1)
