"""
Generalized assigment problem: solve constrained optimal binning problem.
LocalSolver implementation.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import numpy as np

from .model_data import model_data

try:
    from localsolver import LocalSolver
    from localsolver import LSSolutionStatus
    LOCALSOLVER_AVAILABLE = True
except ImportError:
    LOCALSOLVER_AVAILABLE = False


class BinningLS:
    def __init__(self, monotonic_trend, min_n_bins, max_n_bins, min_bin_size,
                 max_bin_size, min_bin_n_event, max_bin_n_event,
                 min_bin_n_nonevent, max_bin_n_nonevent, min_event_rate_diff,
                 max_pvalue, max_pvalue_policy, user_splits_fixed, time_limit):

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
        self.user_splits_fixed = user_splits_fixed
        self.time_limit = time_limit

        self.solver_ = None

        self._n = None
        self._x = None

    def build_model(self, divergence, n_nonevent, n_event, trend_change):
        # Parameters
        M = int(1e6)
        D, V, NE, E, pvalue_violation_indices = model_data(
            divergence, n_nonevent, n_event, self.max_pvalue,
            self.max_pvalue_policy, M, True)

        n = len(n_nonevent)

        # Initialize model
        if not LOCALSOLVER_AVAILABLE:
            raise ImportError('Cannot import localsolver. Install LocalSolver '
                              'or choose another solver, options are "cp" and '
                              '"mip".')

        ls = LocalSolver()
        model = ls.model

        array_V = model.array(model.array(V[i][::-1]) for i in range(n))
        array_D = model.array(model.array(D[i][::-1]) for i in range(n))
        array_NE = model.array(model.array(NE[i][::-1]) for i in range(n))
        array_E = model.array(model.array(E[i][::-1]) for i in range(n))

        # Decision variables
        x = [model.bool() for i in range(n)]
        a = [model.int(0, n) for i in range(n)]
        z = [model.int(0, n) for i in range(n)]

        if self.monotonic_trend in ("peak", "valley"):
            y = [model.bool() for i in range(n)]
            t = model.int(0, n)

        if self.min_n_bins is not None and self.max_n_bins is not None:
            n_bin_diff = self.max_n_bins - self.min_n_bins

            # Range constraints auxiliary variables
            d = model.int(0, n_bin_diff)

        if self.min_bin_size is not None and self.max_bin_size is not None:
            bin_size_diff = self.max_bin_size - self.min_bin_size

            # Range constraints auxiliary variables
            u = [model.int(0, bin_size_diff) for i in range(n)]

        # Constraints: basic search space
        model.constraint(x[n-1] == 1)

        for i in range(n):
            model.constraint(a[i] == (a[i - 1] + 1) * (1 - x[i]))
            model.constraint(z[i] == a[i - 1] * x[i] * (1 - x[i - 1]))

        # Constraints: monotonicity
        if self.monotonic_trend == "ascending":
            self.add_constraint_monotonic_ascending(
                model, n, array_D, D, x, z, M)

        elif self.monotonic_trend == "descending":
            self.add_constraint_monotonic_descending(
                model, n, array_D, D, x, z, M)

        elif self.monotonic_trend == "concave":
            self.add_constraint_monotonic_concave(
                model, n, array_D, D, x, z, M)

        elif self.monotonic_trend == "convex":
            self.add_constraint_monotonic_convex(
                model, n, array_D, D, x, z, M)

        elif self.monotonic_trend in ("peak", "valley"):
            for i in range(n):
                model.constraint(t >= i - n * (1 - y[i]))
                model.constraint(t <= i + n * y[i])

            if self.monotonic_trend == "peak":
                self.add_constraint_monotonic_peak(
                    model, n, array_D, D, x, z, y, M)

            elif self.monotonic_trend == "valley":
                self.add_constraint_monotonic_valley(
                    model, n, array_D, D, x, z, y, M)

        elif self.monotonic_trend == "peak_heuristic":
            self.add_constraint_monotonic_peak_heuristic(
                model, n, array_D, D, x, z, trend_change, M)

        elif self.monotonic_trend == "valley_heuristic":
            self.add_constraint_monotonic_valley_heuristic(
                model, n, array_D, D, x, z, trend_change, M)

        # Constraint: min / max bins
        if self.min_n_bins is not None or self.max_n_bins is not None:
            total_bins = sum([x[i] for i in range(n)])

            if self.min_n_bins is not None and self.max_n_bins is not None:
                model.constraint(d + total_bins - self.max_n_bins == 0)
            elif self.min_n_bins is not None:
                model.constraint(total_bins >= self.min_n_bins)
            elif self.max_n_bins is not None:
                model.constraint(total_bins <= self.max_n_bins)

        # Constraint: min / max bin size
        if self.min_bin_size is not None or self.max_bin_size is not None:
            for i in range(n):
                bin_size = x[i] * (model.at(array_NE, i, z[i]) +
                                   model.at(array_E, i, z[i]))

                if (self.min_bin_size is not None and
                        self.max_bin_size is not None):
                    model.constraint(u[i] + bin_size -
                                     self.max_bin_size * x[i] == 0)
                    model.constraint(u[i] <= bin_size_diff * x[i])
                elif self.min_bin_size is not None:
                    model.constraint(bin_size >= self.min_bin_size * x[i])
                elif self.max_bin_size is not None:
                    model.constraint(bin_size <= self.max_bin_size * x[i])

        # Constraint: min / max n_nonevent per bin
        if (self.min_bin_n_nonevent is not None or
                self.max_bin_n_nonevent is not None):
            for i in range(n):
                bin_ne_size = x[i] * model.at(array_NE, i, z[i])

                if self.min_bin_n_nonevent is not None:
                    model.constraint(
                        bin_ne_size >= self.min_bin_n_nonevent * x[i])

                if self.max_bin_n_nonevent is not None:
                    model.constraint(
                        bin_ne_size <= self.max_bin_n_nonevent * x[i])

        # Constraint: min / max n_event per bin
        if (self.min_bin_n_event is not None or
                self.max_bin_n_event is not None):
            for i in range(n):
                bin_e_size = x[i] * model.at(array_E, i, z[i])

                if self.min_bin_n_event is not None:
                    model.constraint(bin_e_size >= self.min_bin_n_event * x[i])

                if self.max_bin_n_event is not None:
                    model.constraint(bin_e_size <= self.max_bin_n_event * x[i])

        # Constraint: fixed splits
        if self.user_splits_fixed is not None:
            for i in range(n - 1):
                if self.user_splits_fixed[i]:
                    model.constraint(x[i] == 1)

        # Objective function
        model.maximize(model.sum(x[i] * model.at(array_V, i, z[i])
                                 for i in range(n)))

        model.close()

        self.solver_ = ls
        self._x = x
        self._n = n

    def solve(self):
        self.solver_.param.time_limit = self.time_limit
        self.solver_.solve()

        status = self.solver_.solution.get_status()

        if status == LSSolutionStatus.OPTIMAL:
            status_name = "OPTIMAL"
        elif status == LSSolutionStatus.FEASIBLE:
            status_name = "FEASIBLE"
        elif status == LSSolutionStatus.INFEASIBLE:
            status_name = "INFEASIBLE"
        elif status == LSSolutionStatus.INCONSISTENT:
            status_name = "INCONSISTENT"

        if status_name in ("FEASIBLE", "OPTIMAL"):
            solution = np.array([self._x[i].value
                                 for i in range(self._n)]).astype(bool)
        else:
            solution = np.zeros(self._n).astype(bool)
            solution[-1] = True

        return status_name, solution

    def add_constraint_monotonic_ascending(self, model, n, DD, D, x, z, M):
        min_event_rate_diff = int(M * self.min_event_rate_diff)
        for i in range(1, n):
            for j in range(i):
                model.constraint(
                    x[i] * model.at(DD, i, z[i]) + M * (1 - x[i]) >=
                    x[j] * model.at(DD, j, z[j]) +
                    min_event_rate_diff * (x[i] + x[j] - 1))

        if self.min_event_rate_diff == 0:
            for i in range(n - 1):
                if D[i+1][i] - D[i+1][i+1] > 0:
                    model.constraint(x[i] == 0)

    def add_constraint_monotonic_descending(self, model, n, DD, D, x, z, M):
        min_event_rate_diff = int(M * self.min_event_rate_diff)
        for i in range(1, n):
            for j in range(i):
                model.constraint(
                    x[i] * model.at(DD, i, z[i]) +
                    min_event_rate_diff * (x[i] + x[j] - 1) <=
                    x[j] * model.at(DD, j, z[j]) + M * (1 - x[j]))

        if self.min_event_rate_diff == 0:
            for i in range(n - 1):
                if D[i+1][i] - D[i+1][i+1] < 0:
                    model.constraint(x[i] == 0)

    def add_constraint_monotonic_concave(self, model, n, DD, D, x, z, M):
        for i in range(2, n):
            for j in range(1, i):
                for k in range(j):
                    model.constraint(
                        - model.at(DD, i, z[i]) * x[i] +
                        2 * model.at(DD, j, z[j]) * x[j] -
                        model.at(DD, k, z[k]) * x[k] >=
                        M * (x[i] + x[j] + x[k] - 3))

    def add_constraint_monotonic_convex(self, model, n, DD, D, x, z, M):
        for i in range(2, n):
            for j in range(1, i):
                for k in range(j):
                    model.constraint(
                        model.at(DD, i, z[i]) * x[i] -
                        2 * model.at(DD, j, z[j]) * x[j] +
                        model.at(DD, k, z[k]) * x[k] >=
                        M * (x[i] + x[j] + x[k] - 3))

    def add_constraint_monotonic_peak(self, model, n, DD, D, x, z, y, M):
        for i in range(1, n):
            for j in range(i):
                model.constraint(
                    M * (y[i] + y[j]) + M +
                    (model.at(DD, j, z[j]) - M) * x[j] -
                    x[i] * model.at(DD, i, z[i]) >= 0)

                model.constraint(
                    M * (2 - y[i] - y[j]) + M +
                    (model.at(DD, i, z[i]) - M) * x[i] -
                    x[j] * model.at(DD, j, z[j]) >= 0)

    def add_constraint_monotonic_valley(self, model, n, DD, D, x, z, y, M):
        for i in range(1, n):
            for j in range(i):
                model.constraint(
                    M * (y[i] + y[j]) + M +
                    (model.at(DD, i, z[i]) - M) * x[i] -
                    x[j] * model.at(DD, j, z[j]) >= 0)

                model.constraint(
                    M * (2 - y[i] - y[j]) + M +
                    (model.at(DD, j, z[j]) - M) * x[j] -
                    x[i] * model.at(DD, i, z[i]) >= 0)

    def add_constraint_monotonic_peak_heuristic(
            self, model, n, DD, D, x, z, tc, M):

        min_event_rate_diff = int(M * self.min_event_rate_diff)
        for i in range(1, tc):
            for j in range(i):
                model.constraint(
                    x[i] * model.at(DD, i, z[i]) + M * (1 - x[i]) >=
                    x[j] * model.at(DD, j, z[j]) +
                    min_event_rate_diff * (x[i] + x[j] - 1))

        if self.min_event_rate_diff == 0:
            for i in range(tc - 1):
                if D[i+1][i] - D[i+1][i+1] > 0:
                    model.constraint(x[i] == 0)

        for i in range(tc, n):
            for j in range(tc, i):
                model.constraint(
                    x[i] * model.at(DD, i, z[i]) +
                    min_event_rate_diff * (x[i] + x[j] - 1) <=
                    x[j] * model.at(DD, j, z[j]) + M * (1 - x[j]))

        if self.min_event_rate_diff == 0:
            for i in range(tc, n - 1):
                if D[i+1][i] - D[i+1][i+1] < 0:
                    model.constraint(x[i] == 0)

    def add_constraint_monotonic_valley_heuristic(
            self, model, n, DD, D, x, z, tc, M):

        min_event_rate_diff = int(M * self.min_event_rate_diff)
        for i in range(1, tc):
            for j in range(i):
                model.constraint(
                    x[i] * model.at(DD, i, z[i]) +
                    min_event_rate_diff * (x[i] + x[j] - 1) <=
                    x[j] * model.at(DD, j, z[j]) + M * (1 - x[j]))

        if self.min_event_rate_diff == 0:
            for i in range(tc - 1):
                if D[i+1][i] - D[i+1][i+1] < 0:
                    model.constraint(x[i] == 0)

        for i in range(tc, n):
            for j in range(tc, i):
                model.constraint(
                    x[i] * model.at(DD, i, z[i]) + M * (1 - x[i]) >=
                    x[j] * model.at(DD, j, z[j]) +
                    min_event_rate_diff * (x[i] + x[j] - 1))

        if self.min_event_rate_diff == 0:
            for i in range(tc, n - 1):
                if D[i+1][i] - D[i+1][i+1] > 0:
                    model.constraint(x[i] == 0)
