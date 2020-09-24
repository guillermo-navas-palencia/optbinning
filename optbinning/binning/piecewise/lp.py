"""
Polynomial function optimization via Linear programming implementation.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import numpy as np

from ortools.linear_solver import pywraplp


class PWPBinningLP:
    def __init__(self, degree, monotonic_trend, lb, ub, lp_solver, time_limit):
        self.degree = degree
        self.monotonic_trend = monotonic_trend
        self.lb = lb
        self.ub = ub
        self.lp_solver = lp_solver
        self.time_limit = time_limit

        self.solver_ = None

        self._n_bins = None
        self._c = None

    def build_model(self, splits, x_subsamples, x_indices, pred_subsamples,
                    trend_change=None):
        # Parameters
        n_subsamples = len(x_subsamples)
        n_splits = len(splits)
        n_bins = n_splits + 1
        order = int(self.degree) + 1

        # Initialize solver
        if self.lp_solver == "glop":
            solver = pywraplp.Solver(
                'PWPBinningLP', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
        elif self.lp_solver == "clp":
            solver = pywraplp.Solver(
                'PWPBinningLP', pywraplp.Solver.CLP_LINEAR_PROGRAMMING)

        # Decision variables
        c = {}
        tp = {}
        tn = {}

        # General case
        for i in range(n_bins):
            for j in range(order):
                c[i, j] = solver.NumVar(-np.inf, np.inf,
                                        name="c[{}, {}]".format(i, j))

        for i in range(n_subsamples):
            tp[i] = solver.NumVar(0, np.inf, name="tp[{}]".format(i))
            tn[i] = solver.NumVar(0, np.inf, name="tn[{}]".format(i))

        # Objective function
        solver.Minimize(solver.Sum([tp[i] + tn[i]
                                    for i in range(n_subsamples)]))

        # Constraint: absolute value and bounds
        if self.lb is not None and self.ub is not None:
            bound_diff = self.ub - self.lb
            d = {}
            for i in range(n_subsamples):
                d[i] = solver.NumVar(0, bound_diff, name="d[{}]".format(i))

        for i in range(n_bins):
            for j in x_indices[i]:
                poly = self.polyeval_unroll(c, x_subsamples[j], i, order)
                solver.Add(tp[j] - tn[j] == pred_subsamples[j] - poly)

                if self.lb is not None and self.ub is not None:
                    solver.Add(d[j] + poly == self.ub)
                elif self.lb is not None:
                    solver.Add(poly >= self.lb)
                elif self.ub is not None:
                    solver.Add(poly <= self.ub)

        # Constraint: continuity
        for i in range(n_splits):
            solver.Add(self.polyeval_unroll(c, splits[i], i, order) ==
                       self.polyeval_unroll(c, splits[i], i + 1, order))

        # Constraints: monotonicity
        if order == 1:
            if self.monotonic_trend == "ascending":
                for i in range(n_bins - 1):
                    solver.Add(c[i, order - 1] <= c[i + 1, order - 1])

            elif self.monotonic_trend == "descending":
                for i in range(n_bins - 1):
                    solver.Add(c[i, order - 1] >= c[i + 1, order - 1])

            elif self.monotonicity == "concave":
                pass

            elif self.monotonicity == "convex":
                pass

        elif order == 2:
            if self.monotonic_trend == "ascending":
                for i in range(n_bins):
                    solver.Add(c[i, order - 1] >= 0)

            elif self.monotonic_trend == "descending":
                for i in range(n_bins):
                    solver.Add(c[i, order - 1] <= 0)

            elif self.monotonic_trend == "concave":
                for i in range(n_bins - 1):
                    solver.Add(c[i + 1, order - 1] <= c[i, order - 1])

            elif self.monotonic_trend == "convex":
                for i in range(n_bins - 1):
                    solver.Add(c[i + 1, order - 1] >= c[i, order - 1])

        elif order > 2:
            if self.monotonic_trend == "ascending":
                self.add_constraint_monotonic_ascending(
                    solver, n_bins, order, c, x_indices, x_subsamples)

            elif self.monotonic_trend == "descending":
                self.add_constraint_monotonic_descending(
                    solver, n_bins, order, c, x_indices, x_subsamples)

        self.solver_ = solver
        self._n_bins = n_bins
        self._c = c

    def solve(self):
        self.solver_.SetTimeLimit(self.time_limit * 1000)
        self.solver_.EnableOutput()
        status = self.solver_.Solve()

        if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            if status == pywraplp.Solver.OPTIMAL:
                status_name = "OPTIMAL"
            else:
                status_name = "FEASIBLE"

            cc = np.zeros((self._n_bins, self.degree + 1))

            for i in range(self._n_bins):
                for j in range(self.degree + 1):
                    cc[i, j] = self._c[i, j].solution_value()
        else:
            if status == pywraplp.Solver.ABNORMAL:
                status_name = "ABNORMAL"
            elif status == pywraplp.Solver.INFEASIBLE:
                status_name = "INFEASIBLE"
            elif status == pywraplp.Solver.UNBOUNDED:
                status_name = "UNBOUNDED"
            else:
                status_name = "UNKNOWN"

            cc = np.zeros((self._n_bins, self.degree + 1))

        return status_name, cc

    def add_constraint_monotonic_ascending(self, solver, n_bins, order, c,
                                           x_indices, x_subsamples):
        for i in range(n_bins):
            for j in x_indices[i]:
                solver.Add(self.polyeval_d_unroll(
                    c, x_subsamples[j], i, order) >= 0)

    def add_constraint_monotonic_descending(self, solver, n_bins, order, c,
                                            x_indices, x_subsamples):
        for i in range(n_bins):
            for j in x_indices[i]:
                solver.Add(self.polyeval_d_unroll(
                    c, x_subsamples[j], i, order) <= 0)

    def polyeval_unroll(self, c, x, i, order):
        if order == 1:
            return c[i, 0]
        elif order == 2:
            return c[i, 0] + c[i, 1] * x
        elif order == 3:
            return (c[i, 2] * x + c[i, 1]) * x + c[i, 0]
        elif order == 4:
            return ((c[i, 3] * x + c[i, 2]) * x + c[i, 1]) * x + c[i, 0]
        elif order == 5:
            return (((c[i, 4] * x + c[i, 3]) * x + c[i, 2]) * x +
                    c[i, 1]) * x + c[i, 0]

    def polyeval_d_unroll(self, c, x, i, order):
        if order == 1:
            return 0
        elif order == 2:
            return c[i, 1]
        elif order == 3:
            return c[i, 2] * 2 * x + c[i, 1]
        elif order == 4:
            return (c[i, 3] * 3 * x + c[i, 2] * 2) * x + c[i, 1]
        elif order == 5:
            return (((c[i, 4] * 4 * x + c[i, 3] * 3) * x + c[i, 2] * 2) * x +
                    c[i, 1])
