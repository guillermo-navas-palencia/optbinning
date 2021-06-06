"""
Generalized assigment problem: solve constrained optimal 2D binning problem.
Constraint programming implementation.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import numpy as np

from ortools.sat.python import cp_model

from .cp_2d import Binning2DCP
from .model_data_2d import continuous_model_data


class ContinuousBinning2DCP(Binning2DCP):
    def __init__(self, monotonic_trend_h, monotonic_trend_v, min_n_bins,
                 max_n_bins, min_bin_size, max_bin_size, min_mean_diff_h,
                 min_mean_diff_v, gamma, time_limit):

        self.monotonic_trend_h = monotonic_trend_h
        self.monotonic_trend_v = monotonic_trend_v
        self.min_n_bins = min_n_bins
        self.max_n_bins = max_n_bins
        self.min_bin_size = min_bin_size
        self.max_bin_size = max_bin_size

        self.min_mean_diff_h = min_mean_diff_h
        self.min_mean_diff_v = min_mean_diff_v
        self.gamma = gamma

        self.time_limit = time_limit

        self.solver_ = None
        self.mean_ = None

        self._model = None
        self._x = None
        self._n_squares = None
        self._n_rectangles = None

    def build_model(self, n_records, sums):
        # Parameters
        scale = int(1e6)

        [n_squares, n_rectangles, columns, V, d_connected_h, d_connected_v,
         mean, sums, n_records] = continuous_model_data(
            n_records, sums, self.monotonic_trend_h,
            self.monotonic_trend_v, scale)

        # Initialize model
        model = cp_model.CpModel()

        # Decision variables
        x, d, u, bin_size_diff = self.decision_variables(model, n_rectangles)

        # Objective function
        if self.gamma:
            total_records = int(n_records.sum())
            regularization = int(np.ceil(scale * self.gamma / total_records))
            pmax = model.NewIntVar(0, total_records, "pmax")
            pmin = model.NewIntVar(0, total_records, "pmin")

            model.Minimize(sum([V[i] * x[i] for i in range(n_rectangles)]) +
                           regularization * (pmax - pmin))
        else:
            model.Minimize(sum([V[i] * x[i] for i in range(n_rectangles)]))

        # Constraint: unique assignment
        self.add_constraint_unique_assignment(model, x, n_squares, columns)

        # Constraint: min / max bins
        self.add_constraint_min_max_bins(model, n_rectangles, x, d)

        # Constraint: min / max bin size
        self.add_constraint_min_max_bin_size(model, n_rectangles, x, u,
                                             n_records, bin_size_diff)

        # Constraint: monotonicity
        invalid = (n_records == 0)
        self.add_constraint_monotonic(
            model, n_rectangles, x, invalid, mean, d_connected_h,
            d_connected_v, self.min_mean_diff_h, self.min_mean_diff_v)

        # Constraint: reduction of dominating bins
        if self.gamma:
            for i in range(n_rectangles):
                bin_size = n_records[i] * x[i]

                model.Add(pmin <= total_records * (1 - x[i]) + bin_size)
                model.Add(pmax >= bin_size)
                model.Add(pmin <= pmax)

        # Save data for post-processing
        self.mean_ = mean

        self._model = model
        self._x = x
        self._n_squares = n_squares
        self._n_rectangles = n_rectangles

    def decision_variables(self, model, n):
        x = {}
        for i in range(n):
            x[i] = model.NewBoolVar("x[{}]".format(i))

        d = None
        u = None
        bin_size_diff = None

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

        return x, d, u, bin_size_diff
