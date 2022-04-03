"""
Generalized assigment problem: solve constrained optimal 2D binning problem.
Constraint programming implementation.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2022

import numpy as np

from ortools.sat.python import cp_model

from .cp_2d import Binning2DCP


class ContinuousBinning2DCP(Binning2DCP):
    def __init__(self, monotonic_trend_x, monotonic_trend_y, min_n_bins,
                 max_n_bins, min_mean_diff_x, min_mean_diff_y, gamma, n_jobs,
                 time_limit):

        self.monotonic_trend_x = monotonic_trend_x
        self.monotonic_trend_y = monotonic_trend_y
        self.min_n_bins = min_n_bins
        self.max_n_bins = max_n_bins
        self.min_mean_diff_x = min_mean_diff_x
        self.min_mean_diff_y = min_mean_diff_y
        self.gamma = gamma

        self.n_jobs = n_jobs
        self.time_limit = time_limit        

        self.solver_ = None

        self._model = None
        self._x = None
        self._n_rectangles = None

    def build_model(self, n_grid, n_rectangles, cols, c, d_connected_x,
                    d_connected_y, mean, n_records):

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

            model.Maximize(sum([c[i] * x[i] for i in range(n_rectangles)]) +
                           regularization * (pmax - pmin))
        else:
            model.Maximize(sum([c[i] * x[i] for i in range(n_rectangles)]))

        # Constraint: unique assignment
        self.add_constraint_unique_assignment(model, x, n_grid, cols)

        # Constraint: min / max bins
        self.add_constraint_min_max_bins(model, n_rectangles, x, d)

        # Constraint: monotonicity
        self.add_constraint_monotonic(
            model, n_rectangles, x, mean, d_connected_x, d_connected_y,
            self.min_mean_diff_x, self.min_mean_diff_y)

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