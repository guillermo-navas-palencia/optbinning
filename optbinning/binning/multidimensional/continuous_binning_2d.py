"""
Optimal binning 2D algorithm for continuous target.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import numbers
import time

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib as mpl

from ..logging import Logger
from .binning_2d import OptimalBinning2D
from .binning_information import print_binning_information
from .continuous_cp_2d import ContinuousBinning2DCP


class ContinuousOptimalBinning2D(OptimalBinning2D):
    def __init__(self, name="", dtype_x="numerical", dtype_y="numerical",
                 prebinning_method="cart", max_n_prebins=20,
                 min_prebin_size=0.05, min_n_bins=None, max_n_bins=None,
                 min_bin_size=None, max_bin_size=None, monotonic_trend_x=None,
                 monotonic_trend_y=None, min_mean_diff_x=0,
                 min_mean_diff_y=0, gamma=0, user_splits_x=None,
                 user_splits_y=None, special_codes=None,
                 split_digits=None, time_limit=100, verbose=False):

        self.name = name
        self.dtype_x = dtype_x
        self.dtype_y = dtype_y
        self.prebinning_method = prebinning_method
        self.solver = "cp"

        self.max_n_prebins = max_n_prebins
        self.min_prebin_size = min_prebin_size

        self.min_n_bins = min_n_bins
        self.max_n_bins = max_n_bins
        self.min_bin_size = min_bin_size
        self.max_bin_size = max_bin_size

        self.monotonic_trend_x = monotonic_trend_x
        self.monotonic_trend_y = monotonic_trend_y
        self.min_mean_diff_x = min_mean_diff_x
        self.min_mean_diff_y = min_mean_diff_y
        self.gamma = gamma

        self.user_splits_x = user_splits_x
        self.user_splits_y = user_splits_y
        self.special_codes = special_codes
        self.split_digits = split_digits

        self.time_limit = time_limit

        self.verbose = verbose

        # auxiliary

        # info

        # timing
        self._time_total = None
        self._time_preprocessing = None
        self._time_prebinning = None
        self._time_solver = None
        self._time_postprocessing = None

        # logger
        self._class_logger = Logger(__name__)
        self._logger = self._class_logger.logger

        self._is_fitted = False

    def fit(self, x, y, z, check_input=False):
        return self._fit(x, y, z, check_input)

    def _fit(self, x, y, z, check_input):
        pass

    def _fit_from_matrices(self, R, S):
        time_init = time.perf_counter()

        self._m, self._n = R.shape

        optimizer = ContinuousBinning2DCP(
            self.monotonic_trend_x, self.monotonic_trend_y, self.min_n_bins,
            self.max_n_bins, self.min_bin_size, self.max_bin_size,
            self.min_mean_diff_x, self.min_mean_diff_y, self.gamma,
            self.time_limit)

        optimizer.build_model(R, S)
        status, solution = optimizer.solve()

        self._solution = solution

        self._optimizer = optimizer
        self._status = status

        self._time_solver = time.perf_counter() - time_init
        self._time_preprocessing = 0
        self._time_prebinning = 0
        self._time_postprocessing = 0
        self._time_total = self._time_solver

        self._n_prebins = R.size
        self._n_refinements = 0

        self._is_fitted = True

    def iv(self):
        iv = self._optimizer.iv_
        return iv[self._solution].sum()

    def _solution_matrices(self):
        mean = self._optimizer.mean_
        m = self._m
        n = self._n

        selected_rows = []
        counter = 0
        for i in range(m):
            for j in range(n):
                for k in range(i + 1, m + 1):
                    for l in range(j + 1, n + 1):
                        if self._solution[counter]:
                            row = [n * ik + jl for ik in range(i, k)
                                   for jl in range(j, l)]
                            selected_rows.append(row)
                        counter += 1

        DD = np.empty(m * n)
        S = np.empty(m * n)
        for i, r in enumerate(selected_rows):
            DD[r] = mean[self._solution][i]
            S[r] = i

        return DD.reshape((m, n)), S.reshape((m, n))
