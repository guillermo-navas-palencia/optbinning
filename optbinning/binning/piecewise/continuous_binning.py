"""
Optimal piecewise binning for continuous target.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import time

import numpy as np

from sklearn.linear_model import LinearRegression

from .base import BasePWBinning
from .binning_statistics import PWContinuousBinningTable
from .transformations import transform_continuous_target


class ContinuousOptimalPWBinning(BasePWBinning):
    def __init__(self, name="", estimator=None, degree=1, continuity=True,
                 prebinning_method="cart", max_n_prebins=20,
                 min_prebin_size=0.05, min_n_bins=None, max_n_bins=None,
                 min_bin_size=None, max_bin_size=None, monotonic_trend="auto",
                 n_subsamples=10000, max_pvalue=None,
                 max_pvalue_policy="consecutive", outlier_detector=None,
                 outlier_params=None, user_splits=None, user_splits_fixed=None,
                 special_codes=None, split_digits=None, solver="clp",
                 time_limit=100, random_state=None, verbose=False):

        super().__init__(name, estimator, degree, continuity,
                         prebinning_method, max_n_prebins, min_prebin_size,
                         min_n_bins, max_n_bins, min_bin_size, max_bin_size,
                         monotonic_trend, n_subsamples, max_pvalue,
                         max_pvalue_policy, outlier_detector, outlier_params,
                         user_splits, user_splits_fixed, special_codes,
                         split_digits, solver, time_limit, random_state,
                         verbose)

        self._problem_type = "regression"

    def fit_transform(self, x, y, metric_special=0, metric_missing=0,
                      lb=None, ub=None, check_input=False):

        return self.fit(x, y, check_input).transform(
            x, metric_special, metric_missing, lb, ub, check_input)

    def transform(self, x, metric_special=0, metric_missing=0,
                  lb=None, ub=None, check_input=False):

        self._check_is_fitted()

        return transform_continuous_target(self._optb.splits, x, self._c,
                                           lb, ub, metric_special,
                                           metric_missing, check_input)

    def _fit(self, x, y, check_input):
        time_init = time.perf_counter()

        self._n_samples = len(x)

        # Pre-processing
        time_preprocessing = time.perf_counter()

        [x_clean, y_clean, x_missing, y_missing, x_special, y_special,
         _, _, _, _, _, _, _] = self._fit_preprocessing(x, y, check_input)

        self._time_preprocessing = time.perf_counter() - time_preprocessing

        # Fit estimator
        time_estimator = time.perf_counter()

        if self.estimator is None:
            self.estimator = LinearRegression()

        mean = y_clean

        self._time_estimator = time.perf_counter() - time_estimator

        # Fit optimal binning algorithm for continuous target. Use optimal
        # split points to compute optimal piecewise functions
        self._fit_binning(x_clean, y_clean, mean, None, None)

        time_postprocessing = time.perf_counter()

        bt = self._optb.binning_table.build(add_totals=False)

        n_records = bt["Count"].values

        self._binning_table = PWContinuousBinningTable(
            self.name, self._optb.splits, self._c, n_records, None, None, None,
            None, None, None, None, x_clean.min(), x_clean.max(), None)

        self._time_postprocessing = time.perf_counter() - time_postprocessing

        # Completed successfully
        self._class_logger.close()
        self._is_fitted = True

        return self
