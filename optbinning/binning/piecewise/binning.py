"""
Optimal piecewise continuous binning algorithm for binary target.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import time

from sklearn.linear_model import LogisticRegression

from .base import BasePWBinning


class OptimalPWBinning(BasePWBinning):
    def __init__(self, name="", estimator=None, degree=1,
                 prebinning_method="cart", max_n_prebins=20,
                 min_prebin_size=0.05, min_n_bins=None, max_n_bins=None,
                 min_bin_size=None, max_bin_size=None, monotonic_trend="auto",
                 n_subsamples=10000, max_pvalue=None,
                 max_pvalue_policy="consecutive", outlier_detector=None,
                 outlier_params=None, user_splits=None, special_codes=None,
                 split_digits=None, solver="clp", time_limit=100,
                 random_state=None, verbose=False):

        super().__init__(name, estimator, degree, prebinning_method,
                         max_n_prebins, min_prebin_size, min_n_bins,
                         max_n_bins, min_bin_size, max_bin_size,
                         monotonic_trend, n_subsamples, max_pvalue,
                         max_pvalue_policy, outlier_detector, outlier_params,
                         user_splits, special_codes, split_digits, solver,
                         time_limit, random_state, verbose)

        # auxiliary
        self._n_event = None
        self._n_nonevent = None
        self._n_nonevent_missing = None
        self._n_event_missing = None
        self._n_nonevent_special = None
        self._n_event_special = None

    def _fit(self, x, y, check_input):
        time_init = time.perf_counter()

        self._n_samples = len(x)

        # Pre-processing
        time_preprocessing = time.perf_counter()

        [x_clean, y_clean, x_missing, y_missing, x_special, y_special,
         _, _, _, _, _, _, _] = self._fit_preprocessing(x, y, check_input)

        self._time_preprocessing = time.perf_counter() - time_preprocessing

        # Fit estimator and compute event_rate = P[Y=1, X=x]
        time_estimator = time.perf_counter()

        self.estimator.fit(x_clean.reshape(-1, 1), y_clean)
        event_rate = self.estimator.predict_proba(x_clean.reshape(-1, 1))[:, 1]

        self._time_estimator = time.perf_counter() - time_estimator

        # Fit optimal binning algorithm for continuous target. Use optimal
        # split points to compute optimal piecewise functions
        self._fit_binning(x_clean, y_clean, event_rate, 0, 1)

        # Post-processing
        self._time_postprocessing = 0

        self._time_total = time.perf_counter() - time_init

        self._is_fitted = True
