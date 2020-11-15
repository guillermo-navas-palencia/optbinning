"""
Optimal piecewise binning algorithm for binary target.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import time

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.metrics import brier_score_loss

from ...binning.binning_statistics import target_info
from ...scorecard.metrics import gini
from .base import _check_parameters
from .base import BasePWBinning
from .binning_statistics import PWBinningTable
from .transformations import transform_binary_target


class OptimalPWBinning(BasePWBinning):
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

        self._problem_type = "classification"

    def fit_transform(self, x, y, metric="woe", metric_special=0,
                      metric_missing=0, check_input=False):

        return self.fit(x, y, check_input).transform(
            x, metric, metric_special, metric_missing, check_input)

    def transform(self, x, metric="woe", metric_special=0, metric_missing=0,
                  check_input=False):

        self._check_is_fitted()

        lb = 1e-8
        ub = 1.0 - lb

        return transform_binary_target(self._optb.splits, x, self._c, lb, ub,
                                       self._t_n_nonevent, self._t_n_event,
                                       self._n_nonevent_special,
                                       self._n_event_special,
                                       self._n_nonevent_missing,
                                       self._n_event_missing,
                                       self.special_codes, metric,
                                       metric_special, metric_missing,
                                       check_input)

    def _fit(self, x, y, check_input):
        time_init = time.perf_counter()

        if self.verbose:
            self._logger.info("Optimal piecewise binning started.")
            self._logger.info("Options: check parameters.")

        _check_parameters(**self.get_params(deep=False),
                          problem_type=self._problem_type)

        # Pre-processing
        if self.verbose:
            self._logger.info("Pre-processing started.")

        self._n_samples = len(x)

        if self.verbose:
            self._logger.info("Pre-processing: number of samples: {}"
                              .format(self._n_samples))

        time_preprocessing = time.perf_counter()

        [x_clean, y_clean, x_missing, y_missing, x_special, y_special,
         _, _, _, _, _, _, _] = self._fit_preprocessing(x, y, check_input)

        self._time_preprocessing = time.perf_counter() - time_preprocessing

        if self.verbose:
            n_clean = len(x_clean)
            n_missing = len(x_missing)
            n_special = len(x_special)

            self._logger.info("Pre-processing: number of clean samples: {}"
                              .format(n_clean))

            self._logger.info("Pre-processing: number of missing samples: {}"
                              .format(n_missing))

            self._logger.info("Pre-processing: number of special samples: {}"
                              .format(n_special))

            if self.outlier_detector is not None:
                n_outlier = self._n_samples-(n_clean + n_missing + n_special)
                self._logger.info("Pre-processing: number of outlier samples: "
                                  "{}".format(n_outlier))

            self._logger.info("Pre-processing terminated. Time: {:.4f}s"
                              .format(self._time_preprocessing))

        # Pre-binning
        # Fit estimator and compute event_rate = P[Y=1, X=x]
        time_estimator = time.perf_counter()

        if self.estimator is None:
            self.estimator = LogisticRegression()

            if self.verbose:
                self._logger.info("Pre-binning: set logistic regression as an "
                                  "estimator.")

        if self.verbose:
            self._logger.info("Pre-binning: estimator fitting started.")

        self.estimator.fit(x_clean.reshape(-1, 1), y_clean)
        event_rate = self.estimator.predict_proba(x_clean.reshape(-1, 1))[:, 1]

        self._time_estimator = time.perf_counter() - time_estimator

        if self.verbose:
            self._logger.info("Pre-binning: estimator terminated. Time "
                              "{:.4f}s.".format(self._time_estimator))

        # Fit optimal binning algorithm for continuous target. Use optimal
        # split points to compute optimal piecewise functions
        self._fit_binning(x_clean, y_clean, event_rate, 0, 1)

        # Post-processing
        if self.verbose:
            self._logger.info("Post-processing started.")
            self._logger.info("Post-processing: compute binning information.")

        time_postprocessing = time.perf_counter()

        # Compute n_nonevent and n_event for special and missing.
        special_target_info = target_info(y_special)
        self._n_nonevent_special = special_target_info[0]
        self._n_event_special = special_target_info[1]

        missing_target_info = target_info(y_missing)
        self._n_nonevent_missing = missing_target_info[0]
        self._n_event_missing = missing_target_info[1]

        indices = np.digitize(x_clean, self._optb.splits, right=False)
        n_nonevent = np.empty(self._n_bins + 2).astype(np.int64)
        n_event = np.empty(self._n_bins + 2).astype(np.int64)

        y0 = (y_clean == 0)
        y1 = ~y0

        for i in range(self._n_bins):
            mask = (indices == i)
            n_nonevent[i] = np.count_nonzero(y0 & mask)
            n_event[i] = np.count_nonzero(y1 & mask)

        n_nonevent[self._n_bins] = self._n_nonevent_special
        n_nonevent[self._n_bins + 1] = self._n_nonevent_missing
        n_event[self._n_bins] = self._n_event_special
        n_event[self._n_bins + 1] = self._n_event_missing

        self._t_n_nonevent = n_nonevent.sum()
        self._t_n_event = n_event.sum()

        # Compute metrics
        if self.verbose:
            self._logger.info("Post-processing: compute performance metrics.")

        d_metrics = {}

        y_pred_proba = transform_binary_target(
            self._optb.splits, x, self._c, 0, 1, self._t_n_nonevent,
            self._t_n_event, self._n_nonevent_special, self._n_event_special,
            self._n_nonevent_missing, self._n_event_missing,
            self.special_codes, "event_rate", "empirical", "empirical")

        d_metrics["Gini Index"] = gini(y, y_pred_proba)
        d_metrics["Avg precision"] = average_precision_score(y, y_pred_proba)
        d_metrics["Brier score"] = brier_score_loss(y, y_pred_proba)

        self._binning_table = PWBinningTable(
            self.name, self._optb.splits, self._c, n_nonevent, n_event,
            x_clean.min(), x_clean.max(), d_metrics)

        self._time_postprocessing = time.perf_counter() - time_postprocessing

        if self.verbose:
            self._logger.info("Post-processing terminated. Time: {:.4f}s"
                              .format(self._time_postprocessing))

        self._time_total = time.perf_counter() - time_init

        if self.verbose:
            self._logger.info("Optimal piecewise binning terminated. "
                              "Status: {}. Time: {:.4f}s"
                              .format(self._status, self._time_total))

        # Completed successfully
        self._class_logger.close()
        self._is_fitted = True

        return self
