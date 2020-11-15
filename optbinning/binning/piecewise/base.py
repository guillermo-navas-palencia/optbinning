"""
Optimal piecewise continuous binning algorithm.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import numbers
import time

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split

from ...binning.continuous_binning import ContinuousOptimalBinning
from ...logging import Logger
from ...preprocessing import split_data
from .binning_information import print_binning_information
from .lp import PWPBinningLP


def _check_parameters(name, estimator, degree, continuity, prebinning_method,
                      max_n_prebins, min_prebin_size, min_n_bins, max_n_bins,
                      min_bin_size, max_bin_size, monotonic_trend,
                      n_subsamples, max_pvalue, max_pvalue_policy,
                      outlier_detector, outlier_params, user_splits,
                      user_splits_fixed, special_codes, split_digits, solver,
                      time_limit, random_state, verbose, problem_type):

    if not isinstance(name, str):
        raise TypeError("name must be a string.")

    if estimator is not None:
        estimator_fit = hasattr(estimator, "fit")

        if problem_type == "classification":
            if not (estimator_fit and hasattr(estimator, "predict_proba")):
                raise TypeError("estimator must be an object with methods fit "
                                "and predict_proba.")

        elif problem_type == "regression":
            if not (estimator_fit and hasattr(estimator, "predict")):
                raise TypeError("estimator must be an object with methods fit "
                                "and predict.")

    if not isinstance(degree, numbers.Integral) or degree < 0:
        raise ValueError("degree must be an integer >= 0; got {}."
                         .format(degree))

    if not isinstance(continuity, bool):
        raise TypeError("continuity must be a boolean; got {}."
                        .format(continuity))

    if prebinning_method not in ("cart", "quantile", "uniform"):
        raise ValueError('Invalid value for prebinning_method. Allowed string '
                         'values are "cart", "quantile" and "uniform".')

    if not 0. < min_prebin_size <= 0.5:
        raise ValueError("min_prebin_size must be in (0, 0.5]; got {}."
                         .format(min_prebin_size))

    if min_n_bins is not None:
        if not isinstance(min_n_bins, numbers.Integral) or min_n_bins <= 0:
            raise ValueError("min_n_bins must be a positive integer; got {}."
                             .format(min_n_bins))

    if max_n_bins is not None:
        if not isinstance(max_n_bins, numbers.Integral) or max_n_bins <= 0:
            raise ValueError("max_n_bins must be a positive integer; got {}."
                             .format(max_n_bins))

    if min_n_bins is not None and max_n_bins is not None:
        if min_n_bins > max_n_bins:
            raise ValueError("min_n_bins must be <= max_n_bins; got {} <= {}."
                             .format(min_n_bins, max_n_bins))

    if min_bin_size is not None:
        if (not isinstance(min_bin_size, numbers.Number) or
                not 0. < min_bin_size <= 0.5):
            raise ValueError("min_bin_size must be in (0, 0.5]; got {}."
                             .format(min_bin_size))

    if max_bin_size is not None:
        if (not isinstance(max_bin_size, numbers.Number) or
                not 0. < max_bin_size <= 1.0):
            raise ValueError("max_bin_size must be in (0, 1.0]; got {}."
                             .format(max_bin_size))

    if min_bin_size is not None and max_bin_size is not None:
        if min_bin_size > max_bin_size:
            raise ValueError("min_bin_size must be <= max_bin_size; "
                             "got {} <= {}.".format(min_bin_size,
                                                    max_bin_size))
    if monotonic_trend is not None:
        if monotonic_trend not in ("auto", "ascending", "descending", "convex",
                                   "concave", "peak", "valley"):
            raise ValueError('Invalid value for monotonic trend. Allowed '
                             'string values are "auto", "ascending", '
                             '"descending", "concave", "convex", "peak" and '
                             '"valley".')

        if monotonic_trend in ("convex", "concave") and degree > 1:
            raise ValueError("Monotonic trend convex and convex are only "
                             "allowed if degree <= 1.")

    if max_pvalue is not None:
        if (not isinstance(max_pvalue, numbers.Number) or
                not 0. < max_pvalue <= 1.0):
            raise ValueError("max_pvalue must be in (0, 1.0]; got {}."
                             .format(max_pvalue))

    if max_pvalue_policy not in ("all", "consecutive"):
        raise ValueError('Invalid value for max_pvalue_policy. Allowed string '
                         'values are "all" and "consecutive".')

    if outlier_detector is not None:
        if outlier_detector not in ("range", "zscore"):
            raise ValueError('Invalid value for outlier_detector. Allowed '
                             'string values are "range" and "zscore".')

        if outlier_params is not None:
            if not isinstance(outlier_params, dict):
                raise TypeError("outlier_params must be a dict or None; "
                                "got {}.".format(outlier_params))

    if user_splits is not None:
        if not isinstance(user_splits, (np.ndarray, list)):
            raise TypeError("user_splits must be a list or numpy.ndarray.")

    if user_splits_fixed is not None:
        if user_splits is None:
            raise ValueError("user_splits must be provided.")
        else:
            if not isinstance(user_splits_fixed, (np.ndarray, list)):
                raise TypeError("user_splits_fixed must be a list or "
                                "numpy.ndarray.")
            elif not all(isinstance(s, bool) for s in user_splits_fixed):
                raise ValueError("user_splits_fixed must be list of boolean.")
            elif len(user_splits) != len(user_splits_fixed):
                raise ValueError("Inconsistent length of user_splits and "
                                 "user_splits_fixed: {} != {}. Lengths must "
                                 "be equal".format(len(user_splits),
                                                   len(user_splits_fixed)))

    if special_codes is not None:
        if not isinstance(special_codes, (np.ndarray, list)):
            raise TypeError("special_codes must be a list or numpy.ndarray.")

    if split_digits is not None:
        if (not isinstance(split_digits, numbers.Integral) or
                not 0 <= split_digits <= 8):
            raise ValueError("split_digist must be an integer in [0, 8]; "
                             "got {}.".format(split_digits))

    if not isinstance(time_limit, numbers.Number) or time_limit < 0:
        raise ValueError("time_limit must be a positive value in seconds; "
                         "got {}.".format(time_limit))

    if random_state is not None:
        if not isinstance(random_state, (int, np.random.RandomState)):
            raise TypeError("random_state must an integer or a RandomState "
                            "instance; got {}.".format(random_state))

    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a boolean; got {}.".format(verbose))


class BasePWBinning(BaseEstimator):
    def __init__(self, name="", estimator=None, degree=1, continuity=True,
                 prebinning_method="cart", max_n_prebins=20,
                 min_prebin_size=0.05, min_n_bins=None, max_n_bins=None,
                 min_bin_size=None, max_bin_size=None, monotonic_trend="auto",
                 n_subsamples=10000, max_pvalue=None,
                 max_pvalue_policy="consecutive", outlier_detector=None,
                 outlier_params=None, user_splits=None, user_splits_fixed=None,
                 special_codes=None, split_digits=None, solver="clp",
                 time_limit=100, random_state=None, verbose=False):

        self.name = name
        self.estimator = estimator
        self.degree = degree
        self.continuity = continuity
        self.prebinning_method = prebinning_method

        self.max_n_prebins = max_n_prebins
        self.min_prebin_size = min_prebin_size

        self.min_n_bins = min_n_bins
        self.max_n_bins = max_n_bins
        self.min_bin_size = min_bin_size
        self.max_bin_size = max_bin_size

        self.monotonic_trend = monotonic_trend
        self.n_subsamples = n_subsamples
        self.max_pvalue = max_pvalue
        self.max_pvalue_policy = max_pvalue_policy

        self.outlier_detector = outlier_detector
        self.outlier_params = outlier_params

        self.user_splits = user_splits
        self.user_splits_fixed = user_splits_fixed
        self.special_codes = special_codes
        self.split_digits = split_digits

        self.solver = solver
        self.time_limit = time_limit
        self.random_state = random_state
        self.verbose = verbose

        # info
        self._optb = None
        self._binning_table = None
        self._n_bins = None
        self._n_samples = None
        self._optimizer = None
        self._splits_optimal = None
        self._status = None
        self._c = None

        # timing
        self._time_total = None
        self._time_preprocessing = None
        self._time_estimator = None
        self._time_prebinning = None
        self._time_solver = None
        self._time_postprocessing = None

        # logger
        self._class_logger = Logger(__name__)
        self._logger = self._class_logger.logger

        self._is_fitted = False

    def fit(self, x, y, check_input=False):
        """Fit the optimal piecewise binning according to the given training
        data.

        Parameters
        ----------
        x : array-like, shape = (n_samples,)
            Training vector, where n_samples is the number of samples.

        y : array-like, shape = (n_samples,)
            Target vector relative to x.

        check_input : bool (default=False)
            Whether to check input arrays.

        Returns
        -------
        self : object
            Fitted optimal piecewise binning.
        """
        return self._fit(x, y, check_input)

    def information(self, print_level=1):
        """Print overview information about the options settings, problem
        statistics, and the solution of the computation.

        Parameters
        ----------
        print_level : int (default=1)
            Level of details.
        """
        self._check_is_fitted()

        if not isinstance(print_level, numbers.Integral) or print_level < 0:
            raise ValueError("print_level must be an integer >= 0; got {}."
                             .format(print_level))

        if self._optimizer is not None:
            solver = self._optimizer.solver_
            time_solver = self._time_solver
        else:
            solver = None
            time_solver = 0

        dict_user_options = self.get_params()

        print_binning_information(print_level, self.name, self._status,
                                  self.solver, solver, self._time_total,
                                  self._time_preprocessing,
                                  self._time_estimator, self._time_prebinning,
                                  time_solver, self._time_postprocessing,
                                  self._n_bins, dict_user_options)

    def _fit_preprocessing(self, x, y, check_input):
        return split_data(dtype="numerical", x=x, y=y,
                          special_codes=self.special_codes,
                          user_splits=self.user_splits,
                          check_input=check_input,
                          outlier_detector=self.outlier_detector,
                          outlier_params=self.outlier_params)

    def _fit_binning(self, x, y, prediction, lb, ub):
        if self.verbose:
            self._logger.info("Pre-binning: optimal binning started.")

        time_prebinning = time.perf_counter()

        # Determine optimal split points
        monotonic_trend = self.monotonic_trend

        if self.monotonic_trend in ("concave", "convex"):
            monotonic_trend = "auto"

        self._optb = ContinuousOptimalBinning(
            name=self.name, dtype="numerical",
            prebinning_method=self.prebinning_method,
            max_n_prebins=self.max_n_prebins,
            min_prebin_size=self.min_prebin_size,
            min_n_bins=self.min_n_bins,
            max_n_bins=self.max_n_bins,
            min_bin_size=self.min_bin_size,
            max_bin_size=self.max_bin_size,
            monotonic_trend=monotonic_trend,
            max_pvalue=self.max_pvalue,
            max_pvalue_policy=self.max_pvalue_policy,
            outlier_detector=self.outlier_detector,
            outlier_params=self.outlier_params,
            user_splits=self.user_splits,
            user_splits_fixed=self.user_splits_fixed,
            split_digits=self.split_digits)

        self._optb.fit(x, prediction)
        splits = self._optb.splits
        n_splits = len(splits)

        if self.verbose:
            self._logger.info("Pre-binning: number of splits: {}."
                              .format(n_splits))

        # Prepare optimization model data
        n_bins = n_splits + 1
        self._n_bins = n_bins

        indices = np.digitize(x, splits, right=False)
        n_subsamples = min(len(x), self.n_subsamples)

        if len(x) == n_subsamples:
            x_subsamples = x
            pred_subsamples = prediction
            y_subsamples = y
            indices_subsamples = indices

            if self.verbose:
                self._logger.info("Pre-binning: no need for subsamples.")
        else:
            [_, x_subsamples, _, pred_subsamples, _, y_subsamples, _,
             indices_subsamples] = train_test_split(
                x, prediction, y, indices, test_size=n_subsamples,
                random_state=self.random_state)

            if self.verbose:
                self._logger.info("Pre-binning: number of subsamples: {}."
                                  .format(n_subsamples))

        x_indices = []
        for i in range(n_bins):
            mask = (indices_subsamples == i)
            x_indices.append(np.arange(n_subsamples)[mask])

        self._time_prebinning = time.perf_counter() - time_prebinning

        if self.verbose:
            self._logger.info("Pre-binning: optimal binning terminated. "
                              "Time {:.4}s.".format(self._time_prebinning))

        # LP problem
        if self.verbose:
            self._logger.info("Optimizer started.")

        time_solver = time.perf_counter()

        optimizer = PWPBinningLP(self.degree, self.monotonic_trend,
                                 self.continuity, lb, ub, self.solver,
                                 self.time_limit)

        if self.verbose:
            self._logger.info("Optimizer: build model...")

        optimizer.build_model(splits, x_subsamples, x_indices, pred_subsamples)

        if self.verbose:
            self._logger.info("Optimizer: solve...")

        self._status, self._c = optimizer.solve()

        self._optimizer = optimizer
        self._splits_optimal = splits

        self._time_solver = time.perf_counter() - time_solver

        if self.verbose:
            self._logger.info("Optimizer terminated. Time: {:.4f}s"
                              .format(self._time_solver))

    def _check_is_fitted(self):
        if not self._is_fitted:
            raise NotFittedError("This {} instance is not fitted yet. Call "
                                 "'fit' with appropriate arguments."
                                 .format(self.__class__.__name__))

    @property
    def binning_table(self):
        """Return an instantiated binning table. Please refer to
        :ref:`Binning table: binary target`.

        Returns
        -------
        binning_table : BinningTable.
        """
        self._check_is_fitted()

        return self._binning_table

    @property
    def splits(self):
        """List of optimal split points when ``dtype`` is set to "numerical" or
        list of optimal bins when ``dtype`` is set to "categorical".

        Returns
        -------
        splits : numpy.ndarray
        """
        self._check_is_fitted()

        return self._optb.splits

    @property
    def status(self):
        """The status of the underlying optimization solver.

        Returns
        -------
        status : str
        """
        self._check_is_fitted()

        return self._status
