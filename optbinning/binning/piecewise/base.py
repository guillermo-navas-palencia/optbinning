"""
Optimal piecewise continuous binning algorithm.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import numbers
import time

import numpy as np

from ropwr import RobustPWRegression
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

from ...binning.auto_monotonic import type_of_monotonic_trend
from ...binning.base import Base
from ...binning.binning import OptimalBinning
from ...binning.continuous_binning import ContinuousOptimalBinning
from ...binning.preprocessing import split_data
from ...logging import Logger
from .binning_information import print_binning_information
from .binning_information import retrieve_status


logger = Logger(__name__).logger


def _check_parameters(name, estimator, objective, degree, continuous,
                      prebinning_method, max_n_prebins, min_prebin_size,
                      min_n_bins, max_n_bins, min_bin_size, max_bin_size,
                      monotonic_trend, n_subsamples, max_pvalue,
                      max_pvalue_policy, outlier_detector, outlier_params,
                      user_splits, user_splits_fixed, special_codes,
                      split_digits, solver, h_epsilon, quantile,
                      regularization, reg_l1, reg_l2, random_state, verbose,
                      problem_type):

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

    if objective not in ("l1", "l2", "huber", "quantile"):
        raise ValueError('Invalid value for objective. Allowed string '
                         'values are "l1", "l2", "huber" and "quantile".')

    if not isinstance(degree, numbers.Integral) or not 0 <= degree <= 5:
        raise ValueError("degree must be an integer in [0, 5].")

    if not isinstance(continuous, bool):
        raise TypeError("continuous must be a boolean; got {}."
                        .format(verbose))

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

    if n_subsamples is not None:
        if not isinstance(n_subsamples, numbers.Integral) or n_subsamples <= 0:
            raise ValueError("n_subsamples must be a positive integer; got {}."
                             .format(n_subsamples))

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

    if solver not in ("auto", "ecos", "osqp", "direct"):
        raise ValueError('Invalid value for solver. Allowed string '
                         'values are "auto", "ecos", "osqp" and "direct".')

    if not isinstance(h_epsilon, numbers.Number) or h_epsilon < 1.0:
        raise ValueError("h_epsilon must a number >= 1.0; got {}."
                         .format(h_epsilon))

    if not isinstance(quantile, numbers.Number) or not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be a value in (0, 1); got {}."
                         .format(quantile))

    if regularization is not None:
        if regularization not in ("l1", "l2"):
            raise ValueError('Invalid value for regularization. Allowed '
                             'string values are "l1" and "l2".')

    if not isinstance(reg_l1, numbers.Number) or reg_l1 < 0.0:
        raise ValueError("reg_l1 must be a positive value; got {}."
                         .format(reg_l1))

    if not isinstance(reg_l2, numbers.Number) or reg_l2 < 0.0:
        raise ValueError("reg_l2 must be a positive value; got {}."
                         .format(reg_l2))

    if random_state is not None:
        if not isinstance(random_state, (int, np.random.RandomState)):
            raise TypeError("random_state must an integer or a RandomState "
                            "instance; got {}.".format(random_state))

    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a boolean; got {}.".format(verbose))


class BasePWBinning(Base, BaseEstimator):
    def __init__(self, name="", estimator=None, objective="l2", degree=1,
                 continuous=True, prebinning_method="cart", max_n_prebins=20,
                 min_prebin_size=0.05, min_n_bins=None, max_n_bins=None,
                 min_bin_size=None, max_bin_size=None, monotonic_trend="auto",
                 n_subsamples=None, max_pvalue=None,
                 max_pvalue_policy="consecutive", outlier_detector=None,
                 outlier_params=None, user_splits=None, user_splits_fixed=None,
                 special_codes=None, split_digits=None, solver="auto",
                 h_epsilon=1.35, quantile=0.5, regularization=None, reg_l1=1.0,
                 reg_l2=1.0, random_state=None, verbose=False):

        self.name = name
        self.estimator = estimator
        self.objective = objective
        self.degree = degree
        self.continuous = continuous
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
        self.h_epsilon = h_epsilon
        self.quantile = quantile
        self.regularization = regularization
        self.reg_l1 = reg_l1
        self.reg_l2 = reg_l2

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

        self._is_fitted = False

    def fit(self, x, y, lb=None, ub=None, check_input=False):
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
        self : BasePWBinning
            Fitted optimal piecewise binning.
        """
        return self._fit(x, y, lb, ub, check_input)

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
            solver = self._optimizer
            time_solver = self._time_solver
        else:
            solver = None
            time_solver = 0

        dict_user_options = self.get_params()

        if self._problem_type == "regression":
            dict_user_options["estimator"] = None

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
            logger.info("Pre-binning: optimal binning started.")

        time_prebinning = time.perf_counter()

        # Determine optimal split points
        monotonic_trend = self.monotonic_trend

        if self.monotonic_trend in ("concave", "convex"):
            monotonic_trend = "auto"

        if self._problem_type == "regression":
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

        elif self._problem_type == "classification":
            self._optb = OptimalBinning(
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

        self._optb.fit(x, y)
        splits = self._optb.splits
        n_splits = len(splits)

        if self.verbose:
            logger.info("Pre-binning: number of splits: {}."
                        .format(n_splits))

        # Prepare optimization model data
        n_bins = n_splits + 1
        self._n_bins = n_bins

        if self.n_subsamples is None or self.n_subsamples > len(x):
            x_subsamples = x
            pred_subsamples = prediction

            if self.verbose:
                logger.info("Pre-binning: no need for subsamples.")
        else:
            indices = np.digitize(x, splits, right=False)
            [_, x_subsamples, _, pred_subsamples,
             _, _, _, _] = train_test_split(
                x, prediction, y, indices, test_size=self.n_subsamples,
                random_state=self.random_state)

            if self.verbose:
                logger.info("Pre-binning: number of subsamples: {}."
                            .format(self.n_subsamples))

        self._time_prebinning = time.perf_counter() - time_prebinning

        if self.verbose:
            logger.info("Pre-binning: optimal binning terminated. Time {:.4}s."
                        .format(self._time_prebinning))

        # LP problem
        if self.verbose:
            logger.info("Optimizer started.")

        if self.monotonic_trend == "auto":
            indices = np.digitize(x, splits, right=False)
            mean = np.array([y[indices == i].mean() for i in range(n_bins)])

            monotonic = type_of_monotonic_trend(mean)
            if monotonic in ("undefined", "no monotonic"):
                monotonic = None
            elif "peak" in monotonic:
                monotonic = "peak"
            elif "valley" in monotonic:
                monotonic = "valley"

            if self.verbose:
                logger.info("Optimizer: {} monotonic trend."
                            .format(monotonic))
        else:
            monotonic = self.monotonic_trend

        time_solver = time.perf_counter()

        optimizer = RobustPWRegression(
            self.objective, self.degree, self.continuous, monotonic,
            self.solver, self.h_epsilon, self.quantile, self.regularization,
            self.reg_l1, self.reg_l1, self.verbose)

        optimizer.fit(x_subsamples, pred_subsamples, splits, lb, ub)

        self._c = optimizer.coef_

        self._optimizer = optimizer
        self._status = retrieve_status(optimizer.status)
        self._splits_optimal = splits

        self._time_solver = time.perf_counter() - time_solver

        if self.verbose:
            logger.info("Optimizer terminated. Time: {:.4f}s"
                        .format(self._time_solver))

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
