"""
Optimal binning algorithm given scenarions. Deterministic equivalent to
stochastic optimal binning.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import logging
import numbers
import time

import numpy as np

# from sklearn.utils import check_array

from ..logging import Logger
from ..preprocessing import split_data
from .binning import OptimalBinning
from .binning_statistics import bin_info
from .binning_statistics import BinningTable
from .binning_statistics import target_info
from .cp import BinningCP
from .prebinning import PreBinning


def _check_parameters(name, prebinning_method, max_n_prebins, min_prebin_size,
                      min_n_bins, max_n_bins, min_bin_size, max_bin_size,
                      monotonic_trend, min_event_rate_diff, max_pvalue,
                      max_pvalue_policy, class_weight, user_splits,
                      user_splits_fixed, special_codes, split_digits,
                      time_limit, verbose):

    if not isinstance(name, str):
        raise TypeError("name must be a string.")

    if prebinning_method not in ("cart", "quantile", "uniform"):
        raise ValueError('Invalid value for prebinning_method. Allowed string '
                         'values are "cart", "quantile" and "uniform".')

    if not isinstance(max_n_prebins, numbers.Integral) or max_n_prebins <= 1:
        raise ValueError("max_prebins must be an integer greater than 1; "
                         "got {}.".format(max_n_prebins))

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
        if monotonic_trend not in ("auto", "auto_asc_desc", "ascending",
                                   "descending", "convex", "concave", "peak",
                                   "valley"):
            raise ValueError('Invalid value for monotonic trend. Allowed '
                             'string values are "auto", "auto_asc_desc", '
                             '"ascending", "descending", "concave", "convex", '
                             '"peak" and "valley"')

    if (not isinstance(min_event_rate_diff, numbers.Number) or
            not 0. <= min_event_rate_diff <= 1.0):
        raise ValueError("min_event_rate_diff must be in [0, 1]; got {}."
                         .format(min_event_rate_diff))

    if max_pvalue is not None:
        if (not isinstance(max_pvalue, numbers.Number) or
                not 0. < max_pvalue <= 1.0):
            raise ValueError("max_pvalue must be in (0, 1.0]; got {}."
                             .format(max_pvalue))

    if max_pvalue_policy not in ("all", "consecutive"):
        raise ValueError('Invalid value for max_pvalue_policy. Allowed string '
                         'values are "all" and "consecutive".')

    if class_weight is not None:
        if not isinstance(class_weight, (dict, str)):
            raise TypeError('class_weight must be dict, "balanced" or None; '
                            'got {}.'.format(class_weight))

        elif isinstance(class_weight, str) and class_weight != "balanced":
            raise ValueError('Invalid value for class_weight. Allowed string '
                             'value is "balanced".')

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
            raise ValueError("split_digits must be an integer in [0, 8]; "
                             "got {}.".format(split_digits))

    if not isinstance(time_limit, numbers.Number) or time_limit < 0:
        raise ValueError("time_limit must be a positive value in seconds; "
                         "got {}.".format(time_limit))

    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a boolean; got {}.".format(verbose))


class DSOptimalBinning(OptimalBinning):
    """Deterministic equivalent of the stochastic optimal binning of a
    numerical variable with respect to a binary target.

    Parameters
    ----------
    """
    def __init__(self, name="", prebinning_method="cart", max_n_prebins=20,
                 min_prebin_size=0.05, min_n_bins=None, max_n_bins=None,
                 min_bin_size=None, max_bin_size=None, monotonic_trend="auto",
                 min_event_rate_diff=0, max_pvalue=None,
                 max_pvalue_policy="consecutive", class_weight=None,
                 user_splits=None, user_splits_fixed=None, special_codes=None,
                 split_digits=None, time_limit=100, verbose=False):

        self.name = name
        self.dtype = "numerical"
        self.prebinning_method = prebinning_method
        self.solver = "cp"

        self.max_n_prebins = max_n_prebins
        self.min_prebin_size = min_prebin_size

        self.min_n_bins = min_n_bins
        self.max_n_bins = max_n_bins
        self.min_bin_size = min_bin_size
        self.max_bin_size = max_bin_size

        self.monotonic_trend = monotonic_trend
        self.min_event_rate_diff = min_event_rate_diff
        self.max_pvalue = max_pvalue
        self.max_pvalue_policy = max_pvalue_policy

        self.class_weight = class_weight

        self.user_splits = user_splits
        self.user_splits_fixed = user_splits_fixed
        self.special_codes = special_codes
        self.split_digits = split_digits

        self.time_limit = time_limit

        self.verbose = verbose

        # auxiliary
        self._n_scenarios = None
        self._n_event = None
        self._n_nonevent = None
        self._n_nonevent_missing = None
        self._n_event_missing = None
        self._n_nonevent_special = None
        self._n_event_special = None
        self._problem_type = "classification"
        self._user_splits = user_splits
        self._user_splits_fixed = user_splits_fixed

        # info
        self._n_refinements = 0

        # timing
        self._time_total = None
        self._time_preprocessing = None
        self._time_prebinning = None
        self._time_solver = None
        self._time_postprocessing = None

        # logger
        self._logger = Logger()

        self._is_fitted = False

    def fit(self, X, Y, weights=None, check_input=False):
        return self._fit(X, Y, weights, check_input)

    def fit_transform(self, X, Y, weights=None, metric="woe", metric_special=0,
                      metric_missing=0, check_input=False):
        """Fit the optimal binning according to the given training data, then
        transform it."""

    def _fit(self, X, Y, weights, check_input):
        time_init = time.perf_counter()

        # _check_parameters(**self.get_params())

        # Check X, Y and weights
        self._n_scenarios = len(X)

        if self.verbose:
            logging.info("Optimal binning started.")
            logging.info("Options: check parameters.")

        _check_parameters(**self.get_params())

        # Pre-processing
        time_preprocessing = time.perf_counter()

        self._n_samples = sum(len(x) for x in X)

        x_clean = []
        y_clean = []
        x_missing = []
        y_missing = []
        x_special = []
        y_special = []

        if weights is None:
            w = None
        else:
            w = []

        for s in range(self._n_scenarios):
            x = X[s]
            y = Y[s]

            x_c, y_c, x_m, y_m, x_s, y_s, _, _, _ = split_data(
                self.dtype, x, y, special_codes=self.special_codes,
                check_input=check_input)

            x_clean.append(x_c)
            y_clean.append(y_c)
            x_missing.append(x_m)
            y_missing.append(y_m)
            x_special.append(x_s)
            y_special.append(y_s)

            if weights is not None:
                w.extend(np.full(len(x_c), weights[s]))

        self._time_preprocessing = time.perf_counter() - time_preprocessing

        # Pre-binning
        time_prebinning = time.perf_counter()

        if self.user_splits is not None:
            pass
        else:
            splits, n_nonevent, n_event = self._fit_prebinning(
                w, x_clean, y_clean, y_missing, y_special, self.class_weight)

        self._n_prebins = len(n_nonevent)

        self._time_prebinning = time.perf_counter() - time_prebinning

        # Optimization
        self._fit_optimizer(splits, n_nonevent, n_event, weights)

        # Post-processing
        time_postprocessing = time.perf_counter()

        if not len(splits):
            pass

        self._n_nonevent = []
        self._n_event = []

        self._binning_tables = []

        t_n_nonevent = 0
        t_n_event = 0

        for s in range(self._n_scenarios):
            s_n_nonevent, s_n_event = bin_info(
                self._solution, n_nonevent[:, s], n_event[:, s],
                self._n_nonevent_missing[s], self._n_event_missing[s],
                self._n_nonevent_special[s], self._n_event_special[s], None,
                None, [])

            self._n_nonevent.append(s_n_nonevent)
            self._n_event.append(s_n_event)

            t_n_nonevent += s_n_nonevent
            t_n_event += s_n_event

            binning_table = BinningTable(
                self.name, self.dtype, self._splits_optimal, s_n_nonevent,
                s_n_event, None, None, self.user_splits)

            self._binning_tables.append(binning_table)

        self._binning_table = BinningTable(
            self.name, self.dtype, self._splits_optimal, t_n_nonevent,
            t_n_event, None, None, self.user_splits)

        self._time_postprocessing = time.perf_counter() - time_postprocessing

        self._time_total = time.perf_counter() - time_init

        # Completed successfully
        self._logger.close()
        self._is_fitted = True

        return self

    def _fit_prebinning(self, weights, x_clean, y_clean, y_missing, y_special,
                        class_weight=None):
        x = []
        y = []
        for s in range(self._n_scenarios):
            x.extend(x_clean[s])
            y.extend(y_clean[s])

        x = np.array(x)
        y = np.array(y)

        min_bin_size = np.int(np.ceil(self.min_prebin_size * self._n_samples))

        prebinning = PreBinning(method=self.prebinning_method,
                                n_bins=self.max_n_prebins,
                                min_bin_size=min_bin_size,
                                problem_type=self._problem_type,
                                class_weight=class_weight).fit(x, y, weights)

        return self._prebinning_refinement(prebinning.splits, x_clean, y_clean,
                                           y_missing, y_special)

    def _prebinning_refinement(self, splits_prebinning, x, y, y_missing,
                               y_special):
        self._n_nonevent_special = []
        self._n_event_special = []
        self._n_nonevent_missing = []
        self._n_event_missing = []
        for s in range(self._n_scenarios):
            s_n_nonevent, s_n_event = target_info(y_special[s])
            m_n_nonevent, m_n_event = target_info(y_missing[s])
            self._n_nonevent_special.append(s_n_nonevent)
            self._n_event_special.append(s_n_event)
            self._n_nonevent_missing.append(m_n_nonevent)
            self._n_event_missing.append(m_n_event)

        n_splits = len(splits_prebinning)

        if not n_splits:
            return splits_prebinning, np.array([]), np.array([])

        if self.split_digits is not None:
            splits_prebinning = np.round(splits_prebinning, self.split_digits)

        splits_prebinning, n_nonevent, n_event = self._compute_prebins(
            splits_prebinning, x, y)

        return splits_prebinning, n_nonevent, n_event

    def _compute_prebins(self, splits_prebinning, x, y):
        n_splits = len(splits_prebinning)

        if not n_splits:
            return splits_prebinning, np.array([]), np.array([])

        n_bins = n_splits + 1
        n_nonevent = np.zeros((n_bins, self._n_scenarios)).astype(np.int)
        n_event = np.zeros((n_bins, self._n_scenarios)).astype(np.int)
        mask_remove = np.zeros(n_bins).astype(np.bool)

        for s in range(self._n_scenarios):
            y0 = (y[s] == 0)
            y1 = ~y0

            indices = np.digitize(x[s], splits_prebinning, right=False)

            for i in range(n_bins):
                mask = (indices == i)
                n_nonevent[i, s] = np.count_nonzero(y0 & mask)
                n_event[i, s] = np.count_nonzero(y1 & mask)

            mask_remove |= (n_nonevent[:, s] == 0) | (n_event[:, s] == 0)

        if np.any(mask_remove):
            self._n_refinements += 1

            mask_splits = np.concatenate(
                [mask_remove[:-2], [mask_remove[-2] | mask_remove[-1]]])

            splits = splits_prebinning[~mask_splits]

            [splits_prebinning, n_nonevent, n_event] = self._compute_prebins(
                splits, x, y)

        return splits_prebinning, n_nonevent, n_event

    def _fit_optimizer(self, splits, n_nonevent, n_event, weights):
        time_init = time.perf_counter()

        if not len(n_nonevent):
            self._status = "OPTIMAL"
            self._splits_optimal = splits
            self._solution = np.zeros(len(splits)).astype(np.bool)

            if self.verbose:
                logging.warning("Optimizer: no bins after pre-binning.")
                logging.warning("Optimizer: solver not run.")

                logging.info("Optimizer terminated. Time: 0s")
            return

        if self.min_bin_size is not None:
            # min_bin_size = np.int(np.ceil(self.min_bin_size))
            pass
        else:
            min_bin_size = self.min_bin_size

        if self.max_bin_size is not None:
            # max_bin_size = np.int(np.ceil(self.max_bin_size))
            pass
        else:
            max_bin_size = self.max_bin_size

        # Monotonic trend
        auto_monotonic_modes = ("auto", "auto_heuristic", "auto_asc_desc")
        if self.monotonic_trend in auto_monotonic_modes:
            pass
        else:
            monotonic = self.monotonic_trend

        optimizer = BinningCP(monotonic, self.min_n_bins, self.max_n_bins,
                              min_bin_size, max_bin_size, None,
                              None, None, None, 0,
                              self.max_pvalue, self.max_pvalue_policy, None,
                              self.user_splits_fixed, self.time_limit)
        if weights is None:
            weights = np.ones(n_event.shape[1], np.int)

        optimizer.build_model_scenarios(n_nonevent, n_event, weights, None)

        status, solution = optimizer.solve()

        self._solution = solution

        self._optimizer = optimizer
        self._status = status

        self._splits_optimal = splits[solution[:-1]]

        self._time_solver = time.perf_counter() - time_init

    @property
    def binning_table_scenarios(self, scenario_id):
        """Return the instantiated binning table corresponding to
        ``scenario_id``. Please refer to :ref:`Binning table: binary target`.

        Parameters
        ----------
        scenario_id

        Returns
        -------
        binning_table : BinningTable.
        """
        return self._binning_tables[scenario_id]

    @property
    def splits(self):
        """List of optimal split points.

        Returns
        -------
        splits : numpy.ndarray
        """
        self._check_is_fitted()

        return self._splits_optimal
