"""
Optimal binning algorithm for continuous target.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import logging
import numbers
import time

from sklearn.utils import check_array

import numpy as np
from .binning import OptimalBinning
from .continuous_cp import ContinuousBinningCP
from .preprocessing import preprocessing_user_splits_categorical
from .preprocessing import split_data
from .transformations import transform_continuous_target

from .auto_monotonic import auto_monotonic_continuous
from .binning_statistics import continuous_bin_info
from .binning_statistics import ContinuousBinningTable
from .logging import Logger


def _check_parameters(name, dtype, prebinning_method, max_n_prebins,
                      min_prebin_size, min_n_bins, max_n_bins, min_bin_size,
                      max_bin_size, monotonic_trend, min_mean_diff, max_pvalue,
                      max_pvalue_policy, cat_cutoff, user_splits,
                      special_codes, split_digits, time_limit, verbose):

    if not isinstance(name, str):
        raise TypeError("name must be a string.")

    if dtype not in ("categorical", "numerical"):
        raise ValueError('Invalid value for dtype. Allowed string '
                         'values are "categorical" and "numerical".')

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
        if monotonic_trend not in ("auto", "ascending", "descending", "convex",
                                   "concave", "peak", "valley"):
            raise ValueError('Invalid value for monotonic trend. Allowed '
                             'string values are "auto", "ascending", '
                             '"descending", "concave", "convex", "peak" and '
                             '"valley".')

    if (not isinstance(min_mean_diff, numbers.Number) or min_mean_diff < 0):
        raise ValueError("min_mean_diff must be >= 0; got {}."
                         .format(min_mean_diff))

    if max_pvalue is not None:
        if (not isinstance(max_pvalue, numbers.Number) or
                not 0. < max_pvalue <= 1.0):
            raise ValueError("max_pvalue must be in (0, 1.0]; got {}."
                             .format(max_pvalue))

    if max_pvalue_policy not in ("all", "consecutive"):
        raise ValueError('Invalid value for max_pvalue_policy. Allowed string '
                         'values are "all" and "consecutive".')

    if cat_cutoff is not None:
        if (not isinstance(cat_cutoff, numbers.Number) or
                not 0. < cat_cutoff <= 1.0):
            raise ValueError("cat_cutoff must be in (0, 1.0]; got {}."
                             .format(cat_cutoff))

    if user_splits is not None:
        if not isinstance(user_splits, (np.ndarray, list)):
            raise TypeError("user_splits must be a list or numpy.ndarray.")

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

    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a boolean; got {}.".format(verbose))


class ContinuousOptimalBinning(OptimalBinning):
    """Optimal binning of a numerical or categorical variable with respect to a
    continuous target.

    Parameters
    ----------
    name : str, optional (default="")
        The variable name.

    dtype : str, optional (default="numerical")
        The variable data type. Supported data types are "numerical" for
        continuous and ordinal variables and "categorical" for categorical
        and nominal variables.

    prebinning_method : str, optional (default="cart")
        The pre-binning method. Supported methods are "cart" for a CART
        decision tree, "quantile" to generate prebins with approximately same
        frequency and "uniform" to generate prebins with equal width. Method
        "cart" uses `sklearn.tree.DecisionTreeRegressor
        <https://scikit-learn.org/stable/modules/generated/sklearn.tree.
        DecisionTreeRegressor.html>`_.

    max_n_prebins : int (default=20)
        The maximum number of bins after pre-binning (prebins).

    min_prebin_size : float (default=0.05)
        The fraction of mininum number of records for each prebin.

    min_n_bins : int or None, optional (default=None)
        The minimum number of bins. If None, then ``min_n_bins`` is
        a value in ``[0, max_n_prebins]``.

    max_n_bins : int or None, optional (default=None)
        The maximum number of bins. If None, then ``max_n_bins`` is
        a value in ``[0, max_n_prebins]``.

    min_bin_size : float or None, optional (default=None)
        The fraction of minimum number of records for each bin. If None,
        ``min_bin_size = min_prebin_size``.

    max_bin_size : float or None, optional (default=None)
        The fraction of maximum number of records for each bin. If None,
        ``max_bin_size = 1.0``.

    monotonic_trend : str or None, optional (default="auto")
        The **mean** monotonic trend. Supported trends are “auto” to
        automatically determine the trend maximizing IV using a machine
        learning classifier, "ascending", "descending", "concave", "convex",
        "peak" to allow a peak change point and "valley" to allow a valley
        change point. If None, then the monotonic constraint is disabled.

    min_mean_diff : float, optional (default=0)
        The minimum mean difference between consecutives bins. This
        option currently only applies when ``monotonic_trend`` is "ascending"
        or "descending".

    max_pvalue : float or None, optional (default=0.05)
        The maximum p-value among bins. The T-test is used to detect bins
        not satisfying the p-value constraint.

    max_pvalue_policy : str, optional (default="consecutive")
        The method to determine bins not satisfying the p-value constraint.
        Supported methods are "consecutive" to compare consecutive bins and
        "all" to compare all bins.

    cat_cutoff : float or None, optional (default=None)
        Generate bin others with categories in which the fraction of
        occurrences is below the  ``cat_cutoff`` value. This option is
        available when ``dtype`` is "categorical".

    user_splits : array-like or None, optional (default=None)
        The list of pre-binning split points when ``dtype`` is "numerical" or
        the list of prebins when ``dtype`` is "categorical".

    special_codes : array-like or None, optional (default=None)
        List of special codes. Use special codes to specify the data values
        that must be treated separately.

    split_digits : int or None, optional (default=None)
        The significant digits of the split points. If ``split_digits`` is set
        to 0, the split points are integers. If None, then all significant
        digits in the split points are considered.

    time_limit : int (default=100)
        The maximum time in seconds to run the optimization solver.

    verbose : int or bool (default=False)
        Enable verbose output.

    Notes
    -----
    The parameter values ``max_n_prebins`` and ``min_prebin_size`` control
    complexity and memory usage. The default values generally produce quality
    results, however, some improvement can be achieved by increasing
    ``max_n_prebins`` and/or decreasing ``min_prebin_size``.

    The T-test uses an estimate of the standard deviation of the contingency
    table to speed up the model generation and reduce memory usage. Therefore,
    it is not guaranteed to obtain bins satisfying the p-value constraint,
    although it may work reasonably well in most cases. To avoid having bins
    with similar bins the parameter ``min_mean_diff`` is recommended.
    """
    def __init__(self, name="", dtype="numerical", prebinning_method="cart",
                 max_n_prebins=20, min_prebin_size=0.05, min_n_bins=None,
                 max_n_bins=None, min_bin_size=None, max_bin_size=None,
                 monotonic_trend="auto", min_mean_diff=0, max_pvalue=None,
                 max_pvalue_policy="consecutive", cat_cutoff=None,
                 user_splits=None, special_codes=None, split_digits=None,
                 time_limit=100, verbose=False):

        self.name = name
        self.dtype = dtype
        self.prebinning_method = prebinning_method
        self.solver = "cp"

        self.max_n_prebins = max_n_prebins
        self.min_prebin_size = min_prebin_size

        self.min_n_bins = min_n_bins
        self.max_n_bins = max_n_bins
        self.min_bin_size = min_bin_size
        self.max_bin_size = max_bin_size

        self.monotonic_trend = monotonic_trend
        self.min_mean_diff = min_mean_diff
        self.max_pvalue = max_pvalue
        self.max_pvalue_policy = max_pvalue_policy

        self.cat_cutoff = cat_cutoff

        self.user_splits = user_splits
        self.special_codes = special_codes
        self.split_digits = split_digits

        self.time_limit = time_limit

        self.verbose = verbose

        # auxiliary
        self._categories = None
        self._cat_others = None
        self._n_records = None
        self._sums = None
        self._n_records_cat_others = None
        self._n_records_missing = None
        self._n_records_special = None
        self._sum_cat_others = None
        self._sum_missing = None
        self._sum_special = None
        self._problem_type = "regression"

        # info
        self._binning_table = None
        self._n_prebins = None
        self._n_refinements = 0
        self._n_samples = None
        self._optimizer = None
        self._splits_optimal = None
        self._status = None

        # timing
        self._time_total = None
        self._time_preprocessing = None
        self._time_prebinning = None
        self._time_solver = None
        self._time_postprocessing = None

        # logger
        self._logger = Logger()

        self._is_fitted = False

    def fit_transform(self, x, y, metric_special=0, metric_missing=0,
                      check_input=False):
        """Fit the optimal binning according to the given training data, then
        transform it.

        Parameters
        ----------
        x : array-like, shape = (n_samples,)
            Training vector, where n_samples is the number of samples.

        y : array-like, shape = (n_samples,)
            Target vector relative to x.

        metric_special : float or str (default=0)
            The metric value to transform special codes in the input vector.
            Supported metrics are "empirical" to use the empirical mean, and
            any numerical value.

        metric_missing : float or str (default=0)
            The metric value to transform missing values in the input vector.
            Supported metrics are "empirical" to use the empirical mean, and
            any numerical value.

        check_input : bool (default=False)
            Whether to check input arrays.

        Returns
        -------
        x_new : numpy array, shape = (n_samples,)
            Transformed array.
        """
        return self.fit(x, y, check_input).transform(
            x, metric_special, metric_missing, check_input)

    def transform(self, x, metric_special=0, metric_missing=0,
                  check_input=False):
        """Transform given data to mean using bins from the fitted
        optimal binning.

        Parameters
        ----------
        x : array-like, shape = (n_samples,)
            Training vector, where n_samples is the number of samples.

        metric_special : float or str (default=0)
            The metric value to transform special codes in the input vector.
            Supported metrics are "empirical" to use the empirical mean, and
            any numerical value.

        metric_missing : float or str (default=0)
            The metric value to transform missing values in the input vector.
            Supported metrics are "empirical" to use the empirical mean, and
            any numerical value.

        check_input : bool (default=False)
            Whether to check input arrays.

        Returns
        -------
        x_new : numpy array, shape = (n_samples,)
            Transformed array.

        Notes
        -----
        Transformation of data including categories not present during training
        return zero mean.
        """
        self._check_is_fitted()

        return transform_continuous_target(self._splits_optimal, self.dtype,
                                           x, self._n_records, self._sums,
                                           self.special_codes,
                                           self._categories, self._cat_others,
                                           metric_special, metric_missing,
                                           self.user_splits, check_input)

    def _fit(self, x, y, check_input):
        time_init = time.perf_counter()

        if self.verbose:
            logging.info("Optimal binning started.")
            logging.info("Options: check parameters.")

        _check_parameters(**self.get_params())

        # Pre-processing
        if self.verbose:
            logging.info("Pre-processing started.")

        self._n_samples = len(x)

        if self.verbose:
            logging.info("Pre-processing: number of samples: {}"
                         .format(self._n_samples))

        time_preprocessing = time.perf_counter()

        [x_clean, y_clean, x_missing, y_missing, x_special, y_special,
         y_others, categories, cat_others] = split_data(
            self.dtype, x, y, self.special_codes, self.cat_cutoff,
            self.user_splits, check_input)

        self._time_preprocessing = time.perf_counter() - time_preprocessing

        if self.verbose:
            n_clean = len(x_clean)
            n_missing = len(x_missing)
            n_special = len(x_special)

            logging.info("Pre-processing: number of clean samples: {}"
                         .format(n_clean))

            logging.info("Pre-processing: number of missing samples: {}"
                         .format(n_missing))

            logging.info("Pre-processing: number of special samples: {}"
                         .format(n_special))

            if self.dtype == "categorical":
                n_categories = len(categories)
                n_categories_others = len(cat_others)
                n_others = len(y_others)

                logging.info("Pre-processing: number of others samples: {}"
                             .format(n_others))

                logging.info("Pre-processing: number of categories: {}"
                             .format(n_categories))

                logging.info("Pre-processing: number of categories others: {}"
                             .format(n_categories_others))

            logging.info("Pre-processing terminated. Time: {:.4f}s"
                         .format(self._time_preprocessing))

        # Pre-binning
        if self.verbose:
            logging.info("Pre-binning started.")

        time_prebinning = time.perf_counter()

        if self.user_splits is not None:
            n_splits = len(self.user_splits)

            if self.verbose:
                logging.info("Pre-binning: user splits supplied: {}"
                             .format(n_splits))

            if not n_splits:
                splits = self.user_splits
                n_records = np.array([])
                sums = np.array([])
                stds = np.array([])
            else:
                if self.dtype == "numerical":
                    user_splits = check_array(
                        self.user_splits, ensure_2d=False, dtype=None,
                        force_all_finite=True)

                    user_splits = np.unique(self.user_splits)
                else:
                    [categories, user_splits, x_clean, y_clean, y_others,
                     cat_others] = preprocessing_user_splits_categorical(
                        self.user_splits, x_clean, y_clean)

                splits, n_records, sums, stds = self._prebinning_refinement(
                    user_splits, x_clean, y_clean, y_missing, y_special,
                    y_others)
        else:
            splits, n_records, sums, stds = self._fit_prebinning(
                x_clean, y_clean, y_missing, y_special, y_others)

        self._n_prebins = len(n_records)

        self._categories = categories
        self._cat_others = cat_others

        self._time_prebinning = time.perf_counter() - time_prebinning

        if self.verbose:
            logging.info("Pre-binning: number of prebins: {}"
                         .format(self._n_prebins))

            logging.info("Pre-binning terminated. Time: {:.4f}s"
                         .format(self._time_prebinning))

        # Optimization
        self._fit_optimizer(splits, n_records, sums, stds)

        # Post-processing
        if self.verbose:
            logging.info("Post-processing started.")
            logging.info("Post-processing: compute binning information.")

        time_postprocessing = time.perf_counter()

        if not len(splits):
            n_records = n_records.sum()
            sums = sums.sum()

        self._n_records, self._sums = continuous_bin_info(
            self._solution, n_records, sums, self._n_records_missing,
            self._sum_missing, self._n_records_special, self._sum_special,
            self._n_records_cat_others, self._sum_cat_others, self._cat_others)

        self._binning_table = ContinuousBinningTable(
            self.name, self.dtype, self._splits_optimal, self._n_records,
            self._sums, self._categories, self._cat_others,
            self.user_splits)

        self._time_postprocessing = time.perf_counter() - time_postprocessing

        if self.verbose:
            logging.info("Post-processing terminated. Time: {:.4f}s"
                         .format(self._time_postprocessing))

        self._time_total = time.perf_counter() - time_init

        if self.verbose:
            logging.info("Optimal binning terminated. Status: {}. "
                         "Time: {:.4f}s".format(
                            self._status, self._time_total))

        # Completed successfully
        self._logger.close()
        self._is_fitted = True

        return self

    def _fit_optimizer(self, splits, n_records, sums, stds):
        if self.verbose:
            logging.info("Optimizer started.")

        time_init = time.perf_counter()

        if not len(n_records):
            self._status = "OPTIMAL"
            self._splits_optimal = splits
            self._solution = np.zeros(len(splits)).astype(np.bool)

            if self.verbose:
                logging.warning("Optimizer: no bins after pre-binning.")
                logging.warning("Optimizer: solver not run.")

                logging.info("Optimizer terminated. Time: 0s")
            return

        if self.min_bin_size is not None:
            min_bin_size = np.int(np.ceil(self.min_bin_size * self._n_samples))
        else:
            min_bin_size = self.min_bin_size

        if self.max_bin_size is not None:
            max_bin_size = np.int(np.ceil(self.max_bin_size * self._n_samples))
        else:
            max_bin_size = self.max_bin_size

        # Monotonic trend
        if self.dtype == "numerical":
            if self.monotonic_trend == "auto":
                monotonic = auto_monotonic_continuous(n_records, sums)

                if self.verbose:
                    logging.info("Optimizer: classifier predicts {} monotonic "
                                 "trend.".format(monotonic))
            else:
                monotonic = self.monotonic_trend

                if self.verbose:
                    if monotonic is None:
                        logging.info("Optimizer: monotonic trend not set.")
                    else:
                        logging.info("Optimizer: monotonic trend set to {}."
                                     .format(monotonic))
        else:
            monotonic = "ascending"

            if self.verbose:
                logging.info("Optimizer: monotonic trend set to ascending for "
                             "categorical dtype.")

        optimizer = ContinuousBinningCP(monotonic, self.min_n_bins,
                                        self.max_n_bins, min_bin_size,
                                        max_bin_size, self.min_mean_diff,
                                        self.max_pvalue,
                                        self.max_pvalue_policy,
                                        self.time_limit)

        if self.verbose:
            logging.info("Optimizer: build model...")

        optimizer.build_model(n_records, sums, stds)

        if self.verbose:
            logging.info("Optimizer: solve...")

        status, solution = optimizer.solve()

        self._solution = solution

        self._optimizer = optimizer
        self._status = status
        self._splits_optimal = splits[solution[:-1]]

        self._time_solver = time.perf_counter() - time_init

        if self.verbose:
            logging.info("Optimizer terminated. Time: {:.4f}s"
                         .format(self._time_solver))

    def _prebinning_refinement(self, splits_prebinning, x, y, y_missing,
                               y_special, y_others):
        n_splits = len(splits_prebinning)

        if not n_splits:
            return splits_prebinning, np.array([]), np.array([])

        if self.split_digits is not None:
            splits_prebinning = np.round(splits_prebinning, self.split_digits)

        indices = np.digitize(x, splits_prebinning, right=False)

        # Compute n_records, sum and std for special, missing and others
        self._n_records_special = len(y_special)
        self._sum_special = np.sum(y_special)

        self._n_records_missing = len(y_missing)
        self._sum_missing = np.sum(y_missing)

        if len(y_others):
            self._n_records_cat_others = len(y_others)
            self._sum_cat_others = np.sum(y_others)

        n_bins = n_splits + 1
        n_records = np.zeros(n_bins).astype(np.int)
        sums = np.zeros(n_bins).astype(np.float)
        stds = np.zeros(n_bins).astype(np.float)

        # Compute prebin information
        for i in range(n_bins):
            mask = (indices == i)
            n_records[i] = np.count_nonzero(mask)
            sums[i] = np.sum(y[mask])
            stds[i] = np.std(y[mask])

        return splits_prebinning, n_records, sums, stds

    @property
    def binning_table(self):
        """Return an instantiated binning table. Please refer to
        :ref:`Binning table: continuous target`.

        Returns
        -------
        binning_table : ContinuousBinningTable.
        """
        self._check_is_fitted()

        return self._binning_table
