"""
Optimal binning algorithm for continuous target.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import numbers
import time

from sklearn.utils import check_array

import numpy as np

from ..information import solver_statistics
from ..logging import Logger
from .auto_monotonic import auto_monotonic_continuous
from .auto_monotonic import peak_valley_trend_change_heuristic
from .binning import OptimalBinning
from .binning_statistics import continuous_bin_info
from .binning_statistics import ContinuousBinningTable
from .binning_statistics import target_info_special_continuous
from .continuous_cp import ContinuousBinningCP
from .preprocessing import preprocessing_user_splits_categorical
from .preprocessing import split_data
from .transformations import transform_continuous_target


logger = Logger(__name__).logger


def _check_parameters(name, dtype, prebinning_method, max_n_prebins,
                      min_prebin_size, min_n_bins, max_n_bins, min_bin_size,
                      max_bin_size, monotonic_trend, min_mean_diff, max_pvalue,
                      max_pvalue_policy, gamma, outlier_detector,
                      outlier_params, cat_cutoff, user_splits,
                      user_splits_fixed, special_codes, split_digits,
                      time_limit, verbose):

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
        if monotonic_trend not in ("auto", "auto_heuristic", "auto_asc_desc",
                                   "ascending", "descending", "convex",
                                   "concave", "peak", "valley",
                                   "peak_heuristic", "valley_heuristic"):
            raise ValueError('Invalid value for monotonic trend. Allowed '
                             'string values are "auto", "auto_heuristic", '
                             '"auto_asc_desc", "ascending", "descending", '
                             '"concave", "convex", "peak", "valley", '
                             '"peak_heuristic" and "valley_heuristic".')

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

    if not isinstance(gamma, numbers.Number) or gamma < 0:
        raise ValueError("gamma must be >= 0; got {}.".format(gamma))

    if outlier_detector is not None:
        if outlier_detector not in ("range", "zscore"):
            raise ValueError('Invalid value for outlier_detector. Allowed '
                             'string values are "range" and "zscore".')

        if outlier_params is not None:
            if not isinstance(outlier_params, dict):
                raise TypeError("outlier_params must be a dict or None; "
                                "got {}.".format(outlier_params))

    if cat_cutoff is not None:
        if (not isinstance(cat_cutoff, numbers.Number) or
                not 0. < cat_cutoff <= 1.0):
            raise ValueError("cat_cutoff must be in (0, 1.0]; got {}."
                             .format(cat_cutoff))

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
        if not isinstance(special_codes, (np.ndarray, list, dict)):
            raise TypeError("special_codes must be a dit, list or "
                            "numpy.ndarray.")

        if isinstance(special_codes, dict) and not len(special_codes):
            raise ValueError("special_codes empty. special_codes dict must "
                             "contain at least one special.")

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
        The **mean** monotonic trend. Supported trends are “auto”,
        "auto_heuristic" and "auto_asc_desc" to automatically determine the
        trend minimize the L1-norm using a machine learning classifier,
        "ascending", "descending", "concave", "convex", "peak" and
        "peak_heuristic" to allow a peak change point, and "valley" and
        "valley_heuristic" to allow a valley change point. Trends
        "auto_heuristic", "peak_heuristic" and "valley_heuristic" use a
        heuristic to determine the change point, and are significantly faster
        for large size instances (``max_n_prebins> 20``). Trend "auto_asc_desc"
        is used to automatically select the best monotonic trend between
        "ascending" and "descending". If None, then the monotonic constraint
        is disabled.

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

    gamma : float, optional (default=0)
        Regularization strength to reduce the number of dominating bins. Larger
        values specify stronger regularization.

        .. versionadded:: 0.14.0

    outlier_detector : str or None, optional (default=None)
        The outlier detection method. Supported methods are "range" to use
        the interquartile range based method or "zcore" to use the modified
        Z-score method.

    outlier_params : dict or None, optional (default=None)
        Dictionary of parameters to pass to the outlier detection method.

    cat_cutoff : float or None, optional (default=None)
        Generate bin others with categories in which the fraction of
        occurrences is below the  ``cat_cutoff`` value. This option is
        available when ``dtype`` is "categorical".

    user_splits : array-like or None, optional (default=None)
        The list of pre-binning split points when ``dtype`` is "numerical" or
        the list of prebins when ``dtype`` is "categorical".

    user_splits_fixed : array-like or None (default=None)
        The list of pre-binning split points that must be fixed.

    special_codes : array-like, dict or None, optional (default=None)
        List of special codes. Use special codes to specify the data values
        that must be treated separately.

    split_digits : int or None, optional (default=None)
        The significant digits of the split points. If ``split_digits`` is set
        to 0, the split points are integers. If None, then all significant
        digits in the split points are considered.

    time_limit : int (default=100)
        The maximum time in seconds to run the optimization solver.

    verbose : bool (default=False)
        Enable verbose output.

    **prebinning_kwargs : keyword arguments
        The pre-binning keywrord arguments.

        .. versionadded:: 0.6.1

    Notes
    -----
    The parameter values ``max_n_prebins`` and ``min_prebin_size`` control
    complexity and memory usage. The default values generally produce quality
    results, however, some improvement can be achieved by increasing
    ``max_n_prebins`` and/or decreasing ``min_prebin_size``.

    The pre-binning refinement phase guarantee that no prebin has zero number
    of records by merging those pure prebins. Pure bins produce infinity mean.
    """
    def __init__(self, name="", dtype="numerical", prebinning_method="cart",
                 max_n_prebins=20, min_prebin_size=0.05, min_n_bins=None,
                 max_n_bins=None, min_bin_size=None, max_bin_size=None,
                 monotonic_trend="auto", min_mean_diff=0, max_pvalue=None,
                 max_pvalue_policy="consecutive", gamma=0,
                 outlier_detector=None, outlier_params=None, cat_cutoff=None,
                 user_splits=None, user_splits_fixed=None, special_codes=None,
                 split_digits=None, time_limit=100, verbose=False,
                 **prebinning_kwargs):

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
        self.gamma = gamma

        self.outlier_detector = outlier_detector
        self.outlier_params = outlier_params

        self.cat_cutoff = cat_cutoff

        self.user_splits = user_splits
        self.user_splits_fixed = user_splits_fixed
        self.special_codes = special_codes
        self.split_digits = split_digits

        self.time_limit = time_limit

        self.verbose = verbose
        self.prebinning_kwargs = prebinning_kwargs

        # auxiliary
        self._categories = None
        self._cat_others = None
        self._n_records = None
        self._sums = None
        self._stds = None
        self._min_target = None
        self._max_target = None
        self._n_zeros = None
        self._n_records_cat_others = None
        self._n_records_missing = None
        self._n_records_special = None
        self._sum_cat_others = None
        self._sum_special = None
        self._sum_missing = None
        self._std_cat_others = None
        self._std_special = None
        self._std_missing = None
        self._min_target_missing = None
        self._min_target_special = None
        self._min_target_others = None
        self._max_target_missing = None
        self._max_target_special = None
        self._max_target_others = None
        self._n_zeros_missing = None
        self._n_zeros_special = None
        self._n_zeros_others = None
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
        self._time_optimizer = None
        self._time_postprocessing = None

        self._is_fitted = False

    def fit(self, x, y, check_input=False):
        """Fit the optimal binning according to the given training data.

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
        self : ContinuousOptimalBinning
            Fitted optimal binning.
        """
        return self._fit(x, y, check_input)

    def fit_transform(self, x, y, metric="mean", metric_special=0,
                      metric_missing=0, show_digits=2, check_input=False):
        """Fit the optimal binning according to the given training data, then
        transform it.

        Parameters
        ----------
        x : array-like, shape = (n_samples,)
            Training vector, where n_samples is the number of samples.

        y : array-like, shape = (n_samples,)
            Target vector relative to x.

        metric : str (default="mean"):
            The metric used to transform the input vector. Supported metrics
            are "mean" to choose the mean, "indices" to assign the
            corresponding indices of the bins and "bins" to assign the
            corresponding bin interval.

        metric_special : float or str (default=0)
            The metric value to transform special codes in the input vector.
            Supported metrics are "empirical" to use the empirical mean, and
            any numerical value.

        metric_missing : float or str (default=0)
            The metric value to transform missing values in the input vector.
            Supported metrics are "empirical" to use the empirical mean, and
            any numerical value.

        show_digits : int, optional (default=2)
            The number of significant digits of the bin column. Applies when
            ``metric="bins"``.

        check_input : bool (default=False)
            Whether to check input arrays.

        Returns
        -------
        x_new : numpy array, shape = (n_samples,)
            Transformed array.
        """
        return self.fit(x, y, check_input).transform(
            x, metric, metric_special, metric_missing, show_digits,
            check_input)

    def transform(self, x, metric="mean", metric_special=0, metric_missing=0,
                  show_digits=2, check_input=False):
        """Transform given data to mean using bins from the fitted
        optimal binning.

        Parameters
        ----------
        x : array-like, shape = (n_samples,)
            Training vector, where n_samples is the number of samples.

        metric : str (default="mean"):
            The metric used to transform the input vector. Supported metrics
            are "mean" to choose the mean, "indices" to assign the
            corresponding indices of the bins and "bins" to assign the
            corresponding bin interval.

        metric_special : float or str (default=0)
            The metric value to transform special codes in the input vector.
            Supported metrics are "empirical" to use the empirical mean, and
            any numerical value.

        metric_missing : float or str (default=0)
            The metric value to transform missing values in the input vector.
            Supported metrics are "empirical" to use the empirical mean, and
            any numerical value.

        show_digits : int, optional (default=2)
            The number of significant digits of the bin column. Applies when
            ``metric="bins"``.

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
                                           metric, metric_special,
                                           metric_missing, self.user_splits,
                                           show_digits, check_input)

    def _fit(self, x, y, check_input):
        time_init = time.perf_counter()

        if self.verbose:
            logger.info("Optimal binning started.")
            logger.info("Options: check parameters.")

        _check_parameters(**self.get_params())

        # Pre-processing
        if self.verbose:
            logger.info("Pre-processing started.")

        self._n_samples = len(x)

        if self.verbose:
            logger.info("Pre-processing: number of samples: {}"
                        .format(self._n_samples))

        time_preprocessing = time.perf_counter()

        [x_clean, y_clean, x_missing, y_missing, x_special, y_special,
         y_others, categories, cat_others, _, _, _, _] = split_data(
            self.dtype, x, y, self.special_codes, self.cat_cutoff,
            self.user_splits, check_input, self.outlier_detector,
            self.outlier_params)

        self._time_preprocessing = time.perf_counter() - time_preprocessing

        if self.verbose:
            n_clean = len(x_clean)
            n_missing = len(x_missing)
            n_special = len(x_special)

            logger.info("Pre-processing: number of clean samples: {}"
                        .format(n_clean))

            logger.info("Pre-processing: number of missing samples: {}"
                        .format(n_missing))

            logger.info("Pre-processing: number of special samples: {}"
                        .format(n_special))

            if self.outlier_detector is not None:
                n_outlier = self._n_samples-(n_clean + n_missing + n_special)
                logger.info("Pre-processing: number of outlier samples: {}"
                            .format(n_outlier))

            if self.dtype == "categorical":
                n_categories = len(categories)
                n_categories_others = len(cat_others)
                n_others = len(y_others)

                logger.info("Pre-processing: number of others samples: {}"
                            .format(n_others))

                logger.info("Pre-processing: number of categories: {}"
                            .format(n_categories))

                logger.info("Pre-processing: number of categories others: {}"
                            .format(n_categories_others))

            logger.info("Pre-processing terminated. Time: {:.4f}s"
                        .format(self._time_preprocessing))

        # Pre-binning
        if self.verbose:
            logger.info("Pre-binning started.")

        time_prebinning = time.perf_counter()

        if self.user_splits is not None:
            n_splits = len(self.user_splits)

            if self.verbose:
                logger.info("Pre-binning: user splits supplied: {}"
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

                    if len(set(user_splits)) != len(user_splits):
                        raise ValueError("User splits are not unique.")

                    sorted_idx = np.argsort(user_splits)
                    user_splits = user_splits[sorted_idx]
                else:
                    [categories, user_splits, x_clean, y_clean, y_others,
                     cat_others, _, _, sorted_idx
                     ] = preprocessing_user_splits_categorical(
                        self.user_splits, x_clean, y_clean, None)

                if self.user_splits_fixed is not None:
                    self.user_splits_fixed = np.asarray(
                        self.user_splits_fixed)[sorted_idx]

                [splits, n_records, sums, ssums, stds, min_t, max_t,
                 n_zeros] = self._prebinning_refinement(
                    user_splits, x_clean, y_clean, y_missing, x_special,
                    y_special, y_others)
        else:
            [splits, n_records, sums, ssums, stds, min_t, max_t,
             n_zeros] = self._fit_prebinning(
                x_clean, y_clean, y_missing, x_special, y_special, y_others)

        self._n_prebins = len(n_records)

        self._categories = categories
        self._cat_others = cat_others

        self._time_prebinning = time.perf_counter() - time_prebinning

        if self.verbose:
            logger.info("Pre-binning: number of prebins: {}"
                        .format(self._n_prebins))

            logger.info("Pre-binning terminated. Time: {:.4f}s"
                        .format(self._time_prebinning))

        # Optimization
        self._fit_optimizer(splits, n_records, sums, ssums, stds)

        # Post-processing
        if self.verbose:
            logger.info("Post-processing started.")
            logger.info("Post-processing: compute binning information.")

        time_postprocessing = time.perf_counter()

        if not len(splits):
            n_records = n_records.sum()
            sums = sums.sum()

        [self._n_records, self._sums, self._stds, self._min_target,
         self._max_target, self._n_zeros] = continuous_bin_info(
            self._solution, n_records, sums, ssums, stds, min_t, max_t,
            n_zeros, self._n_records_missing, self._sum_missing,
            self._std_missing, self._min_target_missing,
            self._max_target_missing, self._n_zeros_missing,
            self._n_records_special, self._sum_special, self._std_special,
            self._min_target_special, self._max_target_special,
            self._n_zeros_special, self._n_records_cat_others,
            self._sum_cat_others, self._std_cat_others,
            self._min_target_others, self._max_target_others,
            self._n_zeros_others, self._cat_others)

        if self.dtype == "numerical":
            min_x = x_clean.min()
            max_x = x_clean.max()
        else:
            min_x = None
            max_x = None

        self._binning_table = ContinuousBinningTable(
            self.name, self.dtype, self.special_codes, self._splits_optimal,
            self._n_records, self._sums, self._stds, self._min_target,
            self._max_target, self._n_zeros, min_x, max_x, self._categories,
            self._cat_others, self.user_splits)

        self._time_postprocessing = time.perf_counter() - time_postprocessing

        if self.verbose:
            logger.info("Post-processing terminated. Time: {:.4f}s"
                        .format(self._time_postprocessing))

        self._time_total = time.perf_counter() - time_init

        if self.verbose:
            logger.info("Optimal binning terminated. Status: {}. Time: {:.4f}s"
                        .format(self._status, self._time_total))

        # Completed successfully
        self._is_fitted = True

        return self

    def _fit_optimizer(self, splits, n_records, sums, ssums, stds):
        if self.verbose:
            logger.info("Optimizer started.")

        time_init = time.perf_counter()

        if len(n_records) <= 1:
            self._status = "OPTIMAL"
            self._splits_optimal = splits
            self._solution = np.zeros(len(splits)).astype(bool)

            if self.verbose:
                logger.warning("Optimizer: {} bins after pre-binning."
                               .format(len(n_records)))
                logger.warning("Optimizer: solver not run.")
                logger.info("Optimizer terminated. Time: 0s")
            return

        if self.min_bin_size is not None:
            min_bin_size = int(np.ceil(self.min_bin_size * self._n_samples))
        else:
            min_bin_size = self.min_bin_size

        if self.max_bin_size is not None:
            max_bin_size = int(np.ceil(self.max_bin_size * self._n_samples))
        else:
            max_bin_size = self.max_bin_size

        # Monotonic trend
        trend_change = None

        if self.dtype == "numerical":
            auto_monotonic_modes = ("auto", "auto_heuristic", "auto_asc_desc")
            if self.monotonic_trend in auto_monotonic_modes:
                monotonic = auto_monotonic_continuous(
                    n_records, sums, self.monotonic_trend)

                if self.monotonic_trend == "auto_heuristic":
                    if monotonic in ("peak", "valley"):
                        if monotonic == "peak":
                            monotonic = "peak_heuristic"
                        else:
                            monotonic = "valley_heuristic"

                        mean = sums / n_records
                        trend_change = peak_valley_trend_change_heuristic(
                            mean, monotonic)

                if self.verbose:
                    logger.info("Optimizer: classifier predicts {} "
                                "monotonic trend.".format(monotonic))
            else:
                monotonic = self.monotonic_trend

                if monotonic in ("peak_heuristic", "valley_heuristic"):
                    mean = sums / n_records
                    trend_change = peak_valley_trend_change_heuristic(
                        mean, monotonic)

        else:
            monotonic = self.monotonic_trend
            if monotonic is not None:
                monotonic = "ascending"

        if self.verbose:
            if monotonic is None:
                logger.info(
                    "Optimizer: monotonic trend not set.")
            else:
                logger.info("Optimizer: monotonic trend set to {}."
                            .format(monotonic))

        optimizer = ContinuousBinningCP(monotonic, self.min_n_bins,
                                        self.max_n_bins, min_bin_size,
                                        max_bin_size, self.min_mean_diff,
                                        self.max_pvalue,
                                        self.max_pvalue_policy, self.gamma,
                                        self.user_splits_fixed,
                                        self.time_limit)

        if self.verbose:
            logger.info("Optimizer: build model...")

        optimizer.build_model(n_records, sums, ssums, trend_change)

        if self.verbose:
            logger.info("Optimizer: solve...")

        status, solution = optimizer.solve()

        self._solution = solution

        self._optimizer, self._time_optimizer = solver_statistics(
            self.solver, optimizer.solver_)
        self._status = status

        if self.dtype == "categorical" and self.user_splits is not None:
            self._splits_optimal = splits[solution]
        else:
            self._splits_optimal = splits[solution[:-1]]

        self._time_solver = time.perf_counter() - time_init

        if self.verbose:
            logger.info("Optimizer terminated. Time: {:.4f}s"
                        .format(self._time_solver))

    def _prebinning_refinement(self, splits_prebinning, x, y, y_missing,
                               x_special, y_special, y_others, sw_clean=None,
                               sw_missing=None, sw_special=None,
                               sw_others=None):
        n_splits = len(splits_prebinning)

        if not n_splits:
            return splits_prebinning, np.array([]), np.array([])

        if self.split_digits is not None:
            splits_prebinning = np.round(splits_prebinning, self.split_digits)

        # Compute n_records, sum and std for special, missing and others
        [self._n_records_special, self._sum_special, self._n_zeros_special,
         self._std_special, self._min_target_special,
         self._max_target_special] = target_info_special_continuous(
            self.special_codes, x_special, y_special)

        self._n_records_missing = len(y_missing)
        self._sum_missing = np.sum(y_missing)
        self._n_zeros_missing = np.count_nonzero(y_missing == 0)
        if len(y_missing):
            self._std_missing = np.std(y_missing)
            self._min_target_missing = np.min(y_missing)
            self._max_target_missing = np.max(y_missing)

        if len(y_others):
            self._n_records_cat_others = len(y_others)
            self._sum_cat_others = np.sum(y_others)
            self._std_cat_others = np.std(y_others)
            self._min_target_others = np.min(y_others)
            self._max_target_others = np.max(y_others)
            self._n_zeros_others = np.count_nonzero(y_others == 0)

        (splits_prebinning, n_records, sums, ssums, stds, min_t, max_t,
         n_zeros) = self._compute_prebins(splits_prebinning, x, y)

        return (splits_prebinning, n_records, sums, ssums, stds, min_t, max_t,
                n_zeros)

    def _compute_prebins(self, splits_prebinning, x, y):
        n_splits = len(splits_prebinning)
        if not n_splits:
            return splits_prebinning, np.array([]), np.array([])

        if self.dtype == "categorical" and self.user_splits is not None:
            indices = np.digitize(x, splits_prebinning, right=True)
            n_bins = n_splits
        else:
            indices = np.digitize(x, splits_prebinning, right=False)
            n_bins = n_splits + 1

        n_records = np.empty(n_bins).astype(np.int64)
        sums = np.empty(n_bins)
        ssums = np.empty(n_bins)
        stds = np.zeros(n_bins)
        n_zeros = np.empty(n_bins).astype(np.int64)
        min_t = np.full(n_bins, -np.inf)
        max_t = np.full(n_bins, np.inf)

        # Compute prebin information
        for i in range(n_bins):
            mask = (indices == i)
            n_records[i] = np.count_nonzero(mask)
            ymask = y[mask]
            sums[i] = np.sum(ymask)
            ssums[i] = np.sum(ymask ** 2)
            n_zeros[i] = np.count_nonzero(ymask == 0)
            if len(ymask):
                stds[i] = np.std(ymask)
                min_t[i] = np.min(ymask)
                max_t[i] = np.max(ymask)

        mask_remove = (n_records == 0)

        if np.any(mask_remove):
            self._n_refinements += 1

            if (self.dtype == "categorical" and
                    self.user_splits is not None):
                mask_splits = mask_remove
            else:
                mask_splits = np.concatenate([
                    mask_remove[:-2], [mask_remove[-2] | mask_remove[-1]]])

            if self.user_splits_fixed is not None:
                user_splits_fixed = np.asarray(self.user_splits_fixed)
                user_splits = np.asarray(self.user_splits)
                fixed_remove = user_splits_fixed & mask_splits

                if any(fixed_remove):
                    raise ValueError(
                        "Fixed user_splits {} are removed "
                        "because produce pure prebins. Provide "
                        "different splits to be fixed."
                        .format(user_splits[fixed_remove]))

                # Update boolean array of fixed user splits.
                self.user_splits_fixed = user_splits_fixed[~mask_splits]
                self.user_splits = user_splits[~mask_splits]

            splits = splits_prebinning[~mask_splits]
            if self.verbose:
                logger.info("Pre-binning: number prebins removed: {}"
                            .format(np.count_nonzero(mask_remove)))

            (splits_prebinning, n_records, sums, ssums, stds, min_t, max_t,
             n_zeros) = self._compute_prebins(splits, x, y)

        return (splits_prebinning, n_records, sums, ssums, stds, min_t, max_t,
                n_zeros)

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
