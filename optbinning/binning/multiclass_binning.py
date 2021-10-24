"""
Optimal binning algorithm for multiclass target.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import numbers
import time

import numpy as np

from sklearn.utils import check_array

from ..information import solver_statistics
from ..logging import Logger
from .auto_monotonic import auto_monotonic
from .auto_monotonic import peak_valley_trend_change_heuristic
from .binning import OptimalBinning
from .binning_statistics import multiclass_bin_info
from .binning_statistics import MulticlassBinningTable
from .binning_statistics import target_info
from .binning_statistics import target_info_special_multiclass
from .multiclass_cp import MulticlassBinningCP
from .multiclass_mip import MulticlassBinningMIP
from .preprocessing import split_data
from .transformations import transform_multiclass_target


logger = Logger(__name__).logger


def _check_parameters(name, prebinning_method, solver, max_n_prebins,
                      min_prebin_size, min_n_bins, max_n_bins, min_bin_size,
                      max_bin_size, monotonic_trend, max_pvalue,
                      max_pvalue_policy, outlier_detector, outlier_params,
                      user_splits, user_splits_fixed, special_codes,
                      split_digits, mip_solver, time_limit, verbose):

    if not isinstance(name, str):
        raise TypeError("name must be a string.")

    if prebinning_method not in ("cart", "quantile", "uniform"):
        raise ValueError('Invalid value for prebinning_method. Allowed string '
                         'values are "cart", "quantile" and "uniform".')

    if solver not in ("cp", "mip"):
        raise ValueError('Invalid value for solver. Allowed string '
                         'values are "cp" and "mip".')

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
        if isinstance(monotonic_trend, list):
            for trend in monotonic_trend:
                if trend not in ("auto", "auto_heuristic", "auto_asc_desc",
                                 "ascending", "descending", "convex",
                                 "concave", "peak", "valley", "peak_heuristic",
                                 "valley_heuristic", None):
                    raise ValueError('Invalid value for monotonic trend. '
                                     'Allowed string values are "auto", '
                                     '"auto_heuristic", "auto_asc_desc", '
                                     '"ascending", "descending", "concave", '
                                     '"convex", "peak" "valley", '
                                     '"peak_heuristic", "valley_heuristic" '
                                     'and None')

        elif (not isinstance(monotonic_trend, str) or
                monotonic_trend not in ("auto", "auto_heuristic",
                                        "auto_asc_desc")):
            raise ValueError("Invalid value for monotonic trend; got {}."
                             .format(monotonic_trend))

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

    if mip_solver not in ("bop", "cbc"):
        raise ValueError('Invalid value for mip_solver. Allowed string '
                         'values are "bop" and "cbc".')

    if not isinstance(time_limit, numbers.Number) or time_limit < 0:
        raise ValueError("time_limit must be a positive value in seconds; "
                         "got {}.".format(time_limit))

    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a boolean; got {}.".format(verbose))


class MulticlassOptimalBinning(OptimalBinning):
    """Optimal binning of a numerical variable with respect to a multiclass or
    multilabel target.

    **Note that the maximum number of classes is set to 100**. To ease
    visualization of the binning table, a set of 100 maximally distinct colors
    is generated using the library `glasbey
    <https://github.com/taketwo/glasbey>`_.

    Parameters
    ----------
    name : str, optional (default="")
        The variable name.

    prebinning_method : str, optional (default="cart")
        The pre-binning method. Supported methods are "cart" for a CART
        decision tree, "quantile" to generate prebins with approximately same
        frequency and "uniform" to generate prebins with equal width. Method
        "cart" uses `sklearn.tree.DecistionTreeClassifier
        <https://scikit-learn.org/stable/modules/generated/sklearn.tree.
        DecisionTreeClassifier.html>`_.

    solver : str, optional (default="cp")
        The optimizer to solve the optimal binning problem. Supported solvers
        are "mip" to choose a mixed-integer programming solver or "cp" to
        choose a constrained programming solver.

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

    monotonic_trend : str, array-like or None, optional (default="auto")
        The **event rate** monotonic trend. Supported trends are “auto”,
        "auto_heuristic" and "auto_asc_desc" to automatically determine the
        trend maximizing IV using a machine learning classifier, a list of
        monotonic trends combining "auto", "auto_heuristic", "auto_asc_desc",
        "ascending", "descending", "concave", "convex", "peak", "valley",
        "peak_heuristic", "valley_heuristic" and None, one for each class.
        If None, then the monotonic constraint is disabled.

    max_pvalue : float or None, optional (default=0.05)
        The maximum p-value among bins. The Z-test is used to detect bins
        not satisfying the p-value constraint.

    max_pvalue_policy : str, optional (default="consecutive")
        The method to determine bins not satisfying the p-value constraint.
        Supported methods are "consecutive" to compare consecutive bins and
        "all" to compare all bins.

    outlier_detector : str or None, optional (default=None)
        The outlier detection method. Supported methods are "range" to use
        the interquartile range based method or "zcore" to use the modified
        Z-score method.

    outlier_params : dict or None, optional (default=None)
        Dictionary of parameters to pass to the outlier detection method.

    user_splits : array-like or None, optional (default=None)
        The list of pre-binning split points.

    user_splits_fixed : array-like or None (default=None)
        The list of pre-binning split points that must be fixed.

    special_codes : array-like, dict or None, optional (default=None)
        List of special codes. Use special codes to specify the data values
        that must be treated separately.

    split_digits : int or None, optional (default=None)
        The significant digits of the split points. If ``split_digits`` is set
        to 0, the split points are integers. If None, then all significant
        digits in the split points are considered.

    mip_solver : str, optional (default="bop")
        The mixed-integer programming solver. Supported solvers are "bop" to
        choose the Google OR-Tools binary optimizer or "cbc" to choose the
        COIN-OR Branch-and-Cut solver CBC.

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

    The pre-binning refinement phase guarantee that no prebin has either zero
    counts of non-events or events by merging those pure prebins. Pure bins
    produce infinity WoE and event rates.
    """
    def __init__(self, name="", prebinning_method="cart", solver="cp",
                 max_n_prebins=20, min_prebin_size=0.05,
                 min_n_bins=None, max_n_bins=None, min_bin_size=None,
                 max_bin_size=None, monotonic_trend="auto", max_pvalue=None,
                 max_pvalue_policy="consecutive", outlier_detector=None,
                 outlier_params=None, user_splits=None, user_splits_fixed=None,
                 special_codes=None, split_digits=None, mip_solver="bop",
                 time_limit=100, verbose=False, **prebinning_kwargs):

        self.name = name
        self.dtype = "numerical"
        self.prebinning_method = prebinning_method
        self.solver = solver

        self.max_n_prebins = max_n_prebins
        self.min_prebin_size = min_prebin_size

        self.min_n_bins = min_n_bins
        self.max_n_bins = max_n_bins
        self.min_bin_size = min_bin_size
        self.max_bin_size = max_bin_size

        self.monotonic_trend = monotonic_trend
        self.max_pvalue = max_pvalue
        self.max_pvalue_policy = max_pvalue_policy

        self.outlier_detector = outlier_detector
        self.outlier_params = outlier_params

        self.user_splits = user_splits
        self.user_splits_fixed = user_splits_fixed
        self.special_codes = special_codes
        self.split_digits = split_digits

        self.mip_solver = mip_solver
        self.time_limit = time_limit

        self.verbose = verbose
        self.prebinning_kwargs = prebinning_kwargs

        # auxiliary
        self._n_event = None
        self._n_event_missing = None
        self._n_event_special = None
        self._problem_type = "classification"

        # info
        self._binning_table = None
        self._classes = None
        self._n_classes = None
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
        self : MulticlassOptimalBinning
            Fitted optimal binning.
        """
        return self._fit(x, y, check_input)

    def fit_transform(self, x, y, metric="mean_woe", metric_special=0,
                      metric_missing=0, show_digits=2, check_input=False):
        """Fit the optimal binning according to the given training data, then
        transform it.

        Parameters
        ----------
        x : array-like, shape = (n_samples,)
            Training vector, where n_samples is the number of samples.

        y : array-like, shape = (n_samples,)
            Target vector relative to x.

        metric : str, optional (default="mean_woe")
            The metric used to transform the input vector. Supported metrics
            are "mean_woe" to choose the mean of Weight of Evidence (WoE),
            "weighted_mean_woe" to choose weighted mean of WoE using the
            number of records per class as weights, "indices" to assign the
            corresponding indices of the bins and "bins" to assign the
            corresponding bin interval.

        metric_special : float or str (default=0)
            The metric value to transform special codes in the input vector.
            Supported metrics are "empirical" to use the empirical mean WoE
            or weighted mean WoE, and any numerical value.

        metric_missing : float or str (default=0)
            The metric value to transform missing values in the input vector.
            Supported metrics are "empirical" to use the empirical mean WoE
            or weighted mean WoE, and any numerical value.

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

    def transform(self, x, metric="mean_woe", metric_special=0,
                  metric_missing=0, show_digits=2, check_input=False):
        """Transform given data to mean Weight of Evidence (WoE) or weighted
        mean WoE using bins from the fitted optimal binning.

        Parameters
        ----------
        x : array-like, shape = (n_samples,)
            Training vector, where n_samples is the number of samples.

        metric : str, optional (default="mean_woe")
            The metric used to transform the input vector. Supported metrics
            are "mean_woe" to choose the mean of Weight of Evidence (WoE),
            "weighted_mean_woe" to choose weighted mean of WoE using the
            number of records per class as weights, "indices" to assign the
            corresponding indices of the bins and "bins" to assign the
            corresponding bin interval.

        metric_special : float or str (default=0)
            The metric value to transform special codes in the input vector.
            Supported metrics are "empirical" to use the empirical mean WoE
            or weighted mean WoE, and any numerical value.

        metric_missing : float or str (default=0)
            The metric value to transform missing values in the input vector.
            Supported metrics are "empirical" to use the empirical mean WoE
            or weighted mean WoE, and any numerical value.

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
        self._check_is_fitted()

        return transform_multiclass_target(self._splits_optimal, x,
                                           self._n_event, self.special_codes,
                                           metric,  metric_special,
                                           metric_missing, show_digits,
                                           check_input)

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
         _, _, _, _, _, _, _] = split_data(
            self.dtype, x, y, special_codes=self.special_codes,
            check_input=check_input, outlier_detector=self.outlier_detector,
            outlier_params=self.outlier_params)

        # Check that x_clean is numerical
        if x_clean.dtype == np.dtype("object"):
            raise ValueError("x array after removing special codes and "
                             "missing values must be numerical.")

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

            user_splits = check_array(self.user_splits, ensure_2d=False,
                                      dtype=None, force_all_finite=True)

            if len(set(user_splits)) != len(user_splits):
                raise ValueError("User splits are not unique.")

            sorted_idx = np.argsort(user_splits)
            user_splits = user_splits[sorted_idx]

            if self.user_splits_fixed is not None:
                self.user_splits_fixed = np.asarray(
                    self.user_splits_fixed)[sorted_idx]

            splits, n_nonevent, n_event = self._prebinning_refinement(
                user_splits, x_clean, y_clean, y_missing, x_special, y_special,
                None)
        else:
            splits, n_nonevent, n_event = self._fit_prebinning(
                x_clean, y_clean, y_missing, x_special, y_special, None)

        self._n_prebins = len(n_nonevent)

        self._time_prebinning = time.perf_counter() - time_prebinning

        if self.verbose:
            logger.info("Pre-binning: number of prebins: {}"
                        .format(self._n_prebins))
            logger.info("Pre-binning: number of refinements: {}"
                        .format(self._n_refinements))

            logger.info("Pre-binning terminated. Time: {:.4f}s"
                        .format(self._time_prebinning))

        # Optimization
        self._fit_optimizer(splits, n_nonevent, n_event)

        # Post-processing
        if self.verbose:
            logger.info("Post-processing started.")
            logger.info("Post-processing: compute binning information.")

        time_postprocessing = time.perf_counter()

        if not len(splits):
            n_event = np.empty(self._n_classes).astype(np.int64)

            for i, cl in enumerate(self._classes):
                n_event[i] = target_info(y_clean, cl)[0]

        self._n_event = multiclass_bin_info(
            self._solution, self._n_classes, n_event, self._n_event_missing,
            self._n_event_special)

        self._binning_table = MulticlassBinningTable(
            self.name, self.special_codes, self._splits_optimal, self._n_event,
            self._classes)

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

    def _prebinning_refinement(self, splits_prebinning, x, y, y_missing,
                               x_special, y_special, y_others=None,
                               sw_clean=None, sw_missing=None, sw_special=None,
                               sw_others=None):

        self._classes = np.unique(y)
        self._n_classes = len(self._classes)

        if self._n_classes > 100:
            raise ValueError("Maximum number of classes exceeded; got {}."
                             .format(self._n_classes))

        self._n_event_special = target_info_special_multiclass(
            self.special_codes, x_special, y_special, self._classes)
        self._n_event_missing = [target_info(y_missing, cl)[0]
                                 for cl in self._classes]

        n_splits = len(splits_prebinning)

        if not n_splits:
            return splits_prebinning, np.array([]), np.array([])

        if self.split_digits is not None:
            splits_prebinning = np.round(splits_prebinning, self.split_digits)

        splits_prebinning, n_nonevent, n_event = self._compute_prebins(
            splits_prebinning, x, y)

        return splits_prebinning, n_nonevent, n_event

    def _fit_optimizer(self, splits, n_nonevent, n_event):
        if self.verbose:
            logger.info("Optimizer started.")

        time_init = time.perf_counter()

        if not len(n_nonevent):
            self._status = "OPTIMAL"
            self._splits_optimal = splits
            self._solution = np.zeros(len(splits)).astype(bool)

            if self.verbose:
                logger.warning("Optimizer: no bins after pre-binning.")
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
        trend_changes = [None] * self._n_classes

        auto_monotonic_modes = ("auto", "auto_heuristic", "auto_asc_desc")
        if self.monotonic_trend in auto_monotonic_modes:
            monotonic = [auto_monotonic(n_nonevent[:, i], n_event[:, i],
                                        self.monotonic_trend)
                         for i in range(len(self._classes))]

            if self.verbose:
                logger.info("Optimizer: classifier predicts {} "
                            "monotonic trends.".format(monotonic))
        elif isinstance(self.monotonic_trend, list):
            if len(self.monotonic_trend) != self._n_classes:
                raise ValueError("List of monotonic trends must be of size "
                                 "n_classes.")

            monotonic = []
            for i, m_trend in enumerate(self.monotonic_trend):
                if m_trend in auto_monotonic_modes:
                    trend = auto_monotonic(n_nonevent[:, i], n_event[:, i],
                                           m_trend)

                    if m_trend == "auto_heuristic":
                        if trend in ("peak", "valley"):
                            if trend == "peak":
                                trend = "peak_heuristic"
                            else:
                                trend = "valley_heuristic"

                            event_rate = n_event[:, i] / (n_nonevent[:, i] +
                                                          n_event[:, i])
                            trend_change = peak_valley_trend_change_heuristic(
                                event_rate, trend)
                            trend_changes[i] = trend_change

                    monotonic.append(trend)

                    if self.verbose:
                        logger.info("Optimizer: classifier predicts {} "
                                    "monotonic trend.".format(trend))
                else:
                    monotonic.append(m_trend)
        elif self.monotonic_trend is None:
            monotonic = [None] * self._n_classes

            if self.verbose:
                logger.info("Optimizer: monotonic trend not set.")

        if self.solver == "cp":
            optimizer = MulticlassBinningCP(monotonic, self.min_n_bins,
                                            self.max_n_bins, min_bin_size,
                                            max_bin_size, self.max_pvalue,
                                            self.max_pvalue_policy,
                                            self.user_splits_fixed,
                                            self.time_limit)
        else:
            optimizer = MulticlassBinningMIP(monotonic, self.min_n_bins,
                                             self.max_n_bins, min_bin_size,
                                             max_bin_size, self.max_pvalue,
                                             self.max_pvalue_policy,
                                             self.mip_solver,
                                             self.user_splits_fixed,
                                             self.time_limit)
        if self.verbose:
            logger.info("Optimizer: build model...")

        optimizer.build_model(n_nonevent, n_event, trend_changes)

        if self.verbose:
            logger.info("Optimizer: solve...")

        status, solution = optimizer.solve()

        self._solution = solution

        self._optimizer, self._time_optimizer = solver_statistics(
            self.solver, optimizer.solver_)
        self._status = status
        self._splits_optimal = splits[solution[:-1]]

        self._time_solver = time.perf_counter() - time_init

        if self.verbose:
            logger.info("Optimizer terminated. Time: {:.4f}s"
                        .format(self._time_solver))

    def _compute_prebins(self, splits_prebinning, x, y):
        n_splits = len(splits_prebinning)

        if not n_splits:
            return splits_prebinning, np.array([]), np.array([])

        indices = np.digitize(x, splits_prebinning, right=False)

        n_bins = n_splits + 1
        n_nonevent = np.empty((n_bins, self._n_classes)).astype(np.int64)
        n_event = np.empty((n_bins, self._n_classes)).astype(np.int64)
        mask_remove = np.zeros(n_bins).astype(bool)

        for idx, cl in enumerate(self._classes):
            y1 = (y == cl)
            y0 = ~y1

            for i in range(n_bins):
                mask = (indices == i)
                n_nonevent[i, idx] = np.count_nonzero(y0 & mask)
                n_event[i, idx] = np.count_nonzero(y1 & mask)

            mask_remove |= (n_nonevent[:, idx] == 0) | (n_event[:, idx] == 0)

        if np.any(mask_remove):
            self._n_refinements += 1

            mask_splits = np.concatenate(
                [mask_remove[:-2], [mask_remove[-2] | mask_remove[-1]]])

            if self.user_splits_fixed is not None:
                user_splits_fixed = np.asarray(self.user_splits_fixed)
                user_splits = np.asarray(self.user_splits)
                fixed_remove = user_splits_fixed & mask_splits

                if any(fixed_remove):
                    raise ValueError("Fixed user_splits {} are removed "
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

            [splits_prebinning, n_nonevent, n_event] = self._compute_prebins(
                splits, x, y)

        return splits_prebinning, n_nonevent, n_event

    @property
    def binning_table(self):
        """Return an instantiated binning table. Please refer to
        :ref:`Binning table: multiclass target`.

        Returns
        -------
        binning_table : MulticlassBinningTable.
        """
        self._check_is_fitted()

        return self._binning_table

    @property
    def classes(self):
        """List of classes.

        Returns
        -------
        classes : numpy.ndarray
        """
        self._check_is_fitted()

        return self._classes

    @property
    def splits(self):
        """List of optimal split points.

        Returns
        -------
        splits : numpy.ndarray
        """
        self._check_is_fitted()

        return self._splits_optimal
