"""
Optimal binning algorithm.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import logging
import numbers
import time

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_array

from ..logging import Logger
from ..preprocessing import preprocessing_user_splits_categorical
from ..preprocessing import split_data
from .auto_monotonic import auto_monotonic
from .auto_monotonic import peak_valley_trend_change_heuristic
from .binning_information import print_binning_information
from .binning_statistics import bin_categorical
from .binning_statistics import bin_info
from .binning_statistics import BinningTable
from .binning_statistics import target_info
from .cp import BinningCP
from .ls import BinningLS
from .mip import BinningMIP
from .prebinning import PreBinning
from .transformations import transform_binary_target


def _check_parameters(name, dtype, prebinning_method, solver, max_n_prebins,
                      min_prebin_size, min_n_bins, max_n_bins, min_bin_size,
                      max_bin_size, min_bin_n_nonevent, max_bin_n_nonevent,
                      min_bin_n_event, max_bin_n_event, monotonic_trend,
                      min_event_rate_diff, max_pvalue, max_pvalue_policy,
                      gamma, outlier_detector, outlier_params, class_weight,
                      cat_cutoff, user_splits, special_codes, split_digits,
                      mip_solver, time_limit, verbose):

    if not isinstance(name, str):
        raise TypeError("name must be a string.")

    if dtype not in ("categorical", "numerical"):
        raise ValueError('Invalid value for dtype. Allowed string '
                         'values are "categorical" and "numerical".')

    if prebinning_method not in ("cart", "quantile", "uniform"):
        raise ValueError('Invalid value for prebinning_method. Allowed string '
                         'values are "cart", "quantile" and "uniform".')

    if solver not in ("cp", "ls", "mip"):
        raise ValueError('Invalid value for solver. Allowed string '
                         'values are "cp", "ls" and "mip".')

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

    if min_bin_n_nonevent is not None:
        if (not isinstance(min_bin_n_nonevent, numbers.Integral) or
                min_bin_n_nonevent <= 0):
            raise ValueError("min_bin_n_nonevent must be a positive integer; "
                             "got {}.".format(min_bin_n_nonevent))

    if max_bin_n_nonevent is not None:
        if (not isinstance(max_bin_n_nonevent, numbers.Integral) or
                max_bin_n_nonevent <= 0):
            raise ValueError("max_bin_n_nonevent must be a positive integer; "
                             "got {}.".format(max_bin_n_nonevent))

    if min_bin_n_nonevent is not None and max_bin_n_nonevent is not None:
        if min_bin_n_nonevent > max_bin_n_nonevent:
            raise ValueError("min_bin_n_nonevent must be <= "
                             "max_bin_n_nonevent; got {} <= {}."
                             .format(min_bin_n_nonevent, max_bin_n_nonevent))

    if min_bin_n_event is not None:
        if (not isinstance(min_bin_n_event, numbers.Integral) or
                min_bin_n_event <= 0):
            raise ValueError("min_bin_n_event must be a positive integer; "
                             "got {}.".format(min_bin_n_event))

    if max_bin_n_event is not None:
        if (not isinstance(max_bin_n_event, numbers.Integral) or
                max_bin_n_event <= 0):
            raise ValueError("max_bin_n_event must be a positive integer; "
                             "got {}.".format(max_bin_n_event))

    if min_bin_n_event is not None and max_bin_n_event is not None:
        if min_bin_n_event > max_bin_n_event:
            raise ValueError("min_bin_n_event must be <= "
                             "max_bin_n_event; got {} <= {}."
                             .format(min_bin_n_event, max_bin_n_event))

    if monotonic_trend is not None:
        if monotonic_trend not in ("auto", "auto_heuristic", "auto_asc_desc",
                                   "ascending", "descending", "convex",
                                   "concave", "peak", "valley",
                                   "peak_heuristic", "valley_heuristic"):
            raise ValueError('Invalid value for monotonic trend. Allowed '
                             'string values are "auto", "auto_heuristic", '
                             '"auto_asc_desc", "ascending", "descending", '
                             '"concave", "convex", "peak" "valley", '
                             '"peak_heuristic" and "valley_heuristic".')

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

    if class_weight is not None:
        if not isinstance(class_weight, (dict, str)):
            raise TypeError('class_weight must be dict, "balanced" or None; '
                            'got {}.'.format(class_weight))

        elif isinstance(class_weight, str) and class_weight != "balanced":
            raise ValueError('Invalid value for class_weight. Allowed string '
                             'value is "balanced".')

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
            raise ValueError("split_digits must be an integer in [0, 8]; "
                             "got {}.".format(split_digits))

    if mip_solver not in ("bop", "cbc"):
        raise ValueError('Invalid value for mip_solver. Allowed string '
                         'values are "bop" and "cbc".')

    if not isinstance(time_limit, numbers.Number) or time_limit < 0:
        raise ValueError("time_limit must be a positive value in seconds; "
                         "got {}.".format(time_limit))

    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a boolean; got {}.".format(verbose))


class OptimalBinning(BaseEstimator):
    """Optimal binning of a numerical or categorical variable with respect to a
    binary target.

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
        "cart" uses `sklearn.tree.DecistionTreeClassifier
        <https://scikit-learn.org/stable/modules/generated/sklearn.tree.
        DecisionTreeClassifier.html>`_.

    solver : str, optional (default="cp")
        The optimizer to solve the optimal binning problem. Supported solvers
        are "mip" to choose a mixed-integer programming solver, "cp" to choose
        a constrained programming solver or "ls" to choose `LocalSolver
        <https://www.localsolver.com/>`_.

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

    min_bin_n_nonevent : int or None, optional (default=None)
        The minimum number of non-event records for each bin. If None,
        ``min_bin_n_nonevent = 1``.

    max_bin_n_nonevent : int or None, optional (default=None)
        The maximum number of non-event records for each bin. If None, then an
        unlimited number of non-event records for each bin.

    min_bin_n_event : int or None, optional (default=None)
        The minimum number of event records for each bin. If None,
        ``min_bin_n_event = 1``.

    max_bin_n_event : int or None, optional (default=None)
        The maximum number of event records for each bin. If None, then an
        unlimited number of event records for each bin.

    monotonic_trend : str or None, optional (default="auto")
        The **event rate** monotonic trend. Supported trends are “auto”,
        "auto_heuristic" and "auto_asc_desc" to automatically determine the
        trend maximizing IV using a machine learning classifier, "ascending",
        "descending", "concave", "convex", "peak" and "peak_heuristic" to allow
        a peak change point, and "valley" and "valley_heuristic" to allow a
        valley change point. Trends "auto_heuristic", "peak_heuristic" and
        "valley_heuristic" use a heuristic to determine the change point,
        and are significantly faster for large size instances (``max_n_prebins
        > 20``). Trend "auto_asc_desc" is used to automatically select the best
        monotonic trend between "ascending" and "descending". If None, then the
        monotonic constraint is disabled.

    min_event_rate_diff : float, optional (default=0)
        The minimum event rate difference between consecutives bins. This
        option currently only applies when ``monotonic_trend`` is "ascending",
        "descending", "peak_heuristic" or "valley_heuristic".

    max_pvalue : float or None, optional (default=0.05)
        The maximum p-value among bins. The Z-test is used to detect bins
        not satisfying the p-value constraint. Option supported by solvers
        "cp" and "mip".

    max_pvalue_policy : str, optional (default="consecutive")
        The method to determine bins not satisfying the p-value constraint.
        Supported methods are "consecutive" to compare consecutive bins and
        "all" to compare all bins.

    gamma : float, optional (default=0)
        Regularization strength to reduce the number of dominating bins. Larger
        values specify stronger regularization. Option supported by solvers
        "cp" and "mip".

        .. versionadded:: 0.3.0

    outlier_detector : str or None, optional (default=None)
        The outlier detection method. Supported methods are "range" to use
        the interquartile range based method or "zcore" to use the modified
        Z-score method.

    outlier_params : dict or None, optional (default=None)
        Dictionary of parameters to pass to the outlier detection method.

    class_weight : dict, "balanced" or None, optional (default=None)
        Weights associated with classes in the form ``{class_label: weight}``.
        If None, all classes are supposed to have weight one. Check
        `sklearn.tree.DecistionTreeClassifier
        <https://scikit-learn.org/stable/modules/generated/sklearn.tree.
        DecisionTreeClassifier.html>`_.

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

    mip_solver : str, optional (default="bop")
        The mixed-integer programming solver. Supported solvers are "bop" to
        choose the Google OR-Tools binary optimizer or "cbc" to choose the
        COIN-OR Branch-and-Cut solver CBC.

    time_limit : int (default=100)
        The maximum time in seconds to run the optimization solver.

    verbose : bool (default=False)
        Enable verbose output.

    Notes
    -----
    The parameter values ``max_n_prebins`` and ``min_prebin_size`` control
    complexity and memory usage. The default values generally produce quality
    results, however, some improvement can be achieved by increasing
    ``max_n_prebins`` and/or decreasing ``min_prebin_size``. A parameter value
    ``max_n_prebins`` greater than 100 is only recommended if ``solver="ls"``.

    The pre-binning refinement phase guarantee that no prebin has either zero
    counts of non-events or events by merging those pure prebins. Pure bins
    produce infinity WoE and IV measures.

    The mathematical formulation when ``solver="ls"`` does **not** currently
    support the max-pvalue constraint.
    """
    def __init__(self, name="", dtype="numerical", prebinning_method="cart",
                 solver="cp", max_n_prebins=20, min_prebin_size=0.05,
                 min_n_bins=None, max_n_bins=None, min_bin_size=None,
                 max_bin_size=None, min_bin_n_nonevent=None,
                 max_bin_n_nonevent=None, min_bin_n_event=None,
                 max_bin_n_event=None, monotonic_trend="auto",
                 min_event_rate_diff=0, max_pvalue=None,
                 max_pvalue_policy="consecutive", gamma=0,
                 outlier_detector=None, outlier_params=None, class_weight=None,
                 cat_cutoff=None, user_splits=None, special_codes=None,
                 split_digits=None, mip_solver="bop", time_limit=100,
                 verbose=False):

        self.name = name
        self.dtype = dtype
        self.prebinning_method = prebinning_method
        self.solver = solver

        self.max_n_prebins = max_n_prebins
        self.min_prebin_size = min_prebin_size

        self.min_n_bins = min_n_bins
        self.max_n_bins = max_n_bins
        self.min_bin_size = min_bin_size
        self.max_bin_size = max_bin_size
        self.min_bin_n_event = min_bin_n_event
        self.max_bin_n_event = max_bin_n_event
        self.min_bin_n_nonevent = min_bin_n_nonevent
        self.max_bin_n_nonevent = max_bin_n_nonevent

        self.monotonic_trend = monotonic_trend
        self.min_event_rate_diff = min_event_rate_diff
        self.max_pvalue = max_pvalue
        self.max_pvalue_policy = max_pvalue_policy
        self.gamma = gamma

        self.outlier_detector = outlier_detector
        self.outlier_params = outlier_params

        self.class_weight = class_weight
        self.cat_cutoff = cat_cutoff

        self.user_splits = user_splits
        self.special_codes = special_codes
        self.split_digits = split_digits

        self.mip_solver = mip_solver
        self.time_limit = time_limit

        self.verbose = verbose

        # auxiliary
        self._categories = None
        self._cat_others = None
        self._n_event = None
        self._n_nonevent = None
        self._n_nonevent_missing = None
        self._n_event_missing = None
        self._n_nonevent_special = None
        self._n_event_special = None
        self._n_nonevent_cat_others = None
        self._n_event_cat_others = None
        self._problem_type = "classification"

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
        self : object
            Fitted optimal binning.
        """
        return self._fit(x, y, check_input)

    def fit_transform(self, x, y, metric="woe", metric_special=0,
                      metric_missing=0, check_input=False):
        """Fit the optimal binning according to the given training data, then
        transform it.

        Parameters
        ----------
        x : array-like, shape = (n_samples,)
            Training vector, where n_samples is the number of samples.

        y : array-like, shape = (n_samples,)
            Target vector relative to x.

        metric : str (default="woe")
            The metric used to transform the input vector. Supported metrics
            are "woe" to choose the Weight of Evidence and "event_rate" to
            choose the event rate.

        metric_special : float or str (default=0)
            The metric value to transform special codes in the input vector.
            Supported metrics are "empirical" to use the empirical WoE or
            event rate, and any numerical value.

        metric_missing : float or str (default=0)
            The metric value to transform missing values in the input vector.
            Supported metrics are "empirical" to use the empirical WoE or
            event rate and any numerical value.

        check_input : bool (default=False)
            Whether to check input arrays.

        Returns
        -------
        x_new : numpy array, shape = (n_samples,)
            Transformed array.
        """
        return self.fit(x, y, check_input).transform(
            x, metric, metric_special, metric_missing, check_input)

    def transform(self, x, metric="woe", metric_special=0,
                  metric_missing=0, check_input=False):
        """Transform given data to Weight of Evidence (WoE) or event rate using
        bins from the fitted optimal binning.

        Parameters
        ----------
        x : array-like, shape = (n_samples,)
            Training vector, where n_samples is the number of samples.

        metric : str (default="woe")
            The metric used to transform the input vector. Supported metrics
            are "woe" to choose the Weight of Evidence and "event_rate" to
            choose the event rate.

        metric_special : float or str (default=0)
            The metric value to transform special codes in the input vector.
            Supported metrics are "empirical" to use the empirical WoE or
            event rate and any numerical value.

        metric_missing : float or str (default=0)
            The metric value to transform missing values in the input vector.
            Supported metrics are "empirical" to use the empirical WoE or
            event rate and any numerical value.

        check_input : bool (default=False)
            Whether to check input arrays.

        Returns
        -------
        x_new : numpy array, shape = (n_samples,)
            Transformed array.

        Notes
        -----
        Transformation of data including categories not present during training
        return zero WoE or event rate.
        """
        self._check_is_fitted()

        return transform_binary_target(self._splits_optimal, self.dtype, x,
                                       self._n_nonevent, self._n_event,
                                       self.special_codes, self._categories,
                                       self._cat_others, metric,
                                       metric_special, metric_missing,
                                       self.user_splits, check_input)

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

        binning_type = self.__class__.__name__.lower()

        if self._optimizer is not None:
            solver = self._optimizer.solver_
            time_solver = self._time_solver
        else:
            solver = None
            time_solver = 0

        dict_user_options = self.get_params()

        print_binning_information(binning_type, print_level, self.name,
                                  self._status, self.solver, solver,
                                  self._time_total, self._time_preprocessing,
                                  self._time_prebinning, time_solver,
                                  self._time_postprocessing, self._n_prebins,
                                  self._n_refinements, dict_user_options)

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
            self.user_splits, check_input, self.outlier_detector,
            self.outlier_params)

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

            if self.outlier_detector is not None:
                n_outlier = self._n_samples-(n_clean + n_missing + n_special)
                logging.info("Pre-processing: number of outlier samples: {}"
                             .format(n_outlier))

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
                n_nonevent = np.array([])
                n_event = np.array([])
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

                splits, n_nonevent, n_event = self._prebinning_refinement(
                    user_splits, x_clean, y_clean, y_missing, y_special,
                    y_others)
        else:
            splits, n_nonevent, n_event = self._fit_prebinning(
                x_clean, y_clean, y_missing, y_special, y_others,
                self.class_weight)

        self._n_prebins = len(n_nonevent)

        self._categories = categories
        self._cat_others = cat_others

        self._time_prebinning = time.perf_counter() - time_prebinning

        if self.verbose:
            logging.info("Pre-binning: number of prebins: {}"
                         .format(self._n_prebins))
            logging.info("Pre-binning: number of refinements: {}"
                         .format(self._n_refinements))

            logging.info("Pre-binning terminated. Time: {:.4f}s"
                         .format(self._time_prebinning))

        # Optimization
        self._fit_optimizer(splits, n_nonevent, n_event)

        # Post-processing
        if self.verbose:
            logging.info("Post-processing started.")
            logging.info("Post-processing: compute binning information.")

        time_postprocessing = time.perf_counter()

        if not len(splits):
            t_info = target_info(y_clean)
            n_nonevent = np.array([t_info[0]])
            n_event = np.array([t_info[1]])

        self._n_nonevent, self._n_event = bin_info(
            self._solution, n_nonevent, n_event, self._n_nonevent_missing,
            self._n_event_missing, self._n_nonevent_special,
            self._n_event_special, self._n_nonevent_cat_others,
            self._n_event_cat_others, cat_others)

        self._binning_table = BinningTable(
            self.name, self.dtype, self._splits_optimal, self._n_nonevent,
            self._n_event, self._categories, self._cat_others,
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

    def _fit_prebinning(self, x, y, y_missing, y_special, y_others,
                        class_weight=None):
        min_bin_size = np.int(np.ceil(self.min_prebin_size * self._n_samples))

        prebinning = PreBinning(method=self.prebinning_method,
                                n_bins=self.max_n_prebins,
                                min_bin_size=min_bin_size,
                                problem_type=self._problem_type,
                                class_weight=class_weight).fit(x, y)

        return self._prebinning_refinement(prebinning.splits, x, y, y_missing,
                                           y_special, y_others)

    def _fit_optimizer(self, splits, n_nonevent, n_event):
        if self.verbose:
            logging.info("Optimizer started.")

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
            min_bin_size = np.int(np.ceil(self.min_bin_size * self._n_samples))
        else:
            min_bin_size = self.min_bin_size

        if self.max_bin_size is not None:
            max_bin_size = np.int(np.ceil(self.max_bin_size * self._n_samples))
        else:
            max_bin_size = self.max_bin_size

        # Monotonic trend
        trend_change = None

        if self.dtype == "numerical":
            auto_monotonic_modes = ("auto", "auto_heuristic", "auto_asc_desc")
            if self.monotonic_trend in auto_monotonic_modes:
                monotonic = auto_monotonic(n_nonevent, n_event,
                                           self.monotonic_trend)

                if self.monotonic_trend == "auto_heuristic":
                    if monotonic in ("peak", "valley"):
                        if monotonic == "peak":
                            monotonic = "peak_heuristic"
                        else:
                            monotonic = "valley_heuristic"

                        event_rate = n_event / (n_nonevent + n_event)
                        trend_change = peak_valley_trend_change_heuristic(
                            event_rate, monotonic)

                if self.verbose:
                    logging.info("Optimizer: classifier predicts {} monotonic "
                                 "trend.".format(monotonic))
            else:
                monotonic = self.monotonic_trend

                if monotonic in ("peak_heuristic", "valley_heuristic"):
                    event_rate = n_event / (n_nonevent + n_event)
                    trend_change = peak_valley_trend_change_heuristic(
                        event_rate, monotonic)

                    if self.verbose:
                        logging.info("Optimizer: trend change position {}."
                                     .format(trend_change))

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

        if self.solver == "cp":
            optimizer = BinningCP(monotonic, self.min_n_bins, self.max_n_bins,
                                  min_bin_size, max_bin_size,
                                  self.min_bin_n_event, self.max_bin_n_event,
                                  self.min_bin_n_nonevent,
                                  self.max_bin_n_nonevent,
                                  self.min_event_rate_diff, self.max_pvalue,
                                  self.max_pvalue_policy, self.gamma,
                                  self.time_limit)
        elif self.solver == "mip":
            optimizer = BinningMIP(monotonic, self.min_n_bins, self.max_n_bins,
                                   min_bin_size, max_bin_size,
                                   self.min_bin_n_event, self.max_bin_n_event,
                                   self.min_bin_n_nonevent,
                                   self.max_bin_n_nonevent,
                                   self.min_event_rate_diff, self.max_pvalue,
                                   self.max_pvalue_policy, self.gamma,
                                   self.mip_solver, self.time_limit)
        elif self.solver == "ls":
            optimizer = BinningLS(monotonic, self.min_n_bins, self.max_n_bins,
                                  min_bin_size, max_bin_size,
                                  self.min_bin_n_event, self.max_bin_n_event,
                                  self.min_bin_n_nonevent,
                                  self.max_bin_n_nonevent,
                                  self.min_event_rate_diff, self.max_pvalue,
                                  self.max_pvalue_policy, self.time_limit)

        if self.verbose:
            logging.info("Optimizer: build model...")

        optimizer.build_model(n_nonevent, n_event, trend_change)

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
        y0 = (y == 0)
        y1 = ~y0

        # Compute n_nonevent and n_event for special, missing and others.
        special_target_info = target_info(y_special)
        self._n_nonevent_special = special_target_info[0]
        self._n_event_special = special_target_info[1]

        missing_target_info = target_info(y_missing)
        self._n_nonevent_missing = missing_target_info[0]
        self._n_event_missing = missing_target_info[1]

        if len(y_others):
            others_target_info = target_info(y_others)
            self._n_nonevent_cat_others = others_target_info[0]
            self._n_event_cat_others = others_target_info[1]

        n_splits = len(splits_prebinning)

        if not n_splits:
            return splits_prebinning, np.array([]), np.array([])

        if self.split_digits is not None:
            splits_prebinning = np.round(splits_prebinning, self.split_digits)

        splits_prebinning, n_nonevent, n_event = self._compute_prebins(
            splits_prebinning, x, y0, y1)

        return splits_prebinning, n_nonevent, n_event

    def _check_is_fitted(self):
        if not self._is_fitted:
            raise NotFittedError("This {} instance is not fitted yet. Call "
                                 "'fit' with appropriate arguments."
                                 .format(self.__class__.__name__))

    def _compute_prebins(self, splits_prebinning, x, y0, y1):
        n_splits = len(splits_prebinning)

        if not n_splits:
            return splits_prebinning, np.array([]), np.array([])

        indices = np.digitize(x, splits_prebinning, right=False)

        n_bins = n_splits + 1
        n_nonevent = np.zeros(n_bins).astype(np.int)
        n_event = np.zeros(n_bins).astype(np.int)

        for i in range(n_bins):
            mask = (indices == i)
            n_nonevent[i] = np.count_nonzero(y0 & mask)
            n_event[i] = np.count_nonzero(y1 & mask)

        mask_remove = (n_nonevent == 0) | (n_event == 0)

        if np.any(mask_remove):
            self._n_refinements += 1

            mask_splits = np.concatenate(
                [mask_remove[:-2], [mask_remove[-2] | mask_remove[-1]]])

            splits = splits_prebinning[~mask_splits]

            if self.verbose:
                logging.info("Pre-binning: number prebins removed: {}"
                             .format(np.count_nonzero(mask_remove)))

            [splits_prebinning, n_nonevent, n_event] = self._compute_prebins(
                splits, x, y0, y1)

        return splits_prebinning, n_nonevent, n_event

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

        if self.dtype == "numerical":
            return self._splits_optimal
        else:
            return bin_categorical(self._splits_optimal, self._categories,
                                   self._cat_others, self.user_splits)

    @property
    def status(self):
        """The status of the underlying optimization solver.

        Returns
        -------
        status : str
        """
        self._check_is_fitted()

        return self._status
