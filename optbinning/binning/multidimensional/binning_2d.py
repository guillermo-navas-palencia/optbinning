"""
Optimal binning 2D algorithm.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2021

import numbers
import time

import numpy as np

from joblib import effective_n_jobs
from sklearn.tree import DecisionTreeClassifier

from ...information import solver_statistics
from ...logging import Logger
from ..binning import OptimalBinning
from ..binning_statistics import target_info
from ..prebinning import PreBinning
from .binning_statistics_2d import BinningTable2D
from .binning_statistics_2d import bin_categorical
from .cp_2d import Binning2DCP
from .mip_2d import Binning2DMIP
from .model_data_2d import model_data
from .model_data_cart_2d import model_data_cart
from .preprocessing_2d import split_data_2d
from .transformations_2d import transform_binary_target


logger = Logger(__name__).logger


def _check_parameters(name_x, name_y, dtype_x, dtype_y, prebinning_method,
                      strategy, solver, divergence, max_n_prebins_x,
                      max_n_prebins_y, min_prebin_size_x, min_prebin_size_y,
                      min_n_bins, max_n_bins, min_bin_size, max_bin_size,
                      min_bin_n_nonevent, max_bin_n_nonevent, min_bin_n_event,
                      max_bin_n_event, monotonic_trend_x, monotonic_trend_y,
                      min_event_rate_diff_x, min_event_rate_diff_y, gamma,
                      special_codes_x, special_codes_y, split_digits, n_jobs,
                      time_limit, verbose):

    if not isinstance(name_x, str):
        raise TypeError("name_x must be a string.")

    if not isinstance(name_y, str):
        raise TypeError("name_y must be a string.")

    if dtype_x not in ("categorical", "numerical"):
        raise ValueError('Invalid value for dtype_x. Allowed string '
                         'values are "categorical" and "numerical" .')

    if dtype_y not in ("categorical", "numerical",):
        raise ValueError('Invalid value for dtype_y. Allowed string '
                         'values are "categorical" and "numerical" .')

    if prebinning_method not in ("cart", "mdlp", "quantile", "uniform"):
        raise ValueError('Invalid value for prebinning_method. Allowed string '
                         'values are "cart", "mdlp", "quantile" '
                         'and "uniform".')

    if strategy not in ("grid", "cart"):
        raise ValueError('Invalid value for strategy. Allowed string '
                         'values are "grid" and "cart".')

    if solver not in ("cp", "mip"):
        raise ValueError('Invalid value for solver. Allowed string '
                         'values are "cp" and "mip".')

    if divergence not in ("iv", "js", "hellinger", "triangular"):
        raise ValueError('Invalid value for divergence. Allowed string '
                         'values are "iv", "js", "helliger" and "triangular".')

    if (not isinstance(max_n_prebins_x, numbers.Integral) or
            max_n_prebins_x <= 1):
        raise ValueError("max_prebins_x must be an integer greater than 1; "
                         "got {}.".format(max_n_prebins_x))

    if (not isinstance(max_n_prebins_y, numbers.Integral) or
            max_n_prebins_y <= 1):
        raise ValueError("max_prebins_y must be an integer greater than 1; "
                         "got {}.".format(max_n_prebins_y))

    if not 0. < min_prebin_size_x <= 0.5:
        raise ValueError("min_prebin_size_x must be in (0, 0.5]; got {}."
                         .format(min_prebin_size_x))

    if not 0. < min_prebin_size_y <= 0.5:
        raise ValueError("min_prebin_size_y must be in (0, 0.5]; got {}."
                         .format(min_prebin_size_y))

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

    if monotonic_trend_x is not None:
        if monotonic_trend_x not in ("ascending", "descending"):
            raise ValueError('Invalid value for monotonic trend x. Allowed '
                             'string values are "ascending" and "descending".')

    if monotonic_trend_y is not None:
        if monotonic_trend_y not in ("ascending", "descending"):
            raise ValueError('Invalid value for monotonic trend y. Allowed '
                             'string values are "ascending" and "descending".')

    if (not isinstance(min_event_rate_diff_x, numbers.Number) or
            not 0. <= min_event_rate_diff_x <= 1.0):
        raise ValueError("min_event_rate_diff_x must be in [0, 1]; got {}."
                         .format(min_event_rate_diff_x))

    if (not isinstance(min_event_rate_diff_y, numbers.Number) or
            not 0. <= min_event_rate_diff_y <= 1.0):
        raise ValueError("min_event_rate_diff_y must be in [0, 1]; got {}."
                         .format(min_event_rate_diff_y))

    if not isinstance(gamma, numbers.Number) or gamma < 0:
        raise ValueError("gamma must be >= 0; got {}.".format(gamma))

    if special_codes_x is not None:
        if not isinstance(special_codes_x, (np.ndarray, list)):
            raise TypeError("special_codes_x must be a list or numpy.ndarray.")

    if special_codes_y is not None:
        if not isinstance(special_codes_y, (np.ndarray, list)):
            raise TypeError("special_codes_y must be a list or numpy.ndarray.")

    if split_digits is not None:
        if (not isinstance(split_digits, numbers.Integral) or
                not 0 <= split_digits <= 8):
            raise ValueError("split_digits must be an integer in [0, 8]; "
                             "got {}.".format(split_digits))
    if n_jobs is not None:
        if not isinstance(n_jobs, numbers.Integral):
            raise ValueError("n_jobs must be an integer or None; got {}."
                             .format(n_jobs))

    if not isinstance(time_limit, numbers.Number) or time_limit < 0:
        raise ValueError("time_limit must be a positive value in seconds; "
                         "got {}.".format(time_limit))

    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a boolean; got {}.".format(verbose))


class OptimalBinning2D(OptimalBinning):
    """Optimal binning of two numerical variables with respect to a binary
    target.

    Parameters
    ----------
    name_x : str, optional (default="")
        The name of variable x.

    name_y : str, optional (default="")
        The name of variable y.

    dtype_x : str, optional (default="numerical")
        The data type of variable x. Supported data types are "numerical" for
        continuous and ordinal variables and "categorical" for categorical
        and nominal variables.

    dtype_y : str, optional (default="numerical")
        The data type of variable y. Supported data types are "numerical" for
        continuous and ordinal variables and "categorical" for categorical
        and nominal variables.

    prebinning_method : str, optional (default="cart")
        The pre-binning method. Supported methods are "cart" for a CART
        decision tree, "mdlp" for Minimum Description Length Principle (MDLP),
        "quantile" to generate prebins with approximately same frequency and
        "uniform" to generate prebins with equal width. Method "cart" uses
        `sklearn.tree.DecistionTreeClassifier
        <https://scikit-learn.org/stable/modules/generated/sklearn.tree.
        DecisionTreeClassifier.html>`_.

    strategy: str, optional (default="grid")
        The strategy used to create the initial prebinning 2D after computing
        prebinning splits on the x and y axis. The strategy "grid" creates a
        prebinning 2D with n_prebins_x times n_prebins_y elements. The strategy
        "cart" (experimental) reduces the number of elements by pruning. The
        latter is recommended when the number of prebins is large.

    solver : str, optional (default="cp")
        The optimizer to solve the optimal binning problem. Supported solvers
        are "mip" to choose a mixed-integer programming solver, and "cp" to
        choose a constrained programming solver.

    divergence : str, optional (default="iv")
        The divergence measure in the objective function to be maximized.
        Supported divergences are "iv" (Information Value or Jeffrey's
        divergence), "js" (Jensen-Shannon), "hellinger" (Hellinger divergence)
        and "triangular" (triangular discrimination).

    max_n_prebins_x : int (default=5)
        The maximum number of bins on variable x after pre-binning (prebins).

    max_n_prebins_y : int (default=5)
        The maximum number of bins on variable y after pre-binning (prebins).

    min_prebin_size_x : float (default=0.05)
        The fraction of mininum number of records for each prebin on variable
        x.

    min_prebin_size_y : float (default=0.05)
        The fraction of mininum number of records for each prebin on variable
        y.

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

    monotonic_trend_x : str or None, optional (default=None)
        The **event rate** monotonic trend on the x axis. Supported trends are
        “ascending”, and "descending". If None, then the monotonic constraint
        is disabled.

    monotonic_trend_y : str or None, optional (default=None)
        The **event rate** monotonic trend on the y axis. Supported trends are
        “ascending”, and "descending". If None, then the monotonic constraint
        is disabled.

    min_event_rate_diff_x : float, optional (default=0)
        The minimum event rate difference between consecutives bins on the x
        axis.

    min_event_rate_diff_y : float, optional (default=0)
        The minimum event rate difference between consecutives bins on the y
        axis.

    gamma : float, optional (default=0)
        Regularization strength to reduce the number of dominating bins. Larger
        values specify stronger regularization.

    special_codes_x : array-like or None, optional (default=None)
        List of special codes for the variable x. Use special codes to specify
        the data values that must be treated separately.

    special_codes_y : array-like or None, optional (default=None)
        List of special codes for the variable y. Use special codes to specify
        the data values that must be treated separately.

    split_digits : int or None, optional (default=None)
        The significant digits of the split points. If ``split_digits`` is set
        to 0, the split points are integers. If None, then all significant
        digits in the split points are considered.

    n_jobs : int or None, optional (default=None)
        Number of cores to run in parallel while binning variables.
        ``None`` means 1 core. ``-1`` means using all processors.

    time_limit : int (default=100)
        The maximum time in seconds to run the optimization solver.

    verbose : bool (default=False)
        Enable verbose output.
    """
    def __init__(self, name_x="", name_y="", dtype_x="numerical",
                 dtype_y="numerical", prebinning_method="cart",
                 strategy="grid", solver="cp", divergence="iv",
                 max_n_prebins_x=5, max_n_prebins_y=5, min_prebin_size_x=0.05,
                 min_prebin_size_y=0.05, min_n_bins=None, max_n_bins=None,
                 min_bin_size=None, max_bin_size=None, min_bin_n_nonevent=None,
                 max_bin_n_nonevent=None, min_bin_n_event=None,
                 max_bin_n_event=None, monotonic_trend_x=None,
                 monotonic_trend_y=None, min_event_rate_diff_x=0,
                 min_event_rate_diff_y=0, gamma=0, special_codes_x=None,
                 special_codes_y=None, split_digits=None, n_jobs=1,
                 time_limit=100, verbose=False):

        self.name_x = name_x
        self.name_y = name_y
        self.dtype_x = dtype_x
        self.dtype_y = dtype_y
        self.prebinning_method = prebinning_method
        self.strategy = strategy
        self.solver = solver
        self.divergence = divergence

        self.max_n_prebins_x = max_n_prebins_x
        self.max_n_prebins_y = max_n_prebins_y
        self.min_prebin_size_x = min_prebin_size_x
        self.min_prebin_size_y = min_prebin_size_y

        self.min_n_bins = min_n_bins
        self.max_n_bins = max_n_bins
        self.min_bin_size = min_bin_size
        self.max_bin_size = max_bin_size
        self.min_bin_n_event = min_bin_n_event
        self.max_bin_n_event = max_bin_n_event
        self.min_bin_n_nonevent = min_bin_n_nonevent
        self.max_bin_n_nonevent = max_bin_n_nonevent

        self.monotonic_trend_x = monotonic_trend_x
        self.monotonic_trend_y = monotonic_trend_y
        self.min_event_rate_diff_x = min_event_rate_diff_x
        self.min_event_rate_diff_y = min_event_rate_diff_y
        self.gamma = gamma

        self.special_codes_x = special_codes_x
        self.special_codes_y = special_codes_y
        self.split_digits = split_digits

        self.n_jobs = n_jobs
        self.time_limit = time_limit

        self.verbose = verbose

        # auxiliary
        self._categories_x = None
        self._categories_y = None
        self._n_event = None
        self._n_nonevent = None
        self._n_event_special = None
        self._n_nonevent_special = None
        self._n_event_missing = None
        self._n_nonevent_missing = None
        self._problem_type = "classification"

        # info
        self._binning_table = None
        self._n_prebins = None
        self._n_refinements = 0
        self._n_samples = None
        self._optimizer = None
        self._solution = None
        self._splits_x_optimal = None
        self._splits_y_optimal = None
        self._status = None

        # timing
        self._time_total = None
        self._time_preprocessing = None
        self._time_prebinning = None
        self._time_solver = None
        self._time_optimizer = None
        self._time_postprocessing = None

        self._is_fitted = False

    def fit(self, x, y, z, check_input=False):
        """Fit the optimal binning 2D according to the given training data.

        Parameters
        ----------
        x : array-like, shape = (n_samples,)
            Training vector x, where n_samples is the number of samples.

        y : array-like, shape = (n_samples,)
            Training vector y, where n_samples is the number of samples.

        z : array-like, shape = (n_samples,)
            Target vector relative to x and y.

        check_input : bool (default=False)
            Whether to check input arrays.

        Returns
        -------
        self : OptimalBinning2D
            Fitted optimal binning 2D.
        """
        return self._fit(x, y, z, check_input)

    def fit_transform(self, x, y, z, metric="woe", metric_special=0,
                      metric_missing=0, show_digits=2, check_input=False):
        """Fit the optimal binning 2D according to the given training data,
        then transform it.

        Parameters
        ----------
        x : array-like, shape = (n_samples,)
            Training vector x, where n_samples is the number of samples.

        y : array-like, shape = (n_samples,)
            Training vector y, where n_samples is the number of samples.

        z : array-like, shape = (n_samples,)
            Target vector relative to x and y.

        metric : str (default="woe")
            The metric used to transform the input vector. Supported metrics
            are "woe" to choose the Weight of Evidence, "event_rate" to
            choose the event rate, "indices" to assign the corresponding
            indices of the bins and "bins" to assign the corresponding
            bin interval.

        metric_special : float or str (default=0)
            The metric value to transform special codes in the input vector.
            Supported metrics are "empirical" to use the empirical WoE or
            event rate, and any numerical value.

        metric_missing : float or str (default=0)
            The metric value to transform missing values in the input vector.
            Supported metrics are "empirical" to use the empirical WoE or
            event rate and any numerical value.

        show_digits : int, optional (default=2)
            The number of significant digits of the bin column. Applies when
            ``metric="bins"``.

        check_input : bool (default=False)
            Whether to check input arrays.

        Returns
        -------
        z_new : numpy array, shape = (n_samples,)
            Transformed array.
        """
        return self.fit(x, y, z, check_input).transform(
            x, y, metric, metric_special, metric_missing, show_digits,
            check_input)

    def transform(self, x, y, metric="woe", metric_special=0, metric_missing=0,
                  show_digits=2, check_input=False):
        """Transform given data to Weight of Evidence (WoE) or event rate using
        bins from the fitted optimal binning 2D.

        Parameters
        ----------
        x : array-like, shape = (n_samples,)
            Training vector x, where n_samples is the number of samples.

        y : array-like, shape = (n_samples,)
            Training vector y, where n_samples is the number of samples.

        metric : str (default="woe")
            The metric used to transform the input vector. Supported metrics
            are "woe" to choose the Weight of Evidence, "event_rate" to
            choose the event rate, "indices" to assign the corresponding
            indices of the bins and "bins" to assign the corresponding
            bin interval.

        metric_special : float or str (default=0)
            The metric value to transform special codes in the input vector.
            Supported metrics are "empirical" to use the empirical WoE or
            event rate and any numerical value.

        metric_missing : float or str (default=0)
            The metric value to transform missing values in the input vector.
            Supported metrics are "empirical" to use the empirical WoE or
            event rate and any numerical value.

        show_digits : int, optional (default=2)
            The number of significant digits of the bin column. Applies when
            ``metric="bins"``.

        check_input : bool (default=False)
            Whether to check input arrays.

        Returns
        -------
        z_new : numpy array, shape = (n_samples,)
            Transformed array.
        """
        self._check_is_fitted()

        return transform_binary_target(
            self.dtype_x, self.dtype_y, self._splits_x_optimal,
            self._splits_y_optimal, x, y, self._n_nonevent, self._n_event,
            self.special_codes_x, self.special_codes_y, metric, metric_special,
            metric_missing, show_digits, check_input)

    def _fit(self, x, y, z, check_input):
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

        [x_clean, y_clean, z_clean, x_missing, y_missing, z_missing,
         x_special, y_special, z_special,
         categories_x, categories_y] = split_data_2d(
            self.dtype_x, self.dtype_y, x, y, z, self.special_codes_x,
            self.special_codes_y, check_input)

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

            if self.dtype_x == "categorical":
                logger.info("Pre-processing: number of categories in x: {}"
                            .format(len(categories_x)))

            if self.dtype_y == "categorical":
                logger.info("Pre-processing: number of categories in y: {}"
                            .format(len(categories_y)))

        if self.verbose:
            logger.info("Pre-processing terminated. Time: {:.4f}s"
                        .format(self._time_preprocessing))

        # Pre-binning
        if self.verbose:
            logger.info("Pre-binning started.")

        time_prebinning = time.perf_counter()

        splits_x = self._fit_prebinning(self.dtype_x, x_clean, z_clean,
                                        self.max_n_prebins_x,
                                        self.min_prebin_size_x)

        splits_y = self._fit_prebinning(self.dtype_y, y_clean, z_clean,
                                        self.max_n_prebins_y,
                                        self.min_prebin_size_y)

        NE, E = self._prebinning_matrices(
            splits_x, splits_y, x_clean, y_clean, z_clean, x_missing,
            y_missing, z_missing, x_special, y_special, z_special)

        if self.strategy == "cart":

            if self.verbose:
                logger.info("Prebinning: applying strategy cart...")

            n_splits_x = len(splits_x)
            n_splits_y = len(splits_y)

            clf_nodes = n_splits_x * n_splits_y

            indices_x = np.digitize(x_clean, splits_x, right=False)
            n_bins_x = n_splits_x + 1

            indices_y = np.digitize(y_clean, splits_y, right=False)
            n_bins_y = n_splits_y + 1

            xt = np.empty(len(x_clean), dtype=int)
            yt = np.empty(len(y_clean), dtype=int)

            for i in range(n_bins_x):
                xt[(indices_x == i)] = i

            for i in range(n_bins_y):
                yt[(indices_y == i)] = i

            xyt = np.c_[xt, yt]

            min_prebin_size = min(self.min_prebin_size_x,
                                  self.min_prebin_size_y) * 0.25

            clf = DecisionTreeClassifier(min_samples_leaf=min_prebin_size,
                                         max_leaf_nodes=clf_nodes)
            clf.fit(xyt, z_clean)

            self._clf = clf

        self._categories_x = categories_x
        self._categories_y = categories_y

        self._time_prebinning = time.perf_counter() - time_prebinning

        self._n_prebins = E.size

        if self.verbose:
            logger.info("Pre-binning: number of prebins: {}"
                        .format(self._n_prebins))

            logger.info("Pre-binning terminated. Time: {:.4f}s"
                        .format(self._time_prebinning))

        # Optimization
        rows, n_nonevent, n_event = self._fit_optimizer(
            splits_x, splits_y, NE, E)

        # Post-processing
        if self.verbose:
            logger.info("Post-processing started.")
            logger.info("Post-processing: compute binning information.")

        time_postprocessing = time.perf_counter()

        # Refinements
        m, n = E.shape
        self._n_refinements = (m * n * (m + 1) * (n + 1)) // 4 - len(rows)

        # solution matrices
        D = np.empty(m * n, dtype=float)
        P = np.empty(m * n, dtype=int)

        selected_rows = np.array(rows, dtype=object)[self._solution]

        self._selected_rows = selected_rows
        self._m, self._n = m, n

        n_selected_rows = selected_rows.shape[0] + 2

        opt_n_nonevent = np.empty(n_selected_rows, dtype=int)
        opt_n_event = np.empty(n_selected_rows, dtype=int)

        for i, r in enumerate(selected_rows):
            _n_nonevent = n_nonevent[self._solution][i]
            _n_event = n_event[self._solution][i]
            _event_rate = _n_event / (_n_event + _n_nonevent)

            P[r] = i
            D[r] = _event_rate
            opt_n_nonevent[i] = _n_nonevent
            opt_n_event[i] = _n_event

        opt_n_nonevent[-2] = self._n_nonevent_special
        opt_n_event[-2] = self._n_event_special

        opt_n_nonevent[-1] = self._n_nonevent_missing
        opt_n_event[-1] = self._n_event_missing

        self._n_nonevent = opt_n_nonevent
        self._n_event = opt_n_event

        D = D.reshape((m, n))
        P = P.reshape((m, n))

        # optimal bins
        splits_x_optimal, splits_y_optimal = self._splits_xy_optimal(
            selected_rows, splits_x, splits_y, P)

        self._splits_x_optimal = splits_x_optimal
        self._splits_y_optimal = splits_y_optimal

        # instatiate binning table
        self._binning_table = BinningTable2D(
            self.name_x, self.name_y, self.dtype_x, self.dtype_y,
            splits_x_optimal, splits_y_optimal, m, n, opt_n_nonevent,
            opt_n_event, D, P, self._categories_x, self._categories_y)

        self.name = "-".join((self.name_x, self.name_y))

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

    def _fit_prebinning(self, dtype, x, z, max_n_prebins, min_prebin_size):
        # Pre-binning algorithm
        min_bin_size = int(np.ceil(min_prebin_size * self._n_samples))

        prebinning = PreBinning(method=self.prebinning_method,
                                n_bins=max_n_prebins,
                                min_bin_size=min_bin_size,
                                problem_type=self._problem_type).fit(x, z)

        return prebinning.splits

    def _prebinning_matrices(self, splits_x, splits_y, x_clean, y_clean,
                             z_clean, x_missing, y_missing, z_missing,
                             x_special, y_special, z_special):
        z0 = z_clean == 0
        z1 = ~z0

        # Compute n_nonevent and n_event for special and missing
        special_target_info = target_info(z_special)
        self._n_nonevent_special = special_target_info[0]
        self._n_event_special = special_target_info[1]

        missing_target_info = target_info(z_missing)
        self._n_nonevent_missing = missing_target_info[0]
        self._n_event_missing = missing_target_info[1]

        n_splits_x = len(splits_x)
        n_splits_y = len(splits_y)

        indices_x = np.digitize(x_clean, splits_x, right=False)
        n_bins_x = n_splits_x + 1

        indices_y = np.digitize(y_clean, splits_y, right=False)
        n_bins_y = n_splits_y + 1

        E = np.empty((n_bins_y, n_bins_x), dtype=int)
        NE = np.empty((n_bins_y, n_bins_x), dtype=int)

        for i in range(n_bins_y):
            mask_y = (indices_y == i)
            for j in range(n_bins_x):
                mask_x = (indices_x == j)
                mask = mask_x & mask_y

                NE[i, j] = np.count_nonzero(z0 & mask)
                E[i, j] = np.count_nonzero(z1 & mask)

        return NE, E

    def _fit_optimizer(self, splits_x, splits_y, NE, E):
        if self.verbose:
            logger.info("Optimizer started.")

        time_init = time.perf_counter()

        # Min/max number of bins (bin size)
        if self.min_bin_size is not None:
            min_bin_size = int(np.ceil(self.min_bin_size * self._n_samples))
        else:
            min_bin_size = self.min_bin_size

        if self.max_bin_size is not None:
            max_bin_size = int(np.ceil(self.max_bin_size * self._n_samples))
        else:
            max_bin_size = self.max_bin_size

        # Number of threads
        n_jobs = effective_n_jobs(self.n_jobs)

        if self.verbose:
            logger.info("Optimizer: {} jobs.".format(n_jobs))

            if self.monotonic_trend_x is None:
                logger.info(
                    "Optimizer: monotonic trend x not set.")
            else:
                logger.info("Optimizer: monotonic trend x set to {}."
                            .format(self.monotonic_trend_x))

            if self.monotonic_trend_y is None:
                logger.info(
                    "Optimizer: monotonic trend y not set.")
            else:
                logger.info("Optimizer: monotonic trend y set to {}."
                            .format(self.monotonic_trend_x))

        if self.solver == "cp":
            scale = int(1e6)

            optimizer = Binning2DCP(
                self.monotonic_trend_x, self.monotonic_trend_y,
                self.min_n_bins, self.max_n_bins, self.min_event_rate_diff_x,
                self.min_event_rate_diff_y, self.gamma, n_jobs,
                self.time_limit)

        elif self.solver == "mip":
            scale = None

            optimizer = Binning2DMIP(
                self.monotonic_trend_x, self.monotonic_trend_y,
                self.min_n_bins, self.max_n_bins, self.min_event_rate_diff_x,
                self.min_event_rate_diff_y, self.gamma, n_jobs,
                self.time_limit)

        if self.verbose:
            logger.info("Optimizer: model data...")

        time_model_data = time.perf_counter()

        if self.strategy == "cart":
            [n_grid, n_rectangles, rows, cols, c, d_connected_x, d_connected_y,
             event_rate, n_event, n_nonevent, n_records] = model_data_cart(
                self._clf, self.divergence, NE, E, self.monotonic_trend_x,
                self.monotonic_trend_y, scale, min_bin_size, max_bin_size,
                self.min_bin_n_event, self.max_bin_n_event,
                self.min_bin_n_nonevent, self.max_bin_n_nonevent)
        else:
            [n_grid, n_rectangles, rows, cols, c, d_connected_x, d_connected_y,
             event_rate, n_event, n_nonevent, n_records] = model_data(
                self.divergence, NE, E, self.monotonic_trend_x,
                self.monotonic_trend_y, scale, min_bin_size, max_bin_size,
                self.min_bin_n_event, self.max_bin_n_event,
                self.min_bin_n_nonevent, self.max_bin_n_nonevent)

        self._time_model_data = time.perf_counter() - time_model_data

        if self.verbose:
            logger.info("Optimizer: model data terminated. Time {:.4f}s"
                        .format(self._time_model_data))

        if self.verbose:
            logger.info("Optimizer: build model...")

        optimizer.build_model(n_grid, n_rectangles, cols, c, d_connected_x,
                              d_connected_y, event_rate, n_records)

        if self.verbose:
            logger.info("Optimizer: solve...")

        status, solution = optimizer.solve()

        self._solution = solution

        self._optimizer, self._time_optimizer = solver_statistics(
            self.solver, optimizer.solver_)
        self._status = status

        self._time_solver = time.perf_counter() - time_init

        if self.verbose:
            logger.info("Optimizer terminated. Time: {:.4f}s"
                        .format(self._time_solver))

        self._cols = cols
        self._rows = rows
        self._c = c

        return rows, n_nonevent, n_event

    def _splits_optimal(self, dtype, selected_rows, splits, P, axis,
                        categories=None):

        if dtype == "numerical":
            bins = np.concatenate([[-np.inf], splits, [np.inf]])
            bins_str = np.array([[bins[i], bins[i+1]]
                                 for i in range(len(bins) - 1)])

            splits_optimal = []
            for i in range(len(selected_rows)):
                if axis == "x":
                    _, pos = np.where(P == i)
                else:
                    pos, _ = np.where(P == i)

                mask = np.arange(pos.min(), pos.max() + 1)
                bin = bins_str[mask]

                splits_optimal.append([bin[0][0], bin[-1][1]])

            return splits_optimal

        else:
            splits = np.ceil(splits).astype(int)
            n_categories = len(categories)

            indices = np.digitize(np.arange(n_categories), splits, right=False)
            n_bins = len(splits) + 1

            bins = []
            for i in range(n_bins):
                mask = (indices == i)
                bins.append(categories[mask])

            return bins

    def _splits_xy_optimal(self, selected_rows, splits_x, splits_y, P):
        bins_x = np.concatenate([[-np.inf], splits_x, [np.inf]])
        bins_y = np.concatenate([[-np.inf], splits_y, [np.inf]])

        bins_str_x = np.array([[bins_x[i], bins_x[i+1]]
                               for i in range(len(bins_x) - 1)])
        bins_str_y = np.array([[bins_y[i], bins_y[i+1]]
                               for i in range(len(bins_y) - 1)])

        splits_x_optimal = []
        splits_y_optimal = []
        for i in range(len(selected_rows)):
            pos_y, pos_x = np.where(P == i)
            mask_x = np.arange(pos_x.min(), pos_x.max() + 1)
            mask_y = np.arange(pos_y.min(), pos_y.max() + 1)
            bin_x = bins_str_x[mask_x]
            bin_y = bins_str_y[mask_y]

            splits_x_optimal.append([bin_x[0][0], bin_x[-1][1]])
            splits_y_optimal.append([bin_y[0][0], bin_y[-1][1]])

        return splits_x_optimal, splits_y_optimal

    @property
    def splits(self):
        """List of optimal split points and bins for axis x and y.

        Returns
        -------
        splits : (numpy.ndarray, numpy.ndarray)
        """
        self._check_is_fitted()

        if self.dtype_x == "categorical":
            splits_x_optimal = bin_categorical(
                self._splits_x_optimal, self._categories_x)
        else:
            splits_x_optimal = self._splits_x_optimal

        if self.dtype_y == "categorical":
            splits_y_optimal = bin_categorical(
                self._splits_y_optimal, self._categories_y)
        else:
            splits_y_optimal = self._splits_y_optimal

        return (splits_x_optimal, splits_y_optimal)
