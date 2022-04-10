"""
Optimal binning sketch algorithm.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import numbers
import time

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from ...binning.auto_monotonic import auto_monotonic
from ...binning.auto_monotonic import peak_valley_trend_change_heuristic
from ...binning.binning_statistics import bin_categorical
from ...binning.binning_statistics import bin_info
from ...binning.binning_statistics import BinningTable
from ...binning.cp import BinningCP
from ...binning.mip import BinningMIP
from ...binning.transformations import transform_binary_target
from ...information import solver_statistics
from ...logging import Logger
from .base import BaseSketch
from .bsketch import BSketch, BCatSketch
from .bsketch_information import print_binning_information
from .plots import plot_progress_divergence

try:
    from pympler import asizeof
    PYMPLER_AVAILABLE = True
except ImportError:
    PYMPLER_AVAILABLE = False


logger = Logger(__name__).logger


def _check_parameters(name, dtype, sketch, eps, K, solver, divergence,
                      max_n_prebins, min_n_bins, max_n_bins, min_bin_size,
                      max_bin_size, min_bin_n_nonevent, max_bin_n_nonevent,
                      min_bin_n_event, max_bin_n_event, monotonic_trend,
                      min_event_rate_diff, max_pvalue, max_pvalue_policy,
                      gamma, cat_cutoff, cat_heuristic, special_codes,
                      split_digits, mip_solver, time_limit, verbose):

    # Check pympler
    if not PYMPLER_AVAILABLE:
        raise ImportError('Cannot import pympler. Install pympler via '
                          'pip install pympler or install optbinning using '
                          'pip install optbinning[distributed]')

    # Check parameters
    if not isinstance(name, str):
        raise TypeError("name must be a string.")

    if dtype not in ("categorical", "numerical"):
        raise ValueError('Invalid value for dtype. Allowed string '
                         'values are "categorical" and "numerical".')

    if sketch not in ("gk", "t-digest"):
        raise ValueError('Invalid value for sketch. Allowed string '
                         'values are "gk" and "t-digest".')

    if not isinstance(eps, numbers.Number) or not 0 <= eps <= 1:
        raise ValueError("eps must be a value in [0, 1]; got {}."
                         .format(eps))

    if not isinstance(K, numbers.Integral) or K <= 0:
        raise ValueError("K must be a positive integer; got {}."
                         .format(K))

    if solver not in ("cp", "mip"):
        raise ValueError('Invalid value for solver. Allowed string '
                         'values are "cp" and "mip".')

    if divergence not in ("iv", "js", "hellinger", "triangular"):
        raise ValueError('Invalid value for divergence. Allowed string '
                         'values are "iv", "js", "helliger" and "triangular".')

    if not isinstance(max_n_prebins, numbers.Integral) or max_n_prebins <= 1:
        raise ValueError("max_prebins must be an integer greater than 1; "
                         "got {}.".format(max_n_prebins))

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
                             '"concave", "convex", "peak", "valley", '
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

    if cat_cutoff is not None:
        if (not isinstance(cat_cutoff, numbers.Number) or
                not 0. < cat_cutoff <= 1.0):
            raise ValueError("cat_cutoff must be in (0, 1.0]; got {}."
                             .format(cat_cutoff))

    if not isinstance(cat_heuristic, bool):
        raise TypeError("cat_heuristic must be a boolean; got {}."
                        .format(cat_heuristic))

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


class OptimalBinningSketch(BaseSketch, BaseEstimator):
    """Optimal binning over data streams of a numerical or categorical
    variable with respect to a binary target.

    Parameters
    ----------
    name : str, optional (default="")
        The variable name.

    dtype : str, optional (default="numerical")
        The variable data type. Supported data types are "numerical" for
        continuous and ordinal variables and "categorical" for categorical
        and nominal variables.

    sketch : str, optional (default="gk")
        Sketch algorithm. Supported algorithms are "gk" (Greenwald-Khanna's)
        and "t-digest" (Ted Dunning) algorithm. Algorithm "t-digest" relies on
        `tdigest <https://github.com/CamDavidsonPilon/tdigest>`_.

    eps : float, optional (default=1e-4)
        Relative error epsilon. For ``sketch="gk"`` this is the absolute
        precision, whereas for ``sketch="t-digest"`` is the relative precision.

    K : int, optional (default=25)
        Parameter excess growth K to compute compress threshold in t-digest.

    solver : str, optional (default="cp")
        The optimizer to solve the optimal binning problem. Supported solvers
        are "mip" to choose a mixed-integer programming solver, "cp" to choose
        a constrained programming solver or "ls" to choose `LocalSolver
        <https://www.localsolver.com/>`_.

    divergence : str, optional (default="iv")
        The divergence measure in the objective function to be maximized.
        Supported divergences are "iv" (Information Value or Jeffrey's
        divergence), "js" (Jensen-Shannon), "hellinger" (Hellinger divergence)
        and "triangular" (triangular discrimination).

    max_n_prebins : int (default=20)
        The maximum number of bins after pre-binning (prebins).

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
        not satisfying the p-value constraint.

    max_pvalue_policy : str, optional (default="consecutive")
        The method to determine bins not satisfying the p-value constraint.
        Supported methods are "consecutive" to compare consecutive bins and
        "all" to compare all bins.

    gamma : float, optional (default=0)
        Regularization strength to reduce the number of dominating bins. Larger
        values specify stronger regularization.

    cat_cutoff : float or None, optional (default=None)
        Generate bin others with categories in which the fraction of
        occurrences is below the  ``cat_cutoff`` value. This option is
        available when ``dtype`` is "categorical".

    cat_heuristic: bool (default=False):
        Whether to merge categories to guarantee max_n_prebins. If True,
        this option will be triggered when the number of categories >=
        max_n_prebins. This option is recommended if the number of categories,
        in the long run, can increase considerably, and recurrent calls to
        method ``solve`` are required.

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
    The parameter ``sketch`` is neglected when ``dtype=categorical``. The
    sketch parameter ``K`` is only applicable when ``sketch=t-digest``.

    Both quantile sketch algorithms produce good results, being the t-digest
    the most accurate. Note, however, the t-digest algorithm implementation is
    significantly slower than the GK implementation, thus, GK is the
    recommended algorithm when handling partitions. **Besides, GK is
    deterministic, therefore returning reproducible results.**
    """
    def __init__(self, name="", dtype="numerical", sketch="gk", eps=1e-4, K=25,
                 solver="cp", divergence="iv", max_n_prebins=20,
                 min_n_bins=None, max_n_bins=None, min_bin_size=None,
                 max_bin_size=None, min_bin_n_nonevent=None,
                 max_bin_n_nonevent=None, min_bin_n_event=None,
                 max_bin_n_event=None, monotonic_trend="auto",
                 min_event_rate_diff=0, max_pvalue=None,
                 max_pvalue_policy="consecutive", gamma=0, cat_cutoff=None,
                 cat_heuristic=False, special_codes=None, split_digits=None,
                 mip_solver="bop", time_limit=100, verbose=False):

        self.name = name
        self.dtype = dtype

        self.sketch = sketch
        self.eps = eps
        self.K = K

        self.solver = solver
        self.divergence = divergence

        self.max_n_prebins = max_n_prebins
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
        self.cat_cutoff = cat_cutoff
        self.cat_heuristic = cat_heuristic

        self.special_codes = special_codes
        self.split_digits = split_digits

        self.mip_solver = mip_solver
        self.time_limit = time_limit

        self.verbose = verbose

        # auxiliary
        self._flag_min_n_event_nonevent = False
        self._categories = None
        self._cat_others = []
        self._n_event = None
        self._n_nonevent = None
        self._n_nonevent_missing = None
        self._n_event_missing = None
        self._n_nonevent_special = None
        self._n_event_special = None
        self._n_event_special = None
        self._n_nonevent_cat_others = None
        self._n_event_cat_others = None

        # data storage
        self._bsketch = None

        # info
        self._binning_table = None
        self._n_refinements = 0
        self._n_prebins = None

        # streaming stats
        self._n_add = 0
        self._n_solve = 0
        self._solve_stats = {}

        # timming
        self._time_streaming_add = 0
        self._time_streaming_solve = 0

        self._time_total = None
        self._time_prebinning = None
        self._time_solver = None
        self._time_optimizer = None
        self._time_postprocessing = None

        self._is_solved = False

        # Check parameters
        _check_parameters(**self.get_params())

    def add(self, x, y, check_input=False):
        """Add new data x, y to the binning sketch.

        Parameters
        ----------
        x : array-like, shape = (n_samples,)
            Training vector, where n_samples is the number of samples.

        y : array-like, shape = (n_samples,)
            Target vector relative to x.

        check_input : bool (default=False)
            Whether to check input arrays.
        """
        if self._bsketch is None:
            if self.dtype == "numerical":
                self._bsketch = BSketch(self.sketch, self.eps, self.K,
                                        self.special_codes)
            else:
                self._bsketch = BCatSketch(self.cat_cutoff, self.special_codes)

        # Add new data stream
        time_add = time.perf_counter()

        self._bsketch.add(x, y, check_input)
        self._n_add += 1

        self._time_streaming_add += time.perf_counter() - time_add

        if self.verbose:
            logger.info("Sketch: added new data.")

    def information(self, print_level=1):
        """Print overview information about the options settings, problem
        statistics, and the solution of the computation.

        Parameters
        ----------
        print_level : int (default=1)
            Level of details.
        """
        self._check_is_solved()

        if not isinstance(print_level, numbers.Integral) or print_level < 0:
            raise ValueError("print_level must be an integer >= 0; got {}."
                             .format(print_level))

        binning_type = self.__class__.__name__.lower()

        # Optimizer
        if self._optimizer is not None:
            solver = self._optimizer
            time_solver = self._time_solver
        else:
            solver = None
            time_solver = 0

        # Sketch memory usage
        memory_usage = asizeof.asizeof(self._bsketch) * 1e-6

        dict_user_options = self.get_params()

        print_binning_information(binning_type, print_level, self.name,
                                  self._status, self.solver, solver,
                                  self._time_total, self._time_prebinning,
                                  time_solver, self._time_optimizer,
                                  self._time_postprocessing, self._n_prebins,
                                  self._n_refinements, self._bsketch.n,
                                  self._n_add, self._time_streaming_add,
                                  self._n_solve, self._time_streaming_solve,
                                  memory_usage, dict_user_options)

    def merge(self, optbsketch):
        """Merge current instance with another OptimalBinningSketch instance.

        Parameters
        ----------
        optbsketch : object
            OptimalBinningSketch instance.
        """
        if not self.mergeable(optbsketch):
            raise Exception("optbsketch does not share signature.")

        self._bsketch.merge(optbsketch._bsketch)

        if self.verbose:
            logger.info("Sketch: current sketch was merged.")

    def mergeable(self, optbsketch):
        """Check whether two OptimalBinningSketch instances can be merged.

        Parameters
        ----------
        optbsketch : object
            OptimalBinningSketch instance.

        Returns
        -------
        mergeable : bool
        """
        return self.get_params() == optbsketch.get_params()

    def plot_progress(self):
        """Plot divergence measure progress."""
        self._check_is_solved()

        df = pd.DataFrame.from_dict(self._solve_stats).T
        plot_progress_divergence(df, self.divergence)

    def solve(self):
        """Solve optimal binning using added data.

        Returns
        -------
        self : OptimalBinningSketch
            Current fitted optimal binning.
        """
        time_init = time.perf_counter()

        # Check if data was added
        if not self._n_add:
            raise NotFittedError("No data was added. Add data before solving.")

        # Pre-binning
        if self.verbose:
            logger.info("Pre-binning started.")

        time_prebinning = time.perf_counter()

        splits, n_nonevent, n_event = self._prebinning_data()
        self._n_prebins = len(splits) + 1

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
            n_nonevent = np.array([self._t_n_nonevent])
            n_event = np.array([self._t_n_event])

        self._n_nonevent, self._n_event = bin_info(
            self._solution, n_nonevent, n_event, self._n_nonevent_missing,
            self._n_event_missing, self._n_nonevent_special,
            self._n_event_special, self._n_nonevent_cat_others,
            self._n_event_cat_others, self._cat_others)

        self._binning_table = BinningTable(
            self.name, self.dtype, self.special_codes, self._splits_optimal,
            self._n_nonevent, self._n_event, None, None, self._categories,
            self._cat_others, None)

        self._time_postprocessing = time.perf_counter() - time_postprocessing

        if self.verbose:
            logger.info("Post-processing terminated. Time: {:.4f}s"
                        .format(self._time_postprocessing))

        self._time_total = time.perf_counter() - time_init
        self._time_streaming_solve += self._time_total
        self._n_solve += 1

        if self.verbose:
            logger.info("Optimal binning terminated. Status: {}. Time: {:.4f}s"
                        .format(self._status, self._time_total))

        # Completed successfully
        self._is_solved = True
        self._update_streaming_stats()

        return self

    def transform(self, x, metric="woe", metric_special=0,
                  metric_missing=0, show_digits=2, check_input=False):
        """Transform given data to Weight of Evidence (WoE) or event rate using
        bins from the current fitted optimal binning.

        Parameters
        ----------
        x : array-like, shape = (n_samples,)
            Training vector, where n_samples is the number of samples.

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
        x_new : numpy array, shape = (n_samples,)
            Transformed array.

        Notes
        -----
        Transformation of data including categories not present during training
        return zero WoE or event rate.
        """
        self._check_is_solved()

        return transform_binary_target(self._splits_optimal, self.dtype, x,
                                       self._n_nonevent, self._n_event,
                                       self.special_codes, self._categories,
                                       self._cat_others, metric,
                                       metric_special, metric_missing,
                                       None, show_digits, check_input)

    def _prebinning_data(self):
        self._n_nonevent_missing = self._bsketch._count_missing_ne
        self._n_nonevent_special = self._bsketch._count_special_ne
        self._n_event_missing = self._bsketch._count_missing_e
        self._n_event_special = self._bsketch._count_special_e

        self._t_n_nonevent = self._bsketch.n_nonevent
        self._t_n_event = self._bsketch.n_event

        if self.dtype == "numerical":
            sketch_all = self._bsketch.merge_sketches()

            if self.sketch == "gk":
                percentiles = np.linspace(0, 1, self.max_n_prebins + 1)

                splits = np.array([sketch_all.quantile(p)
                                   for p in percentiles[1:-1]])
            elif self.sketch == "t-digest":
                percentiles = np.linspace(0, 100, self.max_n_prebins + 1)

                splits = np.array([sketch_all.percentile(p)
                                   for p in percentiles[1:-1]])

            splits, n_nonevent, n_event = self._compute_prebins(splits)
        else:
            [splits, categories, n_nonevent, n_event, cat_others,
             n_nonevent_others, n_event_others] = self._bsketch.bins()

            self._categories = categories
            self._cat_others = cat_others
            self._n_nonevent_cat_others = n_nonevent_others
            self._n_event_cat_others = n_event_others

            [splits, categories, n_nonevent,
             n_event] = self._compute_cat_prebins(splits, categories,
                                                  n_nonevent, n_event)

        self._splits_prebinning = splits

        return splits, n_nonevent, n_event

    def _compute_prebins(self, splits):
        self._n_refinements = 0

        n_event, n_nonevent = self._bsketch.bins(splits)
        mask_remove = (n_nonevent == 0) | (n_event == 0)

        if np.any(mask_remove):
            if self.divergence in ("hellinger", "triangular"):
                self._flag_min_n_event_nonevent = True
            else:
                self._n_refinements += 1

                mask_splits = np.concatenate(
                    [mask_remove[:-2], [mask_remove[-2] | mask_remove[-1]]])

                splits = splits[~mask_splits]
                splits, n_nonevent, n_event = self._compute_prebins(splits)

        return splits, n_nonevent, n_event

    def _compute_cat_prebins(self, splits, categories, n_nonevent, n_event):
        self._n_refinements = 0
        mask_remove = (n_nonevent == 0) | (n_event == 0)

        if self.cat_heuristic and len(categories) > self.max_n_prebins:
            n_records = n_nonevent + n_event
            mask_size = n_records < self._bsketch.n / self.max_n_prebins
            mask_remove |= mask_size

        if np.any(mask_remove):
            if self.divergence in ("hellinger", "triangular"):
                self._flag_min_n_event_nonevent = True

                if self.cat_heuristic:
                    mask_remove = mask_size
            else:
                self._n_refinements += 1

            mask_splits = np.concatenate(
                [mask_remove[:-2], [mask_remove[-2] | mask_remove[-1]]])

            splits = splits[~mask_splits]

            splits_int = np.ceil(splits).astype(int)
            indices = np.digitize(np.arange(len(categories)), splits_int,
                                  right=False)
            n_bins = len(splits) + 1

            new_nonevent = np.empty(n_bins, dtype=np.int64)
            new_event = np.empty(n_bins, dtype=np.int64)
            new_categories = []
            for i in range(n_bins):
                mask = (indices == i)
                new_categories.append(categories[mask])
                new_nonevent[i] = n_nonevent[mask].sum()
                new_event[i] = n_event[mask].sum()

            new_categories = np.array(new_categories, dtype=object)

            [splits, categories, n_nonevent,
             n_event] = self._compute_cat_prebins(
                splits, new_categories, new_nonevent, new_event)

        return splits, categories, n_nonevent, n_event

    def _fit_optimizer(self, splits, n_nonevent, n_event):
        if self.verbose:
            logger.info("Optimizer started.")

        time_init = time.perf_counter()

        if len(n_nonevent) <= 1:
            self._status = "OPTIMAL"
            self._splits_optimal = splits
            self._solution = np.zeros(len(splits)).astype(bool)

            if self.verbose:
                logger.warning("Optimizer: {} bins after pre-binning."
                               .format(len(n_nonevent)))
                logger.warning("Optimizer: solver not run.")

                logger.info("Optimizer terminated. Time: 0s")
            return

        # Min/max number of bins
        if self.min_bin_size is not None:
            min_bin_size = int(np.ceil(self.min_bin_size * self._bsketch.n))
        else:
            min_bin_size = self.min_bin_size

        if self.max_bin_size is not None:
            max_bin_size = int(np.ceil(self.max_bin_size * self._bsketch.n))
        else:
            max_bin_size = self.max_bin_size

        # Min number of event and nonevent per bin
        if (self.divergence in ("hellinger", "triangular") and
                self._flag_min_n_event_nonevent):
            if self.min_bin_n_nonevent is None:
                min_bin_n_nonevent = 1
            else:
                min_bin_n_nonevent = max(self.min_bin_n_nonevent, 1)

            if self.min_bin_n_event is None:
                min_bin_n_event = 1
            else:
                min_bin_n_event = max(self.min_bin_n_event, 1)
        else:
            min_bin_n_nonevent = self.min_bin_n_nonevent
            min_bin_n_event = self.min_bin_n_event

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
                    logger.info("Optimizer: classifier predicts {} "
                                "monotonic trend.".format(monotonic))
            else:
                monotonic = self.monotonic_trend

                if monotonic in ("peak_heuristic", "valley_heuristic"):
                    event_rate = n_event / (n_nonevent + n_event)
                    trend_change = peak_valley_trend_change_heuristic(
                        event_rate, monotonic)

                    if self.verbose:
                        logger.info("Optimizer: trend change position {}."
                                    .format(trend_change))

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

        if self.solver == "cp":
            optimizer = BinningCP(monotonic, self.min_n_bins, self.max_n_bins,
                                  min_bin_size, max_bin_size,
                                  min_bin_n_event, self.max_bin_n_event,
                                  min_bin_n_nonevent, self.max_bin_n_nonevent,
                                  self.min_event_rate_diff, self.max_pvalue,
                                  self.max_pvalue_policy, self.gamma,
                                  None, self.time_limit)
        elif self.solver == "mip":
            optimizer = BinningMIP(monotonic, self.min_n_bins, self.max_n_bins,
                                   min_bin_size, max_bin_size,
                                   min_bin_n_event, self.max_bin_n_event,
                                   min_bin_n_nonevent, self.max_bin_n_nonevent,
                                   self.min_event_rate_diff, self.max_pvalue,
                                   self.max_pvalue_policy, self.gamma,
                                   None, self.mip_solver, self.time_limit)

        if self.verbose:
            logger.info("Optimizer: build model...")

        optimizer.build_model(self.divergence, n_nonevent, n_event,
                              trend_change)

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

    def _update_streaming_stats(self):
        self._binning_table.build()

        if self.divergence == "iv":
            dv = self._binning_table.iv
        elif self.divergence == "js":
            dv = self._binning_table.js
        elif self.divergence == "hellinger":
            dv = self._binning_table.hellinger
        elif self.divergence == "triangular":
            dv = self._binning_table.triangular

        self._solve_stats[self._n_solve] = {
            "n_add": self._n_add,
            "n_records": self._bsketch.n,
            "divergence".format(self.divergence): dv
        }

    @property
    def binning_table(self):
        """Return an instantiated binning table. Please refer to
        :ref:`Binning table: binary target`.

        Returns
        -------
        binning_table : BinningTable.
        """
        self._check_is_solved()

        return self._binning_table

    @property
    def splits(self):
        """List of optimal split points when ``dtype`` is set to "numerical" or
        list of optimal bins when ``dtype`` is set to "categorical".

        Returns
        -------
        splits : numpy.ndarray
        """
        self._check_is_solved()

        if self.dtype == "numerical":
            return self._splits_optimal
        else:
            return bin_categorical(self._splits_optimal, self._categories,
                                   self._cat_others, None)

    @property
    def status(self):
        """The status of the underlying optimization solver.

        Returns
        -------
        status : str
        """
        self._check_is_solved()

        return self._status
