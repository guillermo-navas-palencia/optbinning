"""
Binning process.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import numbers
import pickle
import time

from warnings import warn

import numpy as np
import pandas as pd

from joblib import Parallel, delayed, effective_n_jobs
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_array
from sklearn.utils import check_consistent_length
from sklearn.utils.multiclass import type_of_target

from ..logging import Logger
from .base import Base
from .binning import OptimalBinning
from .binning_process_information import print_binning_process_information
from .continuous_binning import ContinuousOptimalBinning
from .multiclass_binning import MulticlassOptimalBinning
from .piecewise.binning import OptimalPWBinning
from .piecewise.continuous_binning import ContinuousOptimalPWBinning


logger = Logger(__name__).logger


_METRICS = {
    "binary": {
        "metrics": ["iv", "js", "gini", "quality_score"],
        "iv": {"min": 0, "max": np.inf},
        "gini": {"min": 0, "max": 1},
        "js": {"min": 0, "max": np.inf},
        "quality_score": {"min": 0, "max": 1}
    },
    "multiclass": {
        "metrics": ["js", "quality_score"],
        "js": {"min": 0, "max": np.inf},
        "quality_score": {"min": 0, "max": 1}
    },
    "continuous": {
        "metrics": ["woe", "quality_score"],
        "woe": {"min": 0, "max": np.inf},
        "quality_score": {"min": 0, "max": 1}
    }
}


_OPTB_TYPES = (OptimalBinning, ContinuousOptimalBinning,
               MulticlassOptimalBinning)


_OPTBPW_TYPES = (OptimalPWBinning, ContinuousOptimalPWBinning)


def _read_column(input_path, extension, column, **kwargs):
    if extension == "csv":
        x = pd.read_csv(input_path, engine='c', usecols=[column],
                        low_memory=False, memory_map=True, **kwargs)
    elif extension == "parquet":
        x = pd.read_parquet(input_path, columns=[column], **kwargs)

    return x.iloc[:, 0].values


def _fit_variable(x, y, name, target_dtype, categorical_variables,
                  binning_fit_params, max_n_prebins, min_prebin_size,
                  min_n_bins, max_n_bins, min_bin_size, max_pvalue,
                  max_pvalue_policy, special_codes, split_digits,
                  sample_weight=None):
    params = {}
    dtype = _check_variable_dtype(x)

    if categorical_variables is not None:
        if name in categorical_variables:
            dtype = "categorical"

    if binning_fit_params is not None:
        params = binning_fit_params.get(name, {})

    if target_dtype == "binary":
        optb = OptimalBinning(
            name=name, dtype=dtype, max_n_prebins=max_n_prebins,
            min_prebin_size=min_prebin_size,
            min_n_bins=min_n_bins, max_n_bins=max_n_bins,
            min_bin_size=min_bin_size, max_pvalue=max_pvalue,
            max_pvalue_policy=max_pvalue_policy,
            special_codes=special_codes,
            split_digits=split_digits)
    elif target_dtype == "continuous":
        optb = ContinuousOptimalBinning(
            name=name, dtype=dtype, max_n_prebins=max_n_prebins,
            min_prebin_size=min_prebin_size,
            min_n_bins=min_n_bins, max_n_bins=max_n_bins,
            min_bin_size=min_bin_size, max_pvalue=max_pvalue,
            max_pvalue_policy=max_pvalue_policy,
            special_codes=special_codes,
            split_digits=split_digits)
    else:
        if dtype == "categorical":
            raise ValueError("MulticlassOptimalBinning does not support "
                             "categorical variables.")
        optb = MulticlassOptimalBinning(
            name=name, max_n_prebins=max_n_prebins,
            min_prebin_size=min_prebin_size,
            min_n_bins=min_n_bins, max_n_bins=max_n_bins,
            min_bin_size=min_bin_size, max_pvalue=max_pvalue,
            max_pvalue_policy=max_pvalue_policy,
            special_codes=special_codes,
            split_digits=split_digits)

    optb.set_params(**params)

    if target_dtype == "binary":
        optb.fit(x, y, sample_weight)
    else:
        optb.fit(x, y)

    return dtype, optb


def _fit_block(X, y, names, target_dtype, categorical_variables,
               binning_fit_params, max_n_prebins, min_prebin_size,
               min_n_bins, max_n_bins, min_bin_size, max_pvalue,
               max_pvalue_policy, special_codes, split_digits,
               sample_weight=None):

    variable_dtypes = {}
    binned_variables = {}

    for i, name in enumerate(names):
        if isinstance(X, np.ndarray):
            dtype, optb = _fit_variable(
                X[:, i], y, name, target_dtype, categorical_variables,
                binning_fit_params, max_n_prebins, min_prebin_size, min_n_bins,
                max_n_bins, min_bin_size, max_pvalue, max_pvalue_policy,
                special_codes, split_digits, sample_weight)
        else:
            dtype, optb = _fit_variable(
                X[name], y, name, target_dtype, categorical_variables,
                binning_fit_params, max_n_prebins, min_prebin_size, min_n_bins,
                max_n_bins, min_bin_size, max_pvalue, max_pvalue_policy,
                special_codes, split_digits, sample_weight)

        variable_dtypes[name] = dtype
        binned_variables[name] = optb

    return variable_dtypes, binned_variables


def _check_selection_criteria(selection_criteria, target_dtype):
    default_metrics_info = _METRICS[target_dtype]
    default_metrics = default_metrics_info["metrics"]

    if not all(m in default_metrics for m in selection_criteria.keys()):
        raise ValueError("metric for {} target must be in {}."
                         .format(target_dtype, default_metrics))

    for metric, info in selection_criteria.items():
        if not isinstance(info, dict):
            raise TypeError("metric {} info is not a dict.".format(metric))

        for key, value in info.items():
            if key == "min":
                min_ref = default_metrics_info[metric][key]
                if value < min_ref:
                    raise ValueError("metric {} min value {} < {}."
                                     .format(metric, value, min_ref))
            elif key == "max":
                max_ref = default_metrics_info[metric][key]
                if value > max_ref:
                    raise ValueError("metric {} max value {} > {}."
                                     .format(metric, value, max_ref))
            elif key == "strategy":
                if value not in ("highest", "lowest"):
                    raise ValueError('strategy value for metric {} must be '
                                     '"highest" or "lowest"; got {}.'
                                     .format(value, metric))
            elif key == "top":
                if isinstance(value, numbers.Integral):
                    if value < 1:
                        raise ValueError("top value must be at least 1 or "
                                         "in (0, 1); got {}.".format(value))
                else:
                    if not 0. < value < 1.:
                        raise ValueError("top value must be at least 1 or "
                                         "in (0, 1); got {}.".format(value))
            else:
                raise KeyError(key)


def _check_parameters(variable_names, max_n_prebins, min_prebin_size,
                      min_n_bins, max_n_bins, min_bin_size, max_bin_size,
                      max_pvalue, max_pvalue_policy, selection_criteria,
                      fixed_variables, categorical_variables, special_codes,
                      split_digits, binning_fit_params,
                      binning_transform_params, n_jobs, verbose):

    if not isinstance(variable_names, (np.ndarray, list)):
        raise TypeError("variable_names must be a list or numpy.ndarray.")

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

    if max_pvalue is not None:
        if (not isinstance(max_pvalue, numbers.Number) or
                not 0. < max_pvalue <= 1.0):
            raise ValueError("max_pvalue must be in (0, 1.0]; got {}."
                             .format(max_pvalue))

    if max_pvalue_policy not in ("all", "consecutive"):
        raise ValueError('Invalid value for max_pvalue_policy. Allowed string '
                         'values are "all" and "consecutive".')

    if selection_criteria is not None:
        if not isinstance(selection_criteria, dict):
            raise TypeError("selection_criteria must be a dict.")

    if fixed_variables is not None:
        if not isinstance(fixed_variables, (np.ndarray, list)):
            raise TypeError("fixed_variables must be a list or numpy.ndarray.")

    if categorical_variables is not None:
        if not isinstance(categorical_variables, (np.ndarray, list)):
            raise TypeError("categorical_variables must be a list or "
                            "numpy.ndarray.")

        if not all(isinstance(c, str) for c in categorical_variables):
            raise TypeError("variables in categorical_variables must be "
                            "strings.")

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
            raise ValueError("split_digits must be an integer in [0, 8]; "
                             "got {}.".format(split_digits))

    if binning_fit_params is not None:
        if not isinstance(binning_fit_params, dict):
            raise TypeError("binning_fit_params must be a dict.")

    if binning_transform_params is not None:
        if not isinstance(binning_transform_params, dict):
            raise TypeError("binning_transform_params must be a dict.")

    if n_jobs is not None:
        if not isinstance(n_jobs, numbers.Integral):
            raise ValueError("n_jobs must be an integer or None; got {}."
                             .format(n_jobs))

    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a boolean; got {}.".format(verbose))


def _check_variable_dtype(x):
    return "categorical" if x.dtype == object else "numerical"


class BaseBinningProcess:
    @classmethod
    def load(cls, path):
        """Load binning process from pickle file.

        Parameters
        ----------
        path : str
            Pickle file path.

        Example
        -------
        >>> from optbinning import BinningProcess
        >>> binning_process = BinningProcess.load("my_binning_process.pkl")
        """
        if not isinstance(path, str):
            raise TypeError("path must be a string.")

        with open(path, "rb") as f:
            return pickle.load(f)

    def save(self, path):
        """Save binning process to pickle file.

        Parameters
        ----------
        path : str
            Pickle file path.
        """
        if not isinstance(path, str):
            raise TypeError("path must be a string.")

        with open(path, "wb") as f:
            pickle.dump(self, f)

    def _support_selection_criteria(self):
        self._support = np.full(self._n_variables, True, dtype=bool)

        if self.selection_criteria is None:
            return

        default_metrics_info = _METRICS[self._target_dtype]
        criteria_metrics = self.selection_criteria.keys()

        binning_metrics = pd.DataFrame.from_dict(self._variable_stats).T

        for metric in default_metrics_info["metrics"]:
            if metric in criteria_metrics:
                metric_info = self.selection_criteria[metric]
                metric_values = binning_metrics[metric].values

                if "min" in metric_info:
                    self._support &= metric_values >= metric_info["min"]
                if "max" in metric_info:
                    self._support &= metric_values <= metric_info["max"]
                if all(m in metric_info for m in ("strategy", "top")):
                    indices_valid = np.where(self._support)[0]
                    metric_values = metric_values[indices_valid]
                    n_valid = len(metric_values)

                    # Auxiliary support
                    support = np.full(self._n_variables, False, dtype=bool)

                    top = metric_info["top"]
                    if not isinstance(top, numbers.Integral):
                        top = int(np.ceil(n_valid * top))
                    n_selected = min(n_valid, top)

                    if metric_info["strategy"] == "highest":
                        mask = np.argsort(-metric_values)[:n_selected]
                    elif metric_info["strategy"] == "lowest":
                        mask = np.argsort(metric_values)[:n_selected]

                    support[indices_valid[mask]] = True
                    self._support &= support

        # Fixed variables
        if self.fixed_variables is not None:
            for fv in self.fixed_variables:
                idfv = list(self.variable_names).index(fv)
                self._support[idfv] = True

    def _binning_selection_criteria(self):
        for i, name in enumerate(self.variable_names):
            optb = self._binned_variables[name]
            optb.binning_table.build()

            n_bins = len(optb.splits)
            if isinstance(optb, OptimalPWBinning) or optb.dtype == "numerical":
                n_bins += 1

            if isinstance(optb, OptimalPWBinning):
                dtype = "numerical"
            else:
                dtype = optb.dtype

            info = {"dtype": dtype,
                    "status": optb.status,
                    "n_bins": n_bins}

            optb.binning_table.analysis(print_output=False)

            if self._target_dtype == "binary":
                metrics = {
                    "iv": optb.binning_table.iv,
                    "gini": optb.binning_table.gini,
                    "js": optb.binning_table.js,
                    "quality_score": optb.binning_table.quality_score}
            elif self._target_dtype == "multiclass":
                metrics = {
                    "js": optb.binning_table.js,
                    "quality_score": optb.binning_table.quality_score}
            elif self._target_dtype == "continuous":
                metrics = {
                    "woe": optb.binning_table.woe,
                    "quality_score": optb.binning_table.quality_score}

            info = {**info, **metrics}
            self._variable_stats[name] = info

        self._support_selection_criteria()


class BinningProcess(Base, BaseEstimator, BaseBinningProcess):
    """Binning process to compute optimal binning of variables in a dataset,
    given a binary, continuous or multiclass target dtype.

    Parameters
    ----------
    variable_names : array-like
        List of variable names.

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

    max_pvalue : float or None, optional (default=0.05)
        The maximum p-value among bins.

    max_pvalue_policy : str, optional (default="consecutive")
        The method to determine bins not satisfying the p-value constraint.
        Supported methods are "consecutive" to compare consecutive bins and
        "all" to compare all bins.

    selection_criteria : dict or None (default=None)
        Variable selection criteria. See notes.

        .. versionadded:: 0.6.0

    fixed_variables : array-like or None
        List of variables to be fixed. The binning process will retain these
        variables if the selection criteria is not satisfied.

        .. versionadded:: 0.12.1

    special_codes : array-like or None, optional (default=None)
        List of special codes. Use special codes to specify the data values
        that must be treated separately.

    split_digits : int or None, optional (default=None)
        The significant digits of the split points. If ``split_digits`` is set
        to 0, the split points are integers. If None, then all significant
        digits in the split points are considered.

    categorical_variables : array-like or None, optional (default=None)
        List of variables numerical variables to be considered categorical.
        These are nominal variables. Not applicable when target type is
        multiclass.

    binning_fit_params : dict or None, optional (default=None)
        Dictionary with optimal binning fitting options for specific variables.
        Example: ``{"variable_1": {"max_n_bins": 4}}``.

    binning_transform_params : dict or None, optional (default=None)
        Dictionary with optimal binning transform options for specific
        variables. Example ``{"variable_1": {"metric": "event_rate"}}``.

    n_jobs : int or None, optional (default=None)
        Number of cores to run in parallel while binning variables.
        ``None`` means 1 core. ``-1`` means using all processors.

        .. versionadded:: 0.7.1

    verbose : bool (default=False)
        Enable verbose output.

    Notes
    -----
    Parameter ``selection_criteria`` allows to specify criteria for
    variable selection. The input is a dictionary as follows

    .. code::

        selection_criteria = {
            "metric_1":
                {
                    "min": 0, "max": 1, "strategy": "highest", "top": 0.25
                },
            "metric_2":
                {
                    "min": 0.02
                }
        }

    where several metrics can be combined. For example, above dictionary
    indicates that top 25% variables with "metric_1" in [0, 1] and "metric_2"
    greater or equal than 0.02 are selected. Supported key values are:

    * keys ``min`` and ``max`` support numerical values.
    * key ``strategy`` supports options "highest" and "lowest".
    * key ``top`` supports an integer or decimal (percentage).


    .. warning::

        If the binning process instance is going to be saved, do not pass the
        option ``"solver": "mip"`` via the ``binning_fit_params`` parameter.

    """
    def __init__(self, variable_names, max_n_prebins=20, min_prebin_size=0.05,
                 min_n_bins=None, max_n_bins=None, min_bin_size=None,
                 max_bin_size=None, max_pvalue=None,
                 max_pvalue_policy="consecutive", selection_criteria=None,
                 fixed_variables=None, categorical_variables=None,
                 special_codes=None, split_digits=None,
                 binning_fit_params=None, binning_transform_params=None,
                 n_jobs=None, verbose=False):

        self.variable_names = variable_names

        self.max_n_prebins = max_n_prebins
        self.min_prebin_size = min_prebin_size
        self.min_n_bins = min_n_bins
        self.max_n_bins = max_n_bins
        self.min_bin_size = min_bin_size
        self.max_bin_size = max_bin_size
        self.max_pvalue = max_pvalue
        self.max_pvalue_policy = max_pvalue_policy

        self.selection_criteria = selection_criteria
        self.fixed_variables = fixed_variables

        self.binning_fit_params = binning_fit_params
        self.binning_transform_params = binning_transform_params

        self.special_codes = special_codes
        self.split_digits = split_digits
        self.categorical_variables = categorical_variables
        self.n_jobs = n_jobs
        self.verbose = verbose

        # auxiliary
        self._n_samples = None
        self._n_variables = None
        self._target_dtype = None
        self._n_numerical = None
        self._n_categorical = None
        self._n_selected = None
        self._binned_variables = {}
        self._variable_dtypes = {}
        self._variable_stats = {}

        self._support = None

        # timing
        self._time_total = None

        self._is_updated = False
        self._is_fitted = False

    def fit(self, X, y, sample_weight=None, check_input=False):
        """Fit the binning process. Fit the optimal binning to all variables
        according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples.

            .. versionchanged:: 0.4.0
            X supports ``numpy.ndarray`` and ``pandas.DataFrame``.

        y : array-like of shape (n_samples,)
            Target vector relative to x.

        sample_weight : array-like of shape (n_samples,) (default=None)
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.
            Only applied if ``prebinning_method="cart"``. This option is only
            available for a binary target.

        check_input : bool (default=False)
            Whether to check input arrays.

        Returns
        -------
        self : BinningProcess
            Fitted binning process.
        """
        return self._fit(X, y, sample_weight, check_input)

    def fit_disk(self, input_path, target, **kwargs):
        """Fit the binning process according to the given training data on
        disk.

        Parameters
        ----------
        input_path : str
            Any valid string path to a file with extension .csv or .parquet.

        target : str
            Target column.

        **kwargs : keyword arguments
            Keyword arguments for ``pandas.read_csv`` or
            ``pandas.read_parquet``.

        Returns
        -------
        self : BinningProcess
            Fitted binning process.
        """
        return self._fit_disk(input_path, target, **kwargs)

    def fit_from_dict(self, dict_optb):
        """Fit the binning process from a dict of OptimalBinning objects
        already fitted.

        Parameters
        ----------
        dict_optb : dict
            Dictionary with OptimalBinning objects for binary, continuous
            or multiclass target. All objects must share the same class.

        Returns
        -------
        self : BinningProcess
            Fitted binning process.
        """
        return self._fit_from_dict(dict_optb)

    def fit_transform(self, X, y, sample_weight=None, metric=None,
                      metric_special=0, metric_missing=0, show_digits=2,
                      check_input=False):
        """Fit the binning process according to the given training data, then
        transform it.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples.

        y : array-like of shape (n_samples,)
            Target vector relative to x.

        sample_weight : array-like of shape (n_samples,) (default=None)
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.
            Only applied if ``prebinning_method="cart"``. This option is only
            available for a binary target.

        metric : str or None, (default=None)
            The metric used to transform the input vector. If None, the default
            transformation metric for each target type is applied. For binary
            target options are: "woe" (default), "event_rate", "indices" and
            "bins". For continuous target options are: "mean" (default),
            "indices" and "bins". For multiclass target options are:
            "mean_woe" (default), "weighted_mean_woe", "indices" and "bins".

        metric_special : float or str (default=0)
            The metric value to transform special codes in the input vector.
            Supported metrics are "empirical" to use the empirical WoE or
            event rate for a binary target, and any numerical value for other
            targets.

        metric_missing : float or str (default=0)
            The metric value to transform missing values in the input vector.
            Supported metrics are "empirical" to use the empirical WoE or
            event rate for a binary target, and any numerical value for other
            targets.

        show_digits : int, optional (default=2)
            The number of significant digits of the bin column. Applies when
            ``metric="bins"``.

        check_input : bool (default=False)
            Whether to check input arrays.

        Returns
        -------
        X_new : numpy array, shape = (n_samples, n_features_new)
            Transformed array.
        """
        return self.fit(X, y, sample_weight, check_input).transform(
            X, metric, metric_special, metric_missing, show_digits,
            check_input)

    def fit_transform_disk(self, input_path, output_path, target, chunksize,
                           metric=None, metric_special=0, metric_missing=0,
                           show_digits=2, **kwargs):
        """Fit the binning process according to the given training data on
        disk, then transform it and save to comma-separated values (csv) file.

        Parameters
        ----------
        input_path : str
            Any valid string path to a file with extension .csv.

        output_path : str
            Any valid string path to a file with extension .csv.

        target : str
            Target column.

        chunksize :
            Rows to read, transform and write at a time.

        metric : str or None, (default=None)
            The metric used to transform the input vector. If None, the default
            transformation metric for each target type is applied. For binary
            target options are: "woe" (default), "event_rate", "indices" and
            "bins". For continuous target options are: "mean" (default),
            "indices" and "bins". For multiclass target options are:
            "mean_woe" (default), "weighted_mean_woe", "indices" and "bins".

        metric_special : float or str (default=0)
            The metric value to transform special codes in the input vector.
            Supported metrics are "empirical" to use the empirical WoE or
            event rate for a binary target, and any numerical value for other
            targets.

        metric_missing : float or str (default=0)
            The metric value to transform missing values in the input vector.
            Supported metrics are "empirical" to use the empirical WoE or
            event rate for a binary target, and any numerical value for other
            targets.

        show_digits : int, optional (default=2)
            The number of significant digits of the bin column. Applies when
            ``metric="bins"``.

        **kwargs : keyword arguments
            Keyword arguments for ``pandas.read_csv``.

        Returns
        -------
        self : BinningProcess
            Fitted binning process.
        """
        return self.fit_disk(input_path, target, **kwargs).transform_disk(
            input_path, output_path, chunksize, metric, metric_special,
            metric_missing, show_digits, **kwargs)

    def transform(self, X, metric=None, metric_special=0, metric_missing=0,
                  show_digits=2, check_input=False):
        """Transform given data to metric using bins from each fitted optimal
        binning.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples.

        metric : str or None, (default=None)
            The metric used to transform the input vector. If None, the default
            transformation metric for each target type is applied. For binary
            target options are: "woe" (default), "event_rate", "indices" and
            "bins". For continuous target options are: "mean" (default),
            "indices" and "bins". For multiclass target options are:
            "mean_woe" (default), "weighted_mean_woe", "indices" and "bins".

        metric_special : float or str (default=0)
            The metric value to transform special codes in the input vector.
            Supported metrics are "empirical" to use the empirical WoE or
            event rate for a binary target, and any numerical value for other
            targets.

        metric_missing : float or str (default=0)
            The metric value to transform missing values in the input vector.
            Supported metrics are "empirical" to use the empirical WoE or
            event rate for a binary target, and any numerical value for other
            targets.

        show_digits : int, optional (default=2)
            The number of significant digits of the bin column. Applies when
            ``metric="bins"``.

        check_input : bool (default=False)
            Whether to check input arrays.

        Returns
        -------
        X_new : numpy array or pandas.DataFrame, shape = (n_samples,
        n_features_new)
            Transformed array.
        """
        self._check_is_fitted()

        return self._transform(X, metric, metric_special, metric_missing,
                               show_digits, check_input)

    def transform_disk(self, input_path, output_path, chunksize, metric=None,
                       metric_special=0, metric_missing=0, show_digits=2,
                       **kwargs):
        """Transform given data on disk to metric using bins from each fitted
        optimal binning. Save to comma-separated values (csv) file.

        Parameters
        ----------
        input_path : str
            Any valid string path to a file with extension .csv.

        output_path : str
            Any valid string path to a file with extension .csv.

        chunksize :
            Rows to read, transform and write at a time.

        metric : str or None, (default=None)
            The metric used to transform the input vector. If None, the default
            transformation metric for each target type is applied. For binary
            target options are: "woe" (default), "event_rate", "indices" and
            "bins". For continuous target options are: "mean" (default),
            "indices" and "bins". For multiclass target options are:
            "mean_woe" (default), "weighted_mean_woe", "indices" and "bins".

        metric_special : float or str (default=0)
            The metric value to transform special codes in the input vector.
            Supported metrics are "empirical" to use the empirical WoE or
            event rate for a binary target, and any numerical value for other
            targets.

        metric_missing : float or str (default=0)
            The metric value to transform missing values in the input vector.
            Supported metrics are "empirical" to use the empirical WoE or
            event rate for a binary target, and any numerical value for other
            targets.

        show_digits : int, optional (default=2)
            The number of significant digits of the bin column. Applies when
            ``metric="bins"``.

        **kwargs : keyword arguments
            Keyword arguments for ``pandas.read_csv``.

        Returns
        -------
        self : BinningProcess
            Fitted binning process.
        """
        self._check_is_fitted()

        return self._transform_disk(input_path, output_path, chunksize, metric,
                                    metric_special, metric_missing,
                                    show_digits, **kwargs)

    def information(self, print_level=1):
        """Print overview information about the options settings and
        statistics.

        Parameters
        ----------
        print_level : int (default=1)
            Level of details.
        """
        self._check_is_fitted()

        if not isinstance(print_level, numbers.Integral) or print_level < 0:
            raise ValueError("print_level must be an integer >= 0; got {}."
                             .format(print_level))

        n_numerical = list(self._variable_dtypes.values()).count("numerical")
        n_categorical = self._n_variables - n_numerical

        self._n_selected = np.count_nonzero(self._support)

        dict_user_options = self.get_params()

        print_binning_process_information(
            print_level, self._n_samples, self._n_variables,
            self._target_dtype, n_numerical, n_categorical,
            self._n_selected, self._time_total, dict_user_options)

    def summary(self):
        """Binning process summary with main statistics for all binned
        variables.

        Parameters
        ----------
        df_summary : pandas.DataFrame
            Binning process summary.
        """
        self._check_is_fitted()

        if self._is_updated:
            self._binning_selection_criteria()
            self._is_updated = False

        df_summary = pd.DataFrame.from_dict(self._variable_stats).T
        df_summary.reset_index(inplace=True)
        df_summary.rename(columns={"index": "name"}, inplace=True)
        df_summary["selected"] = self._support

        columns = ["name", "dtype", "status", "selected", "n_bins"]
        columns += _METRICS[self._target_dtype]["metrics"]

        return df_summary[columns]

    def get_binned_variable(self, name):
        """Return optimal binning object for a given variable name.

        Parameters
        ----------
        name : string
            The variable name.
        """
        self._check_is_fitted()

        if not isinstance(name, str):
            raise TypeError("name must be a string.")

        if name in self.variable_names:
            return self._binned_variables[name]
        else:
            raise ValueError("name {} does not match a binned variable."
                             .format(name))

    def update_binned_variable(self, name, optb):
        """Update optimal binning object for a given variable.

        Parameters
        ----------
        name : string
            The variable name.

        optb : object
            The optimal binning object already fitted.
        """
        self._check_is_fitted()

        if not isinstance(name, str):
            raise TypeError("name must be a string.")

        if name not in self.variable_names:
            raise ValueError("name {} does not match a binned variable."
                             .format(name))

        optb_types = _OPTB_TYPES + _OPTBPW_TYPES

        if not isinstance(optb, optb_types):
            raise TypeError("Object {} must be of type ({}); got {}"
                            .format(name, optb_types, type(optb)))

        # Check current class
        if self._target_dtype == "binary":
            optb_binary = (OptimalBinning, OptimalPWBinning)
            if not isinstance(optb, optb_binary):
                raise TypeError("target is binary and Object {} must be of "
                                "type {}.".format(optb, optb_binary))
        elif self._target_dtype == "continuous":
            optb_continuous = (ContinuousOptimalBinning,
                               ContinuousOptimalPWBinning)
            if not isinstance(optb, optb_continuous):
                raise TypeError("target is continuous and Object {} must be "
                                "of type {}.".format(optb, optb_continuous))
        elif self._target_dtype == "multiclass":
            if not isinstance(optb, MulticlassOptimalBinning):
                raise TypeError("target is multiclass and Object {} must be "
                                "of type {}.".format(
                                    optb, MulticlassOptimalBinning))

        optb_old = self._binned_variables[name]
        if optb_old.name and optb_old.name != optb.name:
            raise ValueError("Update object name must match old object name; "
                             "{} != {}.".format(optb_old.name, optb.name))

        if optb.name and name != optb.name:
            raise ValueError("name and object name must coincide.")

        self._binned_variables[name] = optb
        self._is_updated = True

    def get_support(self, indices=False, names=False):
        """Get a mask, or integer index, or names of the variables selected.

        Parameters
        ----------
        indices : boolean (default=False)
            If True, the return value will be an array of integers, rather
            than a boolean mask.

        names : boolean (default=False)
            If True, the return value will be an array of strings, rather
            than a boolean mask.

        Returns
        -------
        support : array
            An index that selects the retained features from a feature vector.
            If `indices` is False, this is a boolean array of shape
            [# input features], in which an element is True iff its
            corresponding feature is selected for retention. If `indices` is
            True, this is an integer array of shape [# output features] whose
            values are indices into the input feature vector. If `names` is
            True, this is an string array of sahpe [# output features], whose
            values are names of the selected features.
        """
        self._check_is_fitted()

        if indices and names:
            raise ValueError("Only indices or names can be True.")

        mask = self._support
        if indices:
            return np.where(mask)[0]
        elif names:
            return np.asarray(self.variable_names)[mask]
        else:
            return mask

    def _fit(self, X, y, sample_weight, check_input):
        time_init = time.perf_counter()

        if self.verbose:
            logger.info("Binning process started.")
            logger.info("Options: check parameters.")

        _check_parameters(**self.get_params())

        # check X dtype
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError("X must be a pandas.DataFrame or numpy.ndarray.")

        # check target dtype
        self._target_dtype = type_of_target(y)

        if self._target_dtype not in ("binary", "continuous", "multiclass"):
            raise ValueError("Target type {} is not supported."
                             .format(self._target_dtype))

        # check sample weight
        if sample_weight is not None and self._target_dtype != "binary":
            raise ValueError("Target type {} does not support sample weight."
                             .format(self._target_dtype))

        if self.selection_criteria is not None:
            _check_selection_criteria(self.selection_criteria,
                                      self._target_dtype)

        # check X and y data
        if check_input:
            X = check_array(X, ensure_2d=False, dtype=None,
                            force_all_finite='allow-nan')

            y = check_array(y, ensure_2d=False, dtype=None,
                            force_all_finite=True)

            check_consistent_length(X, y)

        self._n_samples, self._n_variables = X.shape

        if self._n_variables != len(self.variable_names):
            raise ValueError("The number of columns must be equal to the"
                             "length of variable_names.")

        if self.verbose:
            logger.info("Dataset: number of samples: {}."
                        .format(self._n_samples))

            logger.info("Dataset: number of variables: {}."
                        .format(self._n_variables))

        # Number of jobs
        n_jobs = effective_n_jobs(self.n_jobs)

        if self.verbose:
            logger.info("Options: number of jobs (cores): {}."
                        .format(n_jobs))

        if n_jobs == 1:
            for i, name in enumerate(self.variable_names):
                if self.verbose:
                    logger.info("Binning variable ({} / {}): {}."
                                .format(i, self._n_variables, name))

                if isinstance(X, np.ndarray):
                    dtype, optb = _fit_variable(
                        X[:, i], y, name, self._target_dtype,
                        self.categorical_variables, self.binning_fit_params,
                        self.max_n_prebins, self.min_prebin_size,
                        self.min_n_bins, self.max_n_bins, self.min_bin_size,
                        self.max_pvalue, self.max_pvalue_policy,
                        self.special_codes, self.split_digits, sample_weight)
                else:
                    dtype, optb = _fit_variable(
                        X[name], y, name, self._target_dtype,
                        self.categorical_variables, self.binning_fit_params,
                        self.max_n_prebins, self.min_prebin_size,
                        self.min_n_bins, self.max_n_bins, self.min_bin_size,
                        self.max_pvalue, self.max_pvalue_policy,
                        self.special_codes, self.split_digits, sample_weight)

                self._variable_dtypes[name] = dtype
                self._binned_variables[name] = optb
        else:
            ids = np.arange(len(self.variable_names))
            id_blocks = np.array_split(ids, n_jobs)
            names = np.asarray(self.variable_names)

            if isinstance(X, np.ndarray):
                blocks = Parallel(n_jobs=n_jobs, prefer="threads")(
                    delayed(_fit_block)(
                        X[:, id_block], y, names[id_block],
                        self._target_dtype, self.categorical_variables,
                        self.binning_fit_params, self.max_n_prebins,
                        self.min_prebin_size, self.min_n_bins,
                        self.max_n_bins, self.min_bin_size,
                        self.max_pvalue, self.max_pvalue_policy,
                        self.special_codes, self.split_digits)
                    for id_block in id_blocks)

            else:
                blocks = Parallel(n_jobs=n_jobs, prefer="threads")(
                    delayed(_fit_block)(
                        X[names[id_block]], y, names[id_block],
                        self._target_dtype, self.categorical_variables,
                        self.binning_fit_params, self.max_n_prebins,
                        self.min_prebin_size, self.min_n_bins,
                        self.max_n_bins, self.min_bin_size,
                        self.max_pvalue, self.max_pvalue_policy,
                        self.special_codes, self.split_digits)
                    for id_block in id_blocks)

            for b in blocks:
                vt, bv = b
                self._variable_dtypes.update(vt)
                self._binned_variables.update(bv)

        if self.verbose:
            logger.info("Binning process variable selection...")

        # Compute binning statistics and decide whether a variable is selected
        self._binning_selection_criteria()

        self._time_total = time.perf_counter() - time_init

        if self.verbose:
            logger.info("Binning process terminated. Time: {:.4f}s"
                        .format(self._time_total))

        # Completed successfully
        self._is_fitted = True

        return self

    def _fit_disk(self, input_path, target, **kwargs):
        time_init = time.perf_counter()

        if self.verbose:
            logger.info("Binning process started.")
            logger.info("Options: check parameters.")

        _check_parameters(**self.get_params())

        # Input file extension
        extension = input_path.split(".")[1]

        # Check extension
        if extension not in ("csv", "parquet"):
            raise ValueError("input_path extension must be csv or parquet; "
                             "got {}.".format(extension))

        # Check target
        if not isinstance(target, str):
            raise TypeError("target must be a string.")

        # Retrieve target and check dtype
        y = _read_column(input_path, extension, target, **kwargs)
        self._target_dtype = type_of_target(y)

        if self._target_dtype not in ("binary", "continuous", "multiclass"):
            raise ValueError("Target type {} is not supported."
                             .format(self._target_dtype))

        if self.selection_criteria is not None:
            _check_selection_criteria(self.selection_criteria,
                                      self._target_dtype)

        if self.fixed_variables is not None:
            for fv in self.fixed_variables:
                if fv not in self.variable_names:
                    raise ValueError("Variable {} to be fixed is not a valid "
                                     "variable name.".format(fv))

        self._n_samples = len(y)
        self._n_variables = len(self.variable_names)

        if self.verbose:
            logger.info("Dataset: number of samples: {}."
                        .format(self._n_samples))

            logger.info("Dataset: number of variables: {}."
                        .format(self._n_variables))

        for name in self.variable_names:
            x = _read_column(input_path, extension, name, **kwargs)

            dtype, optb = _fit_variable(
                x, y, name, self._target_dtype, self.categorical_variables,
                self.binning_fit_params, self.max_n_prebins,
                self.min_prebin_size, self.min_n_bins, self.max_n_bins,
                self.min_bin_size, self.max_pvalue, self.max_pvalue_policy,
                self.special_codes, self.split_digits)

            self._variable_dtypes[name] = dtype
            self._binned_variables[name] = optb

        if self.verbose:
            logger.info("Binning process variable selection...")

        # Compute binning statistics and decide whether a variable is selected
        self._binning_selection_criteria()

        self._time_total = time.perf_counter() - time_init

        if self.verbose:
            logger.info("Binning process terminated. Time: {:.4f}s"
                        .format(self._time_total))

        # Completed successfully
        self._is_fitted = True

        return self

    def _fit_from_dict(self, dict_optb):
        time_init = time.perf_counter()

        if self.verbose:
            logger.info("Binning process started.")
            logger.info("Options: check parameters.")

        _check_parameters(**self.get_params())

        if not isinstance(dict_optb, dict):
            raise TypeError("dict_optb must be a dict.")

        # Check variable names
        if set(dict_optb.keys()) != set(self.variable_names):
            raise ValueError("dict_optb keys and variable names must "
                             "coincide.")

        # Check objects class
        optb_types = _OPTB_TYPES
        types = set()
        for name, optb in dict_optb.items():
            if not isinstance(name, str):
                raise TypeError("Object key must be a string.")

            if not isinstance(optb, optb_types):
                raise TypeError("Object {} must be of type ({}); got {}"
                                .format(name, optb_types, type(optb)))

            types.add(type(optb).__name__)
            if len(types) > 1:
                raise TypeError("All binning objects must be of the same "
                                "class.")

            # Check if fitted
            if not optb._is_fitted:
                raise NotFittedError("Object with key={} is not fitted yet. "
                                     "Call 'fit' for this object before "
                                     "passing to a binning process."
                                     .format(name))

            # Check if name was provided and matches dict_optb key.
            if optb.name and optb.name != name:
                raise ValueError("Object with key={} has attribute name={}. "
                                 "If object has a name those must coincide."
                                 .format(name, optb.name))

        obj_class = types.pop()
        if obj_class == "OptimalBinning":
            self._target_dtype = "binary"
        elif obj_class == "ContinuousOptimalBinning":
            self._target_dtype = "continuous"
        elif obj_class == "MulticlassOptimalBinning":
            self._target_dtype = "multiclass"

        if self.selection_criteria is not None:
            _check_selection_criteria(self.selection_criteria,
                                      self._target_dtype)

        self._n_samples = 0
        self._n_variables = len(self.variable_names)

        for name, optb in dict_optb.items():
            self._variable_dtypes[name] = optb.dtype
            self._binned_variables[name] = optb

        # Compute binning statistics and decide whether a variable is selected
        self._binning_selection_criteria()

        self._time_total = time.perf_counter() - time_init

        if self.verbose:
            logger.info("Binning process terminated. Time: {:.4f}s"
                        .format(self._time_total))

        # Completed successfully
        self._is_fitted = True

        return self

    def _transform(self, X, metric, metric_special, metric_missing,
                   show_digits, check_input):

        # Check X dtype
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError("X must be a pandas.DataFrame or numpy.ndarray.")

        n_samples, n_variables = X.shape

        mask = self.get_support()
        if not mask.any():
            warn("No variables were selected: either the data is"
                 " too noisy or the selection_criteria too strict.",
                 UserWarning)
            return np.empty(0).reshape((n_samples, 0))

        if isinstance(X, np.ndarray) and len(mask) != n_variables:
            raise ValueError("X has a different shape that during fitting.")

        if isinstance(X, pd.DataFrame):
            selected_variables = self.get_support(names=True)
            for name in selected_variables:
                if name not in X.columns:
                    raise ValueError("Selected variable {} must be a column "
                                     "in the input dataframe.".format(name))

        # Check metric
        if metric in ("indices", "bins"):
            if any(isinstance(optb, _OPTBPW_TYPES)
                   for optb in self._binned_variables.values()):
                raise TypeError("metric {} not supported for piecewise "
                                "optimal binning objects.".format(metric))

        indices_selected_variables = self.get_support(indices=True)
        n_selected_variables = len(indices_selected_variables)

        if metric == "indices":
            X_transform = np.full(
                (n_samples, n_selected_variables), -1, dtype=int)
        elif metric == "bins":
            X_transform = np.full(
                (n_samples, n_selected_variables), "", dtype=object)
        else:
            X_transform = np.zeros((n_samples, n_selected_variables))

        for i, idx in enumerate(indices_selected_variables):
            name = self.variable_names[idx]
            optb = self._binned_variables[name]

            if isinstance(X, np.ndarray):
                x = X[:, idx]
            else:
                x = X[name]

            params = {}
            if self.binning_transform_params is not None:
                params = self.binning_transform_params.get(name, {})

            metric = params.get("metric", metric)
            metric_missing = params.get("metric_missing", metric_missing)
            metric_special = params.get("metric_special", metric_special)

            tparams = {
                "x": x,
                "metric": metric,
                "metric_special": metric_special,
                "metric_missing": metric_missing,
                "check_input": check_input,
                "show_digits": show_digits
                }

            if isinstance(optb, _OPTBPW_TYPES):
                tparams.pop("show_digits")

            if metric is None:
                tparams.pop("metric")

            X_transform[:, i] = optb.transform(**tparams)

        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(X_transform, columns=selected_variables)

        return X_transform

    def _transform_disk(self, input_path, output_path, chunksize, metric,
                        metric_special, metric_missing, show_digits, **kwargs):

        # check input_path and output_path extensions
        input_extension = input_path.split(".")[1]
        output_extension = output_path.split(".")[1]

        if input_extension != "csv" or output_extension != "csv":
            raise ValueError("input_path and output_path must be csv files.")

        # check chunksize
        if not isinstance(chunksize, numbers.Integral) or chunksize <= 0:
            raise ValueError("chunksize must be a positive integer; got {}."
                             .format(chunksize))

        # Check metric
        if metric in ("indices", "bins"):
            if any(isinstance(optb, _OPTBPW_TYPES)
                   for optb in self._binned_variables.values()):
                raise TypeError("metric {} not supported for piecewise "
                                "optimal binning objects.".format(metric))

        selected_variables = self.get_support(names=True)
        n_selected_variables = len(selected_variables)

        chunks = pd.read_csv(input_path, engine='c', chunksize=chunksize,
                             usecols=selected_variables, **kwargs)

        for k, chunk in enumerate(chunks):
            n_samples, n_variables = chunk.shape

            if metric == "indices":
                X_transform = np.full(
                    (n_samples, n_selected_variables), -1, dtype=int)
            elif metric == "bins":
                X_transform = np.full(
                    (n_samples, n_selected_variables), "", dtype=object)
            else:
                X_transform = np.zeros((n_samples, n_selected_variables))

            for i, name in enumerate(selected_variables):
                optb = self._binned_variables[name]

                params = {}
                if self.binning_transform_params is not None:
                    params = self.binning_transform_params.get(name, {})

                metric = params.get("metric", metric)
                metric_missing = params.get("metric_missing", metric_missing)
                metric_special = params.get("metric_special", metric_special)

                tparams = {
                    "x": chunk[name],
                    "metric": metric,
                    "metric_special": metric_special,
                    "metric_missing": metric_missing,
                    "show_digits": show_digits
                    }

                if isinstance(optb, _OPTBPW_TYPES):
                    tparams.pop("show_digits")

                if metric is None:
                    tparams.pop("metric")

                X_transform[:, i] = optb.transform(**tparams)

            df = pd.DataFrame(X_transform, columns=selected_variables)
            df.to_csv(output_path, mode='a', index=False, header=(k == 0))

        return self
