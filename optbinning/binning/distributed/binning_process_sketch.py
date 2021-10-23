"""
Binning process sketch.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2021

import numbers
import time

from warnings import warn

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator

from ...binning.binning_process import _check_selection_criteria
from ...binning.binning_process import _METRICS
from ...binning.binning_process import BaseBinningProcess
from ...exceptions import NotDataAddedError
from ...logging import Logger
from .base import BaseSketch
from .binning_process_sketch_information import (
    print_binning_process_sketch_information)
from .binning_sketch import OptimalBinningSketch


logger = Logger(__name__).logger


def _check_parameters(variable_names, max_n_prebins, min_n_bins, max_n_bins,
                      min_bin_size, max_bin_size, max_pvalue,
                      max_pvalue_policy, selection_criteria,
                      categorical_variables, special_codes, split_digits,
                      binning_fit_params, binning_transform_params, verbose):

    if not isinstance(variable_names, (np.ndarray, list)):
        raise TypeError("variable_names must be a list or numpy.ndarray.")

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

    if categorical_variables is not None:
        if not isinstance(categorical_variables, (np.ndarray, list)):
            raise TypeError("categorical_variables must be a list or "
                            "numpy.ndarray.")

        if not all(isinstance(c, str) for c in categorical_variables):
            raise TypeError("variables in categorical_variables must be "
                            "strings.")

    if special_codes is not None:
        if not isinstance(special_codes, (np.ndarray, list)):
            raise TypeError("special_codes must be a list or numpy.ndarray.")

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

    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a boolean; got {}.".format(verbose))


class BinningProcessSketch(BaseSketch, BaseEstimator, BaseBinningProcess):
    """Binning process over data streams to compute optimal binning of
    variables with respect to a binary target.

    Parameters
    ----------
    variable_names : array-like
        List of variable names.

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

    max_pvalue : float or None, optional (default=0.05)
        The maximum p-value among bins.

    max_pvalue_policy : str, optional (default="consecutive")
        The method to determine bins not satisfying the p-value constraint.
        Supported methods are "consecutive" to compare consecutive bins and
        "all" to compare all bins.

    selection_criteria : dict or None (default=None)
        Variable selection criteria. See notes.

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
        option ``"solver": "mip"`` via the binning_fit_params parameter.

    """
    def __init__(self, variable_names, max_n_prebins=20, min_n_bins=None,
                 max_n_bins=None, min_bin_size=None, max_bin_size=None,
                 max_pvalue=None, max_pvalue_policy="consecutive",
                 selection_criteria=None, categorical_variables=None,
                 special_codes=None, split_digits=None,
                 binning_fit_params=None, binning_transform_params=None,
                 verbose=False):

        self.variable_names = variable_names

        self.max_n_prebins = max_n_prebins
        self.min_n_bins = min_n_bins
        self.max_n_bins = max_n_bins
        self.min_bin_size = min_bin_size
        self.max_bin_size = max_bin_size
        self.max_pvalue = max_pvalue
        self.max_pvalue_policy = max_pvalue_policy

        self.selection_criteria = selection_criteria

        self.binning_fit_params = binning_fit_params
        self.binning_transform_params = binning_transform_params

        self.special_codes = special_codes
        self.split_digits = split_digits
        self.categorical_variables = categorical_variables
        self.verbose = verbose

        # target information to reuse BaseBinningProcess
        self._target_dtype = "binary"

        # auxiliary
        self._n_samples = None
        self._n_variables = None
        self._n_numerical = None
        self._n_categorical = None
        self._n_selected = None
        self._binned_variables = {}
        self._variable_dtypes = {}
        self._variable_stats = {}

        self._support = None

        # streaming stats
        self._n_add = 0
        self._n_solve = 0

        # timing
        self._time_streaming_add = 0
        self._time_streaming_solve = 0

        # flags
        self._is_started = False
        self._is_solved = False

        # Check parameters
        _check_parameters(**self.get_params())

    def add(self, X, y, check_input=False):
        """Add new data X, y to the binning sketch of each variable.

        Parameters
        ----------
        X : pandas.DataFrame, shape (n_samples, n_features)

        y : array-like of shape (n_samples,)
            Target vector relative to x.

        check_input : bool (default=False)
            Whether to check input arrays.

        Returns
        -------
        self : BinningProcessSketch
            Binning process with new data.
        """
        if not self._is_started:
            self._n_samples = 0
            self._n_variables = len(self.variable_names)

            if self.categorical_variables is not None:
                self._n_categorical = len(self.categorical_variables)
            else:
                self._n_categorical = 0

            self._n_numerical = self._n_variables - self._n_categorical

            # Check selection criteria
            if self.selection_criteria is not None:
                _check_selection_criteria(self.selection_criteria,
                                          self._target_dtype)

            # Initialize bsketch for each variable. To avoid mixed dtypes
            # the user must provide a dtype for all variables. This differs
            # from the BinningProcess, where dtypes are inferred.
            for name in self.variable_names:
                if (self.categorical_variables is not None and
                        name in self.categorical_variables):
                    dtype = "categorical"
                else:
                    dtype = "numerical"

                optb = OptimalBinningSketch(
                        name=name, dtype=dtype,
                        max_n_prebins=self.max_n_prebins,
                        min_n_bins=self.min_n_bins,
                        max_n_bins=self.max_n_bins,
                        min_bin_size=self.min_bin_size,
                        max_pvalue=self.max_pvalue,
                        max_pvalue_policy=self.max_pvalue_policy,
                        special_codes=self.special_codes,
                        split_digits=self.split_digits)

                if self.binning_fit_params is not None:
                    params = self.binning_fit_params.get(name, {})
                else:
                    params = {}

                optb.set_params(**params)

                self._variable_dtypes[name] = dtype
                self._binned_variables[name] = optb

            self._is_started = True

        # Add new data stream
        time_add = time.perf_counter()

        # Add data to variables that appear in X. During training the
        # data columns might change, for example, not all data sources
        # contain the same variables.
        for name in X.columns:
            if name in self.variable_names:
                if self.verbose:
                    logger.info("Add variable: {}.".format(name))

                self._binned_variables[name].add(X[name], y, check_input)

        # Update count samples and addition operations
        self._n_samples += X.shape[0]
        self._n_add += 1

        self._time_streaming_add += time.perf_counter() - time_add

        if self.verbose:
            logger.info("Sketch: added new data.")

        return self

    def information(self, print_level=1):
        """Print overview information about the options settings and
        statistics.

        Parameters
        ----------
        print_level : int (default=1)
            Level of details.
        """
        self._check_is_solved()

        if not isinstance(print_level, numbers.Integral) or print_level < 0:
            raise ValueError("print_level must be an integer >= 0; got {}."
                             .format(print_level))

        self._n_selected = np.count_nonzero(self._support)

        dict_user_options = self.get_params()

        print_binning_process_sketch_information(
            print_level, self._n_samples, self._n_variables,
            self._target_dtype, self._n_numerical, self._n_categorical,
            self._n_selected, self._n_add, self._time_streaming_add,
            self._n_solve, self._time_streaming_solve, dict_user_options)

    def summary(self):
        """Binning process summary with main statistics for all binned
        variables.

        Parameters
        ----------
        df_summary : pandas.DataFrame
            Binning process summary.
        """
        self._check_is_solved()

        df_summary = pd.DataFrame.from_dict(self._variable_stats).T
        df_summary.reset_index(inplace=True)
        df_summary.rename(columns={"index": "name"}, inplace=True)
        df_summary["selected"] = self._support

        columns = ["name", "dtype", "status", "selected", "n_bins"]
        columns += _METRICS[self._target_dtype]["metrics"]

        return df_summary[columns]

    def merge(self, bpsketch):
        """Merge current instance with another BinningProcessSketch instance.

        Parameters
        ----------
        bpsketch : object
            BinningProcessSketch instance.
        """
        if not self.mergeable(bpsketch):
            raise Exception("bpsketch does not share signature.")

        for name in self.variable_names:
            self._binned_variables[name].merge(
                bpsketch._binned_variables[name])

        if self.verbose:
            logger.info("Sketch: current sketch was merged.")

    def mergeable(self, bpsketch):
        """Check whether two BinningProcessSketch instances can be merged.

        Parameters
        ----------
        bpsketch : object
            BinningProcessSketch instance.

        Returns
        -------
        mergeable : bool
        """
        return self.get_params() == bpsketch.get_params()

    def solve(self):
        """Solve optimal binning for all variables using added data.

        Returns
        -------
        self : BinningProcessSketch
            Current fitted binning process.
        """
        time_init = time.perf_counter()

        # Check if data was added
        if not self._n_add:
            raise NotDataAddedError(
                "No data was added. Add data before solving.")

        for i, name in enumerate(self.variable_names):
            if self.verbose:
                logger.info("Binning variable ({} / {}): {}."
                            .format(i, self._n_variables, name))
            self._binned_variables[name].solve()

        if self.verbose:
            logger.info("Binning process variable selection...")

        # Compute binning statistics and decide whether a variable is selected
        self._binning_selection_criteria()

        self._time_streaming_solve += time.perf_counter() - time_init
        self._n_solve += 1

        # Completed successfully
        self._is_solved = True

        return self

    def transform(self, X, metric="woe", metric_special=0, metric_missing=0,
                  show_digits=2, check_input=False):
        """Transform given data to metric using bins from each fitted optimal
        binning.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
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
        X_new : pandas.DataFrame, shape = (n_samples, n_features_new)
            Transformed array.
        """
        self._check_is_solved()

        # Check X dtype
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas.DataFrame.")

        n_samples, n_variables = X.shape

        # Check metric
        if metric not in ("event_rate", "woe"):
            raise ValueError('Invalid value for metric. Allowed string '
                             'values are "event_rate" and "woe".')

        mask = self.get_support()
        if not mask.any():
            warn("No variables were selected: either the data is"
                 " too noisy or the selection_criteria too strict.",
                 UserWarning)
            return np.empty(0).reshape((n_samples, 0))

        selected_variables = self.get_support(names=True)
        for name in selected_variables:
            if name not in X.columns:
                raise ValueError("Selected variable {} must be a column "
                                 "in the input dataframe.".format(name))

        n_selected_variables = len(selected_variables)

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
            x = X[name]

            params = {}
            if self.binning_transform_params is not None:
                params = self.binning_transform_params.get(name, {})

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

            if metric is not None:
                tparams["metric"] = params.get("metric", metric)
            else:
                tparams.pop("metric")

            X_transform[:, i] = optb.transform(**tparams)

        return pd.DataFrame(X_transform, columns=selected_variables)

    def get_binned_variable(self, name):
        """Return optimal binning sketch object for a given variable name.

        Parameters
        ----------
        name : string
            The variable name.
        """
        self._check_is_solved()

        if not isinstance(name, str):
            raise TypeError("name must be a string.")

        if name in self.variable_names:
            return self._binned_variables[name]
        else:
            raise ValueError("name {} does not match a binned variable."
                             .format(name))

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
        self._check_is_solved()

        if indices and names:
            raise ValueError("Only indices or names can be True.")

        mask = self._support
        if indices:
            return np.where(mask)[0]
        elif names:
            return np.asarray(self.variable_names)[mask]
        else:
            return mask
