"""
Binning process.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import logging
import numbers
import time

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_array
from sklearn.utils import check_consistent_length
from sklearn.utils.multiclass import type_of_target

from ..logging import Logger
from .binning import OptimalBinning
from .binning_process_information import print_binning_process_information
from .continuous_binning import ContinuousOptimalBinning
from .multiclass_binning import MulticlassOptimalBinning


def _check_parameters(variable_names, max_n_prebins, min_prebin_size,
                      min_n_bins, max_n_bins, min_bin_size, max_bin_size,
                      max_pvalue, max_pvalue_policy, min_iv, max_iv, min_js,
                      max_js, quality_score_cutoff, special_codes,
                      split_digits, categorical_variables, binning_fit_params,
                      binning_transform_params, verbose):

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

    if min_iv is not None:
        if not isinstance(min_iv, numbers.Number) or min_iv < 0:
            raise ValueError("min_iv must be >= 0; got {}.".format(min_iv))

    if max_iv is not None:
        if not isinstance(max_iv, numbers.Number) or max_iv < 0:
            raise ValueError("max_iv must be >= 0; got {}.".format(max_iv))

    if min_iv is not None and max_iv is not None:
        if min_iv > max_iv:
            raise ValueError("min_iv must be <= max_iv; got {} <= {}."
                             .format(min_iv, max_iv))

    if min_js is not None:
        if not isinstance(min_js, numbers.Number) or min_js < 0:
            raise ValueError("min_js must be >= 0; got {}.".format(min_js))

    if max_js is not None:
        if not isinstance(max_js, numbers.Number) or max_js < 0:
            raise ValueError("max_js must be >= 0; got {}.".format(max_js))

    if min_js is not None and max_js is not None:
        if min_js > max_js:
            raise ValueError("min_js must be <= max_js; got {} <= {}."
                             .format(min_iv, max_iv))

    if quality_score_cutoff is not None:
        if (not isinstance(quality_score_cutoff, numbers.Number) or
                not 0 <= quality_score_cutoff <= 1.0):
            raise ValueError("quality_score_cutoff must be in [0, 1.0]; "
                             "got {}.".format(quality_score_cutoff))

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


def _check_variable_dtype(x):
    return "categorical" if x.dtype == np.object else "numerical"


class BinningProcess(BaseEstimator):
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

    min_iv : float or None, optional (default=None)
        The minimum information value. Applicable if target type is binary.

    max_iv : float or None, optional (default=None)
        The maximum information value. Applicable if target type is binary.

    min_js : float or None, optional (default=None)
        The minimum Jensen-Shannon divergence value. Applicable if target type
        is binary or multiclass.

    max_js : float or None, optional (default=None)
        The maximum Jensen-Shannon divergence value. Applicable if target type
        is binary or multiclass.

    quality_score_cutoff : float or None, optional (default=None)
        The quality score cutoff value. Applicable if target type is binary or
        multiclass.

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
    """
    def __init__(self, variable_names, max_n_prebins=20, min_prebin_size=0.05,
                 min_n_bins=None, max_n_bins=None, min_bin_size=None,
                 max_bin_size=None, max_pvalue=None,
                 max_pvalue_policy="consecutive", min_iv=None, max_iv=None,
                 min_js=None, max_js=None, quality_score_cutoff=None,
                 special_codes=None, split_digits=None,
                 categorical_variables=None, binning_fit_params=None,
                 binning_transform_params=None, verbose=False):

        self.variable_names = variable_names

        self.max_n_prebins = max_n_prebins
        self.min_prebin_size = min_prebin_size
        self.min_n_bins = min_n_bins
        self.max_n_bins = max_n_bins
        self.min_bin_size = min_bin_size
        self.max_bin_size = max_bin_size
        self.max_pvalue = max_pvalue
        self.max_pvalue_policy = max_pvalue_policy

        self.min_iv = min_iv
        self.max_iv = max_iv
        self.min_js = min_js
        self.max_js = max_js
        self.quality_score_cutoff = quality_score_cutoff

        self.binning_fit_params = binning_fit_params
        self.binning_transform_params = binning_transform_params

        self.special_codes = special_codes
        self.split_digits = split_digits
        self.categorical_variables = categorical_variables
        self.verbose = verbose

        # auxiliary
        self._n_samples = None
        self._n_variables = None
        self._target_dtype = None
        self._n_numerical = None
        self._n_categorical = None
        self._n_selected = None
        self._binned_variables = {}
        self._variables_dtype = {}
        self._variables_stats = {}

        self._support = None

        # timing
        self._time_total = None

        # logger
        self._logger = Logger()

        self._is_fitted = False

    def fit(self, X, y, check_input=False):
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

        check_input : bool (default=False)
            Whether to check input arrays.

        Returns
        -------
        self : object
            Fitted binning process.
        """
        return self._fit(X, y, check_input)

    def fit_transform(self, X, y, metric=None, metric_special=0,
                      metric_missing=0, check_input=False):
        """Fit the binning process according to the given training data, then
        transform it.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples.

        y : array-like of shape (n_samples,)
            Target vector relative to x.

        metric : str (default="woe")
            The metric used to transform the input vector.

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
        X_new : numpy array, shape = (n_samples, n_features_new)
            Transformed array.
        """
        return self.fit(X, y, check_input).transform(X, None, metric,
                                                     metric_special,
                                                     metric_missing,
                                                     check_input)

    def transform(self, X, variable_names=None, metric=None,
                  metric_special=0, metric_missing=0, check_input=False):
        """Transform given data to metric using bins from each fitted optimal
        binning.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples.

        variable_names : array-like or None, optional (default=None)
            List of selected variables to apply transformation. If None all
            ``variable_names`` are transformed.

        metric : str (default="woe")
            The metric used to transform the input vector.

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
        X_new : numpy array, shape = (n_samples, n_features_new)
            Transformed array.
        """
        self._check_is_fitted()

        return self._transform(X, variable_names, metric, metric_special,
                               metric_missing, check_input)

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

        n_numerical = list(self._variables_dtype.values()).count("numerical")
        n_categorical = self._n_variables - n_numerical

        self._n_selected = self._n_variables

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

        df_summary = pd.DataFrame.from_dict(self._variables_stats).T
        df_summary.reset_index(inplace=True)
        df_summary.rename(columns={"index": "name"}, inplace=True)

        columns = ["name", "dtype", "status", "selected", "n_bins"]

        if self._target_dtype == "binary":
            columns += ["iv", "gini", "js", "quality score"]
        elif self._target_dtype == "multiclass":
            columns += ["js", "quality score"]
        elif self._target_dtype == "continuous":
            pass

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
            return self.variable_names[mask]
        else:
            return mask

    def _binning_variables_selection(self):
        self._support = np.zeros(self._n_variables).astype(np.bool)

        for i, name in enumerate(self.variable_names):
            optb = self._binned_variables[name]
            optb.binning_table.build()

            dtype = optb.dtype
            splits = optb.splits
            status = optb.status

            n_bins = len(splits) + 1 if dtype == "numerical" else len(splits)

            info = {
                "dtype": dtype, "status": status,
                "n_bins": n_bins
            }

            if self._target_dtype == "binary":
                optb.binning_table.analysis(print_output=False)
                iv = optb.binning_table.iv
                gini = optb.binning_table.gini
                js = optb.binning_table.js
                quality_score = optb.binning_table.quality_score

                selected = True
                selected &= self._binning_metric_eval(self.min_iv, iv, 0)
                selected &= self._binning_metric_eval(self.max_iv, iv, 1)
                selected &= self._binning_metric_eval(self.min_js, js, 0)
                selected &= self._binning_metric_eval(self.max_js, js, 1)
                selected &= self._binning_metric_eval(
                    self.quality_score_cutoff, quality_score, 0)

                info = {**info, **{"iv": iv, "gini": gini, "js": js,
                                   "quality score": quality_score,
                                   "selected": selected}}
            elif self._target_dtype == "continuous":
                selected = True
                info = {**info, **{"selected": selected}}
            elif self._target_dtype == "multiclass":
                optb.binning_table.analysis(print_output=False)
                js = optb.binning_table.js
                quality_score = optb.binning_table.quality_score

                selected = True
                selected &= self._binning_metric_eval(self.max_js, js, 1)
                selected &= self._binning_metric_eval(
                    self.quality_score_cutoff, quality_score, 0)

                info = {**info, **{"js": js, "quality score": quality_score,
                                   "selected": selected}}

            self._variables_stats[name] = info
            self._support[i] = selected

    def _binning_metric_eval(self, metric, binning_metric, min_max):
        if metric is None:
            return True

        if min_max == 0:
            if binning_metric < metric:
                return False
        elif binning_metric > metric:
            return False

        return True

    def _check_is_fitted(self):
        if not self._is_fitted:
            raise NotFittedError("This {} instance is not fitted yet. Call "
                                 "'fit' with appropriate arguments."
                                 .format(self.__class__.__name__))

    def _fit(self, X, y, check_input):
        time_init = time.perf_counter()

        if self.verbose:
            logging.info("Binning process started.")
            logging.info("Options: check parameters.")

        _check_parameters(**self.get_params())

        # check X dtype
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError("X must be a pandas.DataFrame or numpy.ndarray.")

        # check target dtype
        self._target_dtype = type_of_target(y)

        if self._target_dtype not in ("binary", "continuous", "multiclass"):
            raise ValueError("Target type {} is not supported."
                             .format(self._target_dtype))

        # check X and y data
        if check_input:
            X = check_array(X, ensure_2d=False, dtype=None,
                            force_all_finite='allow-nan')

            y = check_array(y, ensure_2d=False, dtype=None,
                            force_all_finite=True)

            check_consistent_length(X, y)

        self._n_samples, self._n_variables = X.shape

        if self.verbose:
            logging.info("Dataset: number of samples: {}."
                         .format(self._n_samples))

            logging.info("Dataset: number of variables: {}."
                         .format(self._n_variables))

        for i, name in enumerate(self.variable_names):
            if isinstance(X, np.ndarray):
                self._fit_variable(X[:, i], y, name)
            else:
                self._fit_variable(X[name], y, name)

        self._time_total = time.perf_counter() - time_init

        if self.verbose:
            logging.info("Binning process variable selection...")

        # Compute binning statistics and decide whether a variable is selected
        self._binning_variables_selection()

        if self.verbose:
            logging.info("Binning process terminated. Time: {:.4f}s"
                         .format(self._time_total))

        # Completed successfully
        self._logger.close()
        self._is_fitted = True

        return self

    def _fit_variable(self, x, y, name):
        params = {}
        dtype = _check_variable_dtype(x)

        if self.verbose:
            logging.info("Binning variable: {}".format(name))

        if self.categorical_variables is not None:
            if name in self.categorical_variables:
                dtype = "categorical"

        self._variables_dtype[name] = dtype

        if self.binning_fit_params is not None:
            params = self.binning_fit_params.get(name, {})

        if self._target_dtype == "binary":
            optb = OptimalBinning(
                name=name, dtype=dtype, max_n_prebins=self.max_n_prebins,
                min_prebin_size=self.min_prebin_size,
                min_n_bins=self.min_n_bins, max_n_bins=self.max_n_bins,
                min_bin_size=self.min_bin_size, max_pvalue=self.max_pvalue,
                max_pvalue_policy=self.max_pvalue_policy,
                special_codes=self.special_codes,
                split_digits=self.split_digits)
        elif self._target_dtype == "continuous":
            optb = ContinuousOptimalBinning(
                name=name, dtype=dtype, max_n_prebins=self.max_n_prebins,
                min_prebin_size=self.min_prebin_size,
                min_n_bins=self.min_n_bins, max_n_bins=self.max_n_bins,
                min_bin_size=self.min_bin_size, max_pvalue=self.max_pvalue,
                max_pvalue_policy=self.max_pvalue_policy,
                special_codes=self.special_codes,
                split_digits=self.split_digits)
        else:
            if dtype == "categorical":
                raise ValueError("MulticlassOptimalBinning does not support "
                                 "categorical variables.")
            optb = MulticlassOptimalBinning(
                name=name, max_n_prebins=self.max_n_prebins,
                min_prebin_size=self.min_prebin_size,
                min_n_bins=self.min_n_bins, max_n_bins=self.max_n_bins,
                min_bin_size=self.min_bin_size, max_pvalue=self.max_pvalue,
                max_pvalue_policy=self.max_pvalue_policy,
                special_codes=self.special_codes,
                split_digits=self.split_digits)

        optb.set_params(**params)
        optb.fit(x, y)

        self._binned_variables[name] = optb

    def _transform(self, X, variable_names, metric, metric_special,
                   metric_missing, check_input):

        # check X dtype
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError("X must be a pandas.DataFrame or numpy.ndarray.")

        if isinstance(X, pd.DataFrame) and variable_names is None:
            raise ValueError("variable_names must be provided if X is of type "
                             "pandas.DataFrame.")

        n_records, n_variables = X.shape

        if variable_names is not None:
            if not isinstance(variable_names, (np.ndarray, list)):
                raise TypeError("variable_names must be a list or "
                                "numpy.ndarray.")

            keys = list(self._binned_variables.keys())
            n_variables = len(variable_names)

        X_transform = np.zeros((n_records, n_variables))

        for i in range(n_variables):
            params = {}

            if variable_names is not None:
                name = variable_names[i]
                if name not in keys:
                    raise ValueError("Variable {} was not previously binned."
                                     .format(name))

                if self.binning_transform_params is not None:
                    params = self.binning_transform_params.get(name, {})

                optb = self._binned_variables[name]

                idx = next(j for j, key in enumerate(keys) if key == name)
            else:
                optb = list(self._binned_variables.values())[i]
                idx = i

            metric_missing = params.get("metric_missing", metric_missing)
            metric_special = params.get("metric_special", metric_special)

            if isinstance(X, np.ndarray):
                x = X[:, idx]
            else:
                x = X[name]

            if metric is not None:
                metric = params.get("metric", metric)

                X_transform[:, i] = optb.transform(x, metric, metric_special,
                                                   metric_missing, check_input)
            else:
                X_transform[:, i] = optb.transform(
                    x, metric_special=metric_special,
                    metric_missing=metric_missing, check_input=check_input)

        return X_transform
