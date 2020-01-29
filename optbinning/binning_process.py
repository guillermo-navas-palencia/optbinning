"""
Binning process.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import logging
import numbers
import time

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_array
from sklearn.utils import check_consistent_length
from sklearn.utils.multiclass import type_of_target

from .binning import OptimalBinning
from .continuous_binning import ContinuousOptimalBinning
from .multiclass_binning import MulticlassOptimalBinning

from .logging import Logger


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
        self._n_records = None
        self._n_variables = None
        self._target_dtype = None
        self._binned_variables = {}

        # timing
        self._time_total = None

        # logger
        self._logger = Logger()

        self._is_fitted = False

    def fit(self, X, y, check_input=False):
        return self._fit(X, y, check_input)

    def fit_transform(self, X, y, metric=None, metric_special=0,
                      metric_missing=0, check_input=False):

        return self.fit(X, y, check_input).transform(X, None, metric,
                                                     metric_special,
                                                     metric_missing,
                                                     check_input)

    def transform(self, X, variable_names=None, metric=None,
                  metric_special=0, metric_missing=0, check_input=False):

        self._check_is_fitted()

        return self._transform(X, variable_names, metric, metric_special,
                               metric_missing, check_input)

    def information(self, print_level=1):
        pass

    def summary(self):
        pass

    def get_binned_variable(self, name):
        self._check_is_fitted()

        if not isinstance(name, str):
            raise TypeError("")

        if name in self.variable_names:
            return self._binned_variables[name]
        else:
            raise ValueError("")

    def get_support(self):
        pass

    def _check_is_fitted(self):
        if not self._is_fitted:
            raise NotFittedError("This {} instance is not fitted yet. Call "
                                 "'fit' with appropriate arguments."
                                 .format(self.__class__.__name__))

    def _fit(self, X, y, check_input):
        time_init = time.perf_counter()

        _check_parameters(**self.get_params())

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

        self._n_records, self._n_variables = X.shape

        for i, name in enumerate(self.variable_names):
            self._fit_variable(X[:, i], y, name)

        self._time_total = time.perf_counter() - time_init

        # Completed successfully
        self._logger.close()
        self._is_fitted = True

        return self

    def _fit_variable(self, x, y, name):
        params = {}
        dtype = _check_variable_dtype(x)

        if self.categorical_variables is not None:
            if name in self.categorical_variables:
                dtype = "categorical"

        if self.binning_fit_params is not None:
            params = self.binning_fit_params.get(name, {})

        if self._target_dtype == "binary":
            optb = OptimalBinning(name=name, dtype=dtype,
                                  max_n_prebins=self.max_n_prebins,
                                  min_prebin_size=self.min_prebin_size,
                                  min_n_bins=self.min_n_bins,
                                  max_n_bins=self.max_n_bins,
                                  min_bin_size=self.min_bin_size,
                                  max_pvalue=self.max_pvalue,
                                  max_pvalue_policy=self.max_pvalue_policy,
                                  special_codes=self.special_codes,
                                  split_digits=self.split_digits,
                                  verbose=self.verbose)
        elif self._target_dtype == "continuous":
            optb = ContinuousOptimalBinning(
                name=name, dtype=dtype, max_n_prebins=self.max_n_prebins,
                min_prebin_size=self.min_prebin_size,
                min_n_bins=self.min_n_bins, max_n_bins=self.max_n_bins,
                min_bin_size=self.min_bin_size, max_pvalue=self.max_pvalue,
                max_pvalue_policy=self.max_pvalue_policy,
                special_codes=self.special_codes,
                split_digits=self.split_digits, verbose=self.verbose)
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
                split_digits=self.split_digits, verbose=self.verbose)

        optb.set_params(**params)
        optb.fit(x, y)

        self._binned_variables[name] = optb

    def _transform(self, X, variable_names, metric, metric_special,
                   metric_missing, check_input):

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

            metric_missing = params.get("metric_missing", metric_special)
            metric_special = params.get("metric_special", metric_missing)

            if metric is not None:
                metric = params.get("metric", metric)

                X_transform[:, i] = optb.transform(X[:, idx], metric,
                                                   metric_special,
                                                   metric_missing, check_input)
            else:
                X_transform[:, i] = optb.transform(
                    X[:, idx], metric_special=metric_special,
                    metric_missing=metric_missing, check_input=check_input)

        return X_transform
