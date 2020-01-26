"""
Binning process.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import logging
import numbers
import time

from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from sklearn.utils import check_consistent_length
from sklearn.utils.multiclass import type_of_target

from .binning import OptimalBinning
from .continuous_binning import ContinuousOptimalBinning
from .multiclass_binning import MulticlassOptimalBinning

from .logging import Logger


def _check_parameters(variable_names, max_n_prebins, min_prebin_size,
                      min_n_bins, max_n_bins, max_pvalue, max_pvalue_policy,
                      min_iv, max_iv, min_js, max_js, min_qs, max_qs,
                      special_codes, split_digits, categorical_variables,
                      binning_options, verbose):
    pass


def _check_dtype(x):
    pass


def _check_binning_target_dtype(binning, target_dtype):
    pass


class BinningProcess(BaseEstimator):
    def __init__(self, variable_names=None, max_n_prebins=20,
                 min_prebin_size=0.05, min_n_bins=None, max_n_bins=None,
                 max_pvalue=None, max_pvalue_policy="consecutive", min_iv=None,
                 max_iv=None, min_js=None, max_js=None, min_qs=None,
                 max_qs=None, special_codes=None, split_digits=None,
                 categorical_variables=None, binning_options=None,
                 verbose=False):

        self.max_n_prebins = max_n_prebins
        self.min_prebin_size = min_prebin_size
        self.min_n_bins = min_n_bins
        self.max_n_bins = max_n_bins
        self.max_pvalue = max_pvalue

        self.min_iv = min_iv
        self.max_iv = max_iv
        self.min_js = min_js
        self.max_js = max_js
        self.min_qs = min_qs
        self.max_qs = max_qs

        self.max_pvalue_policy = max_pvalue_policy
        self.split_digits = split_digits
        self.verbose = verbose

        # auxiliary
        self._n_records = None
        self._n_variables = None
        self._target_dtype = None
        self._binning = {}

        # timing
        self._time_total = None

        # logger
        self._logger = Logger()

        self._is_fitted = False

    def fit(self, X, y, check_input=False):
        pass

    def fit_transform(self, X, y, metric, metric_special=0, metric_missing=0,
                      check_input=False):
        pass

    def transform(self, X, metric, metric_special=0, metric_missing=0,
                  check_input=False):
        pass

    def _fit(self, X, y, check_input):
        time_init = time.perf_counter()

        _check_parameters(**self.get_params())

        # check target dtype
        self._target_dtype = type_of_target(y)

        if self._target_dtype not in ("binary", "continuous", "multiclass"):
            raise ValueError("")

        # check X and y data
        self._n_records, self._n_variables = X.shape

        if self._target_dtype in ("binary", "continuous"):
            # check if 0/1

            for i in range(self._n_variables):
                x = X[:, i]

                options = {}
                dtype = _check_dtype(x)

                if self.variable_names is not None:
                    name = self.variable_names[i]

                    if self.categorical_variables is not None:
                        if name in self.categorical_variables:
                            dtype = "categorical"

                    if self.binning_options is not None:
                        if name in self.binning_options.keys():
                            options = self.binning_options[name]
                else:
                    name = ""
                    if self.categorical_variables is not None:
                        if i in self.categorical_variables:
                            dtype = "categorical"

                if self._target_dtype == "binary":
                    optb = OptimalBinning(name=name, dtype=dtype)
                else:
                    optb = ContinuousOptimalBinning(name=name, dtype=dtype)
                optb.set_params(**options)
                optb.fit(x, y)

                self._binning[i] = optb
        else:
            for i in range(self._n_variables):
                x = X[:, i]

                options = {}

                dtype = _check_dtype(x)
                if dtype == "categorical":
                    raise ValueError()

                if self.variable_names is not None:
                    name = self.variable_names[i]

                    if self.binning_options is not None:
                        if name in self.binning_options.keys():
                            options = self.binning_options[name]
                else:
                    name = ""

                optb = MulticlassOptimalBinning(name=name)
                optb.set_params(**options)
                optb.fit(x, y)

                self._binning[i] = optb

        self._time_total = time.perf_counter() - time_init

        # Completed successfully
        self._logger.close()
        self._is_fitted = True

        return self

    def get_binning_variable(self, id):
        if isinstance(id, str):
            if self.variable_names is not None:
                if id in self.variable_names:
                    return self._binning[id]
                else:
                    raise ValueError("")
            else:
                raise ValueError("")
        elif isinstance(id, numbers.Integral) and 0 <= id < self._n_variables:
            return self._binning.values()[id]
        else:
            raise ValueError("")

    def set_binning_variable(self, id, binning):
        _check_binning_target_dtype(binning, self._target_dtype)

        if isinstance(id, str):
            if self.variable_names is not None:
                if id in self.variable_names:
                    self._binning[id] = binning
                else:
                    raise ValueError("")
            else:
                raise ValueError("")
        elif isinstance(id, numbers.Integral) and 0 <= id < self._n_variables:
            return self._binning.values()[id]
        else:
            raise ValueError("")
