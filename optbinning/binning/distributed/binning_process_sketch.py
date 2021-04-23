"""
Binning process sketch.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2021

import time

from sklearn.base import BaseEstimator

from ...binning.base import Base
from ...binning.binning_process import _check_selection_criteria
from ...binning.binning_process import _METRICS
from ...logging import Logger
from .binning_sketch import OptimalBinningSketch


def _fit_variable(x, y):
    pass


def _fit_block():
    pass


def _check_parameters():
    pass


class BinningProcessSketch(Base, BaseEstimator):
    def __init__(self, variable_names, max_n_prebins=20, min_n_bins=None,
                 max_n_bins=None, min_bin_size=None, max_bin_size=None,
                 max_pvalue=None, max_pvalue_policy="consecutive",
                 selection_criteria=None, categorical_variables=None,
                 special_codes=None, split_digits=None,
                 binning_fit_params=None, binning_transform_params=None,
                 verbose=True):

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

        # logger
        self._class_logger = Logger(__name__)
        self._logger = self._class_logger.logger

        # flags
        self._is_started = False
        self._is_fitted = False

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
        self : object
            Binning process with new data.
        """
        if self._started:
            self._n_samples = 0
            self._n_variables = len(self.variable_names)
            self._n_categorical = len(self.categorical_variables)
            self._n_numerical = self._n_variables - self._n_categorical

            # Initialize bsketch for each variable. To avoid mixed dtypes
            # the user must provide a dtype for all variables. This differs
            # from the BinningProcess, where dtypes are inferred.
            for name in self.variable_names:
                if name in self.categorical_variables:
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

            self._started = True

        # Add new data stream
        time_add = time.perf_counter()

        # Add data to variables that appear in X. During training the
        # data columns might change, for example, not all data sources
        # contain the same variables.
        for name in X.columns:
            if name in self.variable_names:
                if self.verbose:
                    self._logger.info("Add variable: {}.".format(name))

                self._binned_variables[name].add(X[name], y, check_input)

        # Update count samples and addition operations
        self._n_samples += X.shape[0]
        self._n_add += 1

        self._time_streaming_add += time.perf_counter() - time_add

        if self.verbose:
            self._logger.info("Sketch: added new data.")

    def information(self, print_level=1):
        pass

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
            self._logger.info("Sketch: current sketch was merged.")

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
        self : object
            Current fitted binning process.
        """
        time_init = time.perf_counter()

        for i, name in enumerate(self.variable_names):
            if self.verbose:
                self._logger.info("Binning variable ({} / {}): {}."
                                  .format(i, self._n_variables, name))
            self._binned_variables[name].solve()

        self._time_streaming_solve += time.perf_counter() - time_init
        self._n_solve += 1

        # Completed successfully
        self._is_fitted = True

        return self

    def transform(self, X, metric="woe", metric_special=0, metric_missing=0,
                  show_digits=2, check_input=False):
        pass
