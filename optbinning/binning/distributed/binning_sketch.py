"""
Optimal binning t-digest data structure.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from ...binning.auto_monotonic import auto_monotonic
from ...binning.auto_monotonic import peak_valley_trend_change_heuristic
from ...binning.binning_information import print_binning_information
from ...binning.binning_statistics import bin_categorical
from ...binning.binning_statistics import bin_info
from ...binning.binning_statistics import BinningTable
from ...binning.binning_statistics import target_info
from ...binning.cp import BinningCP
from ...binning.mip import BinningMIP

from .bsketch import BSketch


class OptimalBinningSketch(BaseEstimator):
    def __init__(self, name="", sketch="gk", eps=1e-4, K=25, solver="cp",
                 max_n_prebins=20, min_n_bins=None, max_n_bins=None,
                 min_bin_size=None, max_bin_size=None, min_bin_n_nonevent=None,
                 max_bin_n_nonevent=None, min_bin_n_event=None,
                 max_bin_n_event=None, monotonic_trend="auto",
                 min_event_rate_diff=0, max_pvalue=None,
                 max_pvalue_policy="consecutive", gamma=0, special_codes=None,
                 mip_solver="bop", time_limit=100, verbose=False):

        self.name = name
        self.sketch = sketch
        self.eps = eps
        self.K = K

        self.solver = solver

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

        self.special_codes = special_codes
        
        self.mip_solver = mip_solver
        self.time_limit = time_limit

        self.verbose = verbose

        # Data storage
        self.bsketch = BSketch(sketch, eps, K, special_codes)

    def add(self, x, y, check_input=False):
        # Add new data stream
        self.bsketch.add(x, y, check_input)

    def merge(self, optbdigest):
        if not self.mergeable(optbdigest):
            raise Exception()

        self.bsketch.merge(optbdigest.bsketch)

    def mergeable(self, optbdigest):
        return self.get_params() == optbdigest.get_params()

    def solve(self):
        splits, n_nonevent, n_event = self._prebinning_data()

        self._n_prebins = len(splits) + 1

        # Optimization
        self._fit_optimizer(splits, n_nonevent, n_event)

        # Post-processing
        if not len(splits):
            n_nonevent = self._t_n_nonevent
            n_event = self._t_n_nonevent

        self._n_nonevent, self._n_event = bin_info(
            self._solution, n_nonevent, n_event, self._n_nonevent_missing,
            self._n_event_missing, self._n_nonevent_special,
            self._n_event_special, None, None, [])

        self._binning_table = BinningTable(
            self.name, "numerical", self._splits_optimal, self._n_nonevent,
            self._n_event, None, None, []) 

        self._is_fitted = True

        return self

    def _prebinning_data(self):
        self._n_nonevent_missing = self.bsketch._count_missing_ne
        self._n_nonevent_special = self.bsketch._count_special_ne
        self._n_event_missing = self.bsketch._count_missing_e
        self._n_event_special = self.bsketch._count_special_e

        self._t_n_nonevent = self.bsketch.n_nonevent
        self._t_n_event = self.bsketch.n_event

        percentiles = np.linspace(0, 1, self.max_n_prebins + 1)
        sketch_all = self.bsketch.merge_sketches()

        if self.sketch == "gk":
            splits = np.array([sketch_all.quantile(p)
                               for p in percentiles[1:-1]])
        elif self.sketch == "t-digest":
            splits = np.array([sketch_all.percentile(p * 100)
                               for p in percentiles[1:-1]])

        self._splits_prebinning = splits

        splits, n_nonevent, n_event = self._compute_prebins(splits)

        return splits, n_nonevent, n_event
    
    def _compute_prebins(self, splits):
        n_event, n_nonevent = self.bsketch.bins(splits)
        mask_remove = (n_nonevent == 0) | (n_event == 0)

        if np.any(mask_remove):
            mask_splits = np.concatenate(
                [mask_remove[:-2], [mask_remove[-2] | mask_remove[-1]]])

            splits = splits[~mask_splits]
            splits, n_nonevent, n_event = self._compute_prebins(splits)

        return splits, n_nonevent, n_event

    def _fit_optimizer(self, splits, n_nonevent, n_event):
        if not len(n_nonevent):
            self._status = "OPTIMAL"
            self._splits_optimal = splits
            self._solution = np.zeros(len(splits)).astype(np.bool)
            return

        if self.min_bin_size is not None:
            min_bin_size = np.int(np.ceil(self.min_bin_size * self.bsketch.n))
        else:
            min_bin_size = self.min_bin_size

        if self.max_bin_size is not None:
            max_bin_size = np.int(np.ceil(self.max_bin_size * self.bsketch.n))
        else:
            max_bin_size = self.max_bin_size

        # Monotonic trend
        trend_change = None

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
        else:
            monotonic = self.monotonic_trend

            if monotonic in ("peak_heuristic", "valley_heuristic"):
                event_rate = n_event / (n_nonevent + n_event)
                trend_change = peak_valley_trend_change_heuristic(
                    event_rate, monotonic)

        if self.solver == "cp":
            optimizer = BinningCP(monotonic, self.min_n_bins, self.max_n_bins,
                                  min_bin_size, max_bin_size,
                                  self.min_bin_n_event, self.max_bin_n_event,
                                  self.min_bin_n_nonevent,
                                  self.max_bin_n_nonevent,
                                  self.min_event_rate_diff, self.max_pvalue,
                                  self.max_pvalue_policy, self.gamma,
                                  None, self.time_limit)
        elif self.solver == "mip":
            optimizer = BinningMIP(monotonic, self.min_n_bins, self.max_n_bins,
                                   min_bin_size, max_bin_size,
                                   self.min_bin_n_event, self.max_bin_n_event,
                                   self.min_bin_n_nonevent,
                                   self.max_bin_n_nonevent,
                                   self.min_event_rate_diff, self.max_pvalue,
                                   self.max_pvalue_policy, self.gamma,
                                   None, self.mip_solver,
                                   self.time_limit)

        optimizer.build_model(n_nonevent, n_event, trend_change)
        status, solution = optimizer.solve()

        self._solution = solution
        self._optimizer = optimizer
        self._status = status
        self._splits_optimal = splits[solution[:-1]]

    @property
    def binning_table(self):
        """Return an instantiated binning table. Please refer to
        :ref:`Binning table: binary target`.

        Returns
        -------
        binning_table : BinningTable.
        """
        return self._binning_table

    @property
    def splits(self):
        """List of optimal split points when ``dtype`` is set to "numerical" or
        list of optimal bins when ``dtype`` is set to "categorical".

        Returns
        -------
        splits : numpy.ndarray
        """
        return self._splits_optimal

    @property
    def status(self):
        """The status of the underlying optimization solver.

        Returns
        -------
        status : str
        """
        return self._status