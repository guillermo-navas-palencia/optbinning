"""
Optimal binning 2D algorithm.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import numbers
import time

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.style
import matplotlib as mpl

from sklearn.tree import DecisionTreeClassifier

from ...logging import Logger
from ..binning import OptimalBinning
from ..binning_information import print_binning_information
from ..prebinning import PreBinning
from .binning_statistics_2d import BinningTable2D
from .cp_2d import Binning2DCP
from .lp_2d import Binning2DLP
from .mip_2d import Binning2DMIP
from .model_data_2d import model_data
from .model_data_cart_2d import model_data_cart
from .preprocessing_2d import split_data_2d
from .utils import check_is_lp


class OptimalBinning2D(OptimalBinning):
    """
    """
    def __init__(self, name_x="", name_y="", dtype_x="numerical",
                 dtype_y="numerical", prebinning_method="cart", solver="mip",
                 divergence="iv", max_n_prebins_x=5, max_n_prebins_y=5,
                 min_prebin_size_x=0.05, min_prebin_size_y=0.05,
                 min_n_bins=None, max_n_bins=None, min_bin_size=None,
                 max_bin_size=None, min_bin_n_nonevent=None,
                 max_bin_n_nonevent=None, min_bin_n_event=None,
                 max_bin_n_event=None, monotonic_trend_x=None,
                 monotonic_trend_y=None, min_event_rate_diff_x=0,
                 min_event_rate_diff_y=0, gamma=0, user_splits_x=None,
                 user_splits_y=None, special_codes_x=None,
                 special_codes_y=None, split_digits=None, n_jobs=1,
                 time_limit=100, verbose=False, strategy=None):
        
        self.name_x = name_x
        self.name_y = name_y
        self.dtype_x = dtype_x
        self.dtype_y = dtype_y
        self.prebinning_method = prebinning_method
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

        self.user_splits_x = user_splits_x
        self.user_splits_y = user_splits_y
        self.special_codes_x = special_codes_x
        self.special_codes_y = special_codes_y
        self.split_digits = split_digits

        self.n_jobs = n_jobs
        self.time_limit = time_limit

        self.verbose = verbose

        self.strategy = strategy

        # auxiliary
        self._problem_type = "classification"

        # info

        # timing
        self._time_total = None
        self._time_preprocessing = None
        self._time_prebinning = None
        self._time_solver = None
        self._time_postprocessing = None

        # logger
        self._class_logger = Logger(__name__)
        self._logger = self._class_logger.logger

        self._is_fitted = False

    def fit(self, x, y, z, check_input=False):
        return self._fit(x, y, z, check_input)

    def _fit(self, x, y, z, check_input):
        time_init = time.perf_counter()

        if self.verbose:
            self._logger.info("Optimal binning started.")
            self._logger.info("Options: check parameters.")

        # Pre-processing
        if self.verbose:
            self._logger.info("Pre-processing started.")

        self._n_samples = len(x)

        if self.verbose:
            self._logger.info("Pre-processing: number of samples: {}"
                              .format(self._n_samples))

        time_preprocessing = time.perf_counter()

        [x_clean, y_clean, z_clean, special_mask_x, special_mask_y,
         missing_mask_x, missing_mask_y, mask_others_x, mask_others_y,
         categories_x, categories_y, others_x, others_y] = split_data_2d(
            self.dtype_x, self.dtype_y, x, y, z, self.special_codes_x,
            self.special_codes_y, self.user_splits_x, self.user_splits_y,
            check_input)

        self._time_preprocessing = time.perf_counter() - time_preprocessing

        if self.verbose:
            self._logger.info("Pre-processing terminated. Time: {:.4f}s"
                              .format(self._time_preprocessing))

        # Pre-binning
        if self.verbose:
            self._logger.info("Pre-binning started.")

        time_prebinning = time.perf_counter()

        if self.dtype_x == "categorical" and self.user_splits_x is not None:
            pass

        if self.dtype_y == "categorical" and self.user_splits_y is not None:
            pass


        splits_x = self._fit_prebinning(self.dtype_x, x_clean, z_clean,
                                        self.max_n_prebins_x,
                                        self.min_prebin_size_x)
        
        splits_y = self._fit_prebinning(self.dtype_y, y_clean, z_clean,
                                        self.max_n_prebins_y,
                                        self.min_prebin_size_y)

        E, NE = self._prebinning_matrices(splits_x, splits_y, x_clean, y_clean,
                                          z_clean)

        if self.strategy == "cart":
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

            min_prebin_size = min(self.min_prebin_size_x, self.min_prebin_size_y) * 0.25

            print(clf_nodes, min_prebin_size)

            clf = DecisionTreeClassifier(min_samples_leaf=min_prebin_size,
                                         max_leaf_nodes=clf_nodes)
            clf.fit(xyt, z)

            self._clf = clf

        self._time_prebinning = time.perf_counter() - time_prebinning

        # Optimization
        rows, n_nonevent, n_event = self._fit_optimizer(
            splits_x, splits_y, NE, E)

        # Post-processing
        time_postprocessing = time.perf_counter()

        # solution matrices
        m, n = E.shape

        D = np.empty(m * n, dtype=float)
        P = np.empty(m * n, dtype=int)

        selected_rows = np.array(rows, dtype=object)[self._solution]

        self._selected_rows = selected_rows
        self._m, self._n = m, n

        n_selected_rows = selected_rows.shape[0]

        opt_n_nonevent = np.empty(n_selected_rows, dtype=int)
        opt_n_event = np.empty(n_selected_rows, dtype=int)
        opt_event_rate = np.empty(n_selected_rows, dtype=float)

        for i, r in enumerate(selected_rows):
            _n_nonevent = n_nonevent[self._solution][i]
            _n_event = n_event[self._solution][i]
            _event_rate = _n_event / (_n_event + _n_nonevent)
            
            P[r] = i
            D[r] = _event_rate
            opt_n_nonevent[i] = _n_nonevent
            opt_n_event[i] = _n_event
            opt_event_rate[i] = _event_rate

        D = D.reshape((m, n))
        P = P.reshape((m, n))

        # optimal bins
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

        self._binning_table = BinningTable2D(
            self.name_x, self.name_y, self.dtype_x, self.dtype_y,
            splits_x_optimal, splits_y_optimal, m, n, opt_n_nonevent,
            opt_n_event, opt_event_rate, D, P)

        self.name = "-".join((self.name_x, self.name_y))

        self._time_postprocessing = time.perf_counter() - time_postprocessing

        self._n_prebins = E.size
        self._n_refinements = (m * n * (m + 1) * (n + 1)) // 4 - len(rows)

        self._time_total = time.perf_counter() - time_init

        # Completed successfully
        self._is_fitted = True

        return self

    def _fit_prebinning(self, dtype, x, z, max_n_prebins, min_prebin_size):
        # Check user splits

        # Pre-binning algorithm
        min_bin_size = int(np.ceil(min_prebin_size * self._n_samples))

        prebinning = PreBinning(method=self.prebinning_method,
                                n_bins=max_n_prebins,
                                min_bin_size=min_bin_size,
                                problem_type=self._problem_type).fit(x, z)

        return prebinning.splits

    def _prebinning_matrices(self, splits_x, splits_y, x_clean, y_clean, z_clean):
        z0 = z_clean == 0
        z1 = ~z0

        # Compute n_nonevent and n_event for special, missing and others
        # self._n_nonevent_special
        # self._n_event_special
        # self._n_nonevent_missing
        # self._n_event_missing


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

        # Check if problem can be formulated as a LP
        is_lp = check_is_lp(self.gamma, self.monotonic_trend_x,
                            self.monotonic_trend_y, self.min_n_bins,
                            self.max_n_bins)

        if self.solver == "auto":
            if is_lp:
                solver = "lp"
            else:
                solver = "cp"
        else:
            solver = self.solver

            if solver == "lp" and not is_lp:
                raise ValueError()

        if solver == "cp":
            scale = int(1e6)

            optimizer = Binning2DCP(
                self.monotonic_trend_x, self.monotonic_trend_y,
                self.min_n_bins, self.max_n_bins, self.min_event_rate_diff_x,
                self.min_event_rate_diff_y, self.gamma, self.n_jobs,
                self.time_limit)

        elif solver == "mip":
            scale = None

            optimizer = Binning2DMIP(
                self.monotonic_trend_x, self.monotonic_trend_y,
                self.min_n_bins, self.max_n_bins, self.min_event_rate_diff_x,
                self.min_event_rate_diff_y, self.gamma, is_lp, self.time_limit)

        elif solver == "lp":
            scale = None

            optimizer = Binning2DLP()

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

        print("model data:", self._time_model_data)

        if solver in ("cp", "mip"):
            optimizer.build_model(n_grid, n_rectangles, cols, c, d_connected_x,
                                  d_connected_y, event_rate, n_records)

            status, solution = optimizer.solve()
        elif solver == "lp":
            status, solution = optimizer.solve(n_grid, n_rectangles, cols, c)

        self._solution = solution

        self._optimizer = optimizer
        self._status = status
        self.solver = solver

        self._time_solver = time.perf_counter() - time_init

        self._cols = cols
        self._rows = rows
        self._c = c

        return rows, n_nonevent, n_event 
