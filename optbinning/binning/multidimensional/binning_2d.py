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

from ...logging import Logger
from ..binning import OptimalBinning
from ..binning_information import print_binning_information
from ..prebinning import PreBinning
from .binning_statistics_2d import BinningTable2D
from .cp_2d import Binning2DCP
from .lp_2d import Binning2DLP
from .mip_2d import Binning2DMIP
from .model_data_2d import model_data
from .preprocessing_2d import split_data_2d
from .utils import check_is_lp


class OptimalBinning2D(OptimalBinning):
    """
    """
    def __init__(self, name_x="", name_y="", dtype_x="numerical",
                 dtype_y="numerical", prebinning_method="cart", solver="mip",
                 max_n_prebins_x=5, max_n_prebins_y=5, min_prebin_size_x=0.05,
                 min_prebin_size_y=0.05, min_n_bins=None, max_n_bins=None,
                 min_bin_size=None, max_bin_size=None, min_bin_n_nonevent=None,
                 max_bin_n_nonevent=None, min_bin_n_event=None,
                 max_bin_n_event=None, monotonic_trend_x=None,
                 monotonic_trend_y=None, min_event_rate_diff_x=0,
                 min_event_rate_diff_y=0, gamma=0, user_splits_x=None,
                 user_splits_y=None, special_codes=None,
                 split_digits=None, n_jobs=1, time_limit=100,
                 verbose=False):
        
        self.name_x = name_x
        self.name_y = name_y
        self.dtype_x = dtype_x
        self.dtype_y = dtype_y
        self.prebinning_method = prebinning_method
        self.solver = solver

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
        self.special_codes = special_codes
        self.split_digits = split_digits

        self.n_jobs = n_jobs
        self.time_limit = time_limit

        self.verbose = verbose

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

        # Pre-processing
        self._n_samples = len(x)

        time_preprocessing = time.perf_counter()

        [x_clean, y_clean, z_clean, special_mask_x, special_mask_y,
         missing_mask_x, missing_mask_y, mask_others_x, mask_others_y,
         categories_x, categories_y, others_x, others_y] = split_data_2d(
            self.dtype_x, self.dtype_y, x, y, z, self.special_codes,
            self.user_splits_x, self.user_splits_y, check_input)

        self._time_preprocessing = time.perf_counter() - time_preprocessing

        # Pre-binning
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

        self._time_prebinning = time.perf_counter() - time_prebinning

        # Optimization
        rows, n_nonevent, n_event = self._fit_optimizer(
            splits_x, splits_y, NE, E)

        # Post-processing
        time_postprocessing = time.perf_counter()

        # solution matrices
        m, n = E.shape

        D = np.empty(m * n)
        P = np.empty(m * n)

        selected_rows = np.array(rows, dtype=object)[self._solution]
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
        P = P.astype(int).reshape((m, n))

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
            bin_x = [bin_x[0][0], bin_x[-1][1]] 
            bin_y = bins_str_y[mask_y]
            bin_y = [bin_y[0][0], bin_y[-1][1]] 

            splits_x_optimal.append(bin_x)
            splits_y_optimal.append(bin_y)

        self._binning_table = BinningTable2D(
            self.name_x, self.name_y, self.dtype_x, self.dtype_y,
            splits_x_optimal, splits_y_optimal, m, n, opt_n_nonevent, opt_n_event,
            opt_event_rate, D, P)

        self.name = "-".join((self.name_x, self.name_y))

        self._time_postprocessing = time.perf_counter() - time_postprocessing

        self._n_prebins = E.size
        self._n_refinements = 0

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

        [n_grid, n_rectangles, rows, cols, c, d_connected_x, d_connected_y,
         iv, event_rate, n_event, n_nonevent, n_records] = model_data(
            NE, E, self.monotonic_trend_x, self.monotonic_trend_y, scale,
            min_bin_size, max_bin_size, self.min_bin_n_event,
            self.max_bin_n_event, self.min_bin_n_nonevent,
            self.max_bin_n_nonevent)

        self._time_model_data = time.perf_counter() - time_model_data

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

        return rows, n_nonevent, n_event 

    def plot_binning(self):
        # paths x: horizontal        
        paths_x = []
        for i in range(self._m):
            path = tuple(dict.fromkeys(self._S[i, :]))
            if not path in paths_x:
                paths_x.append(path)
        
        # paths y: vertical     
        paths_y = []
        for j in range(self._n):
            path = tuple(dict.fromkeys(self._S[:, j]))
            if not path in paths_y:
                paths_y.append(path) 

        fig, ax = plt.subplots(figsize=(7, 7))

        divider = make_axes_locatable(ax)
        axtop = divider.append_axes("top", size=2.5, pad=0.1, sharex=ax)
        axright = divider.append_axes("right", size=2.5, pad=0.1, sharey=ax)
        # Hide x labels and tick labels for top plots and y ticks for right plots.

        d = self._d
        S = self._S
        D = self._D

        # Position [0, 0]
        for path in paths_x:
            er = sum([[d[p]] * np.count_nonzero(S == p, axis=1).max() for p in path], [])
            er = er + [er[-1]]
            axtop.step(np.arange(S.shape[1] + 1) - 0.5, er, label=path, where="post")
            
        for i in range(S.shape[1]):
            axtop.axvline(i + 0.5, color="grey", linestyle="--", alpha=0.5)    

        axtop.get_xaxis().set_visible(False)
        axtop.set_ylabel("Event rate", fontsize=12)

        # Position [1, 0]
        pos = ax.matshow(D, cmap=plt.cm.bwr)
        for j in range(self._n):
            for i in range(self._m):
                c = int(S[i, j])
                ax.text(j, i, str(c), va='center', ha='center')

        fig.colorbar(pos, ax=ax, orientation="horizontal", fraction=0.025, pad=0.125)        

        ax.xaxis.set_label_position("bottom")
        ax.xaxis.tick_bottom()
        ax.set_ylabel("Bin ID - y", fontsize=12)
        ax.set_xlabel("Bin ID - x", fontsize=12)

        # Position [1, 1]
        for j, path in enumerate(paths_y):
            er = sum([[d[p]] * (np.count_nonzero(S == p, axis=0).max()) for p in path], [])
            er = er + [er[-1]]
            axright.step(er, np.arange(S.shape[0] + 1) - 0.5, label=path, where="pre")
                
        for j in range(S.shape[0]):
            axright.axhline(j -0.5, color="grey", linestyle="--", alpha=0.5)
            
        axright.get_yaxis().set_visible(False)
        axright.set_xlabel("Event rate", fontsize=12)

            
        #adjust margins
        axright.margins(y=0)
        axtop.margins(x=0)
        plt.tight_layout()

        axtop.legend(bbox_to_anchor=(1, 1))
        axright.legend(bbox_to_anchor=(1, 1))

        plt.show()
