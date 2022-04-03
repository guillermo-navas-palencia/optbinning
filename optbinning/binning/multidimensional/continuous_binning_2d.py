"""
Optimal binning 2D algorithm for continuous target.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2022

import numbers
import time

import numpy as np

from joblib import effective_n_jobs

from ...information import solver_statistics
from ...logging import Logger
from .binning_2d import OptimalBinning2D
from .binning_statistics_2d import ContinuousBinningTable2D
from .continuous_cp_2d import ContinuousBinning2DCP
from .model_data_2d import continuous_model_data
from .preprocessing_2d import split_data_2d


logger = Logger(__name__).logger


def _check_parameters():
    pass


class ContinuousOptimalBinning2D(OptimalBinning2D):
    def __init__(self, name_x="", name_y="", dtype_x="numerical",
                 dtype_y="numerical", prebinning_method="cart",
                 strategy="grid", solver="cp", max_n_prebins_x=5,
                 max_n_prebins_y=5, min_prebin_size_x=0.05,
                 min_prebin_size_y=0.05, min_n_bins=None, max_n_bins=None,
                 min_bin_size=None, max_bin_size=None, monotonic_trend_x=None,
                 monotonic_trend_y=None, min_mean_diff_x=0, min_mean_diff_y=0,
                 gamma=0, special_codes_x=None, special_codes_y=None,
                 split_digits=None, n_jobs=1, time_limit=100, verbose=False):

        self.name_x = name_x
        self.name_y = name_y
        self.dtype_x = dtype_x
        self.dtype_y = dtype_y
        self.prebinning_method = prebinning_method
        self.strategy = strategy
        self.solver = solver

        self.max_n_prebins_x = max_n_prebins_x
        self.max_n_prebins_y = max_n_prebins_y
        self.min_prebin_size_x = min_prebin_size_x
        self.min_prebin_size_y = min_prebin_size_y

        self.min_n_bins = min_n_bins
        self.max_n_bins = max_n_bins
        self.min_bin_size = min_bin_size
        self.max_bin_size = max_bin_size

        self.monotonic_trend_x = monotonic_trend_x
        self.monotonic_trend_y = monotonic_trend_y
        self.min_mean_diff_x = min_mean_diff_x
        self.min_mean_diff_y = min_mean_diff_y
        self.gamma = gamma

        self.special_codes_x = special_codes_x
        self.special_codes_y = special_codes_y
        self.split_digits = split_digits

        self.n_jobs = n_jobs
        self.time_limit = time_limit

        self.verbose = verbose

        # auxiliary
        self._problem_type = "regression"

        # info
        self._binning_table = None
        self._n_prebins = None
        self._n_refinements = 0
        self._n_samples = None
        self._optimizer = None
        self._solution = None
        self._splits_x_optimal = None
        self._splits_y_optimal = None
        self._status = None

        # timing
        self._time_total = None
        self._time_preprocessing = None
        self._time_prebinning = None
        self._time_solver = None
        self._time_optimizer = None
        self._time_postprocessing = None

        self._is_fitted = False

    def fit(self, x, y, z, check_input=False):
        return self._fit(x ,y, z, check_input)

    def fit_transform(self, x, y, z, metric="mean", metric_special=0,
                      metric_missing=0, show_digits=2, check_input=False):
        pass

    def transform(self, x, y, metric="mean", metric_special=0, metric_missing=0,
                  show_digits=2, check_input=False):
        pass

    def _fit(self, x, y, z, check_input):
        time_init = time.perf_counter()

        if self.verbose:
            logger.info("Optimal binning started.")
            logger.info("Options: check parameters.")

        # _check_parameters(**self.get_params())

        # Pre-processing
        if self.verbose:
            logger.info("Pre-processing started.")

        self._n_samples = len(x)

        if self.verbose:
            logger.info("Pre-processing: number of samples: {}"
                        .format(self._n_samples))

        time_preprocessing = time.perf_counter()         

        [x_clean, y_clean, z_clean, x_missing, y_missing, z_missing,
         x_special, y_special, z_special] = split_data_2d(
            self.dtype_x, self.dtype_y, x, y, z, self.special_codes_x,
            self.special_codes_y, check_input)

        self._time_preprocessing = time.perf_counter() - time_preprocessing

        if self.verbose:
            n_clean = len(x_clean)
            n_missing = len(x_missing)
            n_special = len(x_special)

            logger.info("Pre-processing: number of clean samples: {}"
                        .format(n_clean))

            logger.info("Pre-processing: number of missing samples: {}"
                        .format(n_missing))

            logger.info("Pre-processing: number of special samples: {}"
                        .format(n_special))
        if self.verbose:
            logger.info("Pre-processing terminated. Time: {:.4f}s"
                        .format(self._time_preprocessing))

        # Pre-binning
        if self.verbose:
            logger.info("Pre-binning started.")

        time_prebinning = time.perf_counter()

        splits_x = self._fit_prebinning(self.dtype_x, x_clean, z_clean,
                                        self.max_n_prebins_x,
                                        self.min_prebin_size_x)

        splits_y = self._fit_prebinning(self.dtype_y, y_clean, z_clean,
                                        self.max_n_prebins_y,
                                        self.min_prebin_size_y)

        R, S, SS, SD = self._prebinning_matrices(
            splits_x, splits_y, x_clean, y_clean, z_clean, x_missing,
            y_missing, z_missing, x_special, y_special, z_special)

        if self.strategy == "cart":
            pass

        self._time_prebinning = time.perf_counter() - time_prebinning

        self._n_prebins = R.size

        if self.verbose:
            logger.info("Pre-binning: number of prebins: {}"
                        .format(self._n_prebins))

            logger.info("Pre-binning terminated. Time: {:.4f}s"
                        .format(self._time_prebinning))

        # Optimization
        rows, n_records, sums, ssums, stds = self._fit_optimizer(
            splits_x, splits_y, R, S, SS)

        # Post-processing
        if self.verbose:
            logger.info("Post-processing started.")
            logger.info("Post-processing: compute binning information.")

        time_postprocessing = time.perf_counter()

        # Refinements
        m, n = R.shape
        self._n_refinements = (m * n * (m + 1) * (n + 1)) // 4 - len(rows)

        # solution matrices
        D = np.empty(m * n, dtype=float)
        P = np.empty(m * n, dtype=int)

        selected_rows = np.array(rows, dtype=object)[self._solution]

        self._selected_rows = selected_rows
        self._m, self._n = m, n

        n_selected_rows = selected_rows.shape[0] + 2        

        opt_sums = np.empty(n_selected_rows, dtype=float)
        opt_n_records = np.empty(n_selected_rows, dtype=int)
        opt_stds = np.zeros(n_selected_rows, dtype=float)

        for i, r in enumerate(selected_rows):
            _n_records = n_records[self._solution][i]
            _sums = sums[self._solution][i]
            _mean = _sums / _n_records
            _stds = stds[self._solution][i]

            P[r] = i
            D[r] = _mean
            opt_sums[i] = _sums
            opt_n_records[i] = _n_records
            opt_stds[i] = _stds

        opt_n_records[-2] = self._n_records_special
        opt_sums[-2] = self._sum_special

        opt_n_records[-1] = self._n_records_missing
        opt_sums[-1] = self._sum_missing

        self._sums = opt_sums
        self._n_records = opt_n_records

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

        self._splits_x_optimal = splits_x_optimal
        self._splits_y_optimal = splits_y_optimal

        self._binning_table = ContinuousBinningTable2D(
            self.name_x, self.name_y, self.dtype_x, self.dtype_y,
            splits_x_optimal, splits_y_optimal, m, n, opt_n_records,
            opt_sums, opt_stds, D, P)

        self.name = "-".join((self.name_x, self.name_y))

        self._time_postprocessing = time.perf_counter() - time_postprocessing

        if self.verbose:
            logger.info("Post-processing terminated. Time: {:.4f}s"
                        .format(self._time_postprocessing))

        self._time_total = time.perf_counter() - time_init

        if self.verbose:
            logger.info("Optimal binning terminated. Status: {}. Time: {:.4f}s"
                        .format(self._status, self._time_total))

        # Completed successfully
        self._is_fitted = True

        return self

    def _prebinning_matrices(self, splits_x, splits_y, x_clean, y_clean,
                             z_clean, x_missing, y_missing, z_missing,
                             x_special, y_special, z_special):
        
        self._n_records_missing = len(z_missing)
        self._n_records_special = len(z_special)
        self._sum_missing = np.sum(z_missing)
        self._sum_special = np.sum(z_special)

        n_splits_x = len(splits_x)
        n_splits_y = len(splits_y)

        indices_x = np.digitize(x_clean, splits_x, right=False)
        n_bins_x = n_splits_x + 1

        indices_y = np.digitize(y_clean, splits_y, right=False)
        n_bins_y = n_splits_y + 1

        R = np.empty((n_bins_x, n_bins_y), dtype=float)
        S = np.empty((n_bins_x, n_bins_y), dtype=float)
        SS = np.empty((n_bins_x, n_bins_y), dtype=float)
        SD = np.empty((n_bins_x, n_bins_y), dtype=float)

        for i in range(n_bins_y):
            mask_y = (indices_y == i)
            for j in range(n_bins_x):
                mask_x = (indices_x == j)
                mask = mask_x & mask_y

                zmask = z_clean[mask]
                R[i, j] = np.count_nonzero(mask)
                S[i, j] = np.sum(zmask)
                SS[i, j] = np.sum(zmask ** 2)

                if len(zmask):
                    SD[i, j] = np.std(zmask)

        return R, S, SS, SD

    def _fit_optimizer(self, splits_x, splits_y, R, S, SS):
        if self.verbose:
            logger.info("Optimizer started.")

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

        # Number of threads
        n_jobs = effective_n_jobs(self.n_jobs)

        if self.verbose:
            logger.info("Optimizer: {} jobs.".format(n_jobs))

            if self.monotonic_trend_x is None:
                logger.info(
                    "Optimizer: monotonic trend x not set.")
            else:
                logger.info("Optimizer: monotonic trend x set to {}."
                            .format(self.monotonic_trend_x))

            if self.monotonic_trend_y is None:
                logger.info(
                    "Optimizer: monotonic trend y not set.")
            else:
                logger.info("Optimizer: monotonic trend y set to {}."
                            .format(self.monotonic_trend_x))

        if self.solver == "cp":
            scale = int(1e6)

            optimizer = ContinuousBinning2DCP(
                self.monotonic_trend_x, self.monotonic_trend_y,
                self.min_n_bins, self.max_n_bins, self.min_mean_diff_x,
                self.min_mean_diff_y, self.gamma, n_jobs, self.time_limit)

        elif self.solver == "mip":
            scale = None

            # optimizer = Binning2DMIP(
            #     self.monotonic_trend_x, self.monotonic_trend_y,
            #     self.min_n_bins, self.max_n_bins, self.min_event_rate_diff_x,
            #     self.min_event_rate_diff_y, self.gamma, n_jobs,
            #     self.time_limit)

        if self.verbose:
            logger.info("Optimizer: model data...")

        time_model_data = time.perf_counter()

        if self.strategy == "cart":
            pass
        else:
            [n_grid, n_rectangles, rows, cols, c, d_connected_x, d_connected_y,
             mean, n_records, sums, ssums, stds] = continuous_model_data(
                R, S, SS, self.monotonic_trend_x, self.monotonic_trend_y, scale,
                min_bin_size, max_bin_size)

        self._time_model_data = time.perf_counter() - time_model_data

        if self.verbose:
            logger.info("Optimizer: model data terminated. Time {:.4f}s"
                        .format(self._time_model_data))

        if self.verbose:
            logger.info("Optimizer: build model...")

        optimizer.build_model(n_grid, n_rectangles, cols, c, d_connected_x,
                              d_connected_y, mean, n_records)

        if self.verbose:
            logger.info("Optimizer: solve...")

        status, solution = optimizer.solve()

        self._solution = solution

        self._optimizer, self._time_optimizer = solver_statistics(
            self.solver, optimizer.solver_)
        self._status = status

        self._time_solver = time.perf_counter() - time_init

        if self.verbose:
            logger.info("Optimizer terminated. Time: {:.4f}s"
                        .format(self._time_solver))

        self._cols = cols
        self._rows = rows
        self._c = c

        return rows, n_records, sums, ssums, stds