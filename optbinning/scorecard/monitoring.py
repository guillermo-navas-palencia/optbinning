"""
Scorecard monitoring (System stability report)

References:
https://mwburke.github.io/data%20science/2018/04/29/population-stability-index.html
https://www.lexjansen.com/wuss/2017/47_Final_Paper_PDF.pdf
https://www.listendata.com/2015/05/population-stability-index.html
http://ucanalytics.com/blogs/population-stability-index-psi-banking-case-study/
http://shichen.name/scorecard/reference/perf_psi.html
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import numbers

import numpy as np
import pandas as pd

from scipy import stats
from sklearn.exceptions import NotFittedError
from sklearn.utils.multiclass import type_of_target

from ..binning.binning_statistics import bin_str_format
from ..binning.metrics import jeffrey
from ..binning.metrics import frequentist_pvalue
from ..binning.prebinning import PreBinning
from ..logging import Logger


class VariableMonitoring:
    def __init__(self, scorecard, show_digits):
        self.scorecard = scorecard
        self.show_digits = show_digits

        # logger
        self._class_logger = Logger(__name__)
        self._logger = self._class_logger.logger

        self._is_fitted = False

    def fit(self, df_actual, df_expected, metric_special=0, metric_missing=0):
        # checks

        if not self.scorecard._is_fitted:
            self.scorecard.fit(df_actual, metric_special, metric_missing,
                               self.show_digits)

        # transformed df_actual
        bins_actual = self.scorecard.binning_process.transform(
            df_actual, "bins", metric_special, metric_missing,
            self.show_digits)

        bins_expected = self.scorecard.binning_process.transform(
            df_expected, "bins", metric_special, metric_missing,
            self.show_digits)

    def report(self):
        """Characteristic analysis report."""
        pass


def print_psi_report(df_psi):

    t_psi = df_psi.PSI.values[-1]
    psi = df_psi.PSI.values[:-1]
    indices = np.digitize(psi, [0.1, 0.25], right=True)

    psi_bins = np.empty(3).astype(np.int64)
    for i in range(2):
        mask = (indices == i)
        psi_bins[i] = len(psi[mask])

    p_psi_bins = psi_bins / sum(psi_bins)

    if t_psi < 0.1:
        psi_veredict = "No significant change"
    elif t_psi < 0.25:
        psi_veredict = "Requires investigation"
    else:
        psi_veredict = "Significance change"

    psi_stats = (
        "Population Stability Index (PSI)\n"
        "----------------------------------------------\n"
        ""
        "\n"
        "PSI total:     {:>7.4f} ({})\n"
        "\n"
        "   PSI bin\n"
        "[0.00, 0.10)        {:>2} ({:>7.2%})\n"
        "[0.10, 0.25)        {:>2} ({:>7.2%})\n"
        "[0.25, Inf+)        {:>2} ({:>7.2%})\n"
        ).format(t_psi, psi_veredict, psi_bins[0], p_psi_bins[0], psi_bins[1],
                 p_psi_bins[1], psi_bins[2], p_psi_bins[2])

    print(psi_stats)


def print_tests_report(df_tests):
    pass


class ScorecardMonitoring:
    def __init__(self, method="cart", n_bins=20, min_bin_size=0.05,
                 show_digits=2, verbose=False):

        self.method = method
        self.n_bins = n_bins
        self.min_bin_size = min_bin_size
        self.show_digits = show_digits
        self.verbose = verbose

        # auxiliary data
        self._splits = None
        self._df_psi = None
        self._df_tests = None
        self._target_dtype = None

        # logger
        self._class_logger = Logger(__name__)
        self._logger = self._class_logger.logger

        self._is_fitted = False

    def fit(self, score_actual, y_actual, score_expected, y_expected):
        """Fit monitoring with actual and expected data.

        Parameters
        ----------
        score_actual : array-like, shape = (n_samples_actual,)
            Score of the training or actual input samples.

        y_actual : array-like, shape = (n_samples_actual,)
            Target of the training or actual input samples.

        score_expected : array-like, shape = (n_samples_expected,)
            Score of the test or expected input samples.

        y_expected : array-like, shape = (n_samples_expected,)
            Target of the training or actual input samples.
        """

        # Check parameters
        if self.method not in ("uniform", "quantile", "cart"):
            raise ValueError('Invalid value for prebinning_method. Allowed '
                             'string values are "cart", "quantile" and '
                             '"uniform".')

        if self.n_bins is not None:
            if (not isinstance(self.n_bins, numbers.Integral) or
                    self.n_bins <= 0):
                raise ValueError("n_bins must be a positive integer; got {}."
                                 .format(self.n_bins))

        if self.min_bin_size is not None:
            if (not isinstance(self.min_bin_size, numbers.Number) or
                    not 0. < self.min_bin_size <= 0.5):
                raise ValueError("min_bin_size must be in (0, 0.5]; got {}."
                                 .format(self.min_bin_size))

        if (not isinstance(self.show_digits, numbers.Integral) or
                not 0 <= self.show_digits <= 8):
            raise ValueError("show_digits must be an integer in [0, 8]; "
                             "got {}.".format(self.show_digits))

        if not isinstance(self.verbose, bool):
            raise TypeError("verbose must be a boolean; got {}."
                            .format(self.verbose))

        target_dtype = type_of_target(y_actual)
        target_dtype_e = type_of_target(y_expected)

        if target_dtype not in ("binary", "continuous"):
            raise ValueError("Target type {} is not supported."
                             .format(target_dtype))

        if target_dtype != target_dtype_e:
            raise ValueError("Target types must coincide; {} != {}."
                             .format(target_dtype, target_dtype_e))

        self._target_dtype = target_dtype

        if target_dtype == "binary":
            problem_type = "classification"
        else:
            problem_type = "regression"

        prebinning = PreBinning(problem_type=problem_type,
                                method=self.method,
                                n_bins=self.n_bins,
                                min_bin_size=self.min_bin_size
                                ).fit(score_actual, y_actual)

        splits = prebinning.splits
        self._splits = splits

        n_splits = len(splits)
        n_bins = n_splits + 1

        indices_a = np.digitize(score_actual, splits, right=True)
        indices_e = np.digitize(score_expected, splits, right=True)

        if target_dtype == "binary":
            n_nonevent_a = np.empty(n_bins).astype(np.int64)
            n_event_a = np.empty(n_bins).astype(np.int64)
            n_nonevent_e = np.empty(n_bins).astype(np.int64)
            n_event_e = np.empty(n_bins).astype(np.int64)

            y0_a = (y_actual == 0)
            y1_a = ~ y0_a

            y0_e = (y_expected == 0)
            y1_e = ~ y0_e

            for i in range(n_bins):
                mask_a = (indices_a == i)
                n_nonevent_a[i] = np.count_nonzero(y0_a & mask_a)
                n_event_a[i] = np.count_nonzero(y1_a & mask_a)

                mask_e = (indices_e == i)
                n_nonevent_e[i] = np.count_nonzero(y0_e & mask_e)
                n_event_e[i] = np.count_nonzero(y1_e & mask_e)

            n_records_a = n_nonevent_a + n_event_a
            n_records_e = n_nonevent_e + n_event_e

        else:
            n_records_a = np.empty(n_bins).astype(np.int64)
            n_records_e = np.empty(n_bins).astype(np.int64)
            mean_a = np.empty(n_bins)
            mean_e = np.empty(n_bins)
            std_a = np.empty(n_bins)
            std_e = np.empty(n_bins)

            for i in range(n_bins):
                mask_a = (indices_a == i)
                n_records_a[i] = np.count_nonzero(mask_a)
                mean_a[i] = y_actual[mask_a].mean()
                std_a[i] = y_actual[mask_a].std()

                mask_e = (indices_e == i)
                n_records_e[i] = np.count_nonzero(mask_e)
                mean_e[i] = y_expected[mask_e].mean()
                std_e[i] = y_expected[mask_e].std()

        # Population Stability Information (PSI)

        t_n_records_a = n_records_a.sum()
        t_n_records_e = n_records_e.sum()
        p_records_a = n_records_a / t_n_records_a
        p_records_e = n_records_e / t_n_records_e

        psi = jeffrey(p_records_a, p_records_e, return_sum=False)
        t_psi = psi.sum()

        bins = np.concatenate([[-np.inf], splits, [np.inf]])
        bin_str = bin_str_format(bins, self.show_digits)

        df_psi = pd.DataFrame({
            "Bin": bin_str,
            "Count A": n_records_a,
            "Count E": n_records_e,
            "Count (%) A": p_records_a,
            "Count (%) E": p_records_e,
            "PSI": psi
            })

        totals = ["", t_n_records_a, t_n_records_e, 1, 1, t_psi]
        df_psi.loc["Totals"] = totals

        self._df_psi = df_psi

        # Significance tests
        t_statistics = []
        p_values = []

        if target_dtype == "binary":
            event_rate_a = n_event_a / n_records_a
            event_rate_e = n_event_e / n_records_e

            for i in range(n_bins):
                obs = np.array([
                    [n_nonevent_a[i], n_nonevent_e[i]],
                    [n_event_a[i], n_event_e[i]]])

                t, p = frequentist_pvalue(obs, "chi2")

                t_statistics.append(t)
                p_values.append(p)

            df_tests = pd.DataFrame({
                "Bin": bin_str,
                "Count A": n_records_a,
                "Count E": n_records_e,
                "Event rate A": event_rate_a,
                "Event rate E": event_rate_e,
                "statistic": t_statistics,
                "p-value": p_values
                })
        else:
            for i in range(n_bins):
                t, p = stats.ttest_ind_from_stats(
                    mean_a[i], std_a[i], n_records_a[i],
                    mean_e[i], std_e[i], n_records_e[i], False)

                t_statistics.append(t)
                p_values.append(p)

            df_tests = pd.DataFrame({
                "Bin": bin_str,
                "Count A": n_records_a,
                "Count E": n_records_e,
                "Mean A": mean_a,
                "Mean E": mean_e,
                "Std A": std_a,
                "Std E": std_e,
                "statistic": t_statistics,
                "p-value": p_values
                })

        self._df_tests = df_tests

        self._is_fitted = True

    def report(self):
        """System stability report."""
        self._check_is_fitted()

        print_psi_report(self._df_psi)

    def psi(self):
        self._check_is_fitted()

        return self._df_psi

    def tests(self):
        self._check_is_fitted()

        return self._df_tests

    def plot_psi(self):
        self._check_is_fitted()

        # Plot depending on target dtype.
        # n_records two bars. => self._df_psi
        # event rate (binary), mean (continuous)
        pass

    def _check_is_fitted(self):
        if not self._is_fitted:
            raise NotFittedError("This {} instance is not fitted yet. Call "
                                 "'fit' with appropriate arguments."
                                 .format(self.__class__.__name__))

    @property
    def splits(self):
        self._check_is_fitted()

        self._splits
