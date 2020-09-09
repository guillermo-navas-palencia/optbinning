"""
Scorecard monitoring (System stability report)
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import numbers

import numpy as np
import pandas as pd

from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils.multiclass import type_of_target

from ..binning.binning_statistics import bin_str_format
from ..binning.metrics import jeffrey
from ..binning.prebinning import PreBinning
from ..formatting import dataframe_to_string
from ..logging import Logger
from .metrics import gini
from .metrics import imbalanced_classification_metrics
from .metrics import regression_metrics
from .scorecard import Scorecard


PSI_VERDICT_MSG = {0: "No significant change",
                   1: "Requires investigation",
                   2: "Significance change"}


def _check_parameters(target, scorecard, psi_method, psi_n_bins,
                      psi_min_bin_size, show_digits, verbose):

    if not isinstance(target, str):
        raise TypeError("target must be a string.")

    if not isinstance(scorecard, Scorecard):
        raise TypeError("scorecard must be a Scorecard instance.")

    if psi_method not in ("uniform", "quantile", "cart"):
        raise ValueError('Invalid value for prebinning_method. Allowed '
                         'string values are "cart", "quantile" and '
                         '"uniform".')

    if psi_n_bins is not None:
        if (not isinstance(psi_n_bins, numbers.Integral) or
                psi_n_bins <= 0):
            raise ValueError("psi_n_bins must be a positive integer; got {}."
                             .format(psi_n_bins))

    if psi_min_bin_size is not None:
        if (not isinstance(psi_min_bin_size, numbers.Number) or
                not 0. < psi_min_bin_size <= 0.5):
            raise ValueError("psi_min_bin_size must be in (0, 0.5]; got {}."
                             .format(psi_min_bin_size))

    if (not isinstance(show_digits, numbers.Integral) or
            not 0 <= show_digits <= 8):
        raise ValueError("show_digits must be an integer in [0, 8]; "
                         "got {}.".format(show_digits))

    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a boolean; got {}.".format(verbose))


def print_psi_report(df_psi):
    t_psi = df_psi.PSI.values[-1]
    psi = df_psi.PSI.values[:-1]

    splits = [0.1, 0.25]
    n_bins = len(splits) + 1
    indices = np.digitize(psi, splits, right=True)

    psi_bins = np.empty(n_bins).astype(np.int64)
    for i in range(n_bins):
        mask = (indices == i)
        psi_bins[i] = len(psi[mask])

    p_psi_bins = psi_bins / psi_bins.sum()
    psi_verdict = PSI_VERDICT_MSG[np.digitize(t_psi, splits)]

    df_psi_string = dataframe_to_string(pd.DataFrame({
        "PSI bin": ["[0.00, 0.10)", "[0.10, 0.25)", "[0.25, Inf+)"],
        "Count": psi_bins,
        "Count (%)": p_psi_bins
        }), tab=4)

    psi_stats = (
        "  Population Stability Index (PSI)\n\n"
        ""
        "\n"
        "    PSI total:     {:>7.4f} ({})\n\n"
        "{}\n").format(t_psi, psi_verdict, df_psi_string)

    print(psi_stats)


def print_tests_report(df_tests):
    pvalues = df_tests["p-value"].values

    splits = [0.05, 0.1, 0.5]
    n_bins = len(splits) + 1
    indices = np.digitize(pvalues, splits, right=True)

    pvalue_bins = np.empty(n_bins).astype(np.int64)
    for i in range(n_bins):
        mask = (indices == i)
        pvalue_bins[i] = len(pvalues[mask])

    p_pvalue_bins = pvalue_bins / pvalue_bins.sum()

    df_tests_string = dataframe_to_string(pd.DataFrame({
        "p-value bin":  ["[0.00, 0.05)", "[0.05, 0.10)", "[0.10, 0.50)",
                         "[0.50, 1.00)"],
        "Count": pvalue_bins,
        "Count (%)": p_pvalue_bins,
        }), tab=4)

    tests_stats = (
        "  Significance tests (H0: actual == expected)\n\n"
        "{}\n").format(df_tests_string)

    print(tests_stats)


def print_target_report(df_target):
    df_target_string = dataframe_to_string(df_target, tab=4)

    target_stats = (
        "  Target analysis\n\n"
        "{}\n").format(df_target_string)

    print(target_stats)


def print_performance_report(df_performance):
    df_performance_string = dataframe_to_string(df_performance, tab=4)

    performance_stats = (
        "  Performance metrics\n\n"
        "{}\n".format(df_performance_string)
        )

    print(performance_stats)


def print_system_report(df_psi, df_tests, df_target_analysis, df_performance):

    print("-----------------------------------\n"
          "Monitoring: System Stability Report\n"
          "-----------------------------------\n")

    print_psi_report(df_psi)
    print_tests_report(df_tests)
    print_target_report(df_target_analysis)
    print_performance_report(df_performance)


class ScorecardMonitoring(BaseEstimator):
    def __init__(self, target, scorecard, psi_method="cart", psi_n_bins=20,
                 psi_min_bin_size=0.05, show_digits=2, verbose=False):

        self.target = target
        self.scorecard = scorecard

        self.psi_method = psi_method
        self.psi_n_bins = psi_n_bins
        self.psi_min_bin_size = psi_min_bin_size

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

        # flags
        self._is_fitted = False

    def fit(self, df_actual, df_expected):
        """Fit monitoring with actual and expected data. If expected is not
        None then actual and expected are compared, otherwise only actual
        data statistics will be shown.

        Parameters
        ----------
        df_actual : pandas.DataFrame
            Score of the training or actual input samples.

        df_expected : pandas.DataFrame
            Trainning data used for fitting the scorecard.
        """

        # Check parameters
        _check_parameters(**self.get_params(deep=False))

        y_actual = df_actual[self.target]
        y_expected = df_expected[self.target]

        target_dtype = type_of_target(y_actual)
        target_dtype_e = type_of_target(y_expected)

        if target_dtype not in ("binary", "continuous"):
            raise ValueError("Target type {} is not supported."
                             .format(target_dtype))

        if target_dtype != target_dtype_e:
            raise ValueError("Target types must coincide; {} != {}."
                             .format(target_dtype, target_dtype_e))

        self._target_dtype = target_dtype

        self._fit_system(df_actual, y_actual, df_expected, y_expected)

        self._fit_variables(df_actual, df_expected)

        self._is_fitted = True

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

    def system_stability_report(self):
        """System stability report."""
        self._check_is_fitted()

        print_system_report(self._df_psi, self._df_tests,
                            self._df_target_analysis, self._df_performance)

    def characteristic_analysis_report(self):
        """Characteristic analysis report."""
        self._check_is_fitted()

    def psi_table(self):
        """"""
        self._check_is_fitted()

        return self._df_psi

    def tests_table(self):
        """"""
        self._check_is_fitted()

        return self._df_tests

    def psi_plot(self, savefig=None):
        """Plot Population Stability Index (PSI).

        Parameters
        ----------
        savefig : str or None (default=None)
            Path to save the plot figure.
        """
        self._check_is_fitted()

    def _fit_system(self, df_actual, y_actual, df_expected, y_expected):

        if self._target_dtype == "binary":
            problem_type = "classification"
        else:
            problem_type = "regression"

        score_actual = self.scorecard.score(df_actual)
        score_expected = self.scorecard.score(df_expected)

        prebinning = PreBinning(problem_type=problem_type,
                                method=self.psi_method,
                                n_bins=self.psi_n_bins,
                                min_bin_size=self.psi_min_bin_size
                                ).fit(score_expected, y_expected)

        splits = prebinning.splits
        n_bins = len(splits) + 1

        # Compute basic metrics
        indices_a = np.digitize(score_actual, splits, right=True)
        indices_e = np.digitize(score_expected, splits, right=True)

        if self._target_dtype == "binary":
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

        bins = np.concatenate([[-np.inf], splits, [np.inf]])
        bin_str = bin_str_format(bins, self.show_digits)

        # Target analysis
        if self._target_dtype == "binary":
            self._system_target_binary(n_records_a, n_event_a, n_nonevent_a,
                                       n_records_e, n_event_e, n_nonevent_e)
        else:
            self._system_target_continuous(y_actual, y_expected)

        # Population Stability Information (PSI)
        self._system_psi(bin_str, n_records_a, n_records_e)

        # Significance tests
        if self._target_dtype == "binary":
            self._system_tests_binary(
                bin_str, n_records_a, n_event_a, n_nonevent_a,
                n_records_e, n_event_e, n_nonevent_e)
        else:
            self._system_tests_continuous(
                bin_str, n_records_a, mean_a, std_a,
                n_records_e, mean_e, std_e)

        # Performance analysis
        if self._target_dtype == "binary":
            self._system_performance_binary(df_actual, df_expected)
        else:
            self._system_performance_continuous(df_actual, df_expected)

    def _system_psi(self, bin_str, n_records_a, n_records_e):
        t_n_records_a = n_records_a.sum()
        t_n_records_e = n_records_e.sum()
        p_records_a = n_records_a / t_n_records_a
        p_records_e = n_records_e / t_n_records_e

        psi = jeffrey(p_records_a, p_records_e, return_sum=False)

        df_psi = pd.DataFrame({
            "Bin": bin_str,
            "Count A": n_records_a,
            "Count E": n_records_e,
            "Count A (%)": p_records_a,
            "Count E (%)": p_records_e,
            "PSI": psi
            })

        totals = ["", t_n_records_a, t_n_records_e, 1, 1, psi.sum()]
        df_psi.loc["Totals"] = totals

        self._df_psi = df_psi

    def _system_tests_binary(self, bin_str, n_records_a, n_event_a,
                             n_nonevent_a, n_records_e, n_event_e,
                             n_nonevent_e):
        t_statistics = []
        p_values = []

        n_bins = len(bin_str)
        event_rate_a = n_event_a / n_records_a
        event_rate_e = n_event_e / n_records_e

        for i in range(n_bins):
            obs = np.array([
                [n_nonevent_a[i], n_nonevent_e[i]],
                [n_event_a[i], n_event_e[i]]])

            t, p, _, _ = stats.chi2_contingency(obs, correction=False)

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

        self._df_tests = df_tests

    def _system_tests_continuous(self, bin_str, n_records_a, mean_a, std_a,
                                 n_records_e, mean_e, std_e):
        t_statistics = []
        p_values = []

        n_bins = len(bin_str)
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

    def _system_target_binary(self, n_records_a, n_event_a, n_nonevent_a,
                              n_records_e, n_event_e, n_nonevent_e):

        t_n_records_a = n_records_a.sum()
        t_n_event_a = n_event_a.sum()
        t_n_nonevent_a = n_nonevent_a.sum()

        t_n_records_e = n_records_e.sum()
        t_n_event_e = n_event_e.sum()
        t_n_nonevent_e = n_nonevent_e.sum()

        event_rate_a = t_n_event_a / t_n_records_a
        event_rate_e = t_n_event_e / t_n_records_e

        df_target = pd.DataFrame({
            "Metric": ["Number of records", "Event records",
                       "Non-event records"],
            "Actual": [t_n_records_a, t_n_event_a, t_n_nonevent_a],
            "Actual (%)": ["-", event_rate_a, 1 - event_rate_a],
            "Expected": [t_n_records_e, t_n_event_e, t_n_nonevent_e],
            "Expected (%)": ["-", event_rate_e, 1 - event_rate_e]
            })

        self._df_target_analysis = df_target

    def _system_target_continuous(self, y_actual, y_expected):

        mean_a = y_actual.mean()
        mean_e = y_expected.mean()
        std_a = y_actual.std()
        std_e = y_expected.std()

        p25_a, median_a, p75_a = np.percentile(y_actual, [25, 50, 75])
        p25_e, median_e, p75_e = np.percentile(y_expected, [25, 50, 75])

        df_target = pd.DataFrame({
            "Metric": ["Mean", "Std", "p25", "Median", "p75"],
            "Actual": [mean_a, std_a, p25_a, median_a, p75_a],
            "Expected": [mean_e, std_e, p25_e, median_e, p75_e]
            })

        self._df_target_analysis = df_target

    def _system_performance_binary(self, df_actual, df_expected):
        # Metrics derived from confusion matrix
        y_true_a = df_actual[self.target]
        y_pred_a = self.scorecard.predict(df_actual)
        d_metrics_a = imbalanced_classification_metrics(y_true_a, y_pred_a)

        y_true_e = df_expected[self.target]
        y_pred_e = self.scorecard.predict(df_expected)
        d_metrics_e = imbalanced_classification_metrics(y_true_e, y_pred_e)

        metric_names = list(d_metrics_a.keys())
        metrics_a = list(d_metrics_a.values())
        metrics_e = list(d_metrics_e.values())

        # Gini
        y_pred_proba_a = self.scorecard.predict_proba(df_actual)[:, 1]
        gini_a = gini(y_true_a, y_pred_proba_a)

        y_pred_proba_e = self.scorecard.predict_proba(df_expected)[:, 1]
        gini_e = gini(y_true_e, y_pred_proba_e)

        metric_names.append("Gini")
        metrics_a.append(gini_a)
        metrics_e.append(gini_e)

        diff = np.array(metrics_a) - np.array(metrics_e)

        df_performance = pd.DataFrame({
            "Metric": metric_names,
            "Actual": metrics_a,
            "Expected": metrics_e,
            "Diff A - E": diff,
            })

        self._df_performance = df_performance

    def _system_performance_continuous(self, df_actual, df_expected):
        y_true_a = df_actual[self.target]
        y_pred_a = self.scorecard.predict(df_actual)
        d_metrics_a = regression_metrics(y_true_a, y_pred_a)

        y_true_e = df_expected[self.target]
        y_pred_e = self.scorecard.predict(df_expected)
        d_metrics_e = regression_metrics(y_true_e, y_pred_e)

        metric_names = list(d_metrics_a.keys())
        metrics_a = list(d_metrics_a.values())
        metrics_e = list(d_metrics_e.values())

        diff = np.array(metrics_a) - np.array(metrics_e)

        df_performance = pd.DataFrame({
            "Metric": metric_names,
            "Actual": metrics_a,
            "Expected": metrics_e,
            "Diff A - E": diff,
            })

        self._df_performance = df_performance

    def _fit_variables(self, df_actual, df_expected):
        pass

    def _check_is_fitted(self):
        if not self._is_fitted:
            raise NotFittedError("This {} instance is not fitted yet. Call "
                                 "'fit' with appropriate arguments."
                                 .format(self.__class__.__name__))

    @property
    def psi_splits(self):
        self._check_is_fitted()

        self._splits
