"""
Scorecard monitoring (System stability report)
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import numbers
import time

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
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
from ..metrics.classification import gini
from ..metrics.classification import imbalanced_classification_metrics
from ..metrics.regression import regression_metrics
from .monitoring_information import print_monitoring_information
from .scorecard import Scorecard


logger = Logger(__name__).logger


PSI_VERDICT_MSG = {0: "No significant change",
                   1: "Requires investigation",
                   2: "Significance change"}


def _check_parameters(scorecard, psi_method, psi_n_bins,
                      psi_min_bin_size, show_digits, verbose):

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
    """Scorecard monitoring.

    Parameters
    ----------
    scorecard : object
        A ``Scorecard`` fitted instance.

    psi_method : str, optional (default="cart")
        The binning method to compute the Population Stability Index (PSI).
        Supported methods are "cart" for a CART
        decision tree, "quantile" to generate prebins with approximately same
        frequency and "uniform" to generate prebins with equal width. Method
        "cart" uses `sklearn.tree.DecistionTreeClassifier
        <https://scikit-learn.org/stable/modules/generated/sklearn.tree.
        DecisionTreeClassifier.html>`_.

    psi_n_bins : int (default=20)
        The maximum number of bins to compute PSI.

    psi_min_bin_size : float (default=0.05)
        The fraction of mininum number of records for PSI bin.

    show_digits : int, optional (default=2)
        The number of significant digits of the bin column.

    verbose : bool (default=False)
        Enable verbose output.
    """
    def __init__(self, scorecard, psi_method="cart", psi_n_bins=20,
                 psi_min_bin_size=0.05, show_digits=2, verbose=False):

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
        self._n_records_a = None
        self._n_records_e = None
        self._metric_a = None
        self._metric_e = None

        # time
        self._time_total = None
        self._time_system = None
        self._time_variables = None

        # flags
        self._is_fitted = False

    def fit(self, X_actual, y_actual, X_expected, y_expected):
        """Fit monitoring with actual and expected data.

        Parameters
        ----------
        X_actual : pandas.DataFrame
            New/actual/test data input samples.

        y_actual : array-like of shape (n_samples,)
            Target vector relative to X actual.

        X_expected : pandas.DataFrame
            Trainning data used for fitting the scorecard.

        y_expected : array-like of shape (n_samples,)
            Target vector relative to X expected.

        Returns
        -------
        self : ScorecardMonitoring
            Fitted monitoring.
        """
        time_init = time.perf_counter()

        if self.verbose:
            logger.info("Monitoring started.")
            logger.info("Options: check parameters.")

        # Check parameters
        _check_parameters(**self.get_params(deep=False))

        # Check if scorecard is fitted
        self.scorecard._check_is_fitted()

        target_dtype = type_of_target(y_actual)
        target_dtype_e = type_of_target(y_expected)

        if target_dtype not in ("binary", "continuous"):
            raise ValueError("Target type (actual) {} is not supported."
                             .format(target_dtype))

        if target_dtype_e not in ("binary", "continuous"):
            raise ValueError("Target type (expected) {} is not supported."
                             .format(target_dtype_e))

        if target_dtype != target_dtype_e:
            raise ValueError("Target types must coincide; {} != {}."
                             .format(target_dtype, target_dtype_e))

        self._target_dtype = target_dtype

        # Check variable names
        if list(X_actual.columns) != list(X_expected.columns):
            raise ValueError("Dataframes X_actual and X_expected must "
                             "have the same columns.")

        # Statistics at system level
        if self.verbose:
            logger.info("System stability analysis started.")

        time_system = time.perf_counter()
        self._fit_system(X_actual, y_actual, X_expected, y_expected)
        self._time_system = time.perf_counter() - time_system

        if self.verbose:
            logger.info("System stability analysis terminated. Time: {:.4f}s"
                        .format(self._time_system))

        # Statistics at variable level
        if self.verbose:
            logger.info("Variable analysis started.")

        time_variable = time.perf_counter()
        self._fit_variables(X_actual, X_expected)
        self._time_variable = time.perf_counter() - time_variable

        if self.verbose:
            logger.info("Variable analysis terminated. Time: {:.4f}s"
                        .format(self._time_variable))

        self._time_total = time.perf_counter() - time_init

        if self.verbose:
            logger.info("Monitoring terminated. Time: {:.4f}s"
                        .format(self._time_total))

        # Completed successfully
        self._is_fitted = True

        return self

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

        n_vars = np.count_nonzero(self.scorecard.binning_process_._support)
        dict_user_options = self.get_params(deep=False)

        print_monitoring_information(print_level, self._n_records_a.sum(),
                                     self._n_records_e.sum(), n_vars,
                                     self._target_dtype, self._time_total,
                                     self._time_system,
                                     self._time_variable,
                                     dict_user_options)

    def system_stability_report(self):
        """Print overview information and statistics about system stability.
        It includes qualitative suggestions regarding the necessity of
        scorecard updates.
        """
        self._check_is_fitted()

        print_system_report(self._df_psi, self._df_tests,
                            self._df_target_analysis, self._df_performance)

    def psi_table(self):
        """System Population Stability Index (PSI) table.

        Returns
        -------
        psi_table : pandas.DataFrame
        """
        self._check_is_fitted()

        return self._df_psi

    def psi_variable_table(self, name=None, style="summary"):
        """Population Stability Index (PSI) at variable level.

        Parameters
        ----------
        name : str or None (default=None)
            The variable name. If name is None, a table with all variables
            is returned.

        style : str, optional (default="summary")
            Supported styles are "summary" and "detailed". Summary only
            includes the total PSI for each variable. Detailed includes the
            PSI for each variable at bin level.

        Returns
        -------
        psi_table : pandas.DataFrame
        """
        self._check_is_fitted()

        if style not in ("summary", "detailed"):
            raise ValueError('Invalid value for style. Allowed string '
                             'values are "summary" and "detailed".')

        if name is not None:
            variables = self.scorecard.binning_process_.get_support(names=True)

            if name not in variables:
                raise ValueError("name {} does not match a binned variable "
                                 "included in the provided scorecard."
                                 .format(name))

            dv = self._df_psi_variable[self._df_psi_variable.Variable == name]

            if style == "summary":
                return dv.groupby(["Variable"])["PSI"].sum()
            else:
                return dv

        if style == "summary":
            return pd.DataFrame(
                self._df_psi_variable.groupby(['Variable'])['PSI'].sum()
                ).reset_index()
        elif style == "detailed":
            return self._df_psi_variable

    def tests_table(self):
        """Compute statistical tests to determine if event rate (Chi-square
        test - binary target) or mean (Student's t-test - continuous target)
        are significantly different. Null hypothesis (actual == expected).

        Returns
        -------
        tests_table : pandas.DataFrame
        """
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

        fig, ax1 = plt.subplots()

        n_bins = len(self._n_records_a)
        indices = np.arange(n_bins)
        width = np.min(np.diff(indices))/3

        p_records_a = self._n_records_a / self._n_records_a.sum() * 100.0
        p_records_e = self._n_records_e / self._n_records_e.sum() * 100.0

        p1 = ax1.bar(indices-width, p_records_a, width, color='tab:red',
                     label="Records Actual", alpha=0.75)
        p2 = ax1.bar(indices, p_records_e, width, color='tab:blue',
                     label="Records Expected", alpha=0.75)

        handles = [p1[0], p2[0]]
        labels = ['Actual', 'Expected']

        ax1.set_xlabel("Bin ID", fontsize=12)
        ax1.set_ylabel("Population distribution", fontsize=13)
        ax1.yaxis.set_major_formatter(mtick.PercentFormatter())

        ax2 = ax1.twinx()

        if self._target_dtype == "binary":
            metric_label = "Event rate"
        elif self._target_dtype == "continuous":
            metric_label = "Mean"

        ax2.plot(indices, self._metric_a, linestyle="solid", marker="o",
                 color='tab:red')
        ax2.plot(indices, self._metric_e,  linestyle="solid", marker="o",
                 color='tab:blue')

        ax2.set_ylabel(metric_label, fontsize=13)
        ax2.xaxis.set_major_locator(mtick.MultipleLocator(1))

        ax2.set_xlim(-width * 2, n_bins - width * 2)

        plt.legend(handles, labels, loc="upper center",
                   bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=12)

        if savefig is None:
            plt.show()
        else:
            if not isinstance(savefig, str):
                raise TypeError("savefig must be a string path; got {}."
                                .format(savefig))
            plt.savefig(savefig)
            plt.close()

    def _fit_system(self, X_actual, y_actual, X_expected, y_expected):
        if self._target_dtype == "binary":
            problem_type = "classification"
        else:
            problem_type = "regression"

        score_actual = self.scorecard.score(X_actual)
        score_expected = self.scorecard.score(X_expected)

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
            self._system_performance_binary(
                X_actual, y_actual, X_expected, y_expected)
        else:
            self._system_performance_continuous(
                X_actual, y_actual, X_expected, y_expected)

        self._splits = splits
        self._n_records_a = n_records_a
        self._n_records_e = n_records_e

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

        self._metric_a = event_rate_a
        self._metric_e = event_rate_e

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

        self._metric_a = mean_a
        self._metric_e = mean_e

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

    def _system_performance_binary(self, X_actual, y_actual, X_expected,
                                   y_expected):
        # Metrics derived from confusion matrix
        y_true_a = y_actual
        y_pred_a = self.scorecard.predict(X_actual)
        d_metrics_a = imbalanced_classification_metrics(y_true_a, y_pred_a)

        y_true_e = y_expected
        y_pred_e = self.scorecard.predict(X_expected)
        d_metrics_e = imbalanced_classification_metrics(y_true_e, y_pred_e)

        metric_names = list(d_metrics_a.keys())
        metrics_a = list(d_metrics_a.values())
        metrics_e = list(d_metrics_e.values())

        # Gini
        y_pred_proba_a = self.scorecard.predict_proba(X_actual)[:, 1]
        gini_a = gini(y_true_a, y_pred_proba_a)

        y_pred_proba_e = self.scorecard.predict_proba(X_expected)[:, 1]
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

    def _system_performance_continuous(self, X_actual, y_actual, X_expected,
                                       y_expected):
        y_true_a = y_actual
        y_pred_a = self.scorecard.predict(X_actual)
        d_metrics_a = regression_metrics(y_true_a, y_pred_a)

        y_true_e = y_expected
        y_pred_e = self.scorecard.predict(X_expected)
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

    def _fit_variables(self, X_actual, X_expected):
        variables = self.scorecard.binning_process_.get_support(names=True)
        sc_table = self.scorecard.table()

        l_df_psi = []

        for name in variables:
            optb = self.scorecard.binning_process_.get_binned_variable(name)
            ta = optb.transform(X_actual[name], metric="indices")
            te = optb.transform(X_expected[name], metric="indices")

            n_bins = te.max() + 1

            n_records_a = np.empty(n_bins).astype(np.int64)
            n_records_e = np.empty(n_bins).astype(np.int64)

            for i in range(n_bins):
                n_records_a[i] = np.count_nonzero(ta == i)
                n_records_e[i] = np.count_nonzero(te == i)

            t_n_records_a = n_records_a.sum()
            t_n_records_e = n_records_e.sum()
            p_records_a = n_records_a / t_n_records_a
            p_records_e = n_records_e / t_n_records_e

            psi = jeffrey(p_records_a, p_records_e, return_sum=False)
            bins = sc_table[sc_table.Variable == name]["Bin"].values[:n_bins]

            df_psi = pd.DataFrame({
                "Variable": [name] * n_bins,
                "Bin": bins,
                "Count A": n_records_a,
                "Count E": n_records_e,
                "Count A (%)": p_records_a,
                "Count E (%)": p_records_e,
                "PSI": psi
                })

            l_df_psi.append(df_psi)

        self._df_psi_variable = pd.concat(l_df_psi)
        self._df_psi_variable.reset_index()

    def _check_is_fitted(self):
        if not self._is_fitted:
            raise NotFittedError("This {} instance is not fitted yet. Call "
                                 "'fit' with appropriate arguments."
                                 .format(self.__class__.__name__))

    @property
    def psi_splits(self):
        """List of splits points used to compute system PSI.

        Returns
        -------
        splits : numpy.ndarray
        """
        self._check_is_fitted()

        return self._splits
