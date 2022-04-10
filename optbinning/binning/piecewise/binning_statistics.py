"""
Binning tables for optimal continuous binning.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import numbers

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import stats

from ...binning.binning_statistics import _check_build_parameters
from ...binning.binning_statistics import _check_is_built
from ...binning.binning_statistics import bin_str_format
from ...binning.binning_statistics import BinningTable
from ...binning.metrics import bayesian_probability
from ...binning.metrics import binning_quality_score
from ...binning.metrics import continuous_binning_quality_score
from ...binning.metrics import chi2_cramer_v
from ...binning.metrics import frequentist_pvalue
from ...binning.metrics import hhi
from ...formatting import dataframe_to_string
from .transformations import transform_binary_target
from .transformations import transform_continuous_target


class PWBinningTable(BinningTable):
    """Piecewise binning table to summarize optimal binning of a numerical
    variable with respecto a binary target.

    Parameters
    ----------
    name : str, optional (default="")
        The variable name.

    splits : numpy.ndarray
        List of split points.

    coef : numpy.ndarray
        Coefficients for each bin.

    n_nonevent : numpy.ndarray
        Number of non-events.

    n_event : numpy.ndarray
        Number of events.

    min_x : float
        Mininum value of x.

    max_x : float
        Maxinum value of x.

    d_metrics : dict
        Dictionary of performance metrics.

    Warning
    -------
    This class is not intended to be instantiated by the user. It is
    preferable to use the class returned by the property ``binning_table``
    available in all optimal binning classes.
    """
    def __init__(self, name, splits, coef, n_nonevent, n_event, min_x, max_x,
                 d_metrics):
        self.name = name
        self.splits = splits
        self.coef = coef
        self.n_nonevent = n_nonevent
        self.n_event = n_event
        self.min_x = min_x
        self.max_x = max_x
        self.d_metrics = d_metrics

        self._n_records = None
        self._t_n_nonevent = None
        self._t_n_event = None
        self._hhi = None
        self._hhi_norm = None
        self._iv = None
        self._js = None
        self._gini = None
        self._quality_score = None
        self._ks = None

        self._is_built = False
        self._is_analyzed = False

    def build(self, show_digits=2, add_totals=True):
        """Build the binning table.

        Parameters
        ----------
        show_digits : int, optional (default=2)
            The number of significant digits of the bin column.

        add_totals : bool (default=True)
            Whether to add a last row with totals.

        Returns
        -------
        binning_table : pandas.DataFrame
        """
        _check_build_parameters(show_digits, add_totals)

        n_nonevent = self.n_nonevent
        n_event = self.n_event

        self._t_n_nonevent = n_nonevent.sum()
        self._t_n_event = n_event.sum()
        n_records = n_event + n_nonevent
        t_n_records = self._t_n_nonevent + self._t_n_event
        p_records = n_records / t_n_records

        mask = (n_event > 0) & (n_nonevent > 0)
        event_rate = np.zeros(len(n_records))
        event_rate[mask] = n_event[mask] / n_records[mask]

        self._n_records = n_records

        # Gini / IV / JS / hellinger / triangular / KS
        self._gini = self.d_metrics["Gini index"]
        self._iv = self.d_metrics["IV (Jeffrey)"]
        self._js = self.d_metrics["JS (Jensen-Shannon)"]
        self._hellinger = self.d_metrics["Hellinger"]
        self._triangular = self.d_metrics["Triangular"]
        self._ks = self.d_metrics["KS"]

        # Compute HHI
        self._hhi = hhi(p_records)
        self._hhi_norm = hhi(p_records, normalized=True)

        bins = np.concatenate([[-np.inf], self.splits, [np.inf]])
        bin_str = bin_str_format(bins, show_digits)

        bin_str.extend(["Special", "Missing"])

        df = pd.DataFrame({
            "Bin": bin_str,
            "Count": n_records,
            "Count (%)": p_records,
            "Non-event": n_nonevent,
            "Event": n_event
            })

        n_coefs = self.coef.shape[1]

        for i in range(n_coefs):
            if i == 0:
                n_nonevent_special = n_nonevent[-2]
                n_event_special = n_event[-2]

                if (n_event_special > 0) & (n_nonevent_special > 0):
                    event_rate_special = n_event_special / n_records[-2]
                else:
                    event_rate_special = 0

                n_nonevent_missing = n_nonevent[-1]
                n_event_missing = n_event[-1]

                if (n_event_missing > 0) & (n_nonevent_missing > 0):
                    event_rate_missing = n_event_missing / n_records[-1]
                else:
                    event_rate_missing = 0

                c_s_m = [event_rate_special, event_rate_missing]

                df["c{}".format(i)] = list(self.coef[:, i]) + c_s_m
            else:
                df["c{}".format(i)] = list(self.coef[:, i]) + [0, 0]

        if add_totals:
            totals = ["", t_n_records, 1, self._t_n_nonevent, self._t_n_event]
            totals += ["-"] * n_coefs
            df.loc["Totals"] = totals

        self._is_built = True

        return df

    def plot(self, metric="woe", add_special=True, add_missing=True,
             n_samples=10000, savefig=None):
        """Plot the binning table.

        Visualize the non-event and event count, and the predicted Weight of
        Evidence or event rate for each bin.

        Parameters
        ----------
        metric : str, optional (default="woe")
            Supported metrics are "woe" to show the Weight of Evidence (WoE)
            measure and "event_rate" to show the event rate.

        add_special : bool (default=True)
            Whether to add the special codes bin.

        add_missing : bool (default=True)
            Whether to add the special values bin.

        n_samples : int (default=10000)
            Number of samples to be represented.

        savefig : str or None (default=None)
            Path to save the plot figure.
        """
        if metric not in ("event_rate", "woe"):
            raise ValueError('Invalid value for metric. Allowed string '
                             'values are "event_rate" and "woe".')

        if not isinstance(add_special, bool):
            raise TypeError("add_special must be a boolean; got {}."
                            .format(add_special))

        if not isinstance(add_missing, bool):
            raise TypeError("add_missing must be a boolean; got {}."
                            .format(add_missing))

        _n_nonevent = self.n_nonevent[:-2]
        _n_event = self.n_event[:-2]

        n_splits = len(self.splits)

        y_pos = np.empty(n_splits + 2)
        y_pos[0] = self.min_x
        y_pos[1:-1] = self.splits
        y_pos[-1] = self.max_x

        width = y_pos[1:] - y_pos[:-1]
        y_pos = y_pos[:-1]

        fig, ax1 = plt.subplots()

        p2 = ax1.bar(y_pos, _n_event, width, color="tab:red", align="edge")
        p1 = ax1.bar(y_pos, _n_nonevent, width, color="tab:blue",
                     bottom=_n_event, align="edge")

        handles = [p1[0], p2[0]]
        labels = ['Non-event', 'Event']

        ax1.set_xlabel("x", fontsize=12)
        ax1.set_ylabel("Bin count", fontsize=13)
        ax1.tick_params(axis='x', labelrotation=45)

        ax2 = ax1.twinx()

        x_samples = np.linspace(self.min_x, self.max_x, n_samples)

        metric_values = transform_binary_target(
            self.splits, x_samples, self.coef, 0, 1, self._t_n_nonevent,
            self._t_n_event, 0, 0, 0, 0, [], metric, 0, 0)

        if metric == "woe":
            metric_label = "WoE"
        elif metric == "event_rate":
            metric_label = "Event rate"

        for split in self.splits:
            ax2.axvline(x=split, color="darkgrey", linestyle="--")

        ax2.plot(x_samples, metric_values, linestyle="solid", color="black")

        ax2.set_ylabel(metric_label, fontsize=13)

        plt.title(self.name, fontsize=14)
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

    def analysis(self, pvalue_test="chi2", n_samples=100, print_output=True):
        """Binning table analysis.

        Statistical analysis of the binning table, computing the statistics
        Gini index, Information Value (IV), Jensen-Shannon divergence, and
        the quality score. Additionally, several statistical significance tests
        between consecutive bins of the contingency table are performed: a
        frequentist test using the Chi-square test or the Fisher's exact test,
        and a Bayesian A/B test using the beta distribution as a conjugate
        prior of the Bernoulli distribution.

        Parameters
        ----------
        pvalue_test : str, optional (default="chi2")
            The statistical test. Supported test are "chi2" to choose the
            Chi-square test and "fisher" to choose the Fisher exact test.

        n_samples : int, optional (default=100)
            The number of samples to run the Bayesian A/B testing between
            consecutive bins to compute the probability of the event rate of
            bin A being greater than the event rate of bin B.

        print_output : bool (default=True)
            Whether to print analysis information.

        Notes
        -----
        The Chi-square test uses `scipy.stats.chi2_contingency
        <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.
        chi2_contingency.html>`_, and the Fisher exact test uses
        `scipy.stats.fisher_exact <https://docs.scipy.org/doc/scipy/reference/
        generated/scipy.stats.fisher_exact.html>`_.
        """
        _check_is_built(self)

        if pvalue_test not in ("chi2", "fisher"):
            raise ValueError('Invalid value for pvalue_test. Allowed string '
                             'values are "chi2" and "fisher".')

        if not isinstance(n_samples, numbers.Integral) or n_samples <= 0:
            raise ValueError("n_samples must be a positive integer; got {}."
                             .format(n_samples))

        # Significance tests
        n_bins = len(self._n_records)
        n_metric = n_bins - 2

        n_nev = self.n_nonevent[:n_metric]
        n_ev = self.n_event[:n_metric]

        if len(n_nev) >= 2:
            chi2, cramer_v = chi2_cramer_v(n_nev, n_ev)
        else:
            cramer_v = 0

        t_statistics = []
        p_values = []
        p_a_b = []
        p_b_a = []
        for i in range(n_metric-1):
            obs = np.array([n_nev[i:i+2] + 0.5, n_ev[i:i+2] + 0.5])
            t_statistic, p_value = frequentist_pvalue(obs, pvalue_test)
            pab, pba = bayesian_probability(obs, n_samples)

            p_a_b.append(pab)
            p_b_a.append(pba)

            t_statistics.append(t_statistic)
            p_values.append(p_value)

        # Quality score
        self._quality_score = binning_quality_score(self._iv, p_values,
                                                    self._hhi_norm)
        df_tests = pd.DataFrame({
                "Bin A": np.arange(n_metric-1),
                "Bin B": np.arange(n_metric-1) + 1,
                "t-statistic": t_statistics,
                "p-value": p_values,
                "P[A > B]": p_a_b,
                "P[B > A]": p_b_a
            })

        if pvalue_test == "fisher":
            df_tests.rename(columns={"t-statistic": "odd ratio"}, inplace=True)

        tab = 4
        if len(df_tests):
            df_tests_string = dataframe_to_string(df_tests, tab)
        else:
            df_tests_string = " " * tab + "None"

        # Metrics
        metrics_string = ""
        for km, kv in self.d_metrics.items():
            metrics_string += "    {:<19} {:>15.8f}\n".format(km, kv)

        report = (
            "---------------------------------------------\n"
            "OptimalBinning: Binary Binning Table Analysis\n"
            "---------------------------------------------\n"
            "\n"
            "  General metrics"
            "\n\n"
            "{}"
            "    HHI                 {:>15.8f}\n"
            "    HHI (normalized)    {:>15.8f}\n"
            "    Cramer's V          {:>15.8f}\n"
            "    Quality score       {:>15.8f}\n"
            "\n"
            "  Significance tests\n\n{}\n"
            ).format(metrics_string, self._hhi, self._hhi_norm, cramer_v,
                     self._quality_score, df_tests_string)

        if print_output:
            print(report)

        self._is_analyzed = True


class PWContinuousBinningTable:
    """Piecewise binning table to summarize optimal binning of a numerical
    variable with respect to a continuous target.

    Parameters
    ----------
    name : str, optional (default="")
        The variable name.

    splits : numpy.ndarray
        List of split points.

    coef : numpy.ndarray
        Coefficients for each bin.

    n_records : numpy.ndarray
        Number of records.

    sums : numpy.ndarray
        Target sums.

    stds : numpy.ndarray
        Target stds.

    min_target : numpy.ndarray
        Target mininum values.

    max_target : numpy.ndarray
        Target maxinum values.

    n_zeros : numpy.ndarray
        Number of zeros.

    min_x : float or None (default=None)
        Mininum value of x.

    max_x : float or None (default=None)
        Maxinum value of x.

    d_metrics : dict
        Dictionary of performance metrics.

    Warning
    -------
    This class is not intended to be instantiated by the user. It is
    preferable to use the class returned by the property ``binning_table``
    available in all optimal binning classes.
    """
    def __init__(self, name, splits, coef, n_records, sums, stds, min_target,
                 max_target, n_zeros, lb, ub, min_x, max_x, d_metrics):

        self.name = name
        self.splits = splits
        self.coef = coef
        self.n_records = n_records
        self.sums = sums
        self.stds = stds
        self.min_target = min_target
        self.max_target = max_target
        self.n_zeros = n_zeros

        self.lb = lb
        self.ub = ub
        self.min_x = min_x
        self.max_x = max_x
        self.d_metrics = d_metrics

        self._mean = None
        self._hhi = None
        self._hhi_norm = None

        self._is_built = False
        self._is_analyzed = False

    def build(self, show_digits=2, add_totals=True):
        """Build the binning table.

        Parameters
        ----------
        show_digits : int, optional (default=2)
            The number of significant digits of the bin column.

        add_totals : bool (default=True)
            Whether to add a last row with totals.

        Returns
        -------
        binning_table : pandas.DataFrame
        """
        _check_build_parameters(show_digits, add_totals)

        t_n_records = np.nansum(self.n_records)
        t_sum = np.nansum(self.sums)
        p_records = self.n_records / t_n_records

        mask = (self.n_records > 0)
        self._mean = np.zeros(len(self.n_records))
        self._mean[mask] = self.sums[mask] / self.n_records[mask]

        # Compute HHI
        self._hhi = hhi(p_records)
        self._hhi_norm = hhi(p_records, normalized=True)

        bins = np.concatenate([[-np.inf], self.splits, [np.inf]])
        bin_str = bin_str_format(bins, show_digits)

        bin_str.extend(["Special", "Missing"])

        df = pd.DataFrame({
            "Bin": bin_str,
            "Count": self.n_records,
            "Count (%)": p_records,
            "Sum": self.sums,
            "Std": self.stds,
            "Min": self.min_target,
            "Max": self.max_target,
            "Zeros count": self.n_zeros,
            })

        n_coefs = self.coef.shape[1]

        for i in range(n_coefs):
            if i == 0:
                n_records_special = self.n_records[-2]

                if n_records_special > 0:
                    meam_special = self.sums[-2] / self.n_records[-2]
                else:
                    meam_special = 0

                n_records_missing = self.n_records[-1]

                if n_records_missing > 0:
                    mean_missing = self.sums[-1] / self.n_records[-1]
                else:
                    mean_missing = 0

                c_s_m = [meam_special, mean_missing]

                df["c{}".format(i)] = list(self.coef[:, i]) + c_s_m
            else:
                df["c{}".format(i)] = list(self.coef[:, i]) + [0, 0]

        if add_totals:
            t_min = np.nanmin(self.min_target)
            t_max = np.nanmax(self.max_target)
            t_n_zeros = self.n_zeros.sum()
            totals = ["", t_n_records, 1, t_sum, "", t_min, t_max, t_n_zeros]
            totals += ["-"] * n_coefs
            df.loc["Totals"] = totals

        self._is_built = True

        return df

    def plot(self, add_special=True, add_missing=True, n_samples=10000,
             savefig=None):
        """Plot the binning table.

        Visualize records count and the prediction for each bin.

        Parameters
        ----------
        add_special : bool (default=True)
            Whether to add the special codes bin.

        add_missing : bool (default=True)
            Whether to add the special values bin.

        n_samples : int (default=10000)
            Number of samples to be represented.

        savefig : str or None (default=None)
            Path to save the plot figure.
        """
        _n_records = self.n_records[:-2]

        n_splits = len(self.splits)

        y_pos = np.empty(n_splits + 2)
        y_pos[0] = self.min_x
        y_pos[1:-1] = self.splits
        y_pos[-1] = self.max_x

        width = y_pos[1:] - y_pos[:-1]
        y_pos = y_pos[:-1]

        fig, ax1 = plt.subplots()

        p1 = ax1.bar(y_pos, _n_records, width, color="tab:blue", align="edge")

        handles = [p1[0]]
        labels = ['Count']

        ax1.set_xlabel("x", fontsize=12)
        ax1.set_ylabel("Bin count", fontsize=13)
        ax1.tick_params(axis='x', labelrotation=45)

        ax2 = ax1.twinx()

        x_samples = np.linspace(self.min_x, self.max_x, n_samples)

        metric_values = transform_continuous_target(
            self.splits, x_samples, self.coef, self.lb, self.ub, 0, 0, 0, 0,
            [], 0, 0)

        metric_label = "Mean"

        for split in self.splits:
            ax2.axvline(x=split, color="darkgrey", linestyle="--")

        ax2.plot(x_samples, metric_values, linestyle="solid", color="black")

        ax2.set_ylabel(metric_label, fontsize=13)

        plt.title(self.name, fontsize=14)
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

    def analysis(self, print_output=True):
        """Binning table analysis.

        Statistical analysis of the binning table, computing the Information
        Value (IV) and Herfindahl-Hirschman Index (HHI).

        Parameters
        ----------
        print_output : bool (default=True)
            Whether to print analysis information.
        """
        _check_is_built(self)

        # Significance tests
        n_bins = len(self.n_records)
        n_metric = n_bins - 2

        n_records = self.n_records[:n_metric]
        mean = self._mean[:n_metric]
        std = self.stds[:n_metric]

        t_statistics = []
        p_values = []

        for i in range(n_metric-1):
            u, u2 = mean[i], mean[i+1]
            s, s2 = std[i], std[i+1]
            r, r2 = n_records[i], n_records[i+1]

            t_statistic, p_value = stats.ttest_ind_from_stats(
                u, s, r, u2, s2, r2, False)

            t_statistics.append(t_statistic)
            p_values.append(p_value)

        df_tests = pd.DataFrame({
                "Bin A": np.arange(n_metric-1),
                "Bin B": np.arange(n_metric-1) + 1,
                "t-statistic": t_statistics,
                "p-value": p_values
            })

        tab = 4
        if len(df_tests):
            df_tests_string = dataframe_to_string(df_tests, tab)
        else:
            df_tests_string = " " * tab + "None"

        # Quality score
        self._quality_score = continuous_binning_quality_score(
            1e6, p_values, self._hhi_norm)

        # Metrics
        metrics_string = ""
        for km, kv in self.d_metrics.items():
            metrics_string += "    {:<21} {:>21.8f}\n".format(km, kv)

        report = (
            "-------------------------------------------------\n"
            "OptimalBinning: Continuous Binning Table Analysis\n"
            "-------------------------------------------------\n"
            "\n"
            "  General metrics"
            "\n\n"
            "{}"
            "    HHI                   {:>21.8f}\n"
            "    HHI (normalized)      {:>21.8f}\n"
            "    Quality score         {:>21.8f}\n"
            "\n"
            "  Significance tests\n\n{}\n"
            ).format(metrics_string, self._hhi, self._hhi_norm,
                     self._quality_score, df_tests_string)

        if print_output:
            print(report)
