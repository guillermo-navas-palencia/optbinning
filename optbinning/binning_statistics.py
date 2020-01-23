"""
Optimal binning algorithm.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import numbers

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from sklearn.exceptions import NotFittedError

from .metrics import bayesian_probability
from .metrics import binning_quality_score
from .metrics import frequentist_pvalue
from .metrics import gini
from .metrics import jeffrey
from .metrics import jensen_shannon
from .metrics import jensen_shannon_multivariate
from .metrics import multiclass_binning_quality_score


COLORS_RGB = [
    (215, 0, 0), (140, 60, 255), (2, 136, 0), (0, 172, 199), (152, 255, 0),
    (255, 127, 209), (108, 0, 79), (255, 165, 48), (0, 0, 157),
    (134, 112, 104), (0, 73, 66), (79, 42, 0), (0, 253, 207), (188, 183, 255),
    (149, 180, 122), (192, 4, 185), (37, 102, 162), (40, 0, 65),
    (220, 179, 175), (254, 245, 144), (80, 69, 91), (164, 124, 0),
    (255, 113, 102), (63, 129, 110), (130, 0, 13), (163, 123, 179),
    (52, 78, 0), (155, 228, 255), (235, 0, 119), (45, 0, 10), (94, 144, 255),
    (0, 199, 32), (88, 1, 170), (0, 30, 0), (154, 71, 0), (150, 159, 166),
    (155, 66, 92), (0, 31, 50), (200, 196, 0), (255, 208, 255), (0, 190, 154),
    (55, 21, 255), (45, 37, 37), (223, 88, 255), (190, 231, 192),
    (127, 69, 152), (82, 79, 60), (216, 102, 0), (100, 116, 56),
    (193, 115, 136), (110, 116, 138), (128, 157, 3), (190, 139, 101),
    (99, 51, 57), (202, 205, 218), (108, 235, 131), (34, 64, 105),
    (162, 127, 255), (254, 3, 203), (118, 188, 253), (217, 195, 130),
    (206, 163, 206), (109, 80, 0), (0, 105, 116), (71, 159, 94),
    (148, 198, 191), (249, 255, 0), (192, 84, 69), (0, 101, 60), (91, 80, 168),
    (83, 32, 100), (79, 95, 255), (126, 143, 119), (185, 8, 250),
    (139, 146, 195), (179, 0, 53), (136, 96, 126), (159, 0, 117),
    (255, 222, 196), (81, 8, 0), (26, 8, 0), (76, 137, 182), (0, 223, 223),
    (200, 255, 250), (48, 53, 21), (255, 39, 71), (255, 151, 170), (4, 0, 26),
    (201, 96, 177), (195, 162, 55), (124, 79, 58), (249, 158, 119),
    (86, 101, 100), (209, 147, 255), (45, 31, 105), (65, 27, 52),
    (175, 147, 152), (98, 158, 153), (255, 255, 255), (0, 0, 0)]


def bin_str_format(bins, show_digits):
    return ["[{0:.{2}f}, {1:.{2}f})".format(bins[i], bins[i+1], show_digits)
            for i in range(len(bins)-1)]


def bin_categorical(splits_categorical, categories, cat_others, user_splits):
    splits = np.ceil(splits_categorical).astype(np.int)
    n_categories = len(categories)
    indices = np.digitize(np.arange(n_categories), splits, right=False)
    n_bins = len(splits) + 1
    bins = []
    for i in range(n_bins):
        mask = (indices == i)
        bins.append(categories[mask])

    if user_splits is not None:
        new_bins = []
        for bin in bins:
            new_bin = []
            for cat in bin:
                new_bin.extend(list(cat))
            new_bins.append(new_bin)

        bins = new_bins

    if len(cat_others):
        bins.append(cat_others)

    return bins


def target_info(y, cl=0):
    if not len(y):
        return 0, 0
    else:
        y0 = (y == cl)
        n_nonevent = np.count_nonzero(y0)
        n_event = np.count_nonzero(~y0)

        return n_nonevent, n_event


def bin_info(solution, n_nonevent, n_event, n_nonevent_missing,
             n_event_missing, n_nonevent_special, n_event_special,
             n_nonevent_cat_others, n_event_cat_others, cat_others):

    n_nev = []
    n_ev = []
    accum_nev = 0
    accum_ev = 0
    for i, selected in enumerate(solution):
        if selected:
            n_nev.append(n_nonevent[i] + accum_nev)
            n_ev.append(n_event[i] + accum_ev)
            accum_nev = 0
            accum_ev = 0
        else:
            accum_nev += n_nonevent[i]
            accum_ev += n_event[i]

    if not len(solution):
        n_ev.append(n_event[0])
        n_nev.append(n_nonevent[0])

    if len(cat_others):
        n_nev.append(n_nonevent_cat_others)
        n_ev.append(n_event_cat_others)

    n_nev.append(n_nonevent_special)
    n_ev.append(n_event_special)

    n_nev.append(n_nonevent_missing)
    n_ev.append(n_event_missing)

    return np.array(n_nev).astype(np.int), np.array(n_ev).astype(np.int)


def multiclass_bin_info(solution, n_classes, n_event, n_event_missing,
                        n_event_special):
    n_ev = []
    accum_ev = np.zeros(n_classes)
    for i, selected in enumerate(solution):
        if selected:
            n_ev.append(n_event[i, :] + accum_ev)
            accum_ev = np.zeros(n_event.shape[1])
        else:
            accum_ev += n_event[i, :]

    if not len(solution):
        n_ev.append(n_event)

    n_ev.append(n_event_special)
    n_ev.append(n_event_missing)

    return np.array(n_ev).astype(np.int)


def continuous_bin_info(solution, n_records, sums, n_records_missing,
                        sum_missing, n_records_special, sum_special,
                        n_records_cat_others, sum_cat_others, cat_others):
    r = []
    s = []
    accum_r = 0
    accum_s = 0
    for i, selected in enumerate(solution):
        if selected:
            r.append(n_records[i] + accum_r)
            s.append(sums[i] + accum_s)
            accum_r = 0
            accum_s = 0
        else:
            accum_r += n_records[i]
            accum_s += sums[i]

    if not len(solution):
        r.append(n_records)
        s.append(sums)

    if len(cat_others):
        r.append(n_records_cat_others)
        s.append(sum_cat_others)

    r.append(n_records_special)
    s.append(sum_special)

    r.append(n_records_missing)
    s.append(sum_missing)

    return np.array(r).astype(np.int), np.array(s).astype(np.float)


def _check_build_parameters(show_digits, add_totals):
    if (not isinstance(show_digits, numbers.Integral) or
            not 0 <= show_digits <= 8):
        raise ValueError("show_digits must be an integer in [0, 8]; "
                         "got {}.".format(show_digits))

    if not isinstance(add_totals, bool):
        raise TypeError("add_totals must be a boolean; got {}."
                        .format(add_totals))


class BinningTable:
    """Binning table to summarize optimal binning of a numerical or categorical
    variable with respect to a binary target.

    Parameters
    ----------
    name : str, optional (default="")
        The variable name.

    dtype : str, optional (default="numerical")
        The variable data type. Supported data types are "numerical" for
        continuous and ordinal variables and "categorical" for categorical
        and nominal variables.

    splits : numpy.ndarray
        List of split points.

    n_nonevent : numpy.ndarray
        Number of non-events.

    n_event : numpy.ndarray
        Number of events.

    categories : list, numpy.ndarray or None, optional (default=None)
        List of categories.

    cat_others : list, numpy.ndarray or None, optional (default=None)
        List of categories in others' bin.

    user_splits: numpy.ndarray
        List of split points pass if prebins were passed by the user.

    Warning
    -------
    This class is not intended to be instantiated by the user. It is
    preferable to use the class returned by the property ``binning_table``
    available in all optimal binning classes.
    """
    def __init__(self, name, dtype, splits, n_nonevent, n_event,
                 categories=None, cat_others=None, user_splits=None):

        self.name = name
        self.dtype = dtype
        self.splits = splits
        self.n_nonevent = n_nonevent
        self.n_event = n_event
        self.categories = categories
        self.cat_others = cat_others if cat_others is not None else []
        self.user_splits = user_splits

        self._n_records = None
        self._event_rate = None
        self._woe = None
        self._iv = None
        self._js = None
        self._gini = None
        self._quality_score = None

        self._is_built = False

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

        n_records = n_event + n_nonevent
        t_n_nonevent = n_nonevent.sum()
        t_n_event = n_event.sum()
        t_n_records = t_n_nonevent + t_n_event
        t_event_rate = t_n_event / t_n_records

        p_records = n_records / t_n_records
        p_event = n_event / t_n_event
        p_nonevent = n_nonevent / t_n_nonevent

        mask = (n_event > 0) & (n_nonevent > 0)
        event_rate = np.zeros(len(n_records))
        woe = np.zeros(len(n_records))
        iv = np.zeros(len(n_records))
        js = np.zeros(len(n_records))

        # Compute weight of evidence and event rate
        event_rate[mask] = n_event[mask] / n_records[mask]
        constant = np.log(t_n_event / t_n_nonevent)
        woe[mask] = np.log(1 / event_rate[mask] - 1) + constant

        # Compute divergence measures
        p_ev = p_event[mask]
        p_nev = p_nonevent[mask]

        iv[mask] = jeffrey(p_ev, p_nev, return_sum=False)
        js[mask] = jensen_shannon(p_ev, p_nev, return_sum=False)
        t_iv = iv.sum()
        t_js = js.sum()

        self._iv = t_iv
        self._js = t_js

        # Keep data for plotting
        self._n_records = n_records
        self._event_rate = event_rate
        self._woe = woe

        if self.dtype == "numerical":
            bins = np.concatenate([[-np.inf], self.splits, [np.inf]])
            bin_str = bin_str_format(bins, show_digits)
        else:
            bin_str = bin_categorical(self.splits, self.categories,
                                      self.cat_others, self.user_splits)

        bin_str.extend(["Special", "Missing"])

        df = pd.DataFrame({
            "Bin": bin_str,
            "Count": n_records,
            "Count (%)": p_records,
            "Non-event": n_nonevent,
            "Event": n_event,
            "Event rate": event_rate,
            "WoE": woe,
            "IV": iv,
            "JS": js
            })

        if add_totals:
            totals = ["", t_n_records, 1, t_n_nonevent, t_n_event,
                      t_event_rate, "", t_iv, t_js]
            df.loc["Totals"] = totals

        self._is_built = True

        return df

    def plot(self, metric="woe"):
        """Plot the binning table.

        Visualize the non-event and event count, and the Weight of Evidence or
        the event rate for each bin.

        Parameters
        ----------
        metric : str, optional (default="woe")
            Supported metrics are "woe" to show the Weight of Evidence (WoE)
            measure and "event_rate" to show the event rate.
        """
        self._check_is_built()

        if metric not in ("event_rate", "woe"):
            raise ValueError('Invalid value for metric. Allowed string '
                             'values are "event_rate" and "woe".')

        n_bins = len(self._n_records)
        n_metric = n_bins - 2

        if len(self.cat_others):
            n_metric -= 1

        fig, ax1 = plt.subplots()

        p2 = ax1.bar(range(n_bins), self.n_event, color="tab:red")
        p1 = ax1.bar(range(n_bins), self.n_nonevent, color="tab:blue",
                     bottom=self.n_event)

        handles = [p1[0], p2[0]]
        labels = ['Non-event', 'Event']

        ax1.set_xlabel("Bin ID", fontsize=12)
        ax1.set_ylabel("Bin count", fontsize=13)

        ax2 = ax1.twinx()

        if metric == "woe":
            metric_values = self._woe
            metric_label = "WoE"
        elif metric == "event_rate":
            metric_values = self._event_rate
            metric_label = "Event rate"

        ax2.plot(range(n_metric), metric_values[:n_metric], linestyle="solid",
                 marker="o", color="black")

        # Add points for others (optional), special and missing bin
        if len(self.cat_others):
            pos_others = n_metric
            pos_special = n_metric + 1
            pos_missing = n_metric + 2

            p1[pos_others].set_alpha(0.5)
            p2[pos_others].set_alpha(0.5)

            ax2.plot(pos_others, metric_values[pos_others], marker="o",
                     color="black")
        else:
            pos_special = n_metric
            pos_missing = n_metric + 1

        p1[pos_special].set_hatch("/")
        p2[pos_special].set_hatch("/")
        p1[pos_missing].set_hatch("\\")
        p2[pos_missing].set_hatch("\\")

        handle_special = mpatches.Patch(hatch="/", alpha=0.1)
        label_special = "Bin special"

        handle_missing = mpatches.Patch(hatch="\\", alpha=0.1)
        label_missing = "Bin missing"

        handles.extend([handle_special, handle_missing])
        labels.extend([label_special, label_missing])

        ax2.plot(pos_special, metric_values[pos_special], marker="o",
                 color="black")
        ax2.plot(pos_missing, metric_values[pos_missing], marker="o",
                 color="black")

        ax2.set_ylabel(metric_label, fontsize=13)
        ax2.xaxis.set_major_locator(mtick.MultipleLocator(1))

        plt.title(self.name, fontsize=14)
        plt.legend(handles, labels, loc="upper center",
                   bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=12)
        plt.show()

    def analysis(self, pvalue_test="chi2", n_samples=100):
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

        Notes
        -----
        The Chi-square test uses `scipy.stats.chi2_contingency
        <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.
        chi2_contingency.html>`_, and the Fisher exact test uses
        `scipy.stats.fisher_exact <https://docs.scipy.org/doc/scipy/reference/
        generated/scipy.stats.fisher_exact.html>`_.
        """
        self._check_is_built()

        if pvalue_test not in ("chi2", "fisher"):
            raise ValueError('Invalid value for pvalue_test. Allowed string '
                             'values are "chi2" and "fisher".')

        if not isinstance(n_samples, numbers.Integral) or n_samples <= 0:
            raise ValueError("n_samples must be a positive integer; got {}."
                             .format(n_samples))

        # General metrics
        gini_ar = gini(self.n_event, self.n_nonevent)
        self._gini = gini_ar

        # Significance tests
        n_bins = len(self._n_records)
        n_metric = n_bins - 2

        if len(self.cat_others):
            n_metric -= 1

        n_nev = self.n_nonevent[:n_metric]
        n_ev = self.n_event[:n_metric]

        t_statistics = []
        p_values = []
        p_a_b = []
        p_b_a = []
        for i in range(n_metric-1):
            obs = np.array([n_nev[i:i+2], n_ev[i:i+2]])
            t_statistic, p_value = frequentist_pvalue(obs, pvalue_test)
            pab, pba = bayesian_probability(obs, n_samples)

            p_a_b.append(pab)
            p_b_a.append(pba)

            t_statistics.append(t_statistic)
            p_values.append(p_value)

        # Quality score
        self._quality_score = binning_quality_score(self._iv, p_values)

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

        tab = "    "
        if len(df_tests):
            df_tests_string = tab + df_tests.to_string(index=False).replace(
                "\n", "\n"+tab)
        else:
            df_tests_string = tab+"None"

        report = (
            "---------------------------------------------\n"
            "OptimalBinning: Binary Binning Table Analysis\n"
            "---------------------------------------------\n"
            "\n"
            "  General metrics"
            "\n\n"
            "    Gini index          {:>15.8f}\n"
            "    IV (Jeffrey)        {:>15.8f}\n"
            "    JS (Jensen-Shannon) {:>15.8f}\n"
            "    Quality score       {:>15.8f}\n"
            "\n"
            "  Significance tests\n\n{}\n"
            ).format(gini_ar, self._iv, self._js, self._quality_score,
                     df_tests_string)

        print(report)

    def _check_is_built(self):
        if not self._is_built:
            raise NotFittedError("This {} instance is not built yet. Call "
                                 "'build' with appropriate arguments."
                                 .format(self.__class__.__name__))

    @property
    def gini(self):
        """The Gini coefficient or Accuracy Ratio.

        The Gini coefficient is a quantitative measure of the discriminatory
        and predictive power of a variable. The Gini coefficient ranges from 0
        to 1.

        Returns
        -------
        gini : float
        """
        self._check_is_built()

        return self._gini

    @property
    def iv(self):
        """The Information Value (IV) or Jeffrey's divergence measure.

        The IV ranges from 0 to Infinity.

        Returns
        -------
        iv : float
        """
        self._check_is_built()

        return self._iv

    @property
    def js(self):
        r"""The Jensen-Shannon divergence measure (JS).

        The JS ranges from 0 to :math:`\log(2)`.

        Returns
        -------
        js : float
        """
        self._check_is_built()

        return self._js

    @property
    def quality_score(self):
        """The quality score (QS).

        The QS is a rating of the quality and discriminatory power of a
        variable. The QS ranges from 0 to 1.

        Returns
        -------
        quality_score : float
        """
        self._check_is_built()

        return self._quality_score


class MulticlassBinningTable:
    """Binning table to summarize optimal binning of a numerical variable with
    respect to a multiclass or multilabel target.

    Parameters
    ----------
    name : str, optional (default="")
        The variable name.

    splits : numpy.ndarray
        List of split points.

    n_event : numpy.ndarray
        Number of events.

    classes : array-like
        List of classes.

    Warning
    -------
    This class is not intended to be instantiated by the user. It is
    preferable to use the class returned by the property ``binning_table``
    available in all optimal binning classes.
    """
    def __init__(self, name, splits, n_event, classes):
        self.name = name
        self.splits = splits
        self.n_event = n_event
        self.classes = classes

        self._n_records = None
        self._event_rate = None
        self._js = None
        self._quality_score = None

        self._is_built = False

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

        n_event = self.n_event

        n_records = n_event.sum(axis=1)
        t_n_records = n_records.sum()

        mask = (n_event > 0)
        event_rate = np.zeros((len(n_records), len(self.classes)))

        for i in range(len(self.classes)):
            event_rate[mask[:, i], i] = n_event[
                mask[:, i], i] / n_records[mask[:, i]]

        # Keep data for plotting
        self._n_records = n_records
        self._event_rate = event_rate

        bins = np.concatenate([[-np.inf], self.splits, [np.inf]])
        bin_str = bin_str_format(bins, show_digits)
        bin_str.extend(["Special", "Missing"])

        dict_event = {"Event_{0}".format(cl): n_event[:, i]
                      for i, cl in enumerate(self.classes)}

        dict_p_event = {"Event_rate_{0}".format(cl): event_rate[:, i]
                        for i, cl in enumerate(self.classes)}

        dict_data = {**{"Bin": bin_str,
                        "Count": n_records,
                        "Count (%)": n_records / t_n_records},
                     **{**dict_event, **dict_p_event}}

        df = pd.DataFrame(dict_data)

        if add_totals:
            t_n_events = self.n_event.sum(axis=0)
            t_n_event_rate_class = t_n_events / t_n_records
            totals = ["", t_n_records, 1] + list(t_n_events)
            totals += list(t_n_event_rate_class)
            df.loc["Totals"] = totals

        self._is_built = True

        return df

    def plot(self):
        """Plot the binning table.

        Visualize event count and event rate values for each class.
        """
        self._check_is_built()

        n_bins = len(self._n_records)
        n_metric = n_bins - 2
        n_classes = len(self.classes)

        fig, ax1 = plt.subplots()

        colors = COLORS_RGB[:n_classes]
        colors = [tuple(c / 255. for c in color) for color in colors]

        p = []
        cum_size = np.zeros(len(self.n_event))

        for i, cl in enumerate(self.classes):
            p.append(ax1.bar(range(n_bins), self.n_event[:, i],
                             color=colors[i], bottom=cum_size))
            cum_size += self.n_event[:, i]

        handles = [_p[0] for _p in p]
        labels = list(self.classes)

        ax1.set_xlabel("Bin ID", fontsize=12)
        ax1.set_ylabel("Bin count", fontsize=13)

        ax2 = ax1.twinx()

        metric_values = self._event_rate
        metric_label = "Event rate"

        for i, cl in enumerate(self.classes):
            ax2.plot(range(n_metric), metric_values[:n_metric, i],
                     linestyle="solid", marker="o", color="black",
                     markerfacecolor=colors[i], markeredgewidth=0.5)

        # Add points for special and missing bin
        pos_special = n_metric
        pos_missing = n_metric + 1

        for _p in p:
            _p[pos_special].set_hatch("/")
            _p[pos_missing].set_hatch("\\")

        handle_special = mpatches.Patch(hatch="/", alpha=0.1)
        label_special = "Bin special"

        handle_missing = mpatches.Patch(hatch="\\", alpha=0.1)
        label_missing = "Bin missing"

        handles.extend([handle_special, handle_missing])
        labels.extend([label_special, label_missing])

        for i, cl in enumerate(self.classes):
            ax2.plot(pos_special, metric_values[pos_special, i], marker="o",
                     color=colors[i])
            ax2.plot(pos_missing, metric_values[pos_missing, i], marker="o",
                     color=colors[i])

        ax2.set_ylabel(metric_label, fontsize=13)
        ax2.xaxis.set_major_locator(mtick.MultipleLocator(1))

        plt.title(self.name, fontsize=14)
        plt.legend(handles, labels, loc="upper center",
                   bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=12)
        plt.show()

    def analysis(self):
        """Binning table analysis.

        Statistical analysis of the binning table, computing the Jensen-shannon
        divergence and the quality score. Additionally, a statistical
        significance test between consecutive bins of the contingency table is
        performed using the Chi-square test.

        Notes
        -----
        The Chi-square test uses `scipy.stats.chi2_contingency
        <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.
        chi2_contingency.html>`_.
        """
        self._check_is_built()

        p_event = self.n_event / self.n_event.sum(axis=0)

        # General metrics
        self._js = jensen_shannon_multivariate(p_event)

        # Significance tests
        n_bins = len(self._n_records)
        n_metric = n_bins - 2

        n_ev = self.n_event[:n_metric, :]

        t_statistics = []
        p_values = []
        for i in range(n_metric-1):
            obs = n_ev[i:i+2, :]
            t_statistic, p_value = frequentist_pvalue(obs, "chi2")

            t_statistics.append(t_statistic)
            p_values.append(p_value)

        # Quality score
        self._quality_score = multiclass_binning_quality_score(
            self._js, len(self.classes), p_values)

        df_tests = pd.DataFrame({
                "Bin A": np.arange(n_metric-1),
                "Bin B": np.arange(n_metric-1) + 1,
                "t-statistic": t_statistics,
                "p-value": p_values
            })

        tab = "    "
        if len(df_tests):
            df_tests_string = tab + df_tests.to_string(index=False).replace(
                "\n", "\n"+tab)
        else:
            df_tests_string = tab+"None"

        report = (
            "-------------------------------------------------\n"
            "OptimalBinning: Multiclass Binning Table Analysis\n"
            "-------------------------------------------------\n"
            "\n"
            "  General metrics"
            "\n\n"
            "    JS (Jensen-Shannon) {:>15.8f}\n"
            "    Quality score       {:>15.8f}\n"
            "\n"
            "  Significance tests\n\n{}\n"
            ).format(self._js, self._quality_score, df_tests_string)

        print(report)

    def _check_is_built(self):
        if not self._is_built:
            raise NotFittedError("This {} instance is not built yet. Call "
                                 "'build' with appropriate arguments."
                                 .format(self.__class__.__name__))

    @property
    def js(self):
        r"""The Jensen-Shannon divergence measure (JS).

        The JS ranges from 0 to :math:`\log(n_{classes})`.

        Returns
        -------
        js : float
        """
        self._check_is_built()

        return self._js

    @property
    def quality_score(self):
        """The quality score (QS).

        The QS is a rating of the quality and discriminatory power of a
        variable. The QS ranges from 0 to 1.

        Returns
        -------
        quality_score : float
        """
        self._check_is_built()

        return self._quality_score


class ContinuousBinningTable:
    """Binning table to summarize optimal binning of a numerical or categorical
    variable with respect to a continuous target.

    Parameters
    ----------
    name : str, optional (default="")
        The variable name.

    dtype : str, optional (default="numerical")
        The variable data type. Supported data types are "numerical" for
        continuous and ordinal variables and "categorical" for categorical
        and nominal variables.

    splits : numpy.ndarray
        List of split points.

    n_records : numpy.ndarray
        Number of records.

    sums : numpy.ndarray
        Target sums.

    categories : list, numpy.ndarray or None, optional (default=None)
        List of categories.

    cat_others : list, numpy.ndarray or None, optional (default=None)
        List of categories in others' bin.

    user_splits: numpy.ndarray
        List of split points pass if prebins were passed by the user.

    Warning
    -------
    This class is not intended to be instantiated by the user. It is
    preferable to use the class returned by the property ``binning_table``
    available in all optimal binning classes.
    """
    def __init__(self, name, dtype, splits, n_records, sums, categories=None,
                 cat_others=None, user_splits=None):

        self.name = name
        self.dtype = dtype
        self.splits = splits
        self.n_records = n_records
        self.sums = sums
        self.categories = categories
        self.cat_others = cat_others if cat_others is not None else []
        self.user_splits = user_splits

        self._mean = None

        self._is_built = False

    def build(self, show_digits=2, add_totals=True):
        """
        Build the binning table.

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

        t_n_records = self.n_records.sum()
        t_sum = self.sums.sum()
        t_mean = t_sum / t_n_records
        p_records = self.n_records / t_n_records

        mask = (self.n_records > 0)
        self._mean = np.zeros(len(self.n_records))
        self._mean[mask] = self.sums[mask] / self.n_records[mask]

        if self.dtype == "numerical":
            bins = np.concatenate([[-np.inf], self.splits, [np.inf]])
            bin_str = bin_str_format(bins, show_digits)
        else:
            bin_str = bin_categorical(self.splits, self.categories,
                                      self.cat_others, self.user_splits)

        bin_str.extend(["Special", "Missing"])

        df = pd.DataFrame({
            "Bin": bin_str,
            "Count": self.n_records,
            "Count (%)": p_records,
            "Sum": self.sums,
            "Mean": self._mean
            })

        if add_totals:
            totals = ["", t_n_records, 1, t_sum, t_mean]
            df.loc["Totals"] = totals

        self._is_built = True

        return df

    def plot(self):
        """Plot the binning table.

        Visualize records count and mean values.
        """
        self._check_is_built()

        n_bins = len(self.n_records)
        n_metric = n_bins - 2

        if len(self.cat_others):
            n_metric -= 1

        fig, ax1 = plt.subplots()

        p1 = ax1.bar(range(n_bins), self.n_records, color="tab:blue")

        handles = [p1[0]]
        labels = ['Count']

        ax1.set_xlabel("Bin ID", fontsize=12)
        ax1.set_ylabel("Bin count", fontsize=13)

        ax2 = ax1.twinx()

        metric_values = self._mean
        metric_label = "Mean"

        ax2.plot(range(n_metric), metric_values[:n_metric], linestyle="solid",
                 marker="o", color="black")

        # Add points for others (optional), special and missing bin
        if len(self.cat_others):
            pos_others = n_metric
            pos_special = n_metric + 1
            pos_missing = n_metric + 2

            p1[pos_others].set_alpha(0.5)

            ax2.plot(pos_others, metric_values[pos_others], marker="o",
                     color="black")
        else:
            pos_special = n_metric
            pos_missing = n_metric + 1

        p1[pos_special].set_hatch("/")
        p1[pos_missing].set_hatch("\\")

        handle_special = mpatches.Patch(hatch="/", alpha=0.1)
        label_special = "Bin special"

        handle_missing = mpatches.Patch(hatch="\\", alpha=0.1)
        label_missing = "Bin missing"

        handles.extend([handle_special, handle_missing])
        labels.extend([label_special, label_missing])

        ax2.plot(pos_special, metric_values[pos_special], marker="o",
                 color="black")
        ax2.plot(pos_missing, metric_values[pos_missing], marker="o",
                 color="black")

        ax2.set_ylabel(metric_label, fontsize=13)
        ax2.xaxis.set_major_locator(mtick.MultipleLocator(1))

        plt.title(self.name, fontsize=14)
        plt.legend(handles, labels, loc="upper center",
                   bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=12)
        plt.show()

    def _check_is_built(self):
        if not self._is_built:
            raise NotFittedError("This {} instance is not built yet. Call "
                                 "'build' with appropriate arguments."
                                 .format(self.__class__.__name__))
