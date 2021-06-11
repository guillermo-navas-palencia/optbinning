"""
Optimal binning algorithm 2D.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2021

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mpl_toolkits.axes_grid1 import make_axes_locatable

from ...formatting import dataframe_to_string
from ..binning_statistics import _check_build_parameters
from ..binning_statistics import _check_is_analyzed
from ..binning_statistics import _check_is_built
from ..metrics import bayesian_probability
from ..metrics import binning_quality_score
from ..metrics import chi2_cramer_v
from ..metrics import frequentist_pvalue
from ..metrics import hhi
from ..metrics import gini
from ..metrics import hellinger
from ..metrics import jeffrey
from ..metrics import jensen_shannon
from ..metrics import triangular


def _bin_fmt(bin, show_digits):
    if np.isinf(bin[0]):
        return "({0:.{2}f}, {1:.{2}f})".format(bin[0], bin[1], show_digits)
    else:
        return "[{0:.{2}f}, {1:.{2}f})".format(bin[0], bin[1], show_digits)

def bin_xy_str_format(bins_x, bins_y, show_digits):
    show_digits = 2 if show_digits is None else show_digits

    bins_xy = []
    for bx, by in zip(bins_x, bins_y):
        _bx = _bin_fmt(bx, show_digits)
        _by = _bin_fmt(by, show_digits)
        bins_xy.append(r"{} $\cup$ {}".format(_bx, _by))

    return bins_xy


def bin_str_format(bins, show_digits):
    show_digits = 2 if show_digits is None else show_digits

    bin_str = []
    for bin in bins:
        bin_str.append(_bin_fmt(bin, show_digits))

    return bin_str


class BinningTable2D:
    def __init__(self, name_x, name_y, dtype_x, dtype_y, splits_x, splits_y,
                 m, n, n_nonevent, n_event, event_rate, D, P,
                 categories_x=None, categories_y=None, cat_others_x=None,
                 cat_others_y=None, user_splits_x=None, user_splits_y=None):

        self.name_x = name_x
        self.name_y = name_y
        self.dtype_x = dtype_x
        self.dtype_y = dtype_y
        self.splits_x = splits_x
        self.splits_y = splits_y
        self.m = m
        self.n = n
        self.n_nonevent = n_nonevent
        self.n_event = n_event
        self.event_rate = event_rate
        self.D = D
        self.P = P

        self.categories_x = categories_x
        self.categories_y = categories_y
        self.cat_others_x = cat_others_x
        self.cat_others_y = cat_others_y

        self.user_splits_x = user_splits_x
        self.user_splits_y = user_splits_y

        self._is_built = False
        self._is_analyzed = False

    def build(self, show_digits=2, show_bin_xy=False, add_totals=True):
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
        W = np.log(1 / self.D - 1) + constant

        # Compute Gini
        self._gini = gini(self.n_event, self.n_nonevent)

        # Compute divergence measures
        p_ev = p_event[mask]
        p_nev = p_nonevent[mask]

        iv[mask] = jeffrey(p_ev, p_nev, return_sum=False)
        js[mask] = jensen_shannon(p_ev, p_nev, return_sum=False)
        t_iv = iv.sum()
        t_js = js.sum()

        self._iv = t_iv
        self._js = t_js
        self._hellinger = hellinger(p_ev, p_nev, return_sum=True)
        self._triangular = triangular(p_ev, p_nev, return_sum=True)

        # Keep data for plotting
        self._n_records = n_records
        self._event_rate = event_rate
        self._woe = woe
        self._W = W

        # Compute KS
        self._ks = np.abs(p_event.cumsum() - p_nonevent.cumsum()).max()

        # Compute HHI
        self._hhi = hhi(p_records)
        self._hhi_norm = hhi(p_records, normalized=True)        

        # Compute paths. This is required for both plot and analysis
        # paths x: horizontal
        self._paths_x = []
        for i in range(self.m):
            path = tuple(dict.fromkeys(self.P[i, :]))
            if not path in self._paths_x:
                self._paths_x.append(path)
        
        # paths y: vertical     
        self._paths_y = []
        for j in range(self.n):
            path = tuple(dict.fromkeys(self.P[:, j]))
            if not path in self._paths_y:
                self._paths_y.append(path)

        if show_bin_xy:
            bin_xy_str = bin_xy_str_format(self.splits_x, self.splits_y,
                                           show_digits)

            df = pd.DataFrame({
                "Bin": bin_xy_str,
                "Count": n_records,
                "Count (%)": p_records,
                "Non-event": n_nonevent,
                "Event": n_event,
                "Event rate": event_rate,
                "WoE": woe,
                "IV": iv,
                "JS": js
                })
        else:
            bin_x_str = bin_str_format(self.splits_x, show_digits)
            bin_y_str = bin_str_format(self.splits_y, show_digits)

            df = pd.DataFrame({
                "Bin x": bin_x_str,
                "Bin y": bin_y_str,
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
            if show_bin_xy:
                totals = ["", t_n_records, 1, t_n_nonevent, t_n_event,
                          t_event_rate, "", t_iv, t_js]
            else:
                totals = ["", "", t_n_records, 1, t_n_nonevent, t_n_event,
                          t_event_rate, "", t_iv, t_js]

            df.loc["Totals"] = totals

        self._is_built = True

        return df

    def plot(self, metric="woe", savefig=None):
        if metric == "woe":
            metric_values = self._woe
            metric_matrix = self._W
            metric_label = "WoE"
        elif metric == "event_rate":
            metric_values = self._event_rate
            metric_matrix = self.D
            metric_label = "Event rate"

        fig, ax = plt.subplots(figsize=(7, 7))

        divider = make_axes_locatable(ax)
        axtop = divider.append_axes("top", size=2.5, pad=0.1, sharex=ax)
        axright = divider.append_axes("right", size=2.5, pad=0.1, sharey=ax)
        # Hide x labels and tick labels for top plots and y ticks for
        # right plots.

        # Position [0, 0]
        for path in self._paths_x:
            er = sum([
                [metric_values[p]] * np.count_nonzero(
                    self.P == p, axis=1).max() for p in path], [])

            er = er + [er[-1]]
            axtop.step(np.arange(self.n + 1) - 0.5, er,
                       label=path, where="post")
            
        for i in range(self.n):
            axtop.axvline(i + 0.5, color="grey", linestyle="--", alpha=0.5)    

        axtop.get_xaxis().set_visible(False)
        axtop.set_ylabel(metric_label, fontsize=12)

        # Position [1, 0]
        pos = ax.matshow(metric_matrix, cmap=plt.cm.bwr)
        for j in range(self.n):
            for i in range(self.m):
                c = int(self.P[i, j])
                ax.text(j, i, str(c), va='center', ha='center')

        fig.colorbar(pos, ax=ax, orientation="horizontal",
                     fraction=0.025, pad=0.125)        

        ax.xaxis.set_label_position("bottom")
        ax.xaxis.tick_bottom()
        ax.set_ylabel("Bin ID - y ({})".format(self.name_x), fontsize=12)
        ax.set_xlabel("Bin ID - x ({})".format(self.name_y), fontsize=12)

        # Position [1, 1]
        for path in self._paths_y:
            er = sum([
                [metric_values[p]] * (np.count_nonzero(
                    self.P == p, axis=0).max()) for p in path], [])

            er = er + [er[-1]]
            axright.step(er, np.arange(self.m + 1) - 0.5, label=path,
                         where="pre")
                
        for j in range(self.m):
            axright.axhline(j -0.5, color="grey", linestyle="--", alpha=0.5)
            
        axright.get_yaxis().set_visible(False)
        axright.set_xlabel(metric_label, fontsize=12)

        #adjust margins
        axright.margins(y=0)
        axtop.margins(x=0)
        plt.tight_layout()

        axtop.legend(bbox_to_anchor=(1, 1))
        axright.legend(bbox_to_anchor=(1, 1))

        plt.show()

    def analysis(self, pvalue_test="chi2", n_samples=100, print_output=True):
        pairs = set()

        for path in self._paths_x:
            tpairs = tuple(zip(path[:-1], path[1:]))
            for tp in tpairs:
                pairs.add(tp)

        for path in self._paths_y:
            tpairs = tuple(zip(path[:-1], path[1:]))
            for tp in tpairs:
                pairs.add(tp)

        pairs = sorted(pairs)

        # Significance tests
        n_bins = len(self._n_records)
        n_metric = n_bins # -2

        # if len(self.cat_others):
        #     n_metric -= 1

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
        for pair in pairs:
            obs = np.array([n_nev[list(pair)], n_ev[list(pair)]])
            t_statistic, p_value = frequentist_pvalue(obs, pvalue_test)
            pab, pba = bayesian_probability(obs, n_samples)

            p_a_b.append(pab)
            p_b_a.append(pba)

            t_statistics.append(t_statistic)
            p_values.append(p_value)

        df_tests = pd.DataFrame({
                "Bin A": np.array([p[0] for p in pairs]),
                "Bin B": np.array([p[1] for p in pairs]),
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

        # Quality score
        self._quality_score = binning_quality_score(self._iv, p_values,
                                                    self._hhi_norm)

        report = (
            "------------------------------------------------\n"
            "OptimalBinning: Binary Binning Table 2D Analysis\n"
            "------------------------------------------------\n"
            "\n"
            "  General metrics"
            "\n\n"
            "    Gini index          {:>15.8f}\n"
            "    IV (Jeffrey)        {:>15.8f}\n"
            "    JS (Jensen-Shannon) {:>15.8f}\n"
            "    Hellinger           {:>15.8f}\n"
            "    Triangular          {:>15.8f}\n"
            "    KS                  {:>15.8f}\n"
            "    HHI                 {:>15.8f}\n"
            "    HHI (normalized)    {:>15.8f}\n"
            "    Cramer's V          {:>15.8f}\n"
            "    Quality score       {:>15.8f}\n"
            "\n"
            "  Significance tests\n\n{}\n"
            ).format(self._gini, self._iv, self._js, self._hellinger,
                     self._triangular, self._ks, self._hhi, self._hhi_norm,
                     cramer_v, self._quality_score, df_tests_string)

        if print_output:
            print(report)

        self._is_analyzed = True

    @property
    def iv(self):
        """The Information Value (IV) or Jeffrey's divergence measure.

        The IV ranges from 0 to Infinity.

        Returns
        -------
        iv : float
        """
        _check_is_built(self)

        return self._iv    
