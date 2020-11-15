"""
Binning tables for optimal continuous binning.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import numbers

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.exceptions import NotFittedError

from ...binning.binning_statistics import bin_str_format
from ...formatting import dataframe_to_string
from .transformations import transform_binary_target


class PWBinningTable:
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
        n_nonevent = self.n_nonevent
        n_event = self.n_event

        self._t_n_nonevent = n_nonevent.sum()
        self._t_n_event = n_event.sum()
        n_records = n_event + n_nonevent
        t_n_records = self._t_n_nonevent + self._t_n_event
        p_records = n_records / t_n_records

        # Keep data for plotting
        self._n_records = n_records

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

                df["ER c{}".format(i)] = list(self.coef[:, i]) + c_s_m
            else:
                df["ER c{}".format(i)] = list(self.coef[:, i]) + [0, 0]

        if add_totals:
            totals = ["", t_n_records, 1, self._t_n_nonevent, self._t_n_event]
            totals += ["-"] * n_coefs
            df.loc["Totals"] = totals

        return df

    def plot(self, metric="woe", add_special=True, add_missing=True,
             n_samples=10000, savefig=None):

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

    def analysis(self, print_output=True):
        report = ""
        for metric, value in self.d_metrics.items():
            report += "    {:<20}{:>15.8f}\n".format(metric, value)

        print(report)


class PWContinuousBinningTable:
    def __init__(self, name, splits, coef, n_records, sums, min_target,
                 max_target, n_zeros):

        self.name = name
        self.splits = splits
        self.coef = coef
        self.n_records = n_records
        self.sums = sums
        self.min_target = min_target
        self.max_target = max_target
        self.n_zeros = n_zeros

    def build(self, show_digits=2, add_totals=True):
        pass

    def plot(self, add_special=True, add_missing=True, savefig=None):
        pass

    def analysis(self, print_output=True):
        pass
