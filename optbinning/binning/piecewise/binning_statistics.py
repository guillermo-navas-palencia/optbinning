"""
Binning tables for optimal continuous binning.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import numbers

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from sklearn.exceptions import NotFittedError

from ...binning.binning_statistics import bin_str_format
from ...formatting import dataframe_to_string


class PWBinningTable:
    def __init__(self, name, splits, coef, n_nonevent, n_event, d_metrics):
        self.name = name
        self.splits = splits
        self.coef = coef
        self.n_nonevent = n_nonevent
        self.n_event = n_event
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

        n_records = n_event + n_nonevent
        t_n_nonevent = n_nonevent.sum()
        t_n_event = n_event.sum()
        t_n_records = t_n_nonevent + t_n_event
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
            totals = ["", t_n_records, 1, t_n_nonevent, t_n_event]
            totals += ["-"] * n_coefs
            df.loc["Totals"] = totals

        return df

    def plot(self, metric="woe", add_special=True, add_missing=True,
             savefig=None):
        pass

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
