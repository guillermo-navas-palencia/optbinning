"""
Population Stability Index (PSI)

References:
https://www.mdpi.com/2227-9091/7/2/53
https://mwburke.github.io/data%20science/2018/04/29/population-stability-index.html
https://www.lexjansen.com/wuss/2017/47_Final_Paper_PDF.pdf
https://www.listendata.com/2015/05/population-stability-index.html
http://ucanalytics.com/blogs/population-stability-index-psi-banking-case-study/
http://shichen.name/scorecard/reference/perf_psi.html
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020


import numpy as np
import pandas as pd

from ..binning.binning_statistics import bin_str_format
from ..binning.metrics import jeffrey
from ..binning.prebinning import PreBinning


class BinningMonitoring:
    def __init__(self):
        pass


class ScorecardMonitoring:
    def __init__(self, method="uniform", n_bins=20, min_bin_size=0.05,
                 show_digits=2):
        
        self.method = method
        self.n_bins = n_bins
        self.min_bin_size = min_bin_size
        self.show_digits = show_digits

        # auxiliary data
        self._splits = None
        self._n_records_a = None
        self._n_records_e = None
        self._t_n_records_a = None
        self._t_n_records_e = None

    def fit(self, score_actual, y_actual, score_expected, y_expected):
        # Fit computes all required information => no self._ are needed.

        prebinning = PreBinning(problem_type="classification",
                                method=self.method,
                                n_bins=self.n_bins,
                                min_bin_size=self.min_bin_size
                                ).fit(score_actual, y_actual)

        splits = prebinning.splits

        n_splits = len(splits)
        n_bins = n_splits + 1

        indices_a = np.digitize(score_actual, splits, right=True)
        indices_e = np.digitize(score_expected, splits, right=True)
        
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

        self._splits = splits

        self._n_records_a = n_nonevent_a + n_event_a
        self._n_records_e = n_nonevent_e + n_event_e
        self._t_n_records_a = self._n_records_a.sum()
        self._t_n_records_e = self._n_records_e.sum()

        self._n_nonevent_a = n_nonevent_a
        self._n_event_a = n_event_a
        self._n_nonevent_e = n_nonevent_e
        self._n_event_e = n_event_e

    def statistics(self):
        # Chi-squared test and KS over binned data.

        # Information PSI: total

        event_rate_a = self._n_event_a / n_records_a
        event_rate_e = self._n_event_e / n_records_e
        t_event_rate_a = self._n_event_a.sum() / self._t_n_records_a
        t_event_rate_e = self._n_event_e.sum() / self._t_n_records_e

    def psi(self):
        p_records_a = self._n_records_a / self._t_n_records_a
        p_records_e = self._n_records_e / self._t_n_records_e

        psi = jeffrey(p_records_a, p_records_e, return_sum=False)
        t_psi = psi.sum()
        self._t_psi = t_psi

        bins = np.concatenate([[-np.inf], self._splits, [np.inf]])
        bin_str = bin_str_format(bins, self.show_digits)

        df = pd.DataFrame({
            "Bin": bin_str,
            "Count A": self._n_records_a,
            "Count E": self._n_records_e,
            "Count (%) A": p_records_a,
            "Count (%) E": p_records_e,
            "PSI": psi
            })

        totals = ["", self._t_n_records_a, self._t_n_records_e, 1, 1, t_psi]
        df.loc["Totals"] = totals

        return df

    def plot_psi(self):
        pass

    @property
    def splits(self):
        self._splits
