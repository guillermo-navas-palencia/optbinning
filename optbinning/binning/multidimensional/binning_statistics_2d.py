"""
Optimal binning algorithm 2D.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2021

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..binning_statistics import bin_str_format
from ..binning_statistics import _check_build_parameters
from ..binning_statistics import _check_is_analyzed
from ..binning_statistics import _check_is_built
from ..metrics import gini
from ..metrics import jeffrey
from ..metrics import jensen_shannon


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

    def build(self, show_digits=2, add_totals=True, bin_x_y=False):
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

        bin_str = ["{} U {}".format(sx, sy) for sx, sy in zip(self.splits_x, self.splits_y)]

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

    def plot(self, metric="woe", savefig=None):
        # paths x: horizontal
        paths_x = []
        for i in range(self.m):
            path = tuple(dict.fromkeys(self.P[i, :]))
            if not path in paths_x:
                paths_x.append(path)
        
        # paths y: vertical     
        paths_y = []
        for j in range(self.n):
            path = tuple(dict.fromkeys(self.P[:, j]))
            if not path in paths_y:
                paths_y.append(path) 

        fig, ax = plt.subplots(figsize=(7, 7))

        divider = make_axes_locatable(ax)
        axtop = divider.append_axes("top", size=2.5, pad=0.1, sharex=ax)
        axright = divider.append_axes("right", size=2.5, pad=0.1, sharey=ax)
        # Hide x labels and tick labels for top plots and y ticks for
        # right plots.

        # Position [0, 0]
        for path in paths_x:
            er = sum([
                [self.event_rate[p]] * np.count_nonzero(
                    self.P == p, axis=1).max() for p in path], [])

            er = er + [er[-1]]
            axtop.step(np.arange(self.n + 1) - 0.5, er,
                       label=path, where="post")
            
        for i in range(self.m):
            axtop.axvline(i + 0.5, color="grey", linestyle="--", alpha=0.5)    

        axtop.get_xaxis().set_visible(False)
        axtop.set_ylabel("Event rate", fontsize=12)

        # Position [1, 0]
        pos = ax.matshow(self.D, cmap=plt.cm.bwr)
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
        for j, path in enumerate(paths_y):
            er = sum([
                [self.event_rate[p]] * (np.count_nonzero(
                    self.P == p, axis=0).max()) for p in path], [])

            er = er + [er[-1]]
            axright.step(er, np.arange(self.m + 1) - 0.5, label=path,
                         where="pre")
                
        for j in range(self.m):
            axright.axhline(j -0.5, color="grey", linestyle="--", alpha=0.5)
            
        axright.get_yaxis().set_visible(False)
        axright.set_xlabel("Event rate", fontsize=12)

        #adjust margins
        axright.margins(y=0)
        axtop.margins(x=0)
        plt.tight_layout()

        axtop.legend(bbox_to_anchor=(1, 1))
        axright.legend(bbox_to_anchor=(1, 1))

        plt.show()

    def analysis(self, print_output=True):
        pass

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
