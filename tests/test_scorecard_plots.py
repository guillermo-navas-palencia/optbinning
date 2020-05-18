"""
Scorecard plots testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import numpy as np

from pytest import raises

from optbinning.scorecard import plot_auc_roc
from optbinning.scorecard import plot_cap
from optbinning.scorecard import plot_ks


y = np.array([0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0])
y_pred = np.array([0.2, 0.1, 0.6, 0.3, 0.7, 0.2, 0.8, 0.1, 0.9, 0.7, 0.3])


def test_params():
    for plot in (plot_auc_roc, plot_cap, plot_ks):
        with raises(ValueError):
            y_pred_wrong = y_pred[:-1]
            plot(y, y_pred_wrong)

        with raises(TypeError):
            plot(y, y_pred, title=1)

        with raises(TypeError):
            plot(y, y_pred, xlabel=1)

        with raises(TypeError):
            plot(y, y_pred, ylabel=1)

        with raises(TypeError):
            plot(y, y_pred, savefig=1)

        with raises(TypeError):
            plot(y, y_pred, fname=1)

        with raises(ValueError):
            plot(y, y_pred, savefig=True, fname=None)


def test_savefig():
    for plot in (plot_auc_roc, plot_cap, plot_ks):
        plot(y, y_pred, savefig=True, fname="{}.png".format(plot.__name__))
