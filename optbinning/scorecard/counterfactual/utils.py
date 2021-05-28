"""
Piecewise linear approximation of logistic function.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2021

import numpy as np

from ropwr import RobustPWRegression


def logistic_pw(min_p, max_p, n_bins):
    xl = np.linspace(min_p, max_p, 100)
    yl = (1.0 / (1 + np.exp(-xl)))

    splits = np.linspace(min_p, max_p, n_bins+1)[1:-1]

    pw = RobustPWRegression(objective="l1", degree=1, monotonic_trend=None)
    pw.fit(xl, yl, splits)

    splits = np.array([min_p] + list(splits) + [max_p])
    b_pw = [(splits[i], splits[i+1]) for i in range(len(splits) - 1)]
    c_pw = pw.coef_

    return b_pw, c_pw
