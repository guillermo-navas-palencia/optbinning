"""
Counterfactual problem data.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2021

import numpy as np


def problem_data(scorecard, X):
    s_vars = X.columns
    n_vars = X.shape[1]

    # Scorecard table
    sc = scorecard.table(style="detailed")

    if scorecard._target_dtype == "binary":
        sc["Points"] = sc["WoE"] * sc["Coefficient"]
    else:
        sc["Points"] = sc["Mean"] * sc["Coefficient"]

    # Linear model coefficients
    intercept = float(scorecard.estimator_.intercept_)
    coef = scorecard.estimator_.coef_.ravel()

    # Big-M parameters (min, max) points.
    # Proximity weights. Inverse value range for each feature
    min_p = 0
    max_p = 0
    wrange = np.empty(n_vars)

    for i, v in enumerate(s_vars):
        v_points = sc[sc["Variable"] == v]["Points"]
        _min = np.min(v_points)
        _max = np.max(v_points)
        min_p += _min
        max_p += _max

        wrange[i] = 1.0 / (_max - _min)

    min_p += intercept
    max_p += intercept

    # Mahalanobis distance
    Xt = scorecard.binning_process_.transform(X).values
    F = np.linalg.cholesky(np.linalg.inv(np.cov(Xt.T)))
    mu = Xt.mean(axis=0)

    return intercept, coef, min_p, max_p, wrange, F, mu
