"""
Optimal piecewise binning metrics.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

from sklearn.metrics import average_precision_score
from sklearn.metrics import brier_score_loss

from ...binning.metrics import jeffrey
from ...binning.metrics import jensen_shannon
from ...binning.metrics import hellinger
from ...binning.metrics import triangular
from ...metrics.classification import gini
from ...metrics.classification import ks
from ...metrics.regression import regression_metrics
from .transformations import transform_binary_target
from .transformations import transform_continuous_target


def _fun_divergence(fun, n, pi, qi, pi_special, qi_special, pi_missing,
                    qi_missing, flag_special, flag_missing):

    div_value = fun(pi, qi, return_sum=True) / n

    if flag_special:
        div_value += fun([pi_special], [qi_special])

    if flag_missing:
        div_value += fun([pi_missing], [qi_missing])

    return float(div_value)


def divergences_asymptotic(event_rate, n_nonevent_special, n_event_special,
                           n_nonevent_missing, n_event_missing, t_n_nonevent,
                           t_n_event):

    n = t_n_nonevent + t_n_event
    p = t_n_event / n

    pi = (1.0 - event_rate) / (1.0 - p)
    qi = event_rate / p

    flag_special = (n_event_special > 0 and n_nonevent_special > 0)
    flag_missing = (n_event_missing > 0 and n_nonevent_missing > 0)

    pi_special = n_nonevent_special / t_n_nonevent
    qi_special = n_event_special / t_n_event

    pi_missing = n_nonevent_missing / t_n_nonevent
    qi_missing = n_event_missing / t_n_event

    d_divergences = {}

    d_divergences["IV (Jeffrey)"] = _fun_divergence(
        jeffrey, n, pi, qi, pi_special, qi_special, pi_missing, qi_missing,
        flag_special, flag_missing)

    d_divergences["JS (Jensen-Shannon)"] = _fun_divergence(
        jensen_shannon, n, pi, qi, pi_special, qi_special, pi_missing,
        qi_missing, flag_special, flag_missing)

    d_divergences["Hellinger"] = _fun_divergence(
        hellinger, n, pi, qi, pi_special, qi_special, pi_missing, qi_missing,
        flag_special, flag_missing)

    d_divergences["Triangular"] = _fun_divergence(
        triangular, n, pi, qi, pi_special, qi_special, pi_missing, qi_missing,
        flag_special, flag_missing)

    return d_divergences


def binary_metrics(x, y, splits, c, t_n_nonevent, t_n_event,
                   n_nonevent_special, n_event_special, n_nonevent_missing,
                   n_event_missing, special_codes):

    d_metrics = {}

    # Metrics using predicted probability of Y=1.
    min_pred = 1e-8
    max_pred = 1 - min_pred

    event_rate = transform_binary_target(
        splits, x, c, min_pred, max_pred, t_n_nonevent, t_n_event,
        n_nonevent_special, n_event_special, n_nonevent_missing,
        n_event_missing, special_codes, "event_rate", "empirical", "empirical")

    d_metrics["Gini index"] = gini(y, event_rate)

    # Divergence metrics
    d_divergences = divergences_asymptotic(
        event_rate, n_nonevent_special, n_event_special, n_nonevent_missing,
        n_event_missing, t_n_nonevent, t_n_event)

    for dk, dv in d_divergences.items():
        d_metrics[dk] = dv

    d_metrics["KS"] = ks(y, event_rate)[0]
    d_metrics["Avg precision"] = average_precision_score(y, event_rate)
    d_metrics["Brier score"] = brier_score_loss(y, event_rate)

    return d_metrics


def continuous_metrics(x, y, splits, c, lb, ub, n_records_special, sum_special,
                       n_records_missing, sum_missing, special_codes):

    y_pred = transform_continuous_target(
        splits, x, c, lb, ub, n_records_special, sum_special,
        n_records_missing, sum_missing, special_codes, "empirical",
        "empirical")

    d_metrics = regression_metrics(y, y_pred)

    return d_metrics
