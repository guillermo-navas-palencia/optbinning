"""
Piecewise binning transformations.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import numpy as np
import pandas as pd

from sklearn.utils import check_array

from ...binning.transformations import transform_event_rate_to_woe
from ...binning.transformations import _check_metric_special_missing


def transform_binary_target(splits, x, c, lb, ub, n_nonevent, n_event,
                            n_event_special, n_nonevent_special,
                            n_event_missing, n_nonevent_missing,
                            special_codes, metric, metric_special,
                            metric_missing, check_input=False):

    if metric not in ("event_rate", "woe"):
        raise ValueError('Invalid value for metric. Allowed string '
                         'values are "event_rate" and "woe".')

    _check_metric_special_missing(metric_special, metric_missing)

    if check_input:
        x = check_array(x, ensure_2d=False, dtype=None,
                        force_all_finite='allow-nan')

    x = np.asarray(x)

    missing_mask = np.isnan(x)

    if special_codes is None:
        clean_mask = ~missing_mask
    else:
        special_mask = pd.Series(x).isin(special_codes).values
        clean_mask = ~missing_mask & ~special_mask

    x_clean = x[clean_mask]

    if len(splits):
        indices = np.digitize(x_clean, splits, right=False)
    else:
        indices = np.zeros(x_clean.shape)

    n_bins = len(splits) + 1

    # Compute event rate for special and missing bin
    if metric_special == "empirical":
        n_records_special = n_event_special + n_nonevent_special

        if n_records_special > 0:
            event_rate_special = n_event_special / n_records_special
        else:
            event_rate_special = 0

        if metric == "woe":
            metric_special = transform_event_rate_to_woe(
                event_rate_special, n_nonevent, n_event)
        else:
            metric_special = event_rate_special

    if metric_missing == "empirical":
        n_records_missing = n_event_missing + n_nonevent_missing

        if n_records_missing > 0:
            event_rate_missing = n_event_missing / n_records_missing
        else:
            event_rate_missing = 0

        if metric == "woe":
            metric_missing = transform_event_rate_to_woe(
                event_rate_missing, n_nonevent, n_event)
        else:
            metric_missing = event_rate_missing

    x_transform = np.zeros(x.shape)
    x_clean_transform = np.zeros(x_clean.shape)

    for i in range(n_bins):
        mask = (indices == i)
        x_clean_transform[mask] = np.polyval(c[i, :][::-1], x_clean[mask])

    # Clip values using LB/UB
    bounded = (lb is not None or ub is not None)
    if bounded:
        x_clean_transform = np.clip(x_clean_transform, lb, ub)

    if metric == "woe":
        x_clean_transform = transform_event_rate_to_woe(
            x_clean_transform, n_nonevent, n_event)

    x_transform[clean_mask] = x_clean_transform

    if special_codes:
        if metric_special == "empirical":
            x_transform[special_mask] = metric_special
        else:
            x_transform[special_mask] = metric_special

    if metric_missing == "empirical":
        x_transform[missing_mask] = metric_missing
    else:
        x_transform[missing_mask] = metric_missing

    return x_transform


def transform_continuous_target(splits, x, c, lb, ub, n_records_special,
                                sum_special, n_records_missing, sum_missing,
                                special_codes, metric_special, metric_missing,
                                check_input=False):

    _check_metric_special_missing(metric_special, metric_missing)

    if check_input:
        x = check_array(x, ensure_2d=False, dtype=None,
                        force_all_finite='allow-nan')

    x = np.asarray(x)

    missing_mask = np.isnan(x)

    if special_codes is None:
        clean_mask = ~missing_mask
    else:
        special_mask = pd.Series(x).isin(special_codes).values
        clean_mask = ~missing_mask & ~special_mask

    x_clean = x[clean_mask]

    if len(splits):
        indices = np.digitize(x_clean, splits, right=False)
    else:
        indices = np.zeros(x_clean.shape)

    n_bins = len(splits) + 1

    # Compute event rate for special and missing bin
    if metric_special == "empirical":
        if n_records_special > 0:
            mean_special = sum_special / n_records_special
        else:
            mean_special = 0

        metric_special = mean_special

    if metric_missing == "empirical":
        if n_records_missing > 0:
            mean_missing = sum_missing / n_records_missing
        else:
            mean_missing = 0

        metric_missing = mean_missing

    x_transform = np.zeros(x.shape)
    x_clean_transform = np.zeros(x_clean.shape)

    for i in range(n_bins):
        mask = (indices == i)
        x_clean_transform[mask] = np.polyval(c[i, :][::-1], x_clean[mask])

    # Clip values using LB/UB
    bounded = (lb is not None or ub is not None)
    if bounded:
        x_clean_transform = np.clip(x_clean_transform, lb, ub)

    x_transform[clean_mask] = x_clean_transform

    if special_codes:
        if metric_special == "empirical":
            x_transform[special_mask] = metric_special
        else:
            x_transform[special_mask] = metric_special

    if metric_missing == "empirical":
        x_transform[missing_mask] = metric_missing
    else:
        x_transform[missing_mask] = metric_missing

    return x_transform
