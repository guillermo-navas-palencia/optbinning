"""
Binning transformations.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import numbers

import numpy as np
import pandas as pd

from sklearn.utils import check_array

from .binning_statistics import bin_categorical


def transform_event_rate_to_woe(event_rate, n_nonevent, n_event):
    """Transform event rate to WoE.

    Parameters
    ----------
    event_rate : array-like or float
        Event rate.

    n_nonevent : int
        Total number of non-events.

    n_event : int
        Total number of events.

    Returns
    -------
    woe : numpy.ndarray or float
        Weight of evidence.
    """
    return np.log((1. / event_rate - 1) * n_event / n_nonevent)


def transform_woe_to_event_rate(woe, n_nonevent, n_event):
    """Transform WoE to event rate.

    Parameters
    ----------
    woe : array-like or float
        Weight of evidence.

    n_nonevent : int
        Total number of non-events.

    n_event : int
        Total number of events.

    Returns
    -------
    event_rate : numpy.ndarray or float
        Event rate.
    """
    return 1.0 / (1.0 + n_nonevent / n_event * np.exp(woe))


def _check_metric_special_missing(metric_special, metric_missing):
    if isinstance(metric_special, str):
        if metric_special != "empirical":
            raise ValueError('Invalidad value for metric_special. Allowed '
                             'value "empirical"; got {}.'
                             .format(metric_special))

    elif not isinstance(metric_special, numbers.Number):
        raise ValueError('Invalidad value for metric_special. Allowed values '
                         'are "empirical" or a numeric value; got {}.'
                         .format(metric_special))

    if isinstance(metric_missing, str):
        if metric_missing != "empirical":
            raise ValueError('Invalidad value for metric_missing. Allowed '
                             'value "empirical"; got {}.'
                             .format(metric_missing))

    elif not isinstance(metric_missing, numbers.Number):
        raise ValueError('Invalidad value for metric_missing. Allowed values '
                         'are "empirical" or a numeric value; got {}.'
                         .format(metric_missing))


def transform_binary_target(splits, dtype, x, n_nonevent, n_event,
                            special_codes, categories, cat_others, metric,
                            metric_special, metric_missing, user_splits,
                            check_input=False):

    if metric not in ("event_rate", "woe"):
        raise ValueError('Invalid value for metric. Allowed string '
                         'values are "event_rate" and "woe".')

    _check_metric_special_missing(metric_special, metric_missing)

    if check_input:
        x = check_array(x, ensure_2d=False, dtype=None,
                        force_all_finite='allow-nan')

    x = np.asarray(x)

    if isinstance(x.dtype, object):
        missing_mask = pd.isnull(x)
    else:
        missing_mask = np.isinan(x)

    if special_codes is None:
        clean_mask = ~missing_mask
    else:
        special_mask = pd.Series(x).isin(special_codes).values
        clean_mask = ~missing_mask & ~special_mask

    x_clean = x[clean_mask]

    if dtype == "numerical":
        if len(splits):
            indices = np.digitize(x_clean, splits, right=False)
        else:
            indices = np.zeros(x_clean.shape)
        n_bins = len(splits) + 1
    else:
        bins = bin_categorical(splits, categories, cat_others, user_splits)
        n_bins = len(bins)

    # Compute event rate and WoE
    n_records = n_event + n_nonevent
    t_n_nonevent = n_nonevent.sum()
    t_n_event = n_event.sum()

    if "empirical" not in (metric_special, metric_missing):
        n_event = n_event[:n_bins]
        n_nonevent = n_nonevent[:n_bins]
        n_records = n_records[:n_bins]

    # default woe and event rate is 0
    mask = (n_event > 0) & (n_nonevent > 0)
    event_rate = np.zeros(len(n_records))
    woe = np.zeros(len(n_records))
    event_rate[mask] = n_event[mask] / n_records[mask]
    constant = np.log(t_n_event / t_n_nonevent)
    woe[mask] = np.log(1 / event_rate[mask] - 1) + constant

    if metric == "woe":
        metric_value = woe
    else:
        metric_value = event_rate

    x_transform = np.zeros(x.shape)

    if dtype == "numerical":
        x_clean_transform = np.zeros(x_clean.shape)
        for i in range(n_bins):
            mask = (indices == i)
            x_clean_transform[mask] = metric_value[i]

        x_transform[clean_mask] = x_clean_transform
    else:
        x_p = pd.Series(x)
        for i in range(n_bins):
            mask = x_p.isin(bins[i])
            x_transform[mask] = metric_value[i]

    if special_codes:
        if metric_special == "empirical":
            x_transform[special_mask] = metric_value[n_bins]
        else:
            x_transform[special_mask] = metric_special

    if metric_missing == "empirical":
        x_transform[missing_mask] = metric_value[n_bins + 1]
    else:
        x_transform[missing_mask] = metric_missing

    return x_transform


def transform_multiclass_target(splits, x, n_event, special_codes, metric,
                                metric_special, metric_missing,
                                check_input=False):

    if metric not in ("mean_woe", "weighted_mean_woe"):
        raise ValueError('Invalid value for metric. Allowed string '
                         'values are "mean_woe" and "weighted_mean_woe".')

    _check_metric_special_missing(metric_special, metric_missing)

    if check_input:
        x = check_array(x, ensure_2d=False, dtype=None,
                        force_all_finite='allow-nan')

    x = np.asarray(x)

    if isinstance(x.dtype, object):
        missing_mask = pd.isnull(x)
    else:
        missing_mask = np.isinan(x)

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

    # Build non-event to compute one-vs-all WoE
    n_classes = n_event.shape[1]
    n_records = np.tile(n_event.sum(axis=1), (n_classes, 1)).T
    n_nonevent = n_records - n_event
    t_n_nonevent = n_nonevent.sum(axis=0)
    t_n_event = n_event.sum(axis=0)

    if "empirical" not in (metric_special, metric_missing):
        n_event = n_event[:n_bins, :]
        n_nonevent = n_nonevent[:n_bins, :]
        n_records = n_records[:n_bins, :]

    event_rate = n_event / n_records
    woe = np.zeros(n_event.shape)

    for i in range(n_classes):
        woe[:,  i] = transform_event_rate_to_woe(
            event_rate[:, i], t_n_nonevent[i], t_n_event[i])

    if metric == "mean_woe":
        metric_value = woe.mean(axis=1)
    elif metric == "weighted_mean_woe":
        metric_value = np.average(woe, weights=t_n_event, axis=1)

    x_transform = np.zeros(x.shape)

    x_clean_transform = np.zeros(x_clean.shape)
    for i in range(n_bins):
        mask = (indices == i)
        x_clean_transform[mask] = metric_value[i]

    x_transform[clean_mask] = x_clean_transform

    if special_codes:
        if metric_special == "empirical":
            x_transform[special_mask] = metric_value[n_bins]
        else:
            x_transform[special_mask] = metric_special

    if metric_missing == "empirical":
        x_transform[missing_mask] = metric_value[n_bins + 1]
    else:
        x_transform[missing_mask] = metric_missing

    return x_transform


def transform_continuous_target(splits, dtype, x, n_records, sums,
                                special_codes, categories, cat_others,
                                metric_special, metric_missing, user_splits,
                                check_input):

    _check_metric_special_missing(metric_special, metric_missing)

    if check_input:
        x = check_array(x, ensure_2d=False, dtype=None,
                        force_all_finite='allow-nan')

    x = np.asarray(x)

    if isinstance(x.dtype, object):
        missing_mask = pd.isnull(x)
    else:
        missing_mask = np.isinan(x)

    if special_codes is None:
        clean_mask = ~missing_mask
    else:
        special_mask = pd.Series(x).isin(special_codes).values
        clean_mask = ~missing_mask & ~special_mask

    x_clean = x[clean_mask]

    if dtype == "numerical":
        if len(splits):
            indices = np.digitize(x_clean, splits, right=False)
        else:
            indices = np.zeros(x_clean.shape)
        n_bins = len(splits) + 1
    else:
        bins = bin_categorical(splits, categories, cat_others, user_splits)
        n_bins = len(bins)

    if "empirical" not in (metric_special, metric_missing):
        n_records = n_records[:n_bins]
        sums = sums[:n_bins]

    metric_value = sums / n_records

    x_transform = np.zeros(x.shape)

    if dtype == "numerical":
        x_clean_transform = np.zeros(x_clean.shape)
        for i in range(n_bins):
            mask = (indices == i)
            x_clean_transform[mask] = metric_value[i]

        x_transform[clean_mask] = x_clean_transform
    else:
        x_p = pd.Series(x)
        for i in range(n_bins):
            mask = x_p.isin(bins[i])
            x_transform[mask] = metric_value[i]

    if special_codes:
        if metric_special == "empirical":
            x_transform[special_mask] = metric_value[n_bins]
        else:
            x_transform[special_mask] = metric_special

    if metric_missing == "empirical":
        x_transform[missing_mask] = metric_value[n_bins + 1]
    else:
        x_transform[missing_mask] = metric_missing

    return x_transform
