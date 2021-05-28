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
from .binning_statistics import bin_str_format


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


def _check_show_digits(show_digits):
    if (not isinstance(show_digits, numbers.Integral) or
            not 0 <= show_digits <= 8):
        raise ValueError("show_digits must be an integer in [0, 8]; "
                         "got {}.".format(show_digits))


def transform_binary_target(splits, dtype, x, n_nonevent, n_event,
                            special_codes, categories, cat_others, metric,
                            metric_special, metric_missing, user_splits,
                            show_digits, check_input=False):

    if metric not in ("event_rate", "woe", "indices", "bins"):
        raise ValueError('Invalid value for metric. Allowed string '
                         'values are "event_rate", "woe", "indices" and '
                         '"bins".')

    _check_metric_special_missing(metric_special, metric_missing)
    _check_show_digits(show_digits)

    if check_input:
        x = check_array(x, ensure_2d=False, dtype=None,
                        force_all_finite='allow-nan')

    x = np.asarray(x)

    if np.issubdtype(x.dtype, np.number):
        missing_mask = np.isnan(x)
    else:
        missing_mask = pd.isnull(x)

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

        bins = np.concatenate([[-np.inf], splits, [np.inf]])
        bins_str = bin_str_format(bins, show_digits)
        n_bins = len(splits) + 1
    else:
        bins = bin_categorical(splits, categories, cat_others, user_splits)
        bins_str = [str(b) for b in bins]
        n_bins = len(bins)

    if metric in ("woe", "event_rate"):
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
    else:
        # Assign corresponding indices or bin intervals
        if metric == "indices":
            metric_value = np.arange(n_bins + 2)
            x_transform = np.full(x.shape, -1, dtype=int)
        elif metric == "bins":
            bins_str.extend(["Special", "Missing"])
            metric_value = bins_str
            x_transform = np.full(x.shape, "", dtype=object)

    if dtype == "numerical":
        if metric == "bins":
            x_clean_transform = np.full(x_clean.shape, "", dtype=object)
        else:
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
        if (metric_special == "empirical" or
            (metric == "indices" and not isinstance(metric_special, int)) or
                metric == "bins"):
            x_transform[special_mask] = metric_value[n_bins]
        else:
            x_transform[special_mask] = metric_special

    if (metric_missing == "empirical" or
        (metric == "indices" and not isinstance(metric_missing, int)) or
            metric == "bins"):
        x_transform[missing_mask] = metric_value[n_bins + 1]
    else:
        x_transform[missing_mask] = metric_missing

    return x_transform


def transform_multiclass_target(splits, x, n_event, special_codes, metric,
                                metric_special, metric_missing, show_digits,
                                check_input=False):

    if metric not in ("mean_woe", "weighted_mean_woe", "indices", "bins"):
        raise ValueError('Invalid value for metric. Allowed string '
                         'values are "mean_woe", "weighted_mean_woe", '
                         '"indices" and "bins".')

    _check_metric_special_missing(metric_special, metric_missing)
    _check_show_digits(show_digits)

    if check_input:
        x = check_array(x, ensure_2d=False, dtype=None,
                        force_all_finite='allow-nan')

    x = np.asarray(x)

    if np.issubdtype(x.dtype, np.number):
        missing_mask = np.isnan(x)
    else:
        missing_mask = pd.isnull(x)

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

    bins = np.concatenate([[-np.inf], splits, [np.inf]])
    bins_str = bin_str_format(bins, show_digits)
    n_bins = len(splits) + 1

    if metric in ("mean_woe", "weighted_mean_woe"):
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
    else:
        # Assign corresponding indices or bin intervals
        if metric == "indices":
            metric_value = np.arange(n_bins + 2)
            x_transform = np.full(x.shape, -1, dtype=int)
        elif metric == "bins":
            bins_str.extend(["Special", "Missing"])
            metric_value = bins_str
            x_transform = np.full(x.shape, "", dtype=object)

        x_clean_transform = np.full(x_clean.shape, "").astype(object)

    for i in range(n_bins):
        mask = (indices == i)
        x_clean_transform[mask] = metric_value[i]

    x_transform[clean_mask] = x_clean_transform

    if special_codes:
        if (metric_special == "empirical" or
            (metric == "indices" and not isinstance(metric_special, int)) or
                metric == "bins"):
            x_transform[special_mask] = metric_value[n_bins]
        else:
            x_transform[special_mask] = metric_special

    if (metric_missing == "empirical" or
        (metric == "indices" and not isinstance(metric_missing, int)) or
            metric == "bins"):
        x_transform[missing_mask] = metric_value[n_bins + 1]
    else:
        x_transform[missing_mask] = metric_missing

    return x_transform


def transform_continuous_target(splits, dtype, x, n_records, sums,
                                special_codes, categories, cat_others, metric,
                                metric_special, metric_missing, user_splits,
                                show_digits, check_input):

    if metric not in ("mean", "indices", "bins"):
        raise ValueError('Invalid value for metric. Allowed string '
                         'values are "mean", "indices" and "bins".')

    _check_metric_special_missing(metric_special, metric_missing)
    _check_show_digits(show_digits)

    if check_input:
        x = check_array(x, ensure_2d=False, dtype=None,
                        force_all_finite='allow-nan')

    x = np.asarray(x)

    if np.issubdtype(x.dtype, np.number):
        missing_mask = np.isnan(x)
    else:
        missing_mask = pd.isnull(x)

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

        bins = np.concatenate([[-np.inf], splits, [np.inf]])
        bins_str = bin_str_format(bins, show_digits)
        n_bins = len(splits) + 1
    else:
        bins = bin_categorical(splits, categories, cat_others, user_splits)
        bins_str = [str(b) for b in bins]
        n_bins = len(bins)

    if "empirical" not in (metric_special, metric_missing):
        n_records = n_records[:n_bins]
        sums = sums[:n_bins]

    if metric == "mean":
        # Compute mean
        metric_value = sums / n_records
        x_transform = np.zeros(x.shape)
    else:
        # Assign corresponding indices or bin intervals
        if metric == "indices":
            metric_value = np.arange(n_bins + 2)
            x_transform = np.full(x.shape, -1, dtype=int)
        elif metric == "bins":
            bins_str.extend(["Special", "Missing"])
            metric_value = bins_str
            x_transform = np.full(x.shape, "", dtype=object)

    if dtype == "numerical":
        if metric == "bins":
            x_clean_transform = np.full(x_clean.shape, "").astype(object)
        else:
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
        if (metric_special == "empirical" or
            (metric == "indices" and not isinstance(metric_special, int)) or
                metric == "bins"):
            x_transform[special_mask] = metric_value[n_bins]
        else:
            x_transform[special_mask] = metric_special

    if (metric_missing == "empirical" or
        (metric == "indices" and not isinstance(metric_missing, int)) or
            metric == "bins"):
        x_transform[missing_mask] = metric_value[n_bins + 1]
    else:
        x_transform[missing_mask] = metric_missing

    return x_transform
