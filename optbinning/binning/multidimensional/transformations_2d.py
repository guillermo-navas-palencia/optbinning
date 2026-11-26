"""
Binning 2D transformations.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2021

import numpy as np
import numpy.typing as npt
import pandas as pd

from sklearn.utils import check_array

from ..transformations import _check_metric_special_missing
from ..transformations import _check_show_digits
from .binning_statistics_2d import bin_categorical
from .binning_statistics_2d import bin_missing_xy_str_format
from .binning_statistics_2d import bin_xy_str_format


def _mask_special_missing(x, special_codes):
    if np.issubdtype(x.dtype, np.number):
        missing_mask = np.isnan(x)
    else:
        missing_mask = pd.isnull(x)

    if special_codes is None:
        special_mask = np.zeros(len(x), dtype=bool)
    else:
        special_mask = pd.Series(x).isin(special_codes).values

    return special_mask, missing_mask


def _mask_special_missing_xy(x, y, special_codes_x, special_codes_y):
    special_mask_x, missing_mask_x = _mask_special_missing(x, special_codes_x)
    special_mask_y, missing_mask_y = _mask_special_missing(y, special_codes_y)

    missing_mask = missing_mask_x | missing_mask_y
    special_mask = special_mask_x | special_mask_y

    clean_mask = ~missing_mask & ~special_mask

    return special_mask, missing_mask, clean_mask


def _mask_missing_x_y(x, y, special_codes_x, special_codes_y):
    """Return the masks of the records with only x or only y missing."""
    special_mask_x, missing_mask_x = _mask_special_missing(x, special_codes_x)
    special_mask_y, missing_mask_y = _mask_special_missing(y, special_codes_y)
    special_mask = special_mask_x | special_mask_y

    missing_x = missing_mask_x & ~missing_mask_y & ~special_mask
    missing_y = missing_mask_y & ~missing_mask_x & ~special_mask

    return missing_x, missing_y


def _missing_bin_indices(dtype, splits, x):
    """Return the bin index of each record of variable x, or -1 if the
    record does not fall into any bin."""
    indices = np.full(len(x), -1)

    if dtype == "categorical":
        x = pd.Series(x)

    for i, split in enumerate(splits):
        if dtype == "numerical":
            mask = (split[0] <= x) & (x < split[1])
        else:
            mask = x.isin(split).values

        indices[mask] = i

    return indices


def _transform_missing(dtype_x, dtype_y, splits_y_missing_x,
                       splits_x_missing_y, x, y, special_codes_x,
                       special_codes_y, metric_value, z_transform,
                       missing_mask, n_bins):
    """Assign the bins of the records with only x or only y missing and
    remove these records from the missing mask."""
    missing_x, missing_y = _mask_missing_x_y(
        x, y, special_codes_x, special_codes_y)

    for dtype, splits, values, mask in (
            (dtype_y, splits_y_missing_x, y, missing_x),
            (dtype_x, splits_x_missing_y, x, missing_y)):

        if not len(splits):
            continue

        indices = _missing_bin_indices(dtype, splits, values[mask])
        z_missing = z_transform[mask]

        for i in range(len(splits)):
            z_missing[indices == i] = metric_value[n_bins + i]

        z_transform[mask] = z_missing
        missing_mask = missing_mask & ~mask
        n_bins += len(splits)

    return z_transform, missing_mask


def _transform_metric_indices_bins(x, metric, n_bins, bins_str):
    # Assign corresponding indices or bin intervals
    if metric == "indices":
        metric_value = np.arange(n_bins + 2)
        z_transform = np.full(x.shape, -1, dtype=int)
    elif metric == "bins":
        bins_str.extend(["Special", "Missing"])
        metric_value = bins_str
        z_transform = np.full(x.shape, "", dtype=object)

    return metric_value, z_transform


def _apply_transform(dtype_x, dtype_y, splits_x, splits_y, special_codes_x,
                     special_codes_y, metric, metric_special, metric_missing,
                     metric_value, clean_mask, special_mask, missing_mask,
                     z_transform, x_clean, y_clean, n_bins,
                     n_bins_missing):

    if metric == "bins":
        z_clean_transform = np.full(x_clean.shape, "", dtype=object)
    else:
        z_clean_transform = np.zeros(x_clean.shape)

    if dtype_x == "categorical":
        x_p = pd.Series(x_clean)

    if dtype_y == "categorical":
        y_p = pd.Series(y_clean)

    for i in range(n_bins):
        if dtype_x == "numerical":
            mask_x = (splits_x[i][0] <= x_clean) & (x_clean < splits_x[i][1])
        else:
            mask_x = x_p.isin(splits_x[i])

        if dtype_y == "numerical":
            mask_y = (splits_y[i][0] <= y_clean) & (y_clean < splits_y[i][1])
        else:
            mask_y = y_p.isin(splits_y[i])

        mask = mask_x & mask_y
        z_clean_transform[mask] = metric_value[i]

    z_transform[clean_mask] = z_clean_transform

    # Special and Missing follow the bins of the records with one missing
    n_bins += n_bins_missing

    if special_codes_x or special_codes_y:
        if (metric_special == "empirical" or
            (metric == "indices" and not isinstance(metric_special, int)) or
                metric == "bins"):
            z_transform[special_mask] = metric_value[n_bins]
        else:
            z_transform[special_mask] = metric_special

    if (metric_missing == "empirical" or
        (metric == "indices" and not isinstance(metric_missing, int)) or
            metric == "bins"):
        z_transform[missing_mask] = metric_value[n_bins + 1]
    else:
        z_transform[missing_mask] = metric_missing

    return z_transform


def transform_binary_target(
    dtype_x: str,
    dtype_y: str,
    splits_x,
    splits_y,
    x: list | npt.NDArray,
    y: list | npt.NDArray,
    n_nonevent: npt.NDArray,
    n_event: npt.NDArray,
    special_codes_x: list | npt.NDArray | None,
    special_codes_y: list | npt.NDArray | None,
    categories_x,
    categories_y,
    metric: str,
    metric_special: float | str,
    metric_missing: float | str,
    show_digits: int,
    check_input: bool = False,
    splits_y_missing_x=(),
    splits_x_missing_y=(),
) -> np.ndarray:

    if metric not in ("event_rate", "woe", "indices", "bins"):
        raise ValueError('Invalid value for metric. Allowed string '
                         'values are "event_rate", "woe", "indices" and '
                         '"bins".')

    _check_metric_special_missing(metric_special, metric_missing)
    _check_show_digits(show_digits)

    if check_input:
        x = check_array(x, ensure_2d=False, dtype=None,
                        ensure_all_finite='allow-nan')

        y = check_array(y, ensure_2d=False, dtype=None,
                        ensure_all_finite='allow-nan')

    x = np.asarray(x)
    y = np.asarray(y)

    special_mask, missing_mask, clean_mask = _mask_special_missing_xy(
        x, y, special_codes_x, special_codes_y)

    x_clean = x[clean_mask]
    y_clean = y[clean_mask]

    bins_str = bin_xy_str_format(dtype_x, dtype_y, splits_x, splits_y,
                                 show_digits, categories_x, categories_y)
    bins_str.extend(bin_missing_xy_str_format(
        dtype_x, dtype_y, splits_y_missing_x, splits_x_missing_y,
        show_digits))
    n_bins = len(splits_x)
    n_bins_missing = len(splits_y_missing_x) + len(splits_x_missing_y)

    if dtype_x == "categorical":
        splits_x = bin_categorical(splits_x, categories_x)

    if dtype_y == "categorical":
        splits_y = bin_categorical(splits_y, categories_y)

    if metric in ("woe", "event_rate"):
        # Compute event rate and WoE
        n_records = n_event + n_nonevent
        t_n_nonevent = n_nonevent.sum()
        t_n_event = n_event.sum()

        if "empirical" not in (metric_special, metric_missing):
            n_event = n_event[:n_bins + n_bins_missing]
            n_nonevent = n_nonevent[:n_bins + n_bins_missing]
            n_records = n_records[:n_bins + n_bins_missing]

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

        z_transform = np.zeros(x.shape)
    else:
        # Assign corresponding indices or bin intervals
        metric_value, z_transform = _transform_metric_indices_bins(
            x, metric, n_bins + n_bins_missing, bins_str)

    if n_bins_missing:
        z_transform, missing_mask = _transform_missing(
            dtype_x, dtype_y, splits_y_missing_x, splits_x_missing_y, x, y,
            special_codes_x, special_codes_y, metric_value, z_transform,
            missing_mask, n_bins)

    z_transform = _apply_transform(
        dtype_x, dtype_y, splits_x, splits_y, special_codes_x, special_codes_y,
        metric, metric_special, metric_missing, metric_value, clean_mask,
        special_mask, missing_mask, z_transform, x_clean, y_clean, n_bins,
        n_bins_missing)

    return z_transform


def transform_continuous_target(
    dtype_x: str,
    dtype_y: str,
    splits_x,
    splits_y,
    x: list | npt.NDArray,
    y: list | npt.NDArray,
    n_records: npt.NDArray,
    sums: npt.NDArray,
    special_codes_x: list | npt.NDArray | None,
    special_codes_y: list | npt.NDArray | None,
    categories_x,
    categories_y,
    metric: str,
    metric_special: float | str,
    metric_missing: float | str,
    show_digits: int,
    check_input: bool = False,
    splits_y_missing_x=(),
    splits_x_missing_y=(),
) -> np.ndarray:

    if metric not in ("mean", "indices", "bins"):
        raise ValueError('Invalid value for metric. Allowed string '
                         'values are "mean", "indices" and "bins".')

    _check_metric_special_missing(metric_special, metric_missing)
    _check_show_digits(show_digits)

    if check_input:
        x = check_array(x, ensure_2d=False, dtype=None,
                        ensure_all_finite='allow-nan')

        y = check_array(y, ensure_2d=False, dtype=None,
                        ensure_all_finite='allow-nan')

    x = np.asarray(x)
    y = np.asarray(y)

    special_mask, missing_mask, clean_mask = _mask_special_missing_xy(
        x, y, special_codes_x, special_codes_y)

    x_clean = x[clean_mask]
    y_clean = y[clean_mask]

    bins_str = bin_xy_str_format(dtype_x, dtype_y, splits_x, splits_y,
                                 show_digits, categories_x, categories_y)
    bins_str.extend(bin_missing_xy_str_format(
        dtype_x, dtype_y, splits_y_missing_x, splits_x_missing_y,
        show_digits))
    n_bins = len(splits_x)
    n_bins_missing = len(splits_y_missing_x) + len(splits_x_missing_y)

    if dtype_x == "categorical":
        splits_x = bin_categorical(splits_x, categories_x)

    if dtype_y == "categorical":
        splits_y = bin_categorical(splits_y, categories_y)

    if "empirical" not in (metric_special, metric_missing):
        n_records = n_records[:n_bins + n_bins_missing]
        sums = sums[:n_bins + n_bins_missing]

    if metric == "mean":
        # Compute mean
        mask = n_records > 0
        metric_value = np.zeros(len(n_records))
        metric_value[mask] = sums[mask] / n_records[mask]
        z_transform = np.zeros(x.shape)
    else:
        # Assign corresponding indices or bin intervals
        metric_value, z_transform = _transform_metric_indices_bins(
            x, metric, n_bins + n_bins_missing, bins_str)

    if n_bins_missing:
        z_transform, missing_mask = _transform_missing(
            dtype_x, dtype_y, splits_y_missing_x, splits_x_missing_y, x, y,
            special_codes_x, special_codes_y, metric_value, z_transform,
            missing_mask, n_bins)

    z_transform = _apply_transform(
        dtype_x, dtype_y, splits_x, splits_y, special_codes_x, special_codes_y,
        metric, metric_special, metric_missing, metric_value, clean_mask,
        special_mask, missing_mask, z_transform, x_clean, y_clean, n_bins,
        n_bins_missing)

    return z_transform
