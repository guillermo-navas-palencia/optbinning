"""
Preprocessing functions.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import numbers

import numpy as np
import pandas as pd

from sklearn.utils import check_array
from sklearn.utils import check_consistent_length

from .outlier import ModifiedZScoreDetector
from .outlier import RangeDetector


def categorical_transform(x, y):
    event_rate = pd.Series(y).groupby(x).mean()
    sorted_categories = event_rate.sort_values().index.values
    d = dict(map(reversed, enumerate(sorted_categories)))

    return sorted_categories, pd.Series(x).map(d).values


def categorical_cutoff(x, y, cutoff=0.01):
    cutoff_count = np.ceil(cutoff * len(x))
    cat_count = pd.value_counts(x)
    cat_others = cat_count[cat_count < cutoff_count].index.values
    mask_others = pd.Series(x).isin(cat_others).values

    if np.count_nonzero(~mask_others) == 0:
        raise ValueError("All categories moved to others' bin. Al least one "
                         "category is needed to perform binning.")

    return mask_others, cat_others


def split_data(dtype, x, y, special_codes=None, cat_cutoff=None,
               user_splits=None, check_input=True, outlier_detector=None,
               outlier_params=None, fix_lb=None, fix_ub=None):
    """Split data into clean, missing and special values data.

    Parameters
    ----------
    dtype : str, optional (default="numerical")
        The variable data type. Supported data types are "numerical" for
        continuous and ordinal variables and "categorical" for categorical
        and nominal variables.

    x : array-like, shape = (n_samples)
        Data samples, where n_samples is the number of samples.

    y : array-like, shape = (n_samples)
        Target vector relative to x.

    special_codes : array-like or None, optional (default=None)
        List of special codes. Use special codes to specify the data values
        that must be treated separately.

    cat_cutoff : float or None, optional (default=None)
        Generate bin others with categories in which the fraction of
        occurrences is below the  ``cat_cutoff`` value. This option is
        available when ``dtype`` is "categorical".

    user_splits : array-like or None, optional (default=None)
        The list of pre-binning split points when ``dtype`` is "numerical" or
        the list of prebins when ``dtype`` is "categorical".

    check_input : bool, (default=True)
        If False, the input arrays x and y will not be checked.

    outlier_detector : str or None (default=None)

    outlier_params : dict or None (default=None)

    fix_lb : float or None (default=None)

    fix_ub : float or None (default=None)

    Returns
    -------
    x_clean : array, shape = (n_clean)
        Clean data samples

    y_clean : array, shape = (n_clean)
        Clean target samples.

    x_missing : array, shape = (n_missing)
        Missing data samples.

    y_missing : array, shape = (n_missing)
        Missing target samples.

    x_special : array, shape = (n_special)
        Special data samples.

    y_special : array, shape = (n_special)
        Special target samples.

    y_others : array, shape = (n_others)
        Others target samples.

    categories : array, shape (n_categories)
        List of categories.

    others : array, shape (n_other_categories)
        List of other categories.
    """
    if outlier_detector is not None:
        if outlier_detector not in ("range", "zscore"):
            raise ValueError('Invalid value for outlier_detector. Allowed '
                             'string values are "range" and "zscore".')

        if outlier_params is not None:
            if not isinstance(outlier_params, dict):
                raise TypeError("outlier_params must be a dict or None; "
                                "got {}.".format(outlier_params))

    if fix_lb is not None:
        if not isinstance(fix_lb, numbers.Number):
            raise ValueError("fix_lb must be a number; got {}.".format(fix_lb))

    if fix_ub is not None:
        if not isinstance(fix_ub, numbers.Number):
            raise ValueError("fix_ub must be a number; got {}.".format(fix_ub))

    if fix_lb is not None and fix_ub is not None:
        if fix_lb > fix_ub:
            raise ValueError("fix_lb must be <= fix_ub; got {} <= {}."
                             .format(fix_lb, fix_ub))

    if check_input:
        x = check_array(x, ensure_2d=False, dtype=None,
                        force_all_finite='allow-nan')

        y = check_array(y, ensure_2d=False, dtype=None,
                        force_all_finite=True)

        check_consistent_length(x, y)

    x = np.asarray(x)
    y = np.asarray(y)

    if isinstance(x.dtype, object) or isinstance(y.dtype, object):
        missing_mask = pd.isnull(x) | pd.isnull(y)
    else:
        missing_mask = np.isinan(x) | np.isnan(y)

    if special_codes is None:
        clean_mask = ~missing_mask

        x_clean = x[clean_mask]
        y_clean = y[clean_mask]
        x_missing = x[missing_mask]
        y_missing = y[missing_mask]
        x_special = []
        y_special = []
    else:
        special_mask = pd.Series(x).isin(special_codes).values

        clean_mask = ~missing_mask & ~special_mask

        x_clean = x[clean_mask]
        y_clean = y[clean_mask]
        x_missing = x[missing_mask]
        y_missing = y[missing_mask]
        x_special = x[special_mask]
        y_special = y[special_mask]

    if dtype == "numerical":
        if outlier_detector is not None:
            if outlier_detector == "range":
                detector = RangeDetector()
            elif outlier_detector == "zscore":
                detector = ModifiedZScoreDetector()

            if outlier_params is not None:
                detector.set_params(**outlier_params)

            mask_outlier = detector.fit(x_clean).get_support()
            x_clean = x_clean[~mask_outlier]
            y_clean = y_clean[~mask_outlier]

        if fix_lb is not None or fix_ub is not None:
            if fix_lb is not None:
                mask = x_clean >= fix_lb
            elif fix_ub is not None:
                mask = x_clean <= fix_ub
            else:
                mask = (x_clean >= fix_lb) & (x_clean <= fix_ub)

            x_clean = x_clean[mask]
            y_clean = y_clean[mask]

    if dtype == "categorical" and user_splits is None:
        if cat_cutoff is not None:
            mask_others, others = categorical_cutoff(
                x_clean, y_clean, cat_cutoff)

            y_others = y_clean[mask_others]
            x_clean = x_clean[~mask_others]
            y_clean = y_clean[~mask_others]
        else:
            y_others = []
            others = []

        categories, x_clean = categorical_transform(x_clean, y_clean)

        return (x_clean, y_clean, x_missing, y_missing, x_special, y_special,
                y_others, categories, others)
    else:
        return (x_clean, y_clean, x_missing, y_missing, x_special, y_special,
                [], [], [])


def preprocessing_user_splits_categorical(user_splits, x, y):
    categories = pd.Series(x).unique()

    n_user_splits = len(user_splits)
    user_splits = np.asarray(user_splits)

    # Check no category is repeated
    user_categories = {}
    for split in user_splits:
        for cat in split:
            if user_categories.get(cat, 0):
                raise ValueError("Category {} is repeated.".format(cat))
            else:
                user_categories[cat] = 1

    unique_user_categories = list(user_categories.keys())

    # If category is not in user_splits, then move category to cat_others
    cat_others = np.array([c for c in categories
                           if c not in unique_user_categories])

    mask_others = pd.Series(x).isin(cat_others).values

    y_others = y[mask_others]
    x_clean = x[~mask_others]
    y_clean = y[~mask_others]

    # Group by user_splits and transform from categorical to nominal
    x_clean_nominal = np.zeros(x_clean.shape)

    event_rate = np.zeros(n_user_splits)
    x_p = pd.Series(x_clean)
    for i, split in enumerate(user_splits):
        event_rate[i] = y_clean[x_p.isin(split)].mean()

    splits_nominal = np.array(range(n_user_splits))
    sorted_idx = np.argsort(event_rate)
    sorted_splits = user_splits[sorted_idx]

    sorted_splits = np.array([np.array(split) for split in sorted_splits])

    for i in range(n_user_splits):
        mask = x_p.isin(user_splits[i])
        x_clean_nominal[mask] = sorted_idx[i]

    return (sorted_splits, splits_nominal, x_clean_nominal, y_clean, y_others,
            cat_others)
