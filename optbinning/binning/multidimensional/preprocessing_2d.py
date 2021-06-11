"""
Preprocessing 2D functions.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2021

import numpy as np
import pandas as pd

from sklearn.utils import check_array
from sklearn.utils import check_consistent_length


def split_data_2d(dtype_x, dtype_y, x, y, z, special_codes_x=None,
                  special_codes_y=None, cat_cutoff=None, user_splits_x=None,
                  user_splits_y=None, check_input=True):
    """Split 2d data into clean, missing and special values data.

    Parameters
    ----------
    dtype_x : str

    dtype_y : str

    x : array-like, shape = (n_samples)

    y : array-like, shape = (n_samples)

    z : array-like, shape = (n_samples)

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

    Returns
    -------
    """
    if check_input:
        x = check_array(x, ensure_2d=False, dtype=None,
                        force_all_finite='allow-nan')

        y = check_array(y, ensure_2d=False, dtype=None,
                        force_all_finite='allow-nan')

        z = check_array(z, ensure_2d=False, dtype=None,
                        force_all_finite=True)

        check_consistent_length(x, y, z)

    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    if isinstance(x.dtype, object) or isinstance(z.dtype, object):
        missing_mask_x = pd.isnull(x) | pd.isnull(z)
    else:
        missing_mask_x = np.isnan(x) | pd.isnull(z)

    if isinstance(x.dtype, object) or isinstance(z.dtype, object):
        missing_mask_y = pd.isnull(y) | pd.isnull(z)
    else:
        missing_mask_y = np.isnan(y) | pd.isnull(z)

    if special_codes_x is not None:
        special_mask_x = pd.Series(x).isin(special_codes_x).values
    else:
        special_mask_x = np.zeros(len(x), dtype=bool)
        
    if special_codes_y is not None:
        special_mask_y = pd.Series(x).isin(special_codes_y).values
    else:
        special_mask_y = np.zeros(len(y), dtype=bool)
    
    clean_mask_x = ~missing_mask_x & ~special_mask_x
    clean_mask_y = ~missing_mask_y & ~special_mask_y

    clean_mask = clean_mask_x & clean_mask_y

    x_clean = x[clean_mask]
    y_clean = y[clean_mask]
    z_clean = z[clean_mask]

    categories_x = None
    categories_y = None
    others_x = None
    others_y = None

    mask_others_x = None
    mask_others_y = None

    if dtype_x == "categorical" and user_splits_x is None:
        if cat_cutoff is not None:
            mask_others_x, others_x = categorical_cutoff(
                x_clean, z_clean, cat_cutoff)
        else:
            mask_others_x = None
            others_x = None

        categories_x, x_clean = categorical_transform(x_clean, z_clean)

    if dtype_y == "categorical" and user_splits_y is None:
        if cat_cutoff is not None:
            mask_others_y, others_y = categorical_cutoff(
                y_clean, z_clean, cat_cutoff)
        else:
            mask_others_y = None
            others_y = None

        categories_y, y_clean = categorical_transform(y_clean, z_clean)

    return (x_clean, y_clean, z_clean, special_mask_x, special_mask_y,
            missing_mask_x, missing_mask_y, mask_others_x, mask_others_y,
            categories_x, categories_y, others_x, others_y)
