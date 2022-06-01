"""
Preprocessing 2D functions.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2021

import numpy as np
import pandas as pd

from sklearn.utils import check_array
from sklearn.utils import check_consistent_length

from ..preprocessing import categorical_transform


def split_data_2d(dtype_x, dtype_y, x, y, z, special_codes_x=None,
                  special_codes_y=None, check_input=True):
    """Split 2d data into clean, missing and special values data.

    Parameters
    ----------
    dtype_x : str, optional (default="numerical")
        The data type of variable x. Supported data type is "numerical" for
        continuous and ordinal variables.

    dtype_y : str, optional (default="numerical")
        The data type of variable y. Supported data type is "numerical" for
        continuous and ordinal variables.

    x : array-like, shape = (n_samples,)
        Training vector x, where n_samples is the number of samples.

    y : array-like, shape = (n_samples,)
        Training vector y, where n_samples is the number of samples.

    z : array-like, shape = (n_samples,)
        Target vector relative to x and y.

    special_codes_x : array-like or None, optional (default=None)
        List of special codes for the variable x. Use special codes to specify
        the data values that must be treated separately.

    special_codes_y : array-like or None, optional (default=None)
        List of special codes for the variable y. Use special codes to specify
        the data values that must be treated separately.

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

    if np.issubdtype(x.dtype, np.number) and np.issubdtype(z.dtype, np.number):
        missing_mask_x = np.isnan(x) | np.isnan(z)
    else:
        missing_mask_x = pd.isnull(x) | pd.isnull(z)

    if np.issubdtype(y.dtype, np.number) and np.issubdtype(z.dtype, np.number):
        missing_mask_y = np.isnan(y) | np.isnan(z)
    else:
        missing_mask_y = pd.isnull(y) | pd.isnull(z)

    if special_codes_x is not None:
        special_mask_x = pd.Series(x).isin(special_codes_x).values
    else:
        special_mask_x = np.zeros(len(x), dtype=bool)

    if special_codes_y is not None:
        special_mask_y = pd.Series(x).isin(special_codes_y).values
    else:
        special_mask_y = np.zeros(len(y), dtype=bool)

    missing_mask = missing_mask_x | missing_mask_y
    special_mask = special_mask_x | special_mask_y

    clean_mask = ~missing_mask & ~special_mask

    x_clean = x[clean_mask]
    y_clean = y[clean_mask]
    z_clean = z[clean_mask]

    x_missing = x[missing_mask]
    y_missing = y[missing_mask]
    z_missing = z[missing_mask]

    x_special = x[special_mask]
    y_special = y[special_mask]
    z_special = z[special_mask]

    if dtype_x == "categorical":
        x_categories, x_clean = categorical_transform(x_clean, z_clean)
    else:
        x_categories = []

    if dtype_y == "categorical":
        y_categories, y_clean = categorical_transform(y_clean, z_clean)
    else:
        y_categories = []

    return (x_clean, y_clean, z_clean, x_missing, y_missing, z_missing,
            x_special, y_special, z_special, x_categories, y_categories)
