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
                  special_codes_y=None, check_input=True):
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

    return (x_clean, y_clean, z_clean, x_missing, y_missing, z_missing,
            x_special, y_special, z_special)
