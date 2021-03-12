"""
Metrics to asses performance of regression models.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2021

import numpy as np

from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score


def mean_absolute_percentage_error(y_true, y_pred):
    """Compute the mean absolute percentage error (MAPE).

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like, shape (n_samples,)
        Estimated target values.

    Returns
    -------
    mape : float
    """
    return np.abs((y_true - y_pred) / y_true).mean()


def median_absolute_percentage_error(y_true, y_pred):
    """Compute the median absolute percentage error (MdAPE).

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like, shape (n_samples,)
        Estimated target values.

    Returns
    -------
    mdape : float
    """
    return np.median(np.abs((y_true - y_pred) / y_true))


def mean_percentage_error(y_true, y_pred):
    """Compute the mean percentage error (MPE).

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like, shape (n_samples,)
        Estimated target values.

    Returns
    -------
    mpe : float
    """
    return ((y_true - y_pred) / y_true).mean()


def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    """Compute the symmetric mean absolute percentage error (SMAPE).

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like, shape (n_samples,)
        Estimated target values.

    Returns
    -------
    smape : float
    """
    e = np.abs(y_true - y_pred)
    return (e / (np.abs(y_true) + np.abs(y_pred))).mean()


def symmetric_median_absolute_percentage_error(y_true, y_pred):
    """Compute the symmetric median absolute percentage error (SMdAPE).

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like, shape (n_samples,)
        Estimated target values.

    Returns
    -------
    smdape : float
    """
    e = np.abs(y_true - y_pred)
    return np.median(e / (np.abs(y_true) + np.abs(y_pred)))


def regression_metrics(y_true, y_pred):
    """Compute regression metrics.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like, shape (n_samples,)
        Estimated target values.

    Returns
    -------
    metrics : dict
        Dictionary of metrics.
    """

    # Explained variance
    variance = explained_variance_score(y_true, y_pred)

    # Mean absolute error
    mae = mean_absolute_error(y_true, y_pred)

    # Mean squared error
    mse = mean_squared_error(y_true, y_pred)

    # Median absolute error
    median_ae = median_absolute_error(y_true, y_pred)

    # R^2 score
    r2 = r2_score(y_true, y_pred)

    # Mean absolute percentage error
    mape = mean_absolute_percentage_error(y_true, y_pred)

    # Mean percentage error
    mpe = mean_percentage_error(y_true, y_pred)

    # Symmetric mean absolute percentage error
    smape = symmetric_mean_absolute_percentage_error(y_true, y_pred)

    # Median absolute percentage error
    mdape = median_absolute_percentage_error(y_true, y_pred)

    # Symmetric meadian absolute percentage error
    smdape = symmetric_median_absolute_percentage_error(y_true, y_pred)

    d_metrics = {
        "Mean absolute error": mae,
        "Mean squared error": mse,
        "Median absolute error": median_ae,
        "Explained variance": variance,
        "R^2": r2,
        "MPE": mpe,
        "MAPE": mape,
        "SMAPE": smape,
        "MdAPE": mdape,
        "SMdAPE": smdape
    }

    return d_metrics
