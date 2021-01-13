"""
Scorecard performance metrics.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import numpy as np

from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import roc_curve


def gini(y_true, y_pred_proba):
    """Compute the Gini Index or Accuracy Ration (AR).

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Ground truth (correct) target values.

    y_pred_proba : array-like, shape (n_samples,)
        Probability estimates of the positive class.

    Returns
    -------
    gini : float
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    return 2 * auc(fpr, tpr) - 1


def ks(y_true, y_pred_proba):
    """Compute the Kolmogorov-Smirnov (KS).

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Ground truth (correct) target values.

    y_pred_proba : array-like, shape (n_samples,)
        Probability estimates of the positive class.

    Returns
    -------
    ks : tuple(ks_score, ks_position)
    """
    n_samples = y_true.shape[0]
    n_event = np.sum(y_true)
    n_nonevent = n_samples - n_event

    idx = np.argsort(y_pred_proba)
    yy = y_true[idx]

    cum_event = np.cumsum(yy)
    cum_population = np.arange(0, n_samples)
    cum_nonevent = cum_population - cum_event

    p_event = cum_event / n_event
    p_nonevent = cum_nonevent / n_nonevent

    p_diff = p_nonevent - p_event
    ks_max_idx = np.argmax(p_diff)
    ks_score = p_diff[ks_max_idx]

    return ks_score, ks_max_idx


def imbalanced_classification_metrics(y_true, y_pred):
    """Compute imbalanced binary classification metrics.

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
    tp, fp, fn, tn = confusion_matrix(y_true, y_pred).ravel()

    # Sensitivity - True positive rate (TPR)
    tpr = tp / (tp + fn)

    # Specificity - True negative rate (TNR)
    tnr = tn / (fp + tn)

    # False positive rate (FPR)
    fpr = 1.0 - tnr

    # False negative rate (FNR)
    fnr = 1.0 - tpr

    # Balanced accuracy
    balanced_accuracy = 0.5 * (tpr + tnr)

    # Discriminant power
    dp = np.sqrt(3) / np.pi * (np.log(tpr / (1-tnr)) + np.log(tnr / (1-tpr)))

    d_metrics = {
        "True positive rate": tpr,
        "True negative rate": tnr,
        "False positive rate": fpr,
        "False negative rate": fnr,
        "Balanced accuracy": balanced_accuracy,
        "Discriminant power": dp
    }

    return d_metrics


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
