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
    pass


def median_absolute_percentage_error(y_true, y_pred):
    pass


def mean_percentage_error(y_true, y_pred):
    pass


def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    pass


def symmetric_median_absolute_percentage_error(y_true, y_pred):
    # https://support.numxl.com/hc/es/articles/115001223503-MdAPE-Error-porcentual-absoluto-medio
    pass


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
