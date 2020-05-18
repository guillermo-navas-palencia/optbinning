"""
Module with plots for visualizing model performance.
"""

# Gabriel S. Gonçalves <gabrielgoncalvesbr@gmail.com>
# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020


import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.utils import check_array
from sklearn.utils import check_consistent_length


def _check_arrays(y, y_pred):
    y = check_array(y, ensure_2d=False, force_all_finite=True)
    y_pred = check_array(y_pred, ensure_2d=False, force_all_finite=True)

    check_consistent_length(y, y_pred)

    return y, y_pred


def _check_parameters(title, xlabel, ylabel, savefig, fname):
    if title is not None and not isinstance(title, str):
        raise TypeError("title must be a string or None; got {}."
                        .format(title))

    if xlabel is not None and not isinstance(xlabel, str):
        raise TypeError("xlabel must be a string or None; got {}."
                        .format(xlabel))

    if ylabel is not None and not isinstance(ylabel, str):
        raise TypeError("ylabel must be a string or None; got {}."
                        .format(ylabel))

    if not isinstance(savefig, bool):
        raise TypeError("savefig must be a boolean; got {}.".format(savefig))

    if fname is not None and not isinstance(fname, str):
        raise TypeError("fname must be a string or None; got {}."
                        .format(fname))

    if savefig is True and fname is None:
        raise ValueError("fname must be provided if savefig is True.")


def plot_auc_roc(y, y_pred, title=None, xlabel=None, ylabel=None,
                 savefig=False, fname=None, **kwargs):
    """Plot Area Under the Receiver Operating Characteristic Curve (AUC ROC).

    Parameters
    ----------
    y : array-like, shape = (n_samples,)
        Array with the target labels.

    y_pred : array-like, shape = (n_samples,)
        Array with predicted probabilities.

    title : str or None, optional (default=None)
        Title for the plot.

    xlabel : str or None, optional (default=None)
        Label for the x-axis.

    ylabel : str or None, optional (default=None)
        Label for the y-axis.

    savefig : bool (default=False)
        Whether to save the figure.

    fname : str or None, optional (default=None)
        Name for the figure file.

    **kwargs : keyword arguments
        Keyword arguments for matplotlib.pyplot.savefig().
    """
    y, y_pred = _check_arrays(y, y_pred)

    _check_parameters(title, xlabel, ylabel, savefig, fname)

    # Define the arrays for plotting
    fpr, tpr, _ = roc_curve(y, y_pred)
    auc_roc = roc_auc_score(y, y_pred)

    # Define the plot settings
    if title is None:
        title = "ROC curve"
    if xlabel is None:
        xlabel = "False Positive Rate"
    if ylabel is None:
        ylabel = "True Positive Rate"

    plt.plot(fpr, fpr, linestyle="--", color="k", label="Random Model")
    plt.plot(fpr, tpr, color="g", label="Model (AUC: {:.5f})".format(auc_roc))
    plt.title(title, fontdict={"fontsize": 14})
    plt.xlabel(xlabel, fontdict={"fontsize": 12})
    plt.ylabel(ylabel, fontdict={"fontsize": 12})
    plt.legend(loc='lower right')

    # Save figure if requested. Pass kwargs.
    if savefig:
        plt.savefig(fname=fname, **kwargs)
        plt.close()


def plot_cap(y, y_pred, title=None, xlabel=None, ylabel=None,
             savefig=False, fname=None, **kwargs):
    """Plot Cumulative Accuracy Profile (CAP).

    Parameters
    ----------
    y : array-like, shape = (n_samples,)
        Array with the target labels.

    y_pred : array-like, shape = (n_samples,)
        Array with predicted probabilities.

    title : str or None, optional (default=None)
        Title for the plot.

    xlabel : str or None, optional (default=None)
        Label for the x-axis.

    ylabel : str or None, optional (default=None)
        Label for the y-axis.

    savefig : bool (default=False)
        Whether to save the figure.

    fname : str or None, optional (default=None)
        Name for the figure file.

    **kwargs : keyword arguments
        Keyword arguments for matplotlib.pyplot.savefig().
    """
    y, y_pred = _check_arrays(y, y_pred)

    _check_parameters(title, xlabel, ylabel, savefig, fname)

    n_samples = y.shape[0]
    n_event = np.sum(y)

    idx = y_pred.argsort()[::-1][:n_samples]
    yy = y[idx]

    p_event = np.append([0], np.cumsum(yy)) / n_event
    p_population = np.arange(0, n_samples + 1) / n_samples

    auroc = roc_auc_score(y, y_pred)
    gini = auroc * 2 - 1

    # Define the plot settings
    if title is None:
        title = "Cumulative Accuracy Profile (CAP)"
    if xlabel is None:
        xlabel = "Fraction of all population"
    if ylabel is None:
        ylabel = "Fraction of event population"

    plt.plot([0, 1], [0, 1], color='k', linestyle='--', label="Random Model")
    plt.plot([0, n_event / n_samples, 1], [0, 1, 1], color='grey',
             linestyle='--', label="Perfect Model")
    plt.plot(p_population, p_event, color="g",
             label="Model (Gini: {:.5f})".format(gini))

    plt.title(title, fontdict={'fontsize': 14})
    plt.xlabel(xlabel, fontdict={'fontsize': 12})
    plt.ylabel(ylabel, fontdict={'fontsize': 12})
    plt.legend(loc='lower right')

    # Save figure if requested. Pass kwargs.
    if savefig:
        plt.savefig(fname=fname, **kwargs)
        plt.close()


def plot_ks(y, y_pred, title=None, xlabel=None, ylabel=None,
            savefig=False, fname=None, **kwargs):
    """Plot Kolmogorov-Smirnov (KS).

    Parameters
    ----------
    y : array-like, shape = (n_samples,)
        Array with the target labels.

    y_pred : array-like, shape = (n_samples,)
        Array with predicted probabilities.

    title : str or None, optional (default=None)
        Title for the plot.

    xlabel : str or None, optional (default=None)
        Label for the x-axis.

    ylabel : str or None, optional (default=None)
        Label for the y-axis.

    savefig : bool (default=False)
        Whether to save the figure.

    fname : str or None, optional (default=None)
        Name for the figure file.

    **kwargs : keyword arguments
        Keyword arguments for matplotlib.pyplot.savefig().
    """
    y, y_pred = _check_arrays(y, y_pred)

    _check_parameters(title, xlabel, ylabel, savefig, fname)

    n_samples = y.shape[0]
    n_event = np.sum(y)
    n_nonevent = n_samples - n_event

    idx = y_pred.argsort()
    yy = y[idx]
    pp = y_pred[idx]

    cum_event = np.cumsum(yy)
    cum_population = np.arange(0, n_samples)
    cum_nonevent = cum_population - cum_event

    p_event = cum_event / n_event
    p_nonevent = cum_nonevent / n_nonevent

    p_diff = p_nonevent - p_event
    ks_score = np.max(p_diff)
    ks_max_idx = np.argmax(p_diff)

    # Define the plot settings
    if title is None:
        title = "Kolmogorov-Smirnov"
    if xlabel is None:
        xlabel = "Threshold"
    if ylabel is None:
        ylabel = "Cumulative probability"

    plt.title(title, fontdict={'fontsize': 14})
    plt.xlabel(xlabel, fontdict={'fontsize': 12})
    plt.ylabel(ylabel, fontdict={'fontsize': 12})

    plt.plot(pp, p_event, color="r", label="Cumulative events")
    plt.plot(pp, p_nonevent, color="b", label="Cumulative non-events")

    plt.vlines(pp[ks_max_idx], ymin=p_event[ks_max_idx],
               ymax=p_nonevent[ks_max_idx], color="k", linestyles="--")

    # Set KS value inside plot
    pos_x = pp[ks_max_idx] + 0.02
    pos_y = 0.5 * (p_nonevent[ks_max_idx] + p_event[ks_max_idx])
    text = "KS: {:.2%} at {:.2f}".format(ks_score, pp[ks_max_idx])
    plt.text(pos_x, pos_y, text, fontsize=12, rotation_mode="anchor")

    plt.legend(loc='lower right')

    # Save figure if requested. Pass kwargs.
    if savefig:
        plt.savefig(fname=fname, **kwargs)
        plt.close()
