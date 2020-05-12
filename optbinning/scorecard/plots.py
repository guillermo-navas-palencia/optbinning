"""
Modules with plots for visualizing model performance.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


def plot_auroc(
    df,
    target_col,
    proba_col,
    title='ROC curve',
    xlabel='False positive rate',
    ylabel='True positive rate',
    roc_color='blue',
    reference_color='black',
    savefig=False,
    figname='auroc.png',
):
    """Plot AUROC for dataframe with probabilities and binary target.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with the columns to be plotted.

    target_col : str
        Column with binary target.

    proba_col : str
        Column with the calculated probabilites.

    title : str (default='ROC curve')
        Title for plot.

    xlabel : str (default='False positive rate')
        Label for the x-axis.

    ylabel : str (default='True positive rate')
        Label for the y-axis.

    roc_color : str (default='blue')
        Color for the roc line.

    reference_color : str (default='red')
        Color for reference line.

    savefig : bool (default=False)
        To save figure.

    figname : str (default='auroc.fig')
        Name for figure file.

    Returns
    -------
    auroc : float
        Calculate AUROC score.
    """
    # Define the arrays for plotting
    fpr, tpr, threshold = roc_curve(df[target_col], df[proba_col])

    # Define the plot settings
    plt.plot(fpr, tpr, color=roc_color)
    plt.plot(fpr, fpr, linestyle="--", color=reference_color)
    plt.title(title, fontdict={'fontsize': 16})
    plt.xlabel(xlabel, fontdict={'fontsize': 16})
    plt.ylabel(ylabel, fontdict={'fontsize': 16})

    if savefig:
        plt.savefig(figname, dpi=300)

    # Set AUROC score
    auroc = roc_auc_score(df[target_col], df[proba_col])
    return auroc


def plot_gini(
    df,
    target_col,
    proba_col,
    title='Gini',
    xlabel='Cumulative % Population',
    ylabel='Cumulative % Bad',
    gini_color='blue',
    reference_color='black',
    savefig=False,
    figname='gini.png',
):
    """Plot Gini for dataframe with probabilities and binary target.

    Parameters
    ----------

    df : pandas.DataFrame
        Dataframe with the columns to be plotted.

    target_col : str
        Column with binary target.

    proba_col : str
        Column with the calculated probabilites.

    title : str (default='Gini'):
        Title for plot.

    xlabel : str (default='Cumulative % Population')
        Label for the x-axis.

    ylabel : str (default='Cumulative % Bad')
        Label for the y-axis.

    roc_color : str (default='blue')
        Color for the Gini line.

    reference_color : str (default='red')
        Color for the reference line.

    savefig : bool (default=False)
        To save figure.

    figname : str (default='gini.fig')
        Name for figure file.

    Returns
    -------

    gini : float
        Calculate Gini Index.
    """
    # Sort dataframe by probabilities and reset index
    df = df.sort_values(proba_col)
    df = df.reset_index()

    # Calculate cumulative columns
    df["cumulative_n_population"] = df.index + 1
    df["cumulative_n_good"] = df[target_col].cumsum()
    df["cumulative_n_bad"] = (
        df["cumulative_n_population"] - df[target_col].cumsum()
    )
    df["cumulative_perc_population"] = df["cumulative_n_population"] / (
        df.shape[0]
    )
    df["cumulative_perc_good"] = df["cumulative_n_good"] / df[target_col].sum()
    df["cumulative_perc_bad"] = df["cumulative_n_bad"] / (
        df.shape[0] - df[target_col].sum()
    )

    # Plot Gini
    plt.plot(
        df["cumulative_perc_population"],
        df["cumulative_perc_bad"],
        color=gini_color,
    )
    plt.plot(
        df["cumulative_perc_population"],
        df["cumulative_perc_population"],
        linestyle="--",
        color=reference_color,
    )
    plt.title(title, fontdict={'fontsize': 16})
    plt.xlabel(xlabel, fontdict={'fontsize': 16})
    plt.ylabel(ylabel, fontdict={'fontsize': 16})

    if savefig:
        plt.savefig(figname, dpi=300)

    # Calculate Gini Index
    auroc = roc_auc_score(df[target_col], df[proba_col])
    gini = auroc * 2 - 1
    return gini


def plot_ks(
    df,
    target_col,
    proba_col,
    title="Kolmogorov-Smirnov",
    xlabel="Estimated Probability for target Good",
    ylabel="Cumulative %",
    good_color="blue",
    bad_color="red",
    savefig=False,
    figname="ks.png",
):
    """Create Kolmogorov-Smirnov plot for probabilities and binary target.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with the columns to be plotted.

    target_col : str
        Column with binary target.

    proba_col : str
        Column with the calculated probabilites.

    title : str (default='Kolmogorov-Smirnov'):
        Title for plot.

    xlabel : str (default='Estimated Probability for target Good')
        Label for the x-axis.

    ylabel : str (default='Cumulative %')
        Label for the y-axis.

    good_color : str (default='blue')
        Color for the event lineplot.

    bad_color : str (default='red')
        Color for the non-event lineplot.

    savefig : bool (default=False)
        To save figure.

    figname : str (default='ks.fig')
        Name for figure file.

    Returns
    -------
    ks_score : float
        Calculate KS Score.
    """
    # Sort dataframe by probabilities and reset index
    df = df.sort_values(proba_col)
    df = df.reset_index()

    # Calculate cumulative columns
    df["cumulative_n_population"] = df.index + 1
    df["cumulative_n_good"] = df[target_col].cumsum()
    df["cumulative_n_bad"] = (
        df["cumulative_n_population"] - df[target_col].cumsum()
    )
    df["cumulative_perc_population"] = df["cumulative_n_population"] / (
        df.shape[0]
    )
    df["cumulative_perc_good"] = df["cumulative_n_good"] / df[target_col].sum()
    df["cumulative_perc_bad"] = df["cumulative_n_bad"] / (
        df.shape[0] - df[target_col].sum()
    )

    # Plot KS
    plt.plot(
        df[proba_col], df["cumulative_perc_bad"], color=bad_color,
    )
    plt.plot(
        df[proba_col], df["cumulative_perc_good"], color=good_color,
    )
    plt.xlabel(
        'Estimated Probability for Good target', fontdict={'fontsize': 16}
    )
    plt.ylabel("Cumulative %", fontdict={'fontsize': 16})
    plt.title("Kolmogorov-Smirnov", fontdict={'fontsize': 16})

    ks_score = max(df["cumulative_perc_bad"] - df["cumulative_perc_good"])
    ks_max_idx = np.argmax(
        df["cumulative_perc_bad"] - df["cumulative_perc_good"]
    )

    # Define the KS line
    plt.vlines(
        df[proba_col].iloc[ks_max_idx],
        ymin=df["cumulative_perc_good"].iloc[ks_max_idx],
        ymax=df["cumulative_perc_bad"].iloc[ks_max_idx],
        color='black',
        linestyles='--',
    )

    # Set KS value inside plot
    plt.text(
        df[proba_col].iloc[ks_max_idx] + 0.1,
        (
            (
                df["cumulative_perc_good"].iloc[ks_max_idx]
                + df["cumulative_perc_bad"].iloc[ks_max_idx]
            )
            * 0.5
        ),
        f'KS = {round(ks_score * 100, 2)}%',
        fontsize=12,
        rotation_mode='anchor',
    )

    if savefig:
        plt.savefig(figname, dpi=300)

    return ks_score
