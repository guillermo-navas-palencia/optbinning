"""
Optimal binning metrics.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import numpy as np

from scipy import special
from scipy import stats


def entropy(x):
    """Calculate the entropy of a discrete probability distribution.

    Parameters
    ----------
    x : array-like
        Discrete probability distribution.

    Returns
    -------
    entropy : float
    """
    x = np.asarray(x)
    return -special.xlogy(x, x).sum()


def gini(event, nonevent):
    """Calculate the Gini coefficient given the number of events and
    non-events.

    Parameters
    ----------
    event : array-like
        Number of events.

    nonevent : array-like
        Number of non-events.

    Returns
    -------
    gini : float
    """
    event = np.asarray(event)
    nonevent = np.asarray(nonevent)

    if event.shape != nonevent.shape:
        raise ValueError("event and nonevent must have same shape.")

    mask = (event + nonevent) > 0
    event = event[mask]
    nonevent = nonevent[mask]

    n = len(event)
    if n <= 1:
        return 0
    else:
        te = event.sum()
        tne = nonevent.sum()

        ner = nonevent / (event + nonevent)
        idx = np.argsort(ner)
        ev = event[idx]
        ne = nonevent[idx]

        s = np.zeros(n)
        s[1:] = 2.0 * ne[:-1].cumsum()

        return 1.0 - np.dot(ev, ne + s) / (te * tne)


def kullback_leibler(x, y, return_sum=False):
    """Calculate the Kullback-Leibler divergence between two distributions.

    Parameters
    ----------
    x : array-like
        Discrete probability distribution.

    y : array-like
        Discrete probability distribution.

    return_sum : bool
        Return sum of kullback-leibler values.

    Returns
    -------
    kullback_leibler : float or numpy.ndarray
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if x.shape != y.shape:
        raise ValueError("x and y must have same shape.")

    if return_sum:
        return special.xlogy(x, x / y).sum()
    else:
        return special.xlogy(x, x / y)


def jeffrey(x, y, return_sum=False):
    """Calculate the Jeffrey's divergence between two distributions.

    Parameters
    ----------
    x : array-like
        Discrete probability distribution.

    y : array-like
        Discrete probability distribution.

    return_sum : bool
        Return sum of jeffrey values.

    Returns
    -------
    jeffrey : float or numpy.ndarray
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if x.shape != y.shape:
        raise ValueError("x and y must have same shape.")

    if return_sum:
        return special.xlogy(x - y, x / y).sum()
    else:
        return special.xlogy(x - y, x / y)


def jensen_shannon(x, y, return_sum=False):
    """Calculate the Jensen-Shannon divergence between two distributions.

    Parameters
    ----------
    x : array-like
        Discrete probability distribution.

    y : array-like
        Discrete probability distribution.

    return_sum : bool
        Return sum of jensen shannon values.

    Returns
    -------
    jensen_shannon : float or numpy.ndarray
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if x.shape != y.shape:
        raise ValueError("x and y must have same shape.")

    m = 0.5 * (x + y)
    return 0.5 * (kullback_leibler(x, m, return_sum) +
                  kullback_leibler(y, m, return_sum))


def jensen_shannon_multivariate(X, weights=None):
    """Calculate Jensen-Shannon divergence between several distributions.

    Parameters
    ----------
    X : array-like, shape = (n_samples, n_distributions)
        Discrete probability distributions.

    weights : array-like, shape = (n_distributions)
        Array of weights associated with the distributions. If None all
        distributions are assumed to have equal weight.

    Returns
    -------
    jensen_shannon : float
    """
    if X.ndim < 2:
        raise ValueError("X must be a 2-D array")

    X = np.asarray(X)
    n = X.shape[1]

    if weights is not None:
        weights = np.asarray(weights)

        if len(weights) != n:
            raise ValueError("Shapes of X and weights differ.")

        if weights.sum() != 1.0:
            raise ValueError("weights must sum 1.")
    else:
        weights = np.ones(n) / n

    js = entropy(np.sum(weights * X, axis=1))
    js -= np.dot(weights, [entropy(X[:, i]) for i in range(n)])

    return js


def frequentist_pvalue(obs, pvalue_method):
    if pvalue_method == "chi2":
        t, p, _, _ = stats.chi2_contingency(obs, correction=False)
        return t, p
    else:
        oddratio, p = stats.fisher_exact(obs)
        return oddratio, p


def bayesian_probability(obs, n_samples):
    aA, aB, bA, bB = obs.ravel()

    r = np.arange(1, n_samples + 1)
    np.random.shuffle(r)
    v = (r - 0.5) / n_samples

    p = special.betainc(aA, bA, stats.beta(aB, bB).ppf(v)).mean()
    return p, 1 - p


def binning_quality_score(iv, p_values):
    # Score 1: Information value
    c = 0.39573882184806863
    score_1 = iv * np.exp(1/2 * (1 - (iv / c) ** 2)) / c

    # Score 2: pairwise p-values
    p_values = np.asarray(p_values)
    score_2 = np.prod(1 - p_values)

    return score_1 * score_2


def multiclass_binning_quality_score(js, n_classes, p_values):
    js_norm = js / np.log(n_classes)

    return binning_quality_score(js_norm, p_values)
