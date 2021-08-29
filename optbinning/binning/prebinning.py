"""
Pre-binning class.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import numpy as np

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import _tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

from .mdlp import MDLP


class PreBinning:
    """Prebinning algorithms.

    Parameters
    ----------
    problem_type:
        The problem type depending on the target type.

    method : str
        Available methods are 'uniform', 'quantile' and 'cart'.

    n_bins : int
        The number of bins to produce.

    min_bin_size : int, float
        The minimum bin size.

    **kwargs : keyword arguments
        Keyword arguments for prebinning method. See notes.

    Notes
    -----
    Keyword arguments are those available in the following classes:

        * ``method="uniform"``: `sklearn.preprocessing.KBinsDiscretizer.

        * ``method="quantile"``: `sklearn.preprocessing.KBinsDiscretizer.

        * ``method="cart"``: sklearn.tree.DecistionTreeClassifier.

        * ``method="mdlp"``: optbinning.binning.mdlp.MDLP.

    """
    def __init__(self, problem_type, method, n_bins, min_bin_size,
                 class_weight=None, **kwargs):

        self.problem_type = problem_type
        self.method = method
        self.n_bins = n_bins
        self.min_bin_size = min_bin_size
        self.class_weight = class_weight
        self.kwargs = kwargs

        self._splits = None

    def fit(self, x, y, sample_weight=None):
        """Fit PreBinning algorithm.

        Parameters
        ----------
        x : array-like, shape = (n_samples)
            Data samples, where n_samples is the number of samples.

        y : array-like, shape = (n_samples)
            Target vector relative to x.

        sample_weight : array-like of shape (n_samples,) (default=None)
            Array of weights that are assigned to individual samples.

        Returns
        -------
        self : PreBinning
        """
        if self.method not in ("uniform", "quantile", "cart", "mdlp"):
            raise ValueError('Invalid value for prebinning method. Allowed '
                             'string values are "cart", "mdlp", "quantile" '
                             'and "uniform".')

        if self.problem_type not in ("classification", "regression"):
            raise ValueError('Invalid value for problem_type. Allowed '
                             'string values are "classification" and '
                             '"regression".')

        if self.problem_type == "regression" and self.method == "mdlp":
            raise ValueError("mdlp method can only handle binary "
                             "classification problems.")

        if self.method in ("uniform", "quantile"):
            unsup_kwargs = {"n_bins": self.n_bins, "strategy": self.method}
            unsup_kwargs.update(**self.kwargs)

            est = KBinsDiscretizer(**unsup_kwargs)
            est.fit(x.reshape(-1, 1), y)
            self._splits = est.bin_edges_[0][1:-1]

        elif self.method == "cart":
            cart_kwargs = {
                    "min_samples_leaf": self.min_bin_size,
                    "max_leaf_nodes": self.n_bins}

            if self.problem_type == "classification":
                cart_kwargs["class_weight"] = self.class_weight
                cart_kwargs.update(**self.kwargs)

                est = DecisionTreeClassifier(**cart_kwargs)
            else:
                cart_kwargs.update(**self.kwargs)
                est = DecisionTreeRegressor(**cart_kwargs)

            est.fit(x.reshape(-1, 1), y, sample_weight=sample_weight)
            splits = np.unique(est.tree_.threshold)
            self._splits = splits[splits != _tree.TREE_UNDEFINED]

        elif self.method == "mdlp":
            mdlp_kwargs = {"min_samples_leaf": self.min_bin_size}
            mdlp_kwargs.update(**self.kwargs)

            est = MDLP(**mdlp_kwargs)
            est.fit(x, y)
            self._splits = est.splits

        return self

    @property
    def splits(self):
        """List of split points

        Returns
        -------
        splits : numpy.ndarray
        """
        return self._splits
