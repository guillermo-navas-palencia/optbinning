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


class PreBinning:
    """
    problem_type:
        The problem type depending on the target type.

    method : str
        Available methods are 'uniform', 'quantile' and 'cart'.

    n_bins : int
        The number of bins to produce.

    min_bin_size : int, float
        The minimum bin size.
    """
    def __init__(self, problem_type, method, n_bins, min_bin_size,
                 class_weight=None):

        self.problem_type = problem_type
        self.method = method
        self.n_bins = n_bins
        self.min_bin_size = min_bin_size
        self.class_weight = class_weight

        self._splits = None

    def fit(self, x, y):
        """
        x : array-like, shape = (n_samples)
            Data samples, where n_samples is the number of samples.

        y : array-like, shape = (n_samples)
            Target vector relative to x.

        Returns
        -------
        self : object
        """
        if self.method not in ("uniform", "quantile", "cart"):
            raise ValueError('Invalid value for prebinning_method. Allowed '
                             'string values are "cart", "quantile" and '
                             '"uniform".')

        if self.problem_type not in ("classification", "regression"):
            raise ValueError('Invalid value for problem_type. Allowed '
                             'string values are "classification" and '
                             '"regression".')

        if self.method in ("uniform", "quantile"):
            est = KBinsDiscretizer(n_bins=self.n_bins, strategy=self.method)
            est.fit(x.reshape(-1, 1), y)

            self._splits = est.bin_edges_[0][1:-1]
        elif self.method == "cart":
            if self.problem_type == "classification":
                est = DecisionTreeClassifier(
                    min_samples_leaf=self.min_bin_size,
                    max_leaf_nodes=self.n_bins, class_weight=self.class_weight)
            else:
                est = DecisionTreeRegressor(
                    min_samples_leaf=self.min_bin_size,
                    max_leaf_nodes=self.n_bins)

            est.fit(x.reshape(-1, 1), y)
            splits = np.unique(est.tree_.threshold)
            self._splits = splits[splits != _tree.TREE_UNDEFINED]

        return self

    @property
    def splits(self):
        return self._splits
