"""
Minimum Description Length Principle (MDLP)
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import numbers

import numpy as np

from scipy import special
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_array


def _check_parameters(min_samples_split, min_samples_leaf, max_candidates):
    if (not isinstance(min_samples_split, numbers.Integral) or
            min_samples_split < 2):
        raise ValueError("min_samples_split must be a positive integer >= 2; "
                         "got {}.".format(min_samples_split))

    if (not isinstance(min_samples_leaf, numbers.Integral) or
            min_samples_leaf < 1):
        raise ValueError("min_samples_leaf must be a positive integer >= 1; "
                         "got {}.".format(min_samples_leaf))

    if not isinstance(max_candidates, numbers.Integral) or max_candidates < 1:
        raise ValueError("max_candidates must be a positive integer >= 1; "
                         "got {}.".format(max_candidates))


class MDLP(BaseEstimator):
    """
    Minimum Description Length Principle (MDLP) discretization algorithm.

    Parameters
    ----------
    min_samples_split : int (default=2)
        The minimum number of samples required to split an internal node.

    min_samples_leaf : int (default=2)
        The minimum number of samples required to be at a leaf node.

    max_candidates : int (default=32)
        The maximum number of split points to evaluate at each partition.

    Notes
    -----
    Implementation of the discretization algorithm in [FI93]. A dynamic
    split strategy based on binning the number of candidate splits [CMR2001]
    is implemented to increase efficiency. For large size datasets, it is
    recommended to use a smaller ``max_candidates`` (e.g. 16) to get a
    significant speed up.

    References
    ----------

    .. [FI93] U. M. Fayyad and K. B. Irani. "Multi-Interval Discretization of
              Continuous-Valued Attributes for Classification Learning".
              International Joint Conferences on Artificial Intelligence,
              13:1022–1027, 1993.

    .. [CMR2001] D. M. Chickering, C. Meek and R. Rounthwaite. "Efficient
                 Determination of Dynamic Split Points in a Decision Tree". In
                 Proceedings of the 2001 IEEE International Conference on Data
                 Mining, 91-98, 2001.
    """
    def __init__(self, min_samples_split=2, min_samples_leaf=2,
                 max_candidates=32):

        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_candidates = max_candidates

        # auxiliary
        self._splits = []

        self._is_fitted = None

    def fit(self, x, y):
        """Fit MDLP discretization algorithm.

        Parameters
        ----------
        x : array-like, shape = (n_samples)
            Data samples, where n_samples is the number of samples.

        y : array-like, shape = (n_samples)
            Target vector relative to x.

        Returns
        -------
        self : MDLP
        """
        return self._fit(x, y)

    def _fit(self, x, y):
        _check_parameters(**self.get_params())

        x = check_array(x, ensure_2d=False, force_all_finite=True)
        y = check_array(y, ensure_2d=False, force_all_finite=True)

        idx = np.argsort(x)
        x = x[idx]
        y = y[idx]

        self._recurse(x, y, 0)

        self._is_fitted = True

        return self

    def _recurse(self, x, y, id):
        u_x = np.unique(x)
        n_x = len(u_x)
        n_y = len(np.bincount(y))

        split = self._find_split(u_x, x, y)

        if split is not None:
            self._splits.append(split)
            t = np.searchsorted(x, split, side="right")

            if not self._terminate(n_x, n_y, y, y[:t], y[t:]):
                self._recurse(x[:t], y[:t], id + 1)
                self._recurse(x[t:], y[t:], id + 2)

    def _find_split(self, u_x, x, y):
        n_x = len(x)
        u_x = np.unique(0.5 * (x[1:] + x[:-1])[(y[1:] - y[:-1]) != 0])

        if len(u_x) > self.max_candidates:
            percentiles = np.linspace(1, 100, self.max_candidates)
            splits = np.percentile(u_x, percentiles)
        else:
            splits = u_x

        max_entropy_gain = 0
        best_split = None

        tt = np.searchsorted(x, splits, side="right")
        for i, t in enumerate(tt):
            samples_l = t >= self.min_samples_leaf
            samples_r = n_x - t >= self.min_samples_leaf

            if samples_l and samples_r:
                entropy_gain = self._entropy_gain(y, y[:t], y[t:])
                if entropy_gain > max_entropy_gain:
                    max_entropy_gain = entropy_gain
                    best_split = splits[i]

        return best_split

    def _entropy(self, x):
        n = len(x)
        ns1 = np.sum(x)
        ns0 = n - ns1
        p = np.array([ns0, ns1]) / n
        return -special.xlogy(p, p).sum()

    def _entropy_gain(self, y, y1, y2):
        n = len(y)
        n1 = len(y1)
        n2 = n - n1
        ent_y = self._entropy(y)
        ent_y1 = self._entropy(y1)
        ent_y2 = self._entropy(y2)
        return ent_y - (n1 * ent_y1 + n2 * ent_y2) / n

    def _terminate(self, n_x, n_y, y, y1, y2):
        splittable = (n_x >= self.min_samples_split) and (n_y >= 2)

        n = len(y)
        n1 = len(y1)
        n2 = n - n1
        ent_y = self._entropy(y)
        ent_y1 = self._entropy(y1)
        ent_y2 = self._entropy(y2)
        gain = ent_y - (n1 * ent_y1 + n2 * ent_y2) / n

        k = len(np.bincount(y))
        k1 = len(np.bincount(y1))
        k2 = len(np.bincount(y2))

        t0 = np.log(3**k - 2)
        t1 = k * ent_y
        t2 = k1 * ent_y1
        t3 = k2 * ent_y2
        delta = t0 - (t1 - t2 - t3)

        return gain <= (np.log(n - 1) + delta) / n or not splittable

    @property
    def splits(self):
        """List of split points

        Returns
        -------
        splits : numpy.ndarray
        """
        if not self._is_fitted:
            raise NotFittedError("This {} instance is not fitted yet. Call "
                                 "'fit' with appropriate arguments."
                                 .format(self.__class__.__name__))

        return np.sort(self._splits)
