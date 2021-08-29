"""
Univariate outlier detection methods.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import numbers

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError


class OutlierDetector:
    """Base class for all outlier detectors."""
    def __init__(self):
        self._support = None

        # flag
        self._is_fitted = False

    def fit(self, x):
        """Fit univariate outlier detector.

        Parameters
        ----------
        x : array-like, shape = (n_samples)

        Returns
        -------
        self : OutlierDetector
        """
        self._fit(x)

        return self

    def get_support(self, indices=False):
        """Get a mask, or integer index, of the samples excluded, i.e, samples
        detected as outliers.

        Parameters
        ----------
        indices : boolean (default False)
            If True, the return value will be an array of integers, rather
            than a boolean mask.

        Returns
        -------
        support : array, shape = (n_samples)
            An index that selects the excluded samples from a vector.
            If `indices` is False, this is a boolean array, in which an element
            is True iff its corresponding sample is excluded. If `indices` is
            True, this is an integer array whose values are indices into the
            input vector.
        """
        if not self._is_fitted:
            raise NotFittedError("This {} instance is not fitted yet. Call "
                                 "'fit' with appropriate arguments."
                                 .format(self.__class__.__name__))

        mask = self._support
        return mask if not indices else np.where(mask)[0]


class RangeDetector(BaseEstimator, OutlierDetector):
    r"""Interquartile range or interval based outlier detection method.

    The default settings compute the usual interquartile range method.

    Parameters
    ----------
    interval_length : float (default=0.5)
        Compute ``interval_length``\% credible interval. This is a value in
        [0, 1].

    k : float (default=1.5)
        Tukey's factor.

    method : str (default="ETI")
        Method to compute credible intervals. Supported methods are Highest
        Density interval (``method="HDI"``) and Equal-tailed interval
        (``method="ETI"``).
    """
    def __init__(self, interval_length=0.5, k=1.5, method="ETI"):
        self.interval_length = interval_length
        self.k = k
        self.method = method

    def _fit(self, x):
        if self.method not in ("ETI", "HDI"):
            raise ValueError('Invalid value for method. Allowed string '
                             'values are "ETI" and "HDI".')

        if (not isinstance(self.interval_length, numbers.Number) or
                not 0 <= self.interval_length <= 1):
            raise ValueError("Interval length must a value in [0, 1]; got {}."
                             .format(self.interval_length))

        if self.method == "ETI":
            lower = 100 * (1 - self.interval_length) / 2
            upper = 100 * (1 + self.interval_length) / 2

            lb, ub = np.percentile(x, [lower, upper])
        else:
            n = len(x)
            xsorted = np.sort(x)
            n_included = int(np.ceil(self.interval_length * n))
            n_ci = n - n_included
            ci = xsorted[n_included:] - xsorted[:n_ci]
            j = np.argmin(ci)
            hdi_min = xsorted[j]
            hdi_max = xsorted[j + n_included]

            lb = hdi_min
            ub = hdi_max

        iqr = ub - lb
        lower_bound = lb - self.k * iqr
        upper_bound = ub + self.k * iqr

        self._support = (x > upper_bound) | (x < lower_bound)

        self._is_fitted = True


class ModifiedZScoreDetector(BaseEstimator, OutlierDetector):
    """Modified Z-score method.

    Parameters
    ----------
    threshold : float (default=3.5)
        Modified Z-scores with an absolute value of greater than the threshold
        are labeled as outliers.

    References
    ----------

    .. [IH93] B. Iglewicz and D. Hoaglin. "Volume 16: How to Detect and Handle
              Outliers", The ASQC Basic References in Quality Control:
              Statistical Techniques, Edward F. Mykytka, Ph.D., Editor, 1993.
    """
    def __init__(self, threshold=3.5):
        self.threshold = threshold

    def _fit(self, x):
        x = np.asarray(x)
        median = np.median(x)
        mad = np.median(np.abs(x - median))
        m_z_score = 0.6745 * (x - median) / mad

        self._support = np.abs(m_z_score) > self.threshold

        self._is_fitted = True
