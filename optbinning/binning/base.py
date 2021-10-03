"""
Base optimal binning algorithm class.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

from abc import ABCMeta
from abc import abstractmethod

from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError


class Base:
    def _check_is_fitted(self):
        if not self._is_fitted:
            raise NotFittedError("This {} instance is not fitted yet. Call "
                                 "'fit' with appropriate arguments."
                                 .format(self.__class__.__name__))


class BaseOptimalBinning(Base, BaseEstimator, metaclass=ABCMeta):
    @abstractmethod
    def fit(self):
        """Fit the optimal binning according to the given training data."""

    @abstractmethod
    def fit_transform(self):
        """Fit the optimal binning according to the given training data, then
        transform it."""

    @abstractmethod
    def transform(self):
        """Transform given data using bins from the fitted optimal binning."""

    @abstractmethod
    def information(self):
        """Print overview information about the options settings, problem
        statistics, and the solution of the computation."""

    @property
    @abstractmethod
    def binning_table(self):
        """Return an instantiated binning table."""

    @property
    @abstractmethod
    def splits(self):
        """List of optimal split points."""

    @property
    @abstractmethod
    def status(self):
        """The status of the underlying optimization solver."""
