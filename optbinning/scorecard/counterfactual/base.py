"""
Base counterfactual algorithm class.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2021

from abc import ABCMeta
from abc import abstractmethod

from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from ...binning.base import Base
from ...logging import Logger


class BaseCounterfactual(Base, BaseEstimator, metaclass=ABCMeta):
    @abstractmethod
    def fit(self):
        """Fit counterfactual with training data."""

    @abstractmethod
    def generate(self):
        """Generate counterfactual explanations."""

    @abstractmethod
    def display(self):
        """Display counterfactual explanations."""

    @property
    @abstractmethod
    def status(self):
        """The status of the underlying optimization solver."""
