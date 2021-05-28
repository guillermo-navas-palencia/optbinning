"""
Base counterfactual algorithm class.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2021

from abc import ABCMeta
from abc import abstractmethod

from sklearn.base import BaseEstimator

from ...binning.base import Base
from ...exceptions import CounterfactualsFoundWarning
from ...exceptions import NotGeneratedError


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

    def _check_is_generated(self):
        if not self._is_generated:
            raise NotGeneratedError("This {} instance has not generated "
                                    "counterfactuals yet. Call "
                                    "'generate' with appropriate arguments."
                                    .format(self.__class__.__name__))

    def _check_counterfactual_is_found(self):
        if not self._cfs:
            raise CounterfactualsFoundWarning(
                "Neither optimal or feasible counterfactuals were found.")
