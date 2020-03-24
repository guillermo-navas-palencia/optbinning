"""
Optimal binning algorithm given scenarions. Deterministic equivalent to
stochastic optimal binning.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import logging
import numbers
import time

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_array


class DSOptimalBinning(BaseEstimator):
    """
    Deterministic equivalent of the stochastic optimal binning problem.

    Parameters
    ----------
    """
    def __init__(self):
        pass


