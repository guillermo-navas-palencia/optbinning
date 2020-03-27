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
    def __init__(self, name, dtype, prebinning_method, max_n_prebins,
                 min_prebin_size, min_n_bins, max_n_bins, min_bin_size,
                 max_bin_size, monotonic_trend, max_pvalue, max_pvalue_policy,
                 outlier_detector, outlier_params, user_splits,
                 user_splits_fixed, special_codes, split_digits, time_limit,
                 verbose):
        pass


    def fit(self, X, Y, weights=None, check_input=False):
        pass

