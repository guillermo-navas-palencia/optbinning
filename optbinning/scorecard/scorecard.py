"""
Scorecard development.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.multiclass import type_of_target

from ..binning.binning_process import BinningProcess


def _check_scorecard_scaling(scaling_method, scaling_method_data):
    if scaling_method is not None:
        if scaling_method == "pd_odds":
            pass
        elif scaling_method == "shift_slope":
            pass
        elif scaling_method == "min_max":
            pass


def compute_scorecard_points(points, binning_tables, method, method_data,
                             intercept):
    """Apply scaling method to scorecard."""
    n = len(binning_tables)

    if method == "pdo_odds":
        pdo = method_data["pdo"]
        odds = method_data["odds"]
        scorecard_points = method_data["scorecard_points"]

        slope = pdo / np.log(2)
        shift = scorecard_points - slope * np.log(odds)

    elif method == "shift_slope":
        shift = method_data["shift"]
        slope = method_data["slope"]

    elif method == "min_max":
        a = method_data["min"]
        b = method_data["max"]

        min_point = np.sum([np.min(bt.Points) for bt in binning_tables])
        max_point = np.sum([np.max(bt.Points) for bt in binning_tables])

        smin = intercept + min_point
        smax = intercept + max_point

        slope = (b - a) / (smax - smin)
        shift = b * smax / (smax - smin)

    base_points = shift - slope * intercept
    new_points = base_points / n - slope * points

    return new_points


def compute_intercept_based(df_scorecard):
    """Compute an intercept-based scorecard.

    All points within a variable are adjusted so that the lowest point is zero.
    """
    scaled_points = np.zeros(df_scorecard.shape[0])
    selected_variables = df_scorecard.Variable.unique()
    intercept = 0
    for variable in selected_variables:
        mask = df_scorecard.Variable == variable
        points = df_scorecard[mask].Points.values
        min_point = np.min(points)
        scaled_points[mask] = points - min_point
        intercept += min_point

    return scaled_points, intercept


class Scorecard(BaseEstimator):
    def __init__(self, target, binning_process, estimator, scaling_method=None,
                 scaling_method_data=None, intercept_based=False):
        """Scorecard.

        Parameters
        ----------
        target : str

        binning_process : object

        estimator : object

        scaling_method : str or None (default=None)

        scaling_method_data : dict or None (default=None)

        intercept_based : bool (default=False)

        Attributes
        ----------
        binning_process_ : object
            The external binning process.

        estimator_ : object
            The external estimator fit on the reduced dataset.

        intercept_ : float
            The intercept if ``intercept_based=True``.
        """
        self.target = target
        self.binning_process = binning_process
        self.estimator = estimator
        self.scaling_method = scaling_method
        self.scaling_method_data = scaling_method_data
        self.intercept_based = intercept_based

        self.binning_process_ = None
        self.estimator_ = None

        self.intercept_ = 0

        # auxiliary
        self._target_dtype = None

    def fit(self, df, metric_special=0, metric_missing=0, show_digits=2,
            check_input=False):
        """Fit scorecard.

        Parameters
        ----------
        metric_special : float or str (default=0)
            The metric value to transform special codes in the input vector.
            Supported metrics are "empirical" to use the empirical WoE or
            event rate, and any numerical value.

        metric_missing : float or str (default=0)
            The metric value to transform missing values in the input vector.
            Supported metrics are "empirical" to use the empirical WoE or
            event rate and any numerical value.

        check_input : bool (default=False)
            Whether to check input arrays.

        Returns
        -------
        self : object
            Fitted scorecard.
        """
        return self._fit(df, metric_special, metric_missing, show_digits,
                         check_input)

    def predict(self, df):
        df_t = df[self.binning_process_.variable_names]
        df_t = self.binning_process_.transform(df_t)
        return self.estimator_.predict(df_t)

    def predict_proba(self, df):
        df_t = df[self.binning_process_.variable_names]
        df_t = self.binning_process_.transform(df_t)
        return self.estimator_.predict_proba(df_t)

    def score(self, df):
        df_t = df[self.binning_process_.variable_names]
        df_t = self.binning_process_.transform(df_t, metric="indices")

        score_ = np.zeros(df_t.shape[0])
        selected_variables = self.binning_process_.get_support(names=True)

        for variable in selected_variables:
            mask = self._df_scorecard.Variable == variable
            points = self._df_scorecard[mask].Points.values
            score_ += points[df_t[variable]]

        return score_ + self.intercept_

    def table(self, style="summary"):
        if style == "summary":
            columns = ["Variable", "Bin", "Points"]
        else:
            main_columns = ["Variable", "Bin id", "Bin"]
            columns = self._df_scorecard.columns
            rest_columns = [col for col in columns if col not in main_columns]
            columns = main_columns + rest_columns

        return self._df_scorecard[columns]

    def _fit(self, df, metric_special, metric_missing, show_digits,
             check_input):

        # Target type and metric
        target = df[self.target]
        self._target_dtype = type_of_target(target)

        if self._target_dtype not in ("binary", "continuous"):
            raise ValueError("Target type {} is not supported."
                             .format(self._target_dtype))

        if self._target_dtype == "binary":
            metric = "woe"
            binning_table_metric = "WoE"
        elif self._target_dtype == "continuous":
            metric = "mean"
            binning_table_metric = "Mean"

        # Fit binning process
        self.binning_process_ = clone(self.binning_process)

        df_t = self.binning_process_.fit_transform(
            df[self.binning_process.variable_names], target,
            metric, metric_special, metric_missing, show_digits,
            check_input)

        # Fit estimator
        self.estimator_ = clone(self.estimator)

        self.estimator_.fit(df_t, df[self.target])
        coef = self.estimator_.coef_
        intercept = self.estimator_.intercept_

        # Build scorecard
        selected_variables = self.binning_process_.get_support(names=True)
        binning_tables = []
        for i, variable in enumerate(selected_variables):
            optb = self.binning_process_.get_binned_variable(variable)
            binning_table = optb.binning_table.build(add_totals=False)
            c = coef.ravel()[i]
            binning_table["Variable"] = variable
            binning_table["Coefficient"] = c
            binning_table["Points"] = binning_table[binning_table_metric] * c
            binning_table.index.names = ['Bin id']
            binning_table.reset_index(level=0, inplace=True)
            binning_tables.append(binning_table)

        df_scorecard = pd.concat(binning_tables)
        df_scorecard.reset_index()

        # Apply score points
        if self.scaling_method is not None:
            points = df_scorecard["Points"]
            scaled_points = compute_scorecard_points(
                points, binning_tables, self.scaling_method,
                self.scaling_method_data, intercept)

            df_scorecard["Points"] = scaled_points

            if self.intercept_based:
                scaled_points, self.intercept_ = compute_intercept_based(
                    df_scorecard)
                df_scorecard["Points"] = scaled_points

        self._df_scorecard = df_scorecard

        return self
