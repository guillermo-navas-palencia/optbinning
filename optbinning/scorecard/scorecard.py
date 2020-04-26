"""
Scorecard development.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from ..binning.binning_process import BinningProcess


def _check_scorecard_method(method):
    pass


def compute_scorecard_points(points, binning_tables, method, method_data, 
                             intercept):
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

        min_point = np.sum([np.min(bt["Points"]) for bt in binning_tables])
        max_point = np.sum([np.max(bt["Points"]) for bt in binning_tables])

        smin = intercept + min_point
        smax = intercept + max_point

        slope = (b - a) / (smax - smin)
        shift = b  * smax / (smax - smin)

    base_points = shift - slope * intercept
    new_points = base_points / n - slope * points

    return new_points


class Scorecard(BaseEstimator):
    def __init__(self, target, binning_process, estimator, scaling_method,
                 scaling_method_data, cutoff=None):
        self.target = target
        self.binning_process = binning_process
        self.estimator = estimator
        self.scaling_method = scaling_method
        self.scaling_method_data = scaling_method_data
        self.cutoff = cutoff

        self.binning_process_ = None
        self.estimator_ = None

    def fit(self, df, metric=None, metric_special=0, metric_missing=0,
            show_digits=2, check_input=False):
        
        return self._fit(df, metric, metric_special, metric_missing,
                         show_digits, check_input)

    def predict(self, df):
        pass

    def predict_proba(self, df):
        df_t = df[self.binning_process.variable_names]
        df_t = self.binning_process.transform(df_t)
        return self.estimator.predict_proba(df_t)

    def score(self, df):
        df_t = df[self.binning_process.variable_names]
        df_t = self.binning_process.transform(df_t, metric="indices")

        score_ = np.zeros(df_t.shape[0])
        selected_variables = self.binning_process.get_support(names=True)

        df_score = self._df_scorecard[["Variable", "Points"]]
        for variable in selected_variables:
            points = df_score[df_score["Variable"] == variable]["Points"].values
            score_ += points[df_t[variable]]

        return score_

    def table(self, style="summary"):
        if style == "summary":
            columns = ["Variable", "Bin", "Points"]
        else:
            main_columns = ["Variable", "Bin ID", "Bin"]
            columns = self._df_scorecard.columns
            rest_columns = [col for col in columns if col not in main_columns]
            columns = main_columns + rest_columns
            
        return self._df_scorecard[columns]

    def _fit(self, df, metric, metric_special, metric_missing, show_digits,
             check_input):
        # Fit binning process
        df_t = self.binning_process.fit_transform(
            df[self.binning_process.variable_names], df[self.target],
            metric, metric_special, metric_missing, show_digits,
            check_input)

        # Fit estimator
        self.estimator.fit(df_t, df[self.target])
        coef = self.estimator.coef_
        intercept = self.estimator.intercept_

        # Build scorecard
        bts = []

        selected_variables = self.binning_process.get_support(names=True)
        for i, variable in enumerate(selected_variables):
            optb = self.binning_process.get_binned_variable(variable)
            bt = optb.binning_table.build(add_totals=False)
            w = coef.ravel()[i]
            bt["Variable"] = variable
            bt["Coefficient"] = w
            bt["Points"] = bt["WoE"] * w
            bt.index.names = ['Bin ID']
            bt.reset_index(level=0, inplace=True)            
            bts.append(bt)

        df_scorecard = pd.concat(bts)
        df_scorecard.reset_index()

        # Apply score points
        if self.scaling_method is None:
            scaled_points = points
        else:
            points = df_scorecard["Points"]
            scaled_points = compute_scorecard_points(
                points, bts, self.scaling_method, self.scaling_method_data,
                intercept)

        df_scorecard["Points"] = scaled_points

        self._df_scorecard = df_scorecard      

        return self
