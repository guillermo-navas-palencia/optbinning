"""
Scorecard testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import pandas as pd
import numpy as np

from pytest import approx, raises

from optbinning import BinningProcess
from optbinning import Scorecard
from optbinning.scorecard import Counterfactual
from sklearn.datasets import load_boston
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


data = load_breast_cancer()
feature_names_binary = data.feature_names
df_binary = pd.DataFrame(data.data, columns=feature_names_binary)
df_binary["target"] = data.target

scorecard_binary = Scorecard(
    target="target",
    binning_process=BinningProcess(feature_names_binary),
    estimator=LogisticRegression()
    ).fit(df_binary)


data = load_boston()
feature_names_continuous = data.feature_names
df_continuous = pd.DataFrame(data.data, columns=feature_names_continuous)
df_continuous["target"] = data.target

scorecard_continuous = Scorecard(
    target="target",
    binning_process=BinningProcess(feature_names_continuous),
    estimator=LinearRegression() 
    ).fit(df_continuous)


def test_params():
    pass


def test_fit():
    pass


def test_generate():
    pass


def test_information():
    pass


def test_display():
    pass


def test_status():
    pass
