"""
Scorecard testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import pandas as pd
import numpy as np

from pytest import approx, raises

from contextlib import redirect_stdout

from optbinning import BinningProcess
from optbinning import Scorecard
from sklearn.datasets import load_boston
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


def test_params():
    data = load_breast_cancer()

    variable_names = data.feature_names
    df = pd.DataFrame(data.data, columns=variable_names)
    df["target"] = data.target

    binning_process = BinningProcess(variable_names)
    estimator = LogisticRegression()

    with raises(TypeError):
        scorecard = Scorecard(target=1, binning_process=binning_process,
                              estimator=estimator)
        scorecard.fit(df)

    with raises(TypeError):
        scorecard = Scorecard(target="target", binning_process=estimator,
                              estimator=estimator)
        scorecard.fit(df)

    with raises(TypeError):
        scorecard = Scorecard(target="target", binning_process=binning_process,
                              estimator=binning_process)
        scorecard.fit(df)

    with raises(ValueError):
        scorecard = Scorecard(target="target", binning_process=binning_process,
                              estimator=estimator, scaling_method="new_method",
                              scaling_method_params=dict())
        scorecard.fit(df)

    with raises(ValueError):
        scorecard = Scorecard(target="target", binning_process=binning_process,
                              estimator=estimator, scaling_method="min_max",
                              scaling_method_params=None)
        scorecard.fit(df)

    with raises(TypeError):
        scorecard = Scorecard(target="target", binning_process=binning_process,
                              estimator=estimator, scaling_method="min_max",
                              scaling_method_params=[])
        scorecard.fit(df)

    with raises(TypeError):
        scorecard = Scorecard(target="target", binning_process=binning_process,
                              estimator=estimator, intercept_based=1)
        scorecard.fit(df)

    with raises(TypeError):
        scorecard = Scorecard(target="target", binning_process=binning_process,
                              estimator=estimator, reverse_scorecard=1)
        scorecard.fit(df)

    with raises(TypeError):
        scorecard = Scorecard(target="target", binning_process=binning_process,
                              estimator=estimator, rounding=1)
        scorecard.fit(df)

    with raises(TypeError):
        scorecard = Scorecard(target="target", binning_process=binning_process,
                              estimator=estimator, verbose=1)
        scorecard.fit(df)


def test_scaling_method_params_continuous_pdo_odds():
    data = load_boston()
    variable_names = data.feature_names
    df = pd.DataFrame(data.data, columns=variable_names)
    df["target"] = data.target

    with raises(ValueError):
        estimator = LinearRegression()
        binning_process = BinningProcess(variable_names)

        scorecard = Scorecard(target="target", binning_process=binning_process,
                              estimator=estimator, scaling_method="pdo_odds",
                              scaling_method_params={})
        scorecard.fit(df)


def test_scaling_params():
    data = load_breast_cancer()

    variable_names = data.feature_names
    df = pd.DataFrame(data.data, columns=variable_names)
    df["target"] = data.target

    binning_process = BinningProcess(variable_names)
    estimator = LogisticRegression()

    with raises(ValueError):
        scorecard = Scorecard(target="target", binning_process=binning_process,
                              estimator=estimator, scaling_method="pdo_odds",
                              scaling_method_params={"pdo": 20})
        scorecard.fit(df)

    with raises(ValueError):
        scorecard = Scorecard(target="target", binning_process=binning_process,
                              estimator=estimator, scaling_method="pdo_odds",
                              scaling_method_params={"pdo": 20, "odds": -2,
                                                     "scorecard_points": -22})
        scorecard.fit(df)

    with raises(ValueError):
        scorecard = Scorecard(target="target", binning_process=binning_process,
                              estimator=estimator, scaling_method="min_max",
                              scaling_method_params={"min": "a", "max": 600})
        scorecard.fit(df)

    with raises(ValueError):
        scorecard = Scorecard(target="target", binning_process=binning_process,
                              estimator=estimator, scaling_method="min_max",
                              scaling_method_params={"min": 900, "max": 600})
        scorecard.fit(df)


def test_default():
    data = load_breast_cancer()

    variable_names = data.feature_names
    df = pd.DataFrame(data.data, columns=variable_names)
    df["target"] = data.target

    binning_process = BinningProcess(variable_names)
    estimator = LogisticRegression()

    scorecard = Scorecard(target="target", binning_process=binning_process,
                          estimator=estimator).fit(df)

    with raises(ValueError):
        sct = scorecard.table(style="new")

    sct = scorecard.table(style="summary")
    sc_min, sc_max = sct.groupby("Variable").agg(
        {'Points': [np.min, np.max]}).sum()

    assert sc_min == approx(-43.65762593147646, rel=1e-6)
    assert sc_max == approx(42.69694657427327, rel=1e-6)


def test_default_continuous():
    data = load_boston()

    variable_names = data.feature_names
    df = pd.DataFrame(data.data, columns=variable_names)
    df["target"] = data.target

    binning_process = BinningProcess(variable_names)
    estimator = LinearRegression()

    scorecard = Scorecard(target="target", binning_process=binning_process,
                          estimator=estimator).fit(df)

    sct = scorecard.table(style="detailed")
    sc_min, sc_max = sct.groupby("Variable").agg(
        {'Points': [np.min, np.max]}).sum()

    assert sc_min == approx(-15.813545796848476, rel=1e-6)
    assert sc_max == approx(85.08156623609487, rel=1e-6)


def test_scaling_method_pdo_odd():
    pass


def test_scaling_method_min_max():
    pass


def test_intercept_based():
    pass


def test_reverse_scorecard():
    pass
