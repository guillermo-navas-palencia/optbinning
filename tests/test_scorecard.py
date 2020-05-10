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
from sklearn.exceptions import NotFittedError
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


def test_input():
    data = load_breast_cancer()
    variable_names = data.feature_names
    df = pd.DataFrame(data.data, columns=variable_names)
    target = data.target
    target[0] = 4
    df["target"] = target

    binning_process = BinningProcess(variable_names)
    estimator = LogisticRegression()

    with raises(ValueError):
        scorecard = Scorecard(target="target", binning_process=binning_process,
                              estimator=estimator)
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
    data = load_breast_cancer()
    variable_names = data.feature_names
    df = pd.DataFrame(data.data, columns=variable_names)
    df["target"] = data.target
    odds = 1 / data.target.mean()

    binning_process = BinningProcess(variable_names)
    estimator = LogisticRegression()

    scaling_method_params = {"pdo": 20, "odds": odds, "scorecard_points": 600}

    scorecard = Scorecard(target="target", binning_process=binning_process,
                          estimator=estimator, scaling_method="pdo_odds",
                          scaling_method_params=scaling_method_params).fit(df)

    sct = scorecard.table(style="summary")
    sc_min, sc_max = sct.groupby("Variable").agg(
        {'Points': [np.min, np.max]}).sum()

    assert sc_min == approx(-612.2266586867094, rel=1e-6)
    assert sc_max == approx(1879.4396115559216, rel=1e-6)


def test_scaling_method_min_max():
    data = load_breast_cancer()
    variable_names = data.feature_names
    df = pd.DataFrame(data.data, columns=variable_names)
    df["target"] = data.target

    binning_process = BinningProcess(variable_names)
    estimator = LogisticRegression()

    scaling_method_params = {"min": 300, "max": 850}

    scorecard = Scorecard(target="target", binning_process=binning_process,
                          estimator=estimator, scaling_method="min_max",
                          scaling_method_params=scaling_method_params).fit(df)

    sct = scorecard.table(style="summary")
    sc_min, sc_max = sct.groupby("Variable").agg(
        {'Points': [np.min, np.max]}).sum()

    assert sc_min == approx(300, rel=1e-6)
    assert sc_max == approx(850, rel=1e-6)


def test_intercept_based():
    data = load_breast_cancer()
    variable_names = data.feature_names
    df = pd.DataFrame(data.data, columns=variable_names)
    df["target"] = data.target

    binning_process = BinningProcess(variable_names)
    estimator = LogisticRegression()

    scaling_method_params = {"min": 300, "max": 850}

    scorecard = Scorecard(target="target", binning_process=binning_process,
                          estimator=estimator, scaling_method="min_max",
                          scaling_method_params=scaling_method_params,
                          intercept_based=True).fit(df)

    sct = scorecard.table(style="summary")
    sc_min, sc_max = sct.groupby("Variable").agg(
        {'Points': [np.min, np.max]}).sum()

    assert sc_min == approx(300 - scorecard.intercept_, rel=1e-6)
    assert sc_max == approx(850 - scorecard.intercept_, rel=1e-6)


def test_reverse_scorecard():
    data = load_breast_cancer()
    variable_names = data.feature_names
    df = pd.DataFrame(data.data, columns=variable_names)
    df["target"] = data.target

    binning_process = BinningProcess(variable_names)
    estimator = LogisticRegression()

    scaling_method_params = {"min": 300, "max": 850}

    scorecard = Scorecard(target="target", binning_process=binning_process,
                          estimator=estimator, scaling_method="min_max",
                          scaling_method_params=scaling_method_params,
                          reverse_scorecard=True).fit(df)

    sct = scorecard.table(style="summary")
    sc_min, sc_max = sct.groupby("Variable").agg(
        {'Points': [np.min, np.max]}).sum()

    assert sc_min == approx(300, rel=1e-6)
    assert sc_max == approx(850, rel=1e-6)


def test_rounding():
    data = load_breast_cancer()
    variable_names = data.feature_names
    df = pd.DataFrame(data.data, columns=variable_names)
    df["target"] = data.target

    binning_process = BinningProcess(variable_names)
    estimator = LogisticRegression()

    scaling_method_params = {"min": 200.52, "max": 850.66}

    scorecard = Scorecard(target="target", binning_process=binning_process,
                          estimator=estimator, scaling_method="min_max",
                          scaling_method_params=scaling_method_params,
                          rounding=True).fit(df)

    sct = scorecard.table(style="summary")
    sc_min, sc_max = sct.groupby("Variable").agg(
        {'Points': [np.min, np.max]}).sum()

    assert sc_min == approx(201, rel=1e-6)
    assert sc_max == approx(851, rel=1e-6)


def test_rounding_pdo_odds():
    data = load_breast_cancer()
    variable_names = data.feature_names
    df = pd.DataFrame(data.data, columns=variable_names)
    df["target"] = data.target
    odds = 1 / data.target.mean()

    binning_process = BinningProcess(variable_names)
    estimator = LogisticRegression()

    scaling_method_params = {"pdo": 20, "odds": odds, "scorecard_points": 600}

    scorecard = Scorecard(target="target", binning_process=binning_process,
                          estimator=estimator, scaling_method="pdo_odds",
                          scaling_method_params=scaling_method_params,
                          rounding=True).fit(df)

    sct = scorecard.table(style="summary")
    sc_min, sc_max = sct.groupby("Variable").agg(
        {'Points': [np.min, np.max]}).sum()

    assert sc_min == approx(-612, rel=1e-6)
    assert sc_max == approx(1880, rel=1e-6)


def test_estimator_not_coef():
    from sklearn.ensemble import RandomForestClassifier

    data = load_breast_cancer()
    variable_names = data.feature_names
    df = pd.DataFrame(data.data, columns=variable_names)
    df["target"] = data.target

    binning_process = BinningProcess(variable_names)
    estimator = RandomForestClassifier()

    scorecard = Scorecard(target="target", binning_process=binning_process,
                          estimator=estimator)

    with raises(RuntimeError):
        scorecard.fit(df)


def test_predict_score():
    data = load_breast_cancer()
    variable_names = data.feature_names
    df = pd.DataFrame(data.data, columns=variable_names)
    df["target"] = data.target

    binning_process = BinningProcess(variable_names)
    estimator = LogisticRegression()
    scaling_method_params = {"min": 300.12, "max": 850.66}

    scorecard = Scorecard(target="target", binning_process=binning_process,
                          estimator=estimator, scaling_method="min_max",
                          scaling_method_params=scaling_method_params)

    with raises(NotFittedError):
        pred = scorecard.predict(df)

    with raises(NotFittedError):
        pred_proba = scorecard.predict_proba(df)

    with raises(NotFittedError):
        score = scorecard.score(df)

    scorecard.fit(df)
    pred = scorecard.predict(df)
    pred_proba = scorecard.predict_proba(df)
    score = scorecard.score(df)

    assert pred[:5] == approx([0, 0, 0, 0, 0])

    assert pred_proba[:5, 1] == approx(
        [1.15260206e-06, 9.79035720e-06, 7.52481206e-08, 1.12438599e-03,
         9.83145644e-06], rel=1e-6)

    assert score[:5] == approx([652.16590046, 638.52659074, 669.56413105,
                                608.27744027, 638.49988325], rel=1e-6)


def test_information():
    data = load_breast_cancer()
    variable_names = data.feature_names
    df = pd.DataFrame(data.data, columns=variable_names)
    df["target"] = data.target

    binning_process = BinningProcess(variable_names)
    estimator = LogisticRegression()
    scorecard = Scorecard(target="target", binning_process=binning_process,
                          estimator=estimator)

    with raises(NotFittedError):
        scorecard.information()

    scorecard.fit(df)

    with raises(ValueError):
        scorecard.information(print_level=-1)

    with open("tests/test_scorecard_information.txt", "w") as f:
        with redirect_stdout(f):
            scorecard.information(print_level=0)
            scorecard.information(print_level=1)
            scorecard.information(print_level=2)


def test_verbose():
    data = load_breast_cancer()
    variable_names = data.feature_names
    df = pd.DataFrame(data.data, columns=variable_names)
    df["target"] = data.target

    binning_process = BinningProcess(variable_names)
    estimator = LogisticRegression()
    scorecard = Scorecard(target="target", binning_process=binning_process,
                          estimator=estimator, verbose=True)

    with open("tests/test_scorecard_verbose.txt", "w") as f:
        with redirect_stdout(f):
            scorecard.fit(df)
