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
from sklearn.datasets import load_breast_cancer
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from tests.datasets import load_boston


def test_params():
    data = load_breast_cancer()
    variable_names = data.feature_names
    X = pd.DataFrame(data.data, columns=variable_names)
    y = data.target

    binning_process = BinningProcess(variable_names)
    estimator = LogisticRegression()

    with raises(TypeError):
        scorecard = Scorecard(binning_process=estimator,
                              estimator=estimator)
        scorecard.fit(X, y)

    with raises(TypeError):
        scorecard = Scorecard(binning_process=binning_process,
                              estimator=binning_process)
        scorecard.fit(X, y)

    with raises(ValueError):
        scorecard = Scorecard(binning_process=binning_process,
                              estimator=estimator, scaling_method="new_method",
                              scaling_method_params=dict())
        scorecard.fit(X, y)

    with raises(ValueError):
        scorecard = Scorecard(binning_process=binning_process,
                              estimator=estimator, scaling_method="min_max",
                              scaling_method_params=None)
        scorecard.fit(X, y)

    with raises(TypeError):
        scorecard = Scorecard(binning_process=binning_process,
                              estimator=estimator, scaling_method="min_max",
                              scaling_method_params=[])
        scorecard.fit(X, y)

    with raises(TypeError):
        scorecard = Scorecard(binning_process=binning_process,
                              estimator=estimator, intercept_based=1)
        scorecard.fit(X, y)

    with raises(ValueError):
        scorecard = Scorecard(binning_process=binning_process,
                              estimator=estimator,
                              scaling_method=None, rounding=True)
        scorecard.fit(X, y)

    with raises(ValueError):
        scorecard = Scorecard(binning_process=binning_process,
                              estimator=estimator, scaling_method="min_max",
                              scaling_method_params={'min': 1.1, 'max': 10},
                              rounding=True)
        scorecard.fit(X, y)

    with raises(ValueError):
        scorecard = Scorecard(binning_process=binning_process,
                              estimator=estimator, scaling_method="min_max",
                              scaling_method_params={'min': 1, 'max': 10.1},
                              rounding=True)
        scorecard.fit(X, y)

    with raises(TypeError):
        scorecard = Scorecard(binning_process=binning_process,
                              estimator=estimator, reverse_scorecard=1)
        scorecard.fit(X, y)

    with raises(TypeError):
        scorecard = Scorecard(binning_process=binning_process,
                              estimator=estimator, rounding=1)
        scorecard.fit(X, y)

    with raises(TypeError):
        scorecard = Scorecard(binning_process=binning_process,
                              estimator=estimator, verbose=1)
        scorecard.fit(X, y)


def test_scaling_method_params_continuous_pdo_odds():
    data = load_boston()
    variable_names = data.feature_names
    X = pd.DataFrame(data.data, columns=variable_names)
    y = data.target

    with raises(ValueError):
        estimator = LinearRegression()
        binning_process = BinningProcess(variable_names)

        scorecard = Scorecard(binning_process=binning_process,
                              estimator=estimator, scaling_method="pdo_odds",
                              scaling_method_params={})
        scorecard.fit(X, y)


def test_scaling_params():
    data = load_breast_cancer()

    variable_names = data.feature_names
    X = pd.DataFrame(data.data, columns=variable_names)
    y = data.target

    binning_process = BinningProcess(variable_names)
    estimator = LogisticRegression()

    with raises(ValueError):
        scorecard = Scorecard(binning_process=binning_process,
                              estimator=estimator, scaling_method="pdo_odds",
                              scaling_method_params={"pdo": 20})
        scorecard.fit(X, y)

    with raises(ValueError):
        scorecard = Scorecard(binning_process=binning_process,
                              estimator=estimator, scaling_method="pdo_odds",
                              scaling_method_params={"pdo": 20, "odds": -2,
                                                     "scorecard_points": -22})
        scorecard.fit(X, y)

    with raises(ValueError):
        scorecard = Scorecard(binning_process=binning_process,
                              estimator=estimator, scaling_method="min_max",
                              scaling_method_params={"min": "a", "max": 600})
        scorecard.fit(X, y)

    with raises(ValueError):
        scorecard = Scorecard(binning_process=binning_process,
                              estimator=estimator, scaling_method="min_max",
                              scaling_method_params={"min": 900, "max": 600})
        scorecard.fit(X, y)


def test_input():
    data = load_breast_cancer()
    variable_names = data.feature_names
    X = pd.DataFrame(data.data, columns=variable_names)
    y = data.target
    y[0] = 4

    binning_process = BinningProcess(variable_names)
    estimator = LogisticRegression()

    with raises(ValueError):
        scorecard = Scorecard(binning_process=binning_process,
                              estimator=estimator)
        scorecard.fit(X, y)


def test_default():
    data = load_breast_cancer()
    variable_names = data.feature_names
    X = pd.DataFrame(data.data, columns=variable_names)
    y = data.target

    binning_process = BinningProcess(variable_names)
    estimator = LogisticRegression()

    scorecard = Scorecard(binning_process=binning_process,
                          estimator=estimator).fit(X, y)

    with raises(ValueError):
        sct = scorecard.table(style="new")

    sct = scorecard.table(style="summary")
    sc_min, sc_max = sct.groupby("Variable").agg(
        {'Points': ['min', 'max']}).sum()

    assert sc_min == approx(-43.5354465187911, rel=1e-6)
    assert sc_max == approx(42.55760963498596, rel=1e-6)


def test_default_continuous():
    data = load_boston()
    variable_names = data.feature_names
    X = pd.DataFrame(data.data, columns=variable_names)
    y = data.target

    binning_process = BinningProcess(variable_names)
    estimator = LinearRegression()

    scorecard = Scorecard(binning_process=binning_process,
                          estimator=estimator).fit(X, y)

    sct = scorecard.table(style="detailed")
    sc_min, sc_max = sct.groupby("Variable").agg(
        {'Points': ['min', 'max']}).sum()

    assert sc_min == approx(-43.261900687199045, rel=1e-6)
    assert sc_max == approx(100.28829019286185, rel=1e-6)


def test_scaling_method_pdo_odd():
    data = load_breast_cancer()
    variable_names = data.feature_names
    X = pd.DataFrame(data.data, columns=variable_names)
    y = data.target
    odds = 1 / data.target.mean()

    binning_process = BinningProcess(variable_names)
    estimator = LogisticRegression()

    scaling_method_params = {"pdo": 20, "odds": odds, "scorecard_points": 600}

    scorecard = Scorecard(binning_process=binning_process, estimator=estimator,
                          scaling_method="pdo_odds",
                          scaling_method_params=scaling_method_params
                          ).fit(X, y)

    sct = scorecard.table(style="summary")
    sc_min, sc_max = sct.groupby("Variable").agg(
        {'Points': ['min', 'max']}).sum()

    assert sc_min == approx(-608.2909715472422, rel=1e-6)
    assert sc_max == approx(1875.829531813342, rel=1e-6)


def test_scaling_method_min_max():
    data = load_breast_cancer()
    variable_names = data.feature_names
    X = pd.DataFrame(data.data, columns=variable_names)
    y = data.target

    binning_process = BinningProcess(variable_names)
    estimator = LogisticRegression()

    scaling_method_params = {"min": 300, "max": 850}

    scorecard = Scorecard(binning_process=binning_process,
                          estimator=estimator, scaling_method="min_max",
                          scaling_method_params=scaling_method_params
                          ).fit(X, y)

    sct = scorecard.table(style="summary")
    sc_min, sc_max = sct.groupby("Variable").agg(
        {'Points': ['min', 'max']}).sum()

    assert sc_min == approx(300, rel=1e-6)
    assert sc_max == approx(850, rel=1e-6)


def test_intercept_based():
    data = load_breast_cancer()
    variable_names = data.feature_names
    X = pd.DataFrame(data.data, columns=variable_names)
    y = data.target

    binning_process = BinningProcess(variable_names)
    estimator = LogisticRegression()

    scaling_method_params = {"min": 300, "max": 850}

    scorecard = Scorecard(binning_process=binning_process,
                          estimator=estimator, scaling_method="min_max",
                          scaling_method_params=scaling_method_params,
                          intercept_based=True).fit(X, y)

    sct = scorecard.table(style="summary")
    sc_min, sc_max = sct.groupby("Variable").agg(
        {'Points': ['min', 'max']}).sum()

    assert sc_min == approx(300 - scorecard.intercept_, rel=1e-6)
    assert sc_max == approx(850 - scorecard.intercept_, rel=1e-6)


def test_reverse_scorecard():
    data = load_breast_cancer()
    variable_names = data.feature_names
    X = pd.DataFrame(data.data, columns=variable_names)
    y = data.target

    binning_process = BinningProcess(variable_names)
    estimator = LogisticRegression()

    scaling_method_params = {"min": 300, "max": 850}

    scorecard = Scorecard(binning_process=binning_process,
                          estimator=estimator, scaling_method="min_max",
                          scaling_method_params=scaling_method_params,
                          reverse_scorecard=True).fit(X, y)

    sct = scorecard.table(style="summary")
    sc_min, sc_max = sct.groupby("Variable").agg(
        {'Points': ['min', 'max']}).sum()

    assert sc_min == approx(300, rel=1e-6)
    assert sc_max == approx(850, rel=1e-6)


def test_rounding():
    data = load_breast_cancer()
    variable_names = data.feature_names
    X = pd.DataFrame(data.data, columns=variable_names)
    y = data.target

    binning_process = BinningProcess(variable_names)
    estimator = LogisticRegression()

    scaling_method_params = {"min": 200, "max": 851}

    scorecard = Scorecard(binning_process=binning_process,
                          estimator=estimator, scaling_method="min_max",
                          scaling_method_params=scaling_method_params,
                          rounding=True).fit(X, y)

    sct = scorecard.table(style="summary")
    sc_min, sc_max = sct.groupby("Variable").agg(
        {'Points': ['min', 'max']}).sum()

    assert sc_min == approx(200, rel=1e-6)
    assert sc_max == approx(851, rel=1e-6)


def test_rounding_pdo_odds():
    data = load_breast_cancer()
    variable_names = data.feature_names
    X = pd.DataFrame(data.data, columns=variable_names)
    y = data.target
    odds = 1 / data.target.mean()

    binning_process = BinningProcess(variable_names)
    estimator = LogisticRegression()

    scaling_method_params = {"pdo": 20, "odds": odds, "scorecard_points": 600}

    scorecard = Scorecard(binning_process=binning_process,
                          estimator=estimator, scaling_method="pdo_odds",
                          scaling_method_params=scaling_method_params,
                          rounding=True).fit(X, y)

    sct = scorecard.table(style="summary")
    sc_min, sc_max = sct.groupby("Variable").agg(
        {'Points': ['min', 'max']}).sum()

    assert sc_min == approx(-609, rel=1e-6)
    assert sc_max == approx(1876, rel=1e-6)


def test_estimator_not_coef():
    from sklearn.ensemble import RandomForestClassifier

    data = load_breast_cancer()
    variable_names = data.feature_names
    X = pd.DataFrame(data.data, columns=variable_names)
    y = data.target

    binning_process = BinningProcess(variable_names)
    estimator = RandomForestClassifier()

    scorecard = Scorecard(binning_process=binning_process, estimator=estimator)

    with raises(RuntimeError):
        scorecard.fit(X, y)


def test_predict_score():
    data = load_breast_cancer()
    variable_names = data.feature_names
    X = pd.DataFrame(data.data, columns=variable_names)
    y = data.target

    binning_process = BinningProcess(variable_names)
    estimator = LogisticRegression()
    scaling_method_params = {"min": 300.12, "max": 850.66}

    scorecard = Scorecard(binning_process=binning_process,
                          estimator=estimator, scaling_method="min_max",
                          scaling_method_params=scaling_method_params)

    with raises(NotFittedError):
        pred = scorecard.predict(X)

    with raises(NotFittedError):
        pred_proba = scorecard.predict_proba(X)

    with raises(NotFittedError):
        score = scorecard.score(X)

    scorecard.fit(X, y)
    pred = scorecard.predict(X)
    pred_proba = scorecard.predict_proba(X)
    score = scorecard.score(X)

    assert pred[:5] == approx([0, 0, 0, 0, 0])

    expected_pred_proba = [
        1.18812864e-06, 
        1.01521192e-05, 
        7.65959946e-08, 
        1.09683243e-03,
        9.99982719e-06
    ]
    assert pred_proba[:5, 1] == approx(expected_pred_proba, rel=1e-6)

    expected_score = [
        652.16890659, 
        638.45026205, 
        669.70058258, 
        608.50009151,
        638.54691686
    ]
    assert score[:5] == approx(expected_score, rel=1e-6)


def test_information():
    data = load_breast_cancer()
    variable_names = data.feature_names
    X = pd.DataFrame(data.data, columns=variable_names)
    y = data.target

    binning_process = BinningProcess(variable_names)
    estimator = LogisticRegression()
    scorecard = Scorecard(binning_process=binning_process, estimator=estimator)

    with raises(NotFittedError):
        scorecard.information()

    scorecard.fit(X, y)

    with raises(ValueError):
        scorecard.information(print_level=-1)

    with open("tests/results/test_scorecard_information.txt", "w") as f:
        with redirect_stdout(f):
            scorecard.information(print_level=0)
            scorecard.information(print_level=1)
            scorecard.information(print_level=2)


def test_verbose():
    data = load_breast_cancer()
    variable_names = data.feature_names
    X = pd.DataFrame(data.data, columns=variable_names)
    y = data.target

    binning_process = BinningProcess(variable_names)
    estimator = LogisticRegression()
    scorecard = Scorecard(binning_process=binning_process, estimator=estimator,
                          verbose=True)

    with open("tests/results/test_scorecard_verbose.txt", "w") as f:
        with redirect_stdout(f):
            scorecard.fit(X, y)


def test_missing_metrics():
    data = pd.DataFrame(
        {'target': np.hstack(
            (np.tile(np.array([0, 1]), 50),
             np.array([0]*90 + [1]*10)
             )
         ),
         'var': [np.nan] * 100 + ['A'] * 100}
    )

    binning_process = BinningProcess(['var'])
    scaling_method_params = {'min': 0, 'max': 100}

    scorecard = Scorecard(
        binning_process=binning_process,
        estimator=LogisticRegression(),
        scaling_method="min_max",
        scaling_method_params=scaling_method_params
    ).fit(data, data.target)

    assert scorecard.table()['Points'].iloc[-1] == approx(0, rel=1e-6)
