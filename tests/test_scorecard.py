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


def test_per_variable_metric_special():
    """Test that per-variable metric_special from binning_transform_params
    is used for scorecard Points calculation."""
    data = load_breast_cancer()
    variable_names = ['mean radius', 'mean texture', 'mean perimeter']
    X = pd.DataFrame(data.data[:, :3], columns=variable_names)
    y = data.target
    
    # Add special codes to variables
    X_with_specials = X.copy()
    X_with_specials.loc[:20, 'mean radius'] = -999  # Special code
    X_with_specials.loc[:20, 'mean texture'] = -888  # Special code
    
    binning_process = BinningProcess(
        variable_names=variable_names,
        binning_fit_params={
            'mean radius': {'special_codes': [-999]},
            'mean texture': {'special_codes': [-888]},
        },
        binning_transform_params={
            'mean radius': {'metric': 'woe', 'metric_special': 'empirical'},
            'mean texture': {'metric': 'woe', 'metric_special': 0.5},
            'mean perimeter': {'metric': 'woe'},  # Will use global default
        }
    )
    
    scorecard = Scorecard(
        binning_process=binning_process,
        estimator=LogisticRegression(),
    )
    
    # Fit with global metric_special=0 (should be overridden per-variable)
    scorecard.fit(X_with_specials, y, metric_special=0)
    
    table = scorecard.table(style='detailed')
    
    # For 'mean radius': metric_special='empirical', so Points should use empirical WoE
    # For 'mean texture': metric_special=0.5, so Points should be 0.5 * coefficient
    # For 'mean perimeter': no binning_transform_params metric_special, so use global=0
    
    # Get the special code rows
    radius_table = table[table['Variable'] == 'mean radius']
    texture_table = table[table['Variable'] == 'mean texture']
    
    # For 'mean radius' with empirical, Points should equal WoE * Coefficient
    radius_special = radius_table[radius_table['Bin'].str.contains('Special', na=False)]
    if len(radius_special) > 0:
        woe_value = radius_special['WoE'].iloc[0]
        coef_value = radius_special['Coefficient'].iloc[0]
        points_value = radius_special['Points'].iloc[0]
        # Points should be WoE * coef (empirical case)
        assert points_value == approx(woe_value * coef_value, rel=1e-6)
    
    # For 'mean texture' with metric_special=0.5, Points should be 0.5 * Coefficient
    texture_special = texture_table[texture_table['Bin'].str.contains('Special', na=False)]
    if len(texture_special) > 0:
        coef_value = texture_special['Coefficient'].iloc[0]
        points_value = texture_special['Points'].iloc[0]
        # Points should be 0.5 * coef
        assert points_value == approx(0.5 * coef_value, rel=1e-6)


def test_per_variable_metric_special_backward_compatibility():
    """Test that global metric_special is used when binning_transform_params
    doesn't specify it (backward compatibility)."""
    data = load_breast_cancer()
    variable_names = ['mean radius', 'mean texture']
    X = pd.DataFrame(data.data[:, :2], columns=variable_names)
    y = data.target
    
    # Add special codes
    X_with_specials = X.copy()
    X_with_specials.loc[:20, 'mean radius'] = -999
    X_with_specials.loc[:20, 'mean texture'] = -888
    
    binning_process = BinningProcess(
        variable_names=variable_names,
        binning_fit_params={
            'mean radius': {'special_codes': [-999]},
            'mean texture': {'special_codes': [-888]},
        },
        # No binning_transform_params specified - should use global values
    )
    
    scorecard = Scorecard(
        binning_process=binning_process,
        estimator=LogisticRegression(),
    )
    
    # Fit with global metric_special=0.3
    scorecard.fit(X_with_specials, y, metric_special=0.3)
    
    table = scorecard.table(style='detailed')
    
    # Both variables should use global metric_special=0.3
    for variable in variable_names:
        var_table = table[table['Variable'] == variable]
        special_rows = var_table[var_table['Bin'].str.contains('Special', na=False)]
        
        if len(special_rows) > 0:
            coef_value = special_rows['Coefficient'].iloc[0]
            points_value = special_rows['Points'].iloc[0]
            # Points should be 0.3 * coef (global value)
            assert points_value == approx(0.3 * coef_value, rel=1e-6)


def test_per_variable_metric_missing():
    """Test that per-variable metric_missing from binning_transform_params
    is used for scorecard Points calculation."""
    data = pd.DataFrame({
        'target': np.hstack((np.tile(np.array([0, 1]), 50), np.array([0]*90 + [1]*10))),
        'var1': [np.nan] * 100 + ['A'] * 100,
        'var2': [np.nan] * 100 + ['B'] * 100,
    })
    
    binning_process = BinningProcess(
        variable_names=['var1', 'var2'],
        binning_transform_params={
            'var1': {'metric': 'woe', 'metric_missing': 'empirical'},
            'var2': {'metric': 'woe', 'metric_missing': 0.25},
        }
    )
    
    scorecard = Scorecard(
        binning_process=binning_process,
        estimator=LogisticRegression(),
    )
    
    # Fit with global metric_missing=0 (should be overridden per-variable)
    scorecard.fit(data, data.target, metric_missing=0)
    
    table = scorecard.table(style='detailed')
    
    # For 'var1': metric_missing='empirical', Points should use empirical WoE
    var1_table = table[table['Variable'] == 'var1']
    var1_missing = var1_table[var1_table['Bin'].str.contains('Missing', na=False)]
    
    if len(var1_missing) > 0:
        woe_value = var1_missing['WoE'].iloc[0]
        coef_value = var1_missing['Coefficient'].iloc[0]
        points_value = var1_missing['Points'].iloc[0]
        # Points should equal WoE * coef (empirical case)
        assert points_value == approx(woe_value * coef_value, rel=1e-6)
    
    # For 'var2': metric_missing=0.25, Points should be 0.25 * Coefficient
    var2_table = table[table['Variable'] == 'var2']
    var2_missing = var2_table[var2_table['Bin'].str.contains('Missing', na=False)]
    
    if len(var2_missing) > 0:
        coef_value = var2_missing['Coefficient'].iloc[0]
        points_value = var2_missing['Points'].iloc[0]
        # Points should be 0.25 * coef
        assert points_value == approx(0.25 * coef_value, rel=1e-6)


def test_woe_points_consistency():
    """Test that WoE and Points columns are consistent when using per-variable
    metric_special from binning_transform_params."""
    data = load_breast_cancer()
    variable_names = ['mean radius', 'mean texture']
    X = pd.DataFrame(data.data[:, :2], columns=variable_names)
    y = data.target
    
    # Add special codes
    X_with_specials = X.copy()
    X_with_specials.loc[:20, 'mean radius'] = -999
    
    binning_process = BinningProcess(
        variable_names=variable_names,
        binning_fit_params={
            'mean radius': {'special_codes': [-999]},
        },
        binning_transform_params={
            'mean radius': {'metric': 'woe', 'metric_special': 'empirical'},
        }
    )
    
    scorecard = Scorecard(
        binning_process=binning_process,
        estimator=LogisticRegression(),
    )
    
    # Fit - WoE display should match Points calculation
    scorecard.fit(X_with_specials, y)
    
    table = scorecard.table(style='detailed')
    
    # For all bins (including special), Points should be WoE * Coefficient
    # This verifies the consistency fix
    radius_table = table[table['Variable'] == 'mean radius']
    
    for _, row in radius_table.iterrows():
        woe = row['WoE']
        coef = row['Coefficient']
        points = row['Points']
        # Points should equal WoE * coef for all bins
        assert points == approx(woe * coef, rel=1e-6)

