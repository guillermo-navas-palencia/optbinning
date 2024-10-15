"""
Scorecard monitoring testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import numpy as np
import pandas as pd

from pytest import approx, raises

from contextlib import redirect_stdout

from optbinning import BinningProcess
from optbinning import Scorecard
from optbinning.scorecard import ScorecardMonitoring
from sklearn.datasets import load_breast_cancer
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tests.datasets import load_boston


def _data(target_dtype):
    if target_dtype == "binary":
        data = load_breast_cancer()
    else:
        data = load_boston()

    variable_names = data.feature_names
    X = pd.DataFrame(data.data, columns=variable_names)
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X_train, y_train, X_test, y_test


def _fit_scorecard(target_dtype, X_train, y_train):
    if target_dtype == "binary":
        estimator = LogisticRegression()
    else:
        estimator = LinearRegression()

    variable_names = list(X_train.columns)
    binning_process = BinningProcess(variable_names)
    scorecard = Scorecard(binning_process=binning_process,
                          estimator=estimator).fit(X_train, y_train)

    return scorecard


def test_params():
    X_train, y_train, X_test, y_test = _data("binary")
    scorecard = _fit_scorecard("binary", X_train, y_train)

    with raises(TypeError):
        monitoring = ScorecardMonitoring(scorecard=None)
        monitoring.fit(X_test, y_test, X_train, y_train)

    with raises(ValueError):
        monitoring = ScorecardMonitoring(scorecard=scorecard,
                                         psi_method="new_method")
        monitoring.fit(X_test, y_test, X_train, y_train)

    with raises(ValueError):
        monitoring = ScorecardMonitoring(scorecard=scorecard, psi_n_bins=-12.2)
        monitoring.fit(X_test, y_test, X_train, y_train)

    with raises(ValueError):
        monitoring = ScorecardMonitoring(scorecard=scorecard,
                                         psi_min_bin_size=0.8)
        monitoring.fit(X_test, y_test, X_train, y_train)

    with raises(ValueError):
        monitoring = ScorecardMonitoring(scorecard=scorecard, show_digits=9)
        monitoring.fit(X_test, y_test, X_train, y_train)

    with raises(TypeError):
        monitoring = ScorecardMonitoring(scorecard=scorecard, verbose=1)
        monitoring.fit(X_test, y_test, X_train, y_train)


def test_input():
    X_train, y_train, X_test, y_test = _data("binary")
    scorecard = _fit_scorecard("binary", X_train, y_train)
    monitoring = ScorecardMonitoring(scorecard=scorecard)

    with raises(ValueError):
        X_test_2 = X_test.copy()
        columns = list(X_test.columns)
        columns[3] = columns[3].upper()
        X_test_2.columns = columns
        monitoring.fit(X_test_2, y_test, X_train, y_train)

    with raises(ValueError):
        X_test_2 = X_test.copy()
        y_test_2 = np.random.randint(0, 3, len(X_test))
        monitoring.fit(X_test_2, y_test_2, X_train, y_train)

    with raises(ValueError):
        y_train_2 = np.random.randint(0, 3, len(X_train))
        monitoring.fit(X_test, y_test, X_train, y_train_2)

    with raises(ValueError):
        y_train_2 = np.random.randn(len(X_train))
        monitoring.fit(X_test, y_test, X_train, y_train_2)


def test_default_binary():
    X_train, y_train, X_test, y_test = _data("binary")
    scorecard = _fit_scorecard("binary", X_train, y_train)

    monitoring = ScorecardMonitoring(scorecard=scorecard)
    monitoring.fit(X_test, y_test, X_train, y_train)

    # Check psi_table
    psi_table = monitoring.psi_table()
    assert psi_table.PSI.sum() == approx(0.002224950381423254)

    # Check psi_variable_table
    with raises(ValueError):
        monitoring.psi_variable_table(style="new_style")

    with raises(ValueError):
        monitoring.psi_variable_table(name="new variable")

    assert monitoring.psi_variable_table(
        name="mean radius", style="summary").values == approx(0.02463385)

    # Check tests table
    tests_table = monitoring.tests_table()
    assert tests_table["p-value"].values[:2] == approx(
        [0.00250006, 0.49480006], rel=1e-4)

    # Check system stability report
    with open("tests/results/test_scorecard_monitoring_default.txt", "w") as f:
        with redirect_stdout(f):
            monitoring.system_stability_report()


def test_default_continuous():
    X_train, y_train, X_test, y_test = _data("continuous")
    scorecard = _fit_scorecard("continuous", X_train, y_train)

    monitoring = ScorecardMonitoring(scorecard=scorecard)
    monitoring.fit(X_test, y_test, X_train, y_train)

    # Check psi_table
    psi_table = monitoring.psi_table()
    assert psi_table.PSI.sum() == approx(0.32608491143830726)

    # Check psi_variable_table
    assert monitoring.psi_variable_table(
        name="CRIM", style="summary").values == approx(0.01065983)

    assert monitoring.psi_variable_table(
        name="CRIM", style="detailed")["PSI"].sum() == approx(0.01065983)

    psi_variable_table_s = monitoring.psi_variable_table(style="summary")
    psi_variable_table_d = monitoring.psi_variable_table(style="detailed")

    psis = psi_variable_table_s[psi_variable_table_s.Variable == "CRIM"]["PSI"]
    psid = psi_variable_table_d[psi_variable_table_d.Variable == "CRIM"]["PSI"]

    assert psis.sum() == approx(0.01065983)
    assert psid.sum() == approx(0.01065983)

    # Check psi splits
    assert monitoring.psi_splits[:4] == approx(
        [16.63161373, 18.33728027, 20.07739162, 21.29977036], rel=1e-4)

    # Check tests table
    tests_table = monitoring.tests_table()
    assert tests_table["p-value"].values[:2] == approx(
        [0.78558541, 0.29332423], rel=1e-4)

    # Check system stability report
    with open("tests/results/test_scorecard_monitoring_default_continuous.txt",
              "w") as f:
        with redirect_stdout(f):
            monitoring.system_stability_report()


def test_information():
    X_train, y_train, X_test, y_test = _data("continuous")
    scorecard = _fit_scorecard("continuous", X_train, y_train)

    monitoring = ScorecardMonitoring(scorecard=scorecard)

    with raises(NotFittedError):
        monitoring.information()

    with raises(NotFittedError):
        monitoring.psi_splits

    monitoring.fit(X_test, y_test, X_train, y_train)

    with raises(ValueError):
        monitoring.information(print_level=-1)

    with open("tests/results/test_scorecard_monitoring_information.txt",
              "w") as f:
        with redirect_stdout(f):
            monitoring.information(print_level=0)
            monitoring.information(print_level=1)
            monitoring.information(print_level=2)


def test_verbose():
    X_train, y_train, X_test, y_test = _data("continuous")
    scorecard = _fit_scorecard("continuous", X_train, y_train)

    monitoring = ScorecardMonitoring(scorecard=scorecard, verbose=True)

    with open("tests/results/test_scorecard_monitoring_verbose.txt", "w") as f:
        with redirect_stdout(f):
            monitoring.fit(X_test, y_test, X_train, y_train)


def test_plot_binary():
    X_train, y_train, X_test, y_test = _data("binary")
    scorecard = _fit_scorecard("binary", X_train, y_train)

    monitoring = ScorecardMonitoring(scorecard=scorecard)
    monitoring.fit(X_test, y_test, X_train, y_train)

    with raises(TypeError):
        monitoring.psi_plot(savefig=1)

    monitoring.psi_plot(savefig="tests/results/psi_plot_binary.png")


def test_plot_continuous():
    X_train, y_train, X_test, y_test = _data("continuous")
    scorecard = _fit_scorecard("continuous", X_train, y_train)

    monitoring = ScorecardMonitoring(scorecard=scorecard)
    monitoring.fit(X_test, y_test, X_train, y_train)
    monitoring.psi_plot(savefig="tests/results/psi_plot_continuous.png")
