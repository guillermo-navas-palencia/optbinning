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
from sklearn.datasets import load_boston
from sklearn.datasets import load_breast_cancer
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def _data(target_dtype):
    if target_dtype == "binary":
        data = load_breast_cancer()
    else:
        data = load_boston()

    variable_names = data.feature_names
    df = pd.DataFrame(data.data, columns=variable_names)
    df["target"] = data.target

    df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)

    return df_train, df_test


def _fit_scorecard(target_dtype, df_train):
    if target_dtype == "binary":
        estimator = LogisticRegression()
    else:
        estimator = LinearRegression()

    variable_names = list(df_train.columns)
    binning_process = BinningProcess(variable_names)
    scorecard = Scorecard(target="target", binning_process=binning_process,
                          estimator=estimator).fit(df_train)

    return scorecard


def test_params():
    df_train, df_test = _data("binary")
    scorecard = _fit_scorecard("binary", df_train)

    with raises(TypeError):
        monitoring = ScorecardMonitoring(target=1, scorecard=scorecard)
        monitoring.fit(df_test, df_train)

    with raises(TypeError):
        monitoring = ScorecardMonitoring(target="target", scorecard=None)
        monitoring.fit(df_test, df_train)

    with raises(ValueError):
        monitoring = ScorecardMonitoring(target="target", scorecard=scorecard,
                                         psi_method="new_method")
        monitoring.fit(df_test, df_train)

    with raises(ValueError):
        monitoring = ScorecardMonitoring(target="target", scorecard=scorecard,
                                         psi_n_bins=-12.2)
        monitoring.fit(df_test, df_train)

    with raises(ValueError):
        monitoring = ScorecardMonitoring(target="target", scorecard=scorecard,
                                         psi_min_bin_size=0.8)
        monitoring.fit(df_test, df_train)

    with raises(ValueError):
        monitoring = ScorecardMonitoring(target="target", scorecard=scorecard,
                                         show_digits=9)
        monitoring.fit(df_test, df_train)

    with raises(TypeError):
        monitoring = ScorecardMonitoring(target="target", scorecard=scorecard,
                                         verbose=1)
        monitoring.fit(df_test, df_train)


def test_input():
    df_train, df_test = _data("binary")
    scorecard = _fit_scorecard("binary", df_train)
    monitoring = ScorecardMonitoring(target="target", scorecard=scorecard)

    with raises(ValueError):
        df_test_2 = df_test.copy()
        columns = list(df_test.columns)
        columns[3] = columns[3].upper()
        df_test_2.columns = columns
        monitoring.fit(df_test_2, df_train)

    with raises(ValueError):
        df_test_2 = df_test.copy()
        df_test_2["target"] = np.random.randint(0, 3, len(df_test))
        monitoring.fit(df_test_2, df_train)

    with raises(ValueError):
        df_train["target"] = np.random.randint(0, 3, len(df_train))
        monitoring.fit(df_test, df_train)

    with raises(ValueError):
        df_train["target"] = np.random.randn(len(df_train))
        monitoring.fit(df_test, df_train)


def test_default_binary():
    df_train, df_test = _data("binary")
    scorecard = _fit_scorecard("binary", df_train)

    monitoring = ScorecardMonitoring(target="target", scorecard=scorecard)
    monitoring.fit(df_test, df_train)

    # Check psi_table
    psi_table = monitoring.psi_table()
    assert psi_table.PSI.sum() == approx(0.003536079105130241)

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
        [0.00077184, 0.51953576], rel=1e-4)

    # Check system stability report
    with open("tests/test_scorecard_monitoring_default.txt", "w") as f:
        with redirect_stdout(f):
            monitoring.system_stability_report()


def test_default_continuous():
    df_train, df_test = _data("continuous")
    scorecard = _fit_scorecard("continuous", df_train)

    monitoring = ScorecardMonitoring(target="target", scorecard=scorecard)
    monitoring.fit(df_test, df_train)

    # Check psi_table
    psi_table = monitoring.psi_table()
    assert psi_table.PSI.sum() == approx(0.24101307906596886)

    # Check psi_variable_table
    assert monitoring.psi_variable_table(
        name="CRIM", style="summary").values == approx(0.02469948)

    assert monitoring.psi_variable_table(
        name="CRIM", style="detailed")["PSI"].sum() == approx(0.02469948)

    psi_variable_table_s = monitoring.psi_variable_table(style="summary")
    psi_variable_table_d = monitoring.psi_variable_table(style="detailed")

    psis = psi_variable_table_s[psi_variable_table_s.Variable == "CRIM"]["PSI"]
    psid = psi_variable_table_d[psi_variable_table_d.Variable == "CRIM"]["PSI"]

    assert psis.sum() == approx(0.02469948)
    assert psid.sum() == approx(0.02469948)

    # Check psi splits
    assert monitoring.psi_splits[:4] == approx(
        [11.936903, 15.63085079, 18.19052601, 19.64347458], rel=1e-4)

    # Check tests table
    tests_table = monitoring.tests_table()
    assert tests_table["p-value"].values[:2] == approx(
        [0.80810407, 0.69939868], rel=1e-4)

    # Check system stability report
    with open("tests/test_scorecard_monitoring_default_continuous.txt",
              "w") as f:
        with redirect_stdout(f):
            monitoring.system_stability_report()


def test_information():
    df_train, df_test = _data("continuous")
    scorecard = _fit_scorecard("continuous", df_train)

    monitoring = ScorecardMonitoring(target="target", scorecard=scorecard)

    with raises(NotFittedError):
        monitoring.information()

    with raises(NotFittedError):
        monitoring.psi_splits

    monitoring.fit(df_test, df_train)

    with raises(ValueError):
        monitoring.information(print_level=-1)

    with open("tests/test_scorecard_monitoring_information.txt", "w") as f:
        with redirect_stdout(f):
            monitoring.information(print_level=0)
            monitoring.information(print_level=1)
            monitoring.information(print_level=2)


def test_verbose():
    df_train, df_test = _data("continuous")
    scorecard = _fit_scorecard("continuous", df_train)

    monitoring = ScorecardMonitoring(target="target", scorecard=scorecard,
                                     verbose=True)

    with open("tests/test_scorecard_monitoring_verbose.txt", "w") as f:
        with redirect_stdout(f):
            monitoring.fit(df_test, df_train)


def test_plot_binary():
    df_train, df_test = _data("binary")
    scorecard = _fit_scorecard("binary", df_train)

    monitoring = ScorecardMonitoring(target="target", scorecard=scorecard)
    monitoring.fit(df_test, df_train)

    with raises(TypeError):
        monitoring.psi_plot(savefig=1)

    monitoring.psi_plot(savefig="psi_plot_binary.png")


def test_plot_continuous():
    df_train, df_test = _data("continuous")
    scorecard = _fit_scorecard("continuous", df_train)

    monitoring = ScorecardMonitoring(target="target", scorecard=scorecard)
    monitoring.fit(df_test, df_train)
    monitoring.psi_plot(savefig="psi_plot_continuous.png")
