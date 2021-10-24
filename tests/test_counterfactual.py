"""
Scorecard testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import pandas as pd

from pytest import raises

from optbinning import BinningProcess
from optbinning import Scorecard
from optbinning.exceptions import NotGeneratedError
from optbinning.exceptions import CounterfactualsFoundWarning
from optbinning.scorecard import Counterfactual

from sklearn.datasets import load_breast_cancer
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from tests.datasets import load_boston


data = load_breast_cancer()
feature_names_binary = data.feature_names
X_binary = pd.DataFrame(data.data, columns=feature_names_binary)
y_binary = data.target

scorecard_binary = Scorecard(
    binning_process=BinningProcess(feature_names_binary),
    estimator=LogisticRegression()
    ).fit(X_binary, y_binary)


data = load_boston()
feature_names_continuous = data.feature_names
X_continuous = pd.DataFrame(data.data, columns=feature_names_continuous)
y_continuous = data.target

scorecard_continuous = Scorecard(
    binning_process=BinningProcess(feature_names_continuous),
    estimator=LinearRegression()
    ).fit(X_continuous, y_continuous)


def test_params():
    with raises(TypeError):
        cf = Counterfactual(scorecard=None)
        cf.fit(X_binary)

    with raises(NotFittedError):
        binning_process = BinningProcess(feature_names_binary)
        estimator = LogisticRegression()
        scorecard = Scorecard(binning_process=binning_process,
                              estimator=estimator)

        cf = Counterfactual(scorecard=scorecard)
        cf.fit(X_binary)

    with raises(TypeError):
        cf = Counterfactual(scorecard_binary, special_missing=1)
        cf.fit(X_binary)

    with raises(ValueError):
        cf = Counterfactual(scorecard_binary, n_jobs=-1)
        cf.fit(X_binary)

    with raises(TypeError):
        cf = Counterfactual(scorecard_binary, verbose=1)
        cf.fit(X_binary)


def test_fit():
    cf = Counterfactual(scorecard_binary)

    with raises(TypeError):
        cf.fit(X_binary.values)

    with raises(ValueError):
        cf.fit(X_binary[feature_names_binary[10:]])


def test_generate_binary_params():
    cf = Counterfactual(scorecard_binary)

    with raises(NotFittedError):
        cf.generate(query=[], y=1, outcome_type="binary", n_cf=1)

    cf.fit(X_binary)

    with raises(TypeError):
        cf.generate(query=[], y=1, outcome_type="binary", n_cf=1)

    query = X_binary.iloc[0, :].to_frame().T
    with raises(TypeError):
        cf.generate(query=query, y="1", outcome_type="binary", n_cf=1)

    with raises(ValueError):
        cf.generate(query=query, y=1, outcome_type="continuous", n_cf=1)

    with raises(ValueError):
        cf.generate(query=query, y=0.5, outcome_type="binary", n_cf=1)

    with raises(ValueError):
        cf.generate(query=query, y=1.5, outcome_type="probability", n_cf=1)

    with raises(ValueError):
        cf.generate(query=query, y=1, outcome_type="binary", n_cf=0)

    with raises(ValueError):
        cf.generate(query=query, y=1, outcome_type="binary", n_cf=1,
                    max_changes=0)

    with raises(TypeError):
        cf.generate(query=query, y=1, outcome_type="binary", n_cf=1,
                    actionable_features={})

    with raises(ValueError):
        cf.generate(query=query, y=1, outcome_type="binary", n_cf=1,
                    actionable_features=["new_variable"])

    with raises(ValueError):
        cf.generate(query=query, y=1, outcome_type="binary", n_cf=1,
                    method="new_method")

    with raises(TypeError):
        cf.generate(query=query, y=1, outcome_type="binary", n_cf=1,
                    objectives=[])

    with raises(ValueError):
        cf.generate(query=query, y=1, outcome_type="binary", n_cf=1,
                    objectives={})

    with raises(ValueError):
        cf.generate(query=query, y=1, outcome_type="binary", n_cf=1,
                    objectives={"new_objective": 1})

    with raises(ValueError):
        cf.generate(query=query, y=1, outcome_type="binary", n_cf=1,
                    objectives={"proximity": -1})

    with raises(TypeError):
        cf.generate(query=query, y=1, outcome_type="binary", n_cf=1,
                    hard_constraints={})

    with raises(ValueError):
        cf.generate(query=query, y=1, outcome_type="binary", n_cf=1,
                    hard_constraints=["diversity_values", "diversity_values"])

    with raises(ValueError):
        cf.generate(query=query, y=1, outcome_type="binary", n_cf=1,
                    hard_constraints=["new_constraint"])

    with raises(TypeError):
        cf.generate(query=query, y=1, outcome_type="binary", n_cf=1,
                    soft_constraints=[])

    with raises(ValueError):
        cf.generate(query=query, y=1, outcome_type="binary", n_cf=1,
                    soft_constraints={"new_constraint": 2})

    with raises(ValueError):
        cf.generate(query=query, y=1, outcome_type="binary", n_cf=1,
                    soft_constraints={"diversity_features": -1})

    with raises(ValueError):
        cf.generate(query=query, y=1, outcome_type="probability", n_cf=1)

    with raises(ValueError):
        cf.generate(query=query, y=1, outcome_type="probability", n_cf=1,
                    hard_constraints=["diversity_values"])

    with raises(ValueError):
        cf.generate(query=query, y=1, outcome_type="binary", n_cf=1,
                    priority_tol=1.1)

    with raises(ValueError):
        cf.generate(query=query, y=1, outcome_type="binary", n_cf=1,
                    time_limit=0)


def test_generate_continuous_params():
    cf = Counterfactual(scorecard_continuous)
    cf.fit(X_continuous)

    query = X_continuous.iloc[0, :].to_frame().T

    with raises(ValueError):
        cf.generate(query=query, y=1, outcome_type="binary", n_cf=1)


def test_generate_binary():
    cf = Counterfactual(scorecard_binary)
    cf.fit(X_binary)

    query = X_binary.iloc[0, :].to_frame().T

    cf.generate(query=query, y=1, outcome_type="binary", n_cf=1,
                max_changes=3, method="weighted")

    assert all(cf.display(show_outcome=True)["outcome"] > 0.5)

    cf.generate(query=query, y=1, outcome_type="binary", n_cf=1,
                max_changes=3, method="hierarchical")

    assert all(cf.display(show_outcome=True)["outcome"] > 0.5)

    cf.generate(query=query, y=1, outcome_type="binary", n_cf=2,
                max_changes=4, method="weighted")

    assert all(cf.display(show_outcome=True)["outcome"] > 0.5)

    cf.generate(query=query, y=1, outcome_type="binary", n_cf=2,
                max_changes=4, method="hierarchical")

    assert all(cf.display(show_outcome=True)["outcome"] > 0.5)

    cf.generate(query=query, y=0.7, outcome_type="probability", n_cf=1,
                max_changes=4, method="hierarchical",
                hard_constraints=["min_outcome"])

    all(cf.display(show_outcome=True)["outcome"] > 0.7)


def test_generate_continuous():
    cf = Counterfactual(scorecard_continuous)
    cf.fit(X_continuous)

    query = X_continuous.iloc[0, :].to_frame().T

    cf.generate(query=query, y=30, outcome_type="continuous", n_cf=1,
                max_changes=4, method="hierarchical",
                hard_constraints=["min_outcome"])

    assert all(cf.display(show_outcome=True)["outcome"] > 30)

    cf.generate(query=query, y=30, outcome_type="continuous", n_cf=1,
                max_changes=4, method="weighted",
                hard_constraints=["min_outcome"])

    assert all(cf.display(show_outcome=True)["outcome"] > 30)

    cf.generate(query=query, y=20, outcome_type="continuous", n_cf=2,
                max_changes=4, method="weighted",
                hard_constraints=["diversity_features", "max_outcome"])

    assert all(cf.display(show_outcome=True)["outcome"] < 20)

    cf.generate(query=query, y=20, outcome_type="continuous", n_cf=2,
                max_changes=4, method="hierarchical",
                hard_constraints=["diversity_features", "max_outcome"])

    assert all(cf.display(show_outcome=True)["outcome"] < 20)


def test_information():
    cf = Counterfactual(scorecard_binary)

    with raises(NotGeneratedError):
        cf.information()

    cf.fit(X_binary)
    query = X_binary.iloc[0, :].to_frame().T
    cf.generate(query=query, y=1, outcome_type="binary", n_cf=1)

    with raises(ValueError):
        cf.information(print_level=-1)

    cf.information(print_level=0)
    cf.information(print_level=1)
    cf.information(print_level=2)


def test_display():
    cf = Counterfactual(scorecard_binary)
    cf.fit(X_binary)

    query = X_binary.iloc[0, :].to_frame().T
    cf.generate(query=query, y=1, outcome_type="binary", n_cf=1)

    with raises(TypeError):
        cf.display(show_only_changes=1)

    with raises(TypeError):
        cf.display(show_outcome=1)

    cf.generate(query=query, y=1, outcome_type="binary", n_cf=3,
                max_changes=1, hard_constraints=["diversity_features"],
                time_limit=1)

    with raises(CounterfactualsFoundWarning):
        cf.display()

    cf.generate(query=query, y=1, outcome_type="binary", n_cf=1)
    df_cf = cf.display()
    assert df_cf.shape[0] == 1

    cf.generate(query=query, y=1, outcome_type="binary", n_cf=2,
                hard_constraints=["diversity_features"])
    df_cf = cf.display()
    assert df_cf.shape[0] == 2

    df_cf = cf.display(show_only_changes=True, show_outcome=True)
    assert "outcome" in df_cf.columns


def test_status():
    cf = Counterfactual(scorecard_binary)
    cf.fit(X_binary)

    query = X_binary.iloc[0, :].to_frame().T
    cf.generate(query=query, y=1, outcome_type="binary", n_cf=1,
                max_changes=4)

    assert cf.status == "OPTIMAL"
