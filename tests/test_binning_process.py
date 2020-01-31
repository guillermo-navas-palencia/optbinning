"""
Binning process testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import pandas as pd
import numpy as np

from pytest import approx, raises

from contextlib import redirect_stdout

from optbinning import BinningProcess
from optbinning import ContinuousOptimalBinning
from optbinning import MulticlassOptimalBinning
from optbinning import OptimalBinning
from sklearn.datasets import load_boston
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_wine
from sklearn.exceptions import NotFittedError


data = load_breast_cancer()

variable_names = data.feature_names
X = data.data
y = data.target


def test_params():
    with raises(TypeError):
        process = BinningProcess(variable_names=1)
        process.fit(X, y)

    with raises(ValueError):
        process = BinningProcess(variable_names=[], max_n_prebins=-2)
        process.fit(X, y)

    with raises(ValueError):
        process = BinningProcess(variable_names=[], min_prebin_size=0.6)
        process.fit(X, y)

    with raises(ValueError):
        process = BinningProcess(variable_names=[], min_n_bins=-2)
        process.fit(X, y)

    with raises(ValueError):
        process = BinningProcess(variable_names=[], max_n_bins=-2.2)
        process.fit(X, y)

    with raises(ValueError):
        process = BinningProcess(variable_names=[], min_n_bins=3, max_n_bins=2)
        process.fit(X, y)

    with raises(ValueError):
        process = BinningProcess(variable_names=[], min_bin_size=0.6)
        process.fit(X, y)

    with raises(ValueError):
        process = BinningProcess(variable_names=[], max_bin_size=-0.6)
        process.fit(X, y)

    with raises(ValueError):
        process = BinningProcess(variable_names=[], min_bin_size=0.5,
                                 max_bin_size=0.3)
        process.fit(X, y)

    with raises(ValueError):
        process = BinningProcess(variable_names=[], max_pvalue=1.1)
        process.fit(X, y)

    with raises(ValueError):
        process = BinningProcess(variable_names=[],
                                 max_pvalue_policy="new_policy")
        process.fit(X, y)

    with raises(ValueError):
        process = BinningProcess(variable_names=[], min_iv=-0.2)
        process.fit(X, y)

    with raises(ValueError):
        process = BinningProcess(variable_names=[], max_iv=-0.2)
        process.fit(X, y)

    with raises(ValueError):
        process = BinningProcess(variable_names=[], min_iv=1.0, max_iv=0.8)
        process.fit(X, y)

    with raises(ValueError):
        process = BinningProcess(variable_names=[], min_js=-0.2)
        process.fit(X, y)

    with raises(ValueError):
        process = BinningProcess(variable_names=[], max_js=-0.2)
        process.fit(X, y)

    with raises(ValueError):
        process = BinningProcess(variable_names=[], min_js=1.0, max_js=0.8)
        process.fit(X, y)

    with raises(ValueError):
        process = BinningProcess(variable_names=[], quality_score_cutoff=-0.1)
        process.fit(X, y)

    with raises(TypeError):
        process = BinningProcess(variable_names=[], special_codes={1, 2, 3})
        process.fit(X, y)

    with raises(ValueError):
        process = BinningProcess(variable_names=[], split_digits=9)
        process.fit(X, y)

    with raises(TypeError):
        process = BinningProcess(variable_names=[], binning_fit_params=[1, 2])
        process.fit(X, y)

    with raises(TypeError):
        process = BinningProcess(variable_names=[],
                                 binning_transform_params=[1, 2])
        process.fit(X, y)

    with raises(TypeError):
        process = BinningProcess(variable_names=[], verbose=1)
        process.fit(X, y)


def test_default():
    process = BinningProcess(variable_names)
    process.fit(X, y, check_input=True)

    with raises(TypeError):
        process.get_binned_variable(1)

    with raises(ValueError):
        process.get_binned_variable("new_variable")

    optb = process.get_binned_variable("mean radius")

    assert optb.status == "OPTIMAL"
    assert optb.splits == approx([11.42500019, 12.32999992, 13.09499979,
                                  13.70499992, 15.04500008, 16.92500019],
                                 rel=1e-6)

    optb.binning_table.build()
    assert optb.binning_table.iv == approx(5.04392547, rel=1e-6)


def test_incorrect_target_type():
    variable_names = ["var_{}".format(i) for i in range(2)]
    X = np.zeros((2, 2))
    y = np.array([[1, 2], [3, 1]])
    process = BinningProcess(variable_names)

    with raises(ValueError):
        process.fit(X, y)


def test_categorical_variables():
    data = load_boston()

    variable_names = data.feature_names
    X = data.data
    y = data.target

    process = BinningProcess(variable_names, categorical_variables=["CHAS"])
    process.fit(X, y, check_input=True)

    df_summary = process.summary()
    assert df_summary[
        df_summary.name == "CHAS"]["dtype"].values[0] == "categorical"


def test_fit_params():
    binning_fit_params = {"mean radius": {"max_n_bins": 4}}

    process = BinningProcess(variable_names=variable_names,
                             binning_fit_params=binning_fit_params)
    process.fit(X, y)

    optb = process.get_binned_variable("mean radius")

    assert optb.status == "OPTIMAL"
    assert len(optb.splits) <= 3


def test_default_transform():
    process = BinningProcess(variable_names)
    with raises(NotFittedError):
        process.transform(X)

    process.fit(X, y)
    X_transform = process.transform(X)

    optb = OptimalBinning()
    x = X[:, 5]
    optb.fit(x, y)

    assert optb.transform(x) == approx(X_transform[:, 5], rel=1e-6)


def test_default_transform_continuous():
    data = load_boston()
    variable_names = data.feature_names
    X = data.data
    y = data.target

    process = BinningProcess(variable_names)
    process.fit(X, y)
    X_transform = process.transform(X)

    optb = process.get_binned_variable(variable_names[0])
    assert isinstance(optb, ContinuousOptimalBinning)

    optb = ContinuousOptimalBinning()
    x = X[:, 5]
    optb.fit(x, y)
    assert optb.transform(x) == approx(X_transform[:, 5], rel=1e-6)


def test_default_transform_multiclass():
    data = load_wine()
    variable_names = data.feature_names
    X = data.data
    y = data.target

    process = BinningProcess(variable_names)
    process.fit(X, y)
    X_transform = process.transform(X)

    optb = process.get_binned_variable(variable_names[0])
    assert isinstance(optb, MulticlassOptimalBinning)

    optb = MulticlassOptimalBinning()
    x = X[:, 5]
    optb.fit(x, y)
    assert optb.transform(x) == approx(X_transform[:, 5], rel=1e-6)


def test_transform_some_variables():
    process = BinningProcess(variable_names)
    process.fit(X, y)

    with raises(TypeError):
        process.transform(X, {})

    with raises(ValueError):
        process.transform(X, ["new_1", "new_2"])

    selected_variables = ['mean area', 'mean smoothness', 'mean compactness',
                          'mean concavity']

    X_transform = process.transform(X, selected_variables)
    assert X_transform.shape[1] == 4

    for i in range(3, 7):
        optb = OptimalBinning()
        x = X[:, i]
        optb.fit(x, y)

        assert optb.transform(x) == approx(X_transform[:, i-3], rel=1e-6)


def test_default_fit_transform():
    process = BinningProcess(variable_names)
    X_transform = process.fit_transform(X, y, metric="event_rate")

    optb = OptimalBinning()
    x = X[:, 5]
    optb.fit(x, y)

    assert optb.transform(x, metric="event_rate") == approx(
        X_transform[:, 5], rel=1e-6)


def test_information():
    data = load_breast_cancer()

    variable_names = data.feature_names
    X = data.data
    y = data.target

    process = BinningProcess(variable_names)
    process.fit(X, y, check_input=True)

    with raises(ValueError):
        process.information(print_level=-1)

    with open("tests/test_binning_process_information.txt", "w") as f:
        with redirect_stdout(f):
            process.information(print_level=0)
            process.information(print_level=1)
            process.information(print_level=2)


def test_summary_get_support():
    data = load_breast_cancer()

    variable_names = data.feature_names
    X = data.data
    y = data.target

    process = BinningProcess(variable_names, min_iv=0.1, max_iv=0.6)

    with raises(ValueError):
        process.summary()

    with raises(ValueError):
        process.get_support()

    process.fit(X, y, check_input=True)

    assert isinstance(process.summary(), pd.DataFrame)

    with raises(ValueError):
        process.get_support(indices=True, names=True)

    assert all(process.get_support() == [
        False, False, False, False, False, False, False, False, False, True,
        False,  True, False, False,  True, False, False, False, True,  True,
        False, False, False, False, False, False, False, False, False,  True])
    assert process.get_support(indices=True) == approx([9, 11, 14, 18, 19, 29])
    assert all(process.get_support(names=True) == [
        'mean fractal dimension', 'texture error', 'smoothness error',
        'symmetry error', 'fractal dimension error',
        'worst fractal dimension'])


def test_verbose():
    process = BinningProcess(variable_names, verbose=True)
    
    with open("tests/test_binning_process_verbose.txt", "w") as f:
        with redirect_stdout(f):    
            process.fit(X, y, check_input=True)
