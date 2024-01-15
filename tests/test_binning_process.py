"""
Binning process testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import os

import pandas as pd
import numpy as np

from pytest import approx, raises

from contextlib import redirect_stdout

from optbinning import BinningProcess
from optbinning import ContinuousOptimalBinning
from optbinning import ContinuousOptimalPWBinning
from optbinning import MulticlassOptimalBinning
from optbinning import OptimalBinning
from optbinning import OptimalPWBinning
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_wine
from sklearn.exceptions import NotFittedError
from tests.datasets import load_boston


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

    with raises(TypeError):
        process = BinningProcess(variable_names=[], selection_criteria=[])
        process.fit(X, y)

    with raises(TypeError):
        process = BinningProcess(variable_names=[],
                                 categorical_variables={})
        process.fit(X, y)

    with raises(TypeError):
        process = BinningProcess(variable_names=[],
                                 categorical_variables=[1, 2])
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

    with raises(ValueError):
        process = BinningProcess(variable_names=[], n_jobs="all")
        process.fit(X, y)

    with raises(TypeError):
        process = BinningProcess(variable_names=[], verbose=1)
        process.fit(X, y)


def test_selection_criteria():
    with raises(ValueError):
        selection_criteria = {"new_metric": {"min": 0}}
        process = BinningProcess(variable_names=[],
                                 selection_criteria=selection_criteria)
        process.fit(X, y)

    with raises(TypeError):
        selection_criteria = {"iv": ["min", 0]}
        process = BinningProcess(variable_names=[],
                                 selection_criteria=selection_criteria)
        process.fit(X, y)

    with raises(ValueError):
        selection_criteria = {"iv": {"min": -10}}
        process = BinningProcess(variable_names=[],
                                 selection_criteria=selection_criteria)
        process.fit(X, y)

    with raises(ValueError):
        selection_criteria = {"quality_score": {"max": 10}}
        process = BinningProcess(variable_names=[],
                                 selection_criteria=selection_criteria)
        process.fit(X, y)

    with raises(ValueError):
        selection_criteria = {"iv": {"strategy": "new_strategy"}}
        process = BinningProcess(variable_names=[],
                                 selection_criteria=selection_criteria)
        process.fit(X, y)

    with raises(ValueError):
        selection_criteria = {"iv": {"top": -2}}
        process = BinningProcess(variable_names=[],
                                 selection_criteria=selection_criteria)
        process.fit(X, y)

    with raises(ValueError):
        selection_criteria = {"iv": {"top": 1.1}}
        process = BinningProcess(variable_names=[],
                                 selection_criteria=selection_criteria)
        process.fit(X, y)

    with raises(KeyError):
        selection_criteria = {"iv": {"new_threshold": 2}}
        process = BinningProcess(variable_names=[],
                                 selection_criteria=selection_criteria)
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


def test_default_pandas():
    df = pd.DataFrame(data.data, columns=data.feature_names)

    process = BinningProcess(variable_names)

    with raises(TypeError):
        process.fit(df.to_dict(), y, check_input=True)

    process.fit(df, y, check_input=True)

    optb = process.get_binned_variable("mean radius")

    assert optb.status == "OPTIMAL"
    assert optb.splits == approx([11.42500019, 12.32999992, 13.09499979,
                                  13.70499992, 15.04500008, 16.92500019],
                                 rel=1e-6)

    optb.binning_table.build()
    assert optb.binning_table.iv == approx(5.04392547, rel=1e-6)


def test_default_disk_csv():
    process = BinningProcess(variable_names, verbose=True)

    with raises(ValueError):
        process.fit_disk(input_path="tests/data/breast_cancer.txt",
                         target="target")

    with raises(TypeError):
        process.fit_disk(input_path="tests/data/breast_cancer.csv", target=0)

    process.fit_disk(input_path="tests/data/breast_cancer.csv",
                     target="target")

    optb = process.get_binned_variable("mean radius")

    assert optb.status == "OPTIMAL"
    assert optb.splits == approx([11.42500019, 12.32999992, 13.09499979,
                                  13.70499992, 15.04500008, 16.92500019],
                                 rel=1e-6)


def test_default_disk_parquet():
    process = BinningProcess(variable_names, verbose=True)
    process.fit_disk(input_path="tests/data/breast_cancer.parquet",
                     target="target")

    optb = process.get_binned_variable("mean radius")

    assert optb.status == "OPTIMAL"
    assert optb.splits == approx([11.42500019, 12.32999992, 13.09499979,
                                  13.70499992, 15.04500008, 16.92500019],
                                 rel=1e-6)


def test_default_from_dict():
    df = pd.DataFrame(data.data, columns=data.feature_names)

    dict_optb = {}
    for name in variable_names:
        optb = OptimalBinning(name=name, dtype="numerical")
        optb.fit(df[name], y)
        dict_optb[name] = optb

    process = BinningProcess(variable_names, verbose=True)

    with raises(TypeError):
        process.fit_from_dict(list(dict_optb.values()))

    with raises(ValueError):
        dict_optb2 = dict_optb.copy()
        dict_optb2.pop("mean radius")
        process.fit_from_dict(dict_optb2)

    process.fit_from_dict(dict_optb)

    optb = process.get_binned_variable("mean radius")

    assert optb.status == "OPTIMAL"
    assert optb.splits == approx([11.42500019, 12.32999992, 13.09499979,
                                  13.70499992, 15.04500008, 16.92500019],
                                 rel=1e-6)


def test_auto_modes():
    df = pd.DataFrame(data.data, columns=data.feature_names)

    binning_fit_params0 = {v: {"monotonic_trend": "auto", "solver": "mip"}
                           for v in data.feature_names}

    binning_fit_params1 = {v: {"monotonic_trend": "auto_heuristic",
                               "solver": "mip"}
                           for v in data.feature_names}

    binning_fit_params2 = {v: {"monotonic_trend": "auto", "solver": "cp"}
                           for v in data.feature_names}

    binning_fit_params3 = {v: {"monotonic_trend": "auto_heuristic",
                               "solver": "cp"}
                           for v in data.feature_names}

    process0 = BinningProcess(variable_names,
                              binning_fit_params=binning_fit_params0)

    process1 = BinningProcess(variable_names,
                              binning_fit_params=binning_fit_params1)

    process2 = BinningProcess(variable_names,
                              binning_fit_params=binning_fit_params2)

    process3 = BinningProcess(variable_names,
                              binning_fit_params=binning_fit_params3)

    process0.fit(df, y)
    process1.fit(df, y)
    process2.fit(df, y)
    process3.fit(df, y)

    assert process0.summary().iv.sum() == process1.summary().iv.sum()
    assert process2.summary().iv.sum() == process3.summary().iv.sum()
    assert process0.summary().iv.sum() == process2.summary().iv.sum()


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
    assert len(optb.splits) <= 4


def test_default_transform():
    process = BinningProcess(variable_names)
    with raises(NotFittedError):
        process.transform(X, metric="woe")

    process.fit(X, y)

    with raises(ValueError):
        X_transform = process.transform(X[:, :3], metric="woe")

    X_transform = process.transform(X)

    optb = OptimalBinning()
    x = X[:, 5]
    optb.fit(x, y)

    assert optb.transform(x, metric="woe") == approx(
        X_transform[:, 5], rel=1e-6)


def test_default_transform_pandas():
    df = pd.DataFrame(data.data, columns=data.feature_names)

    process = BinningProcess(variable_names)
    process.fit(df, y)

    with raises(TypeError):
        X_transform = process.transform(df.to_dict(), metric="woe")

    X_transform = process.transform(df, metric="woe")

    optb = OptimalBinning()
    x = X[:, 5]
    optb.fit(x, y)

    assert optb.transform(x, metric="woe") == approx(
        X_transform.values[:, 5], rel=1e-6)


def test_default_transform_continuous():
    data = load_boston()
    variable_names = data.feature_names
    X = data.data
    y = data.target

    process = BinningProcess(variable_names)
    process.fit(X, y)
    X_transform = process.transform(X, metric="mean")

    optb = process.get_binned_variable(variable_names[0])
    assert isinstance(optb, ContinuousOptimalBinning)

    optb = ContinuousOptimalBinning()
    x = X[:, 5]
    optb.fit(x, y)
    assert optb.transform(x, metric="mean") == approx(
        X_transform[:, 5], rel=1e-6)


def test_default_transform_multiclass():
    data = load_wine()
    variable_names = data.feature_names
    X = data.data
    y = data.target

    process = BinningProcess(variable_names)
    process.fit(X, y)
    X_transform = process.transform(X, metric="mean_woe")

    optb = process.get_binned_variable(variable_names[0])
    assert isinstance(optb, MulticlassOptimalBinning)

    optb = MulticlassOptimalBinning()
    x = X[:, 5]
    optb.fit(x, y)
    assert optb.transform(x, metric="mean_woe") == approx(
        X_transform[:, 5], rel=1e-6)


def test_default_transform_disk():
    input_csv = "tests/data/breast_cancer.csv"
    input_parquet = "tests/data/breast_cancer.parquet"
    output_csv = "tests/results/breast_cancer_woe.csv"

    process = BinningProcess(variable_names, verbose=True)
    process.fit_disk(input_path=input_parquet, target="target")

    with raises(ValueError):
        process.transform_disk(input_path=input_parquet,
                               output_path=output_csv, chunksize=1000)

    with raises(ValueError):
        process.transform_disk(input_path=input_csv, output_path=output_csv,
                               chunksize=0)

    if os.path.exists(output_csv):
        os.remove(output_csv)

    process.transform_disk(input_path=input_csv, output_path=output_csv,
                           chunksize=100)


def test_default_fit_transform():
    process = BinningProcess(variable_names)
    X_transform = process.fit_transform(X, y, metric="indices")

    optb = OptimalBinning()
    x = X[:, 5]
    optb.fit(x, y)

    assert optb.transform(x, metric="indices") == approx(
        X_transform[:, 5])


def test_default_fit_transform_no_selected_variables():
    selection_criteria = {"quality_score": {"min": 0.99}}
    process = BinningProcess(variable_names,
                             selection_criteria=selection_criteria)

    X_transform = process.fit_transform(X, y, metric="event_rate")
    assert X_transform == approx(np.empty(0).reshape((X.shape[0], 0)))


def test_default_fit_transform_disk():
    input_csv = "tests/data/breast_cancer.csv"
    output_csv = "tests/results/breast_cancer_woe_2.csv"

    process = BinningProcess(variable_names, verbose=True)
    process.fit_transform_disk(input_path=input_csv, output_path=output_csv,
                               target="target", chunksize=100)


def test_binning_transform_params():
    btp = {variable_names[0]: {"metric": "bins"},
           variable_names[1]: {"metric": "woe"}}

    process = BinningProcess(variable_names[:3],
                             binning_transform_params=btp)

    with raises(ValueError):
        X_transform = process.fit_transform(X[:, :3], y)


def test_update_binned_variable():
    process = BinningProcess(variable_names)
    process.fit(X, y, check_input=True)

    optb = OptimalPWBinning()
    x = X[:, 5]
    optb.fit(x, y)

    with raises(TypeError):
        process.update_binned_variable(1, optb)

    with raises(ValueError):
        process.update_binned_variable('new_variable', optb)

    with raises(TypeError):
        process.update_binned_variable('mean compactness', None)

    with raises(TypeError):
        coptb = ContinuousOptimalPWBinning()
        coptb.fit(x, y)
        process.update_binned_variable('mean compactness', coptb)

    with raises(ValueError):
        optb = OptimalPWBinning(name="new_name")
        optb.fit(x, y)
        process.update_binned_variable('mean compactness', optb)

    with raises(ValueError):
        optb = OptimalPWBinning(name='mean compactness')
        optb.fit(x, y)
        process.update_binned_variable('mean radius', optb)


def test_information():
    data = load_breast_cancer()

    variable_names = data.feature_names
    X = data.data
    y = data.target

    process = BinningProcess(variable_names)
    process.fit(X, y, check_input=True)

    with raises(ValueError):
        process.information(print_level=-1)

    with open("tests/results/test_binning_process_information.txt", "w") as f:
        with redirect_stdout(f):
            process.information(print_level=0)
            process.information(print_level=1)
            process.information(print_level=2)


def test_summary_get_support():
    data = load_breast_cancer()

    variable_names = data.feature_names
    X = data.data
    y = data.target

    selection_criteria = {"iv": {"min": 0.1, "max": 0.6,
                                 "strategy": "highest", "top": 10}}

    process = BinningProcess(variable_names=variable_names,
                             selection_criteria=selection_criteria)

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

    with open("tests/results/test_binning_process_verbose.txt", "w") as f:
        with redirect_stdout(f):
            process.fit(X, y, check_input=True)


def test_dataframe_index():
    process = BinningProcess(variable_names)
    X_train = pd.DataFrame(X, columns=variable_names, index=[2 * i for i in range(len(X))])
    X_transform = process.fit_transform(X_train, y, metric="indices")
    pd.testing.assert_index_equal(X_train.index, X_transform.index)
