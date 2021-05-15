"""
BinningProcessSketch testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2021

import pandas as pd

from pytest import approx, raises

from optbinning import BinningProcessSketch
from optbinning import OptimalBinningSketch
from optbinning.exceptions import NotSolvedError
from optbinning.exceptions import NotDataAddedError
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
variable_names = data.feature_names
df = pd.DataFrame(data.data, columns=variable_names)
y = data.target


def test_params():
    with raises(TypeError):
        BinningProcessSketch(variable_names=1)

    with raises(ValueError):
        BinningProcessSketch(variable_names=[], max_n_prebins=-2)

    with raises(ValueError):
        BinningProcessSketch(variable_names=[], min_n_bins=-2)

    with raises(ValueError):
        BinningProcessSketch(variable_names=[], max_n_bins=-2.2)

    with raises(ValueError):
        BinningProcessSketch(variable_names=[], min_n_bins=3, max_n_bins=2)

    with raises(ValueError):
        BinningProcessSketch(variable_names=[], min_bin_size=0.6)

    with raises(ValueError):
        BinningProcessSketch(variable_names=[], max_bin_size=-0.6)

    with raises(ValueError):
        BinningProcessSketch(variable_names=[], min_bin_size=0.5,
                             max_bin_size=0.3)

    with raises(ValueError):
        BinningProcessSketch(variable_names=[], max_pvalue=1.1)

    with raises(ValueError):
        BinningProcessSketch(variable_names=[], max_pvalue_policy="new_policy")

    with raises(TypeError):
        BinningProcessSketch(variable_names=[], selection_criteria=[])

    with raises(TypeError):
        BinningProcessSketch(variable_names=[], categorical_variables={})

    with raises(TypeError):
        BinningProcessSketch(variable_names=[], categorical_variables=[1, 2])

    with raises(TypeError):
        BinningProcessSketch(variable_names=[], special_codes={1, 2, 3})

    with raises(ValueError):
        BinningProcessSketch(variable_names=[], split_digits=9)

    with raises(TypeError):
        BinningProcessSketch(variable_names=[], binning_fit_params=[1, 2])

    with raises(TypeError):
        BinningProcessSketch(variable_names=[],
                             binning_transform_params=[1, 2])

    with raises(TypeError):
        BinningProcessSketch(variable_names=[], verbose=1)


def test_default():
    bpsketch = BinningProcessSketch(variable_names)
    bpsketch.add(df, y)
    bpsketch.solve()

    optb = bpsketch.get_binned_variable("mean radius")

    assert optb.status == "OPTIMAL"

    optb.binning_table.build()
    assert optb.binning_table.iv == approx(5.04392547, rel=1e-2)


def test_default_merge():
    bpsketch_1 = BinningProcessSketch(variable_names)
    bpsketch_2 = BinningProcessSketch(variable_names)

    df_1, y_1 = df.iloc[:200, :], y[:200]
    df_2, y_2 = df.iloc[200:, :], y[200:]

    bpsketch_1.add(df_1, y_1)
    bpsketch_2.add(df_2, y_2)
    bpsketch_1.merge(bpsketch_2)

    bpsketch_1.solve()

    optb = bpsketch_1.get_binned_variable("mean radius")

    assert optb.status == "OPTIMAL"

    optb.binning_table.build()
    assert optb.binning_table.iv == approx(5.04392547, rel=1e-2)


def test_default_tdigest_merge():
    binning_fit_params = {v: {"sketch": "t-digest"} for v in variable_names}

    bpsketch_1 = BinningProcessSketch(variable_names,
                                      binning_fit_params=binning_fit_params)
    bpsketch_2 = BinningProcessSketch(variable_names,
                                      binning_fit_params=binning_fit_params)

    df_1, y_1 = df.iloc[:200, :], y[:200]
    df_2, y_2 = df.iloc[200:, :], y[200:]

    bpsketch_1.add(df_1, y_1)
    bpsketch_2.add(df_2, y_2)
    bpsketch_1.merge(bpsketch_2)

    bpsketch_1.solve()

    optb = bpsketch_1.get_binned_variable("mean radius")

    assert optb.status == "OPTIMAL"

    optb.binning_table.build()
    assert optb.binning_table.iv == approx(5.04392547, rel=1e-2)


def test_default_transform():
    bpsketch = BinningProcessSketch(variable_names)
    bpsketch.add(df, y)

    with raises(NotSolvedError):
        bpsketch.transform(df, metric="woe")

    bpsketch.solve()

    with raises(TypeError):
        X_transform = bpsketch.transform(df.values, metric="woe")

    with raises(ValueError):
        X_transform = bpsketch.transform(df, metric="new_woe")

    X_transform = bpsketch.transform(df)

    optb = OptimalBinningSketch()
    x = df["mean radius"]
    optb.add(x, y)
    optb.solve()

    assert optb.transform(x, metric="woe") == approx(
        X_transform["mean radius"], rel=1e-6)


def test_information():
    bpsketch = BinningProcessSketch(variable_names)

    with raises(NotDataAddedError):
        bpsketch.solve()

    bpsketch.add(df, y)

    with raises(NotSolvedError):
        bpsketch.information()

    bpsketch.solve()

    with raises(ValueError):
        bpsketch.information(print_level=-1)

    bpsketch.information(print_level=0)
    bpsketch.information(print_level=1)
    bpsketch.information(print_level=2)
