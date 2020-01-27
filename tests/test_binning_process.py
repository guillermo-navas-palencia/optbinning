"""
Binning process testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

from pytest import approx, raises

from optbinning import BinningProcess
from sklearn.datasets import load_breast_cancer
from sklearn.exceptions import NotFittedError


data = load_breast_cancer()

variable_names = data.feature_names
X = data.data
y = data.target


def test_params():
    with raises(TypeError):
        process = BinningProcess(variable_names=1)
        process.fit(X, y)


def test_default():
    process = BinningProcess(variable_names)
    process.fit(X, y)

    optb = process.get_binned_variable("mean radius")

    assert optb.status == "OPTIMAL"
    assert optb.splits == approx([11.42500019, 12.32999992, 13.09499979,
                                  13.70499992, 15.04500008, 16.92500019],
                                 rel=1e-6)

    optb.binning_table.build()
    assert optb.binning_table.iv == approx(5.04392547, rel=1e-6)


def test_default_transform():
    process = BinningProcess(variable_names)
    with raises(NotFittedError):
        process.transform(X)
