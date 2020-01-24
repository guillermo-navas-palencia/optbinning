"""
OptimalBinning testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import pandas as pd

from pytest import raises

from optbinning import OptimalBinning
from sklearn.datasets import load_breast_cancer


data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)

variable = "mean radius"
x = df[variable].values
y = data.target


def test_params():
    with raises(TypeError):
        optb = OptimalBinning(name=1)
        optb.fit(x, y)

    with raises(ValueError):
        optb = OptimalBinning(dtype="nominal")
        optb.fit(x, y)

    with raises(ValueError):
        optb = OptimalBinning(prebinning_method="new_method")
        optb.fit(x, y)

    with raises(ValueError):
        optb = OptimalBinning(solver="new_solver")
        optb.fit(x, y)
