"""
MDLP testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import numpy as np
import pandas as pd

from pytest import approx, raises

from optbinning import MDLP
from sklearn.datasets import load_breast_cancer
from sklearn.exceptions import NotFittedError


data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)

variable = "mean radius"
x = df[variable].values
y = data.target


def test_params():
    with raises(ValueError):
        mdlp = MDLP(min_samples_split=-1)
        mdlp.fit(x, y)

    with raises(ValueError):
        mdlp = MDLP(min_samples_leaf=-1)
        mdlp.fit(x, y)

    with raises(ValueError):
        mdlp = MDLP(max_candidates=-1)
        mdlp.fit(x, y)


# def test_numerical_default():
#     mdlp = MDLP()
#     mdlp.fit(x, y)

#     assert mdlp.splits == approx([10.945, 13.08729032, 15.00163870,
#                                   15.10030322, 16.925, 17.88], rel=1e-6)


# def test_numerical_practical():
#     min_samples_leaf = int(np.ceil(len(x) * 0.05))
#     mdlp = MDLP(max_candidates=128, min_samples_leaf=min_samples_leaf)
#     mdlp.fit(x, y)

#     assert mdlp.splits == approx([10.945, 12.995, 13.71, 15.045, 16.325,
#                                   17.88], rel=1e-6)


def test_splits():
    mdlp = MDLP()

    with raises(NotFittedError):
        mdlp.splits
