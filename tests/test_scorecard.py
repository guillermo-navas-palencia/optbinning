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
from sklearn.datasets import load_boston
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression



def test_params():
    data = load_breast_cancer()

    variable_names = data.feature_names
    df = pd.DataFrame(data.data, columns=variable_names)
    df["target"] = data.target

    binning_process = BinningProcess(variable_names)
    estimator = LogisticRegression()

    with raises(TypeError):
        scorecard = Scorecard(target=1, binning_process=binning_process,
                              estimator=estimator)
        scorecard.fit(df)

    with raises(TypeError):
        scorecard = Scorecard(target="target", binning_process=estimator,
                              estimator=estimator)
        scorecard.fit(df)

    with raises(TypeError):
        scorecard = Scorecard(target="target", binning_process=binning_process,
                              estimator=binning_process)
        scorecard.fit(df)

    with raises(ValueError):
        scorecard = Scorecard(target="target", binning_process=binning_process,
                              estimator=estimator, scaling_method="new_method",
                              scaling_method_params=dict())
        scorecard.fit(df)

    with raises(ValueError):
        scorecard = Scorecard(target="target", binning_process=binning_process,
                              estimator=estimator, scaling_method="min_max",
                              scaling_method_params=None)
        scorecard.fit(df)

    with raises(TypeError):
        scorecard = Scorecard(target="target", binning_process=binning_process,
                              estimator=estimator, scaling_method="min_max",
                              scaling_method_params=[])
        scorecard.fit(df)

    with raises(TypeError):
        scorecard = Scorecard(target="target", binning_process=binning_process,
                              estimator=estimator, intercept_based=1)
        scorecard.fit(df)

    with raises(TypeError):
        scorecard = Scorecard(target="target", binning_process=binning_process,
                              estimator=estimator, reverse_scorecard=1)
        scorecard.fit(df)

    with raises(TypeError):
        scorecard = Scorecard(target="target", binning_process=binning_process,
                              estimator=estimator, rounding=1)
        scorecard.fit(df)

    with raises(TypeError):
        scorecard = Scorecard(target="target", binning_process=binning_process,
                              estimator=estimator, verbose=1)
        scorecard.fit(df)


def test_scaling_method_params():
    pass


def test_default():
    pass