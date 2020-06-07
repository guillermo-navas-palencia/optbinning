"""
Scenario-based stochastic optimal binning testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import numpy as np
import pandas as pd

from pytest import approx, raises

from optbinning.binning.binning_statistics import BinningTable
from optbinning.binning.uncertainty import SBOptimalBinning
from sklearn.datasets import load_breast_cancer
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split


data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)

variable = "mean radius"
x = df[variable].values
y = data.target

x1, x2, y1, y2 = train_test_split(x, y, test_size=0.5, random_state=42)
x3, x4, y3, y4 = train_test_split(x1, y1, test_size=0.2, random_state=42)
x_s = [x1, x3, x4]
y_s = [y1, y3, y4]


def test_params():
    with raises(TypeError):
        sboptb = SBOptimalBinning(name=1)
        sboptb.fit(x_s, y_s)

    with raises(ValueError):
        sboptb = SBOptimalBinning(prebinning_method="new_method")
        sboptb.fit(x_s, y_s)

    with raises(ValueError):
        sboptb = SBOptimalBinning(max_n_prebins=-2)
        sboptb.fit(x_s, y_s)

    with raises(ValueError):
        sboptb = SBOptimalBinning(min_prebin_size=0.6)
        sboptb.fit(x_s, y_s)

    with raises(ValueError):
        sboptb = SBOptimalBinning(min_n_bins=-2)
        sboptb.fit(x_s, y_s)

    with raises(ValueError):
        sboptb = SBOptimalBinning(max_n_bins=-2.2)
        sboptb.fit(x_s, y_s)

    with raises(ValueError):
        sboptb = SBOptimalBinning(min_n_bins=3, max_n_bins=2)
        sboptb.fit(x_s, y_s)

    with raises(ValueError):
        sboptb = SBOptimalBinning(min_bin_size=0.6)
        sboptb.fit(x_s, y_s)

    with raises(ValueError):
        sboptb = SBOptimalBinning(max_bin_size=-0.6)
        sboptb.fit(x_s, y_s)

    with raises(ValueError):
        sboptb = SBOptimalBinning(min_bin_size=0.5, max_bin_size=0.3)
        sboptb.fit(x_s, y_s)

    with raises(ValueError):
        sboptb = SBOptimalBinning(monotonic_trend="new_trend")
        sboptb.fit(x_s, y_s)

    with raises(ValueError):
        sboptb = SBOptimalBinning(min_event_rate_diff=1.1)
        sboptb.fit(x_s, y_s)

    with raises(ValueError):
        sboptb = SBOptimalBinning(max_pvalue=1.1)
        sboptb.fit(x_s, y_s)

    with raises(ValueError):
        sboptb = SBOptimalBinning(max_pvalue_policy="new_policy")
        sboptb.fit(x_s, y_s)

    with raises(TypeError):
        sboptb = SBOptimalBinning(class_weight=[0, 1])
        sboptb.fit(x_s, y_s)

    with raises(ValueError):
        sboptb = SBOptimalBinning(class_weight="unbalanced")
        sboptb.fit(x_s, y_s)

    with raises(TypeError):
        sboptb = SBOptimalBinning(user_splits={"a": [1, 2]})
        sboptb.fit(x_s, y_s)

    with raises(ValueError):
        sboptb = SBOptimalBinning(user_splits_fixed=[True, False])
        sboptb.fit(x_s, y_s)

    with raises(TypeError):
        sboptb = SBOptimalBinning(user_splits=[2], user_splits_fixed={0: True})
        sboptb.fit(x_s, y_s)

    with raises(ValueError):
        sboptb = SBOptimalBinning(user_splits=[2, 3],
                                  user_splits_fixed=[True, 0])
        sboptb.fit(x_s, y_s)

    with raises(ValueError):
        sboptb = SBOptimalBinning(user_splits=[2, 3],
                                  user_splits_fixed=[True])
        sboptb.fit(x_s, y_s)

    with raises(TypeError):
        sboptb = SBOptimalBinning(special_codes={1, 2, 3})
        sboptb.fit(x_s, y_s)

    with raises(ValueError):
        sboptb = SBOptimalBinning(split_digits=9)
        sboptb.fit(x_s, y_s)

    with raises(ValueError):
        sboptb = SBOptimalBinning(time_limit=-2)
        sboptb.fit(x_s, y_s)

    with raises(TypeError):
        sboptb = SBOptimalBinning(verbose=1)
        sboptb.fit(x_s, y_s)


def test_input_scenarios():
    with raises(TypeError):
        sboptb = SBOptimalBinning()
        sboptb.fit(x1, y_s)

    with raises(TypeError):
        sboptb = SBOptimalBinning()
        sboptb.fit(x_s, y1)

    with raises(ValueError):
        sboptb = SBOptimalBinning()
        sboptb.fit([x1, x3], [y1])

    with raises(ValueError):
        sboptb = SBOptimalBinning()
        sboptb.fit(x_s, y_s, [0.2, 0.8])


def test_default():
    sboptb = SBOptimalBinning(monotonic_trend="descending")
    sboptb.fit(x_s, y_s)

    assert sboptb.status == "OPTIMAL"
    assert sboptb.splits == approx([13.09499979, 14.14999962, 15.24499989],
                                   rel=1e-6)


def test_default_transform():
    sboptb = SBOptimalBinning(monotonic_trend="descending")
    sboptb.fit(x_s, y_s)

    x_transform = sboptb.transform([12, 14, 15, 21], metric="woe")
    assert x_transform == approx([-2.46667422, -0.55591146, 0.24544931,
                                  2.8963411], rel=1e-6)


def test_default_fit_transform():
    sboptb = SBOptimalBinning(monotonic_trend="descending")

    x_transform = sboptb.fit_transform([12, 14, 15, 21], x_s, y_s)
    assert x_transform == approx([-2.46667422, -0.55591146, 0.24544931,
                                  2.8963411], rel=1e-6)


def test_user_splits():
    user_splits = [11, 12, 13, 14, 15, 16, 17]

    sboptb = SBOptimalBinning(monotonic_trend="descending",
                              user_splits=user_splits)
    sboptb.fit(x_s, y_s)

    assert sboptb.status == "OPTIMAL"
    assert sboptb.splits == approx([12, 14, 15], rel=1e-6)


def test_user_splits_non_unique():
    user_splits = [11, 12, 13, 13, 15, 16, 17]

    sboptb = SBOptimalBinning(monotonic_trend="descending",
                              user_splits=user_splits)

    with raises(ValueError):
        sboptb.fit(x_s, y_s)


def test_user_splits_fixed():
    user_splits = [11, 12, 13, 14, 14.7, 15.5, 17]
    user_splits_fixed = [False, False, False, False, False, True, False]

    sboptb = SBOptimalBinning(monotonic_trend="descending",
                              user_splits=user_splits,
                              user_splits_fixed=user_splits_fixed)
    sboptb.fit(x_s, y_s)

    assert sboptb.status == "OPTIMAL"
    assert 15.5 in sboptb.splits


def test_min_bin_size():
    sboptb = SBOptimalBinning(monotonic_trend="descending", min_bin_size=0.1)
    sboptb.fit(x_s, y_s)

    n_scenarios = len(x_s)

    for s in range(n_scenarios):
        bt_s = sboptb.binning_table_scenario(s)
        bt = bt_s.build()

        count = bt["Count (%)"].values[:-3]
        assert np.all(count >= 0.1)


def test_max_bin_size():
    sboptb = SBOptimalBinning(monotonic_trend="descending", max_bin_size=0.6)
    sboptb.fit(x_s, y_s)

    assert sboptb.status == "OPTIMAL"

    n_scenarios = len(x_s)

    for s in range(n_scenarios):
        bt_s = sboptb.binning_table_scenario(s)
        bt = bt_s.build()

        count = bt["Count (%)"].values[:-3]
        assert np.all(count <= 0.6)


def test_binning_table_scenario():
    sboptb = SBOptimalBinning(monotonic_trend="descending")

    with raises(NotFittedError):
        sboptb.binning_table_scenario(scenario_id=2)

    sboptb.fit(x_s, y_s)

    with raises(ValueError):
        bt = sboptb.binning_table_scenario(scenario_id=4)

    bt = sboptb.binning_table_scenario(scenario_id=2)

    assert isinstance(bt, BinningTable)


def test_verbose():
    sboptb = SBOptimalBinning(monotonic_trend="descending", verbose=True)
    sboptb.fit(x_s, y_s)

    assert sboptb.status == "OPTIMAL"
