"""
OptimalBinning testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import pandas as pd

from pytest import approx, raises

from optbinning import OptimalBinning
from sklearn.datasets import load_breast_cancer
from sklearn.exceptions import NotFittedError


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

    with raises(ValueError):
        optb = OptimalBinning(max_n_prebins=-2)
        optb.fit(x, y)

    with raises(ValueError):
        optb = OptimalBinning(min_prebin_size=0.6)
        optb.fit(x, y)

    with raises(ValueError):
        optb = OptimalBinning(min_n_bins=-2)
        optb.fit(x, y)

    with raises(ValueError):
        optb = OptimalBinning(max_n_bins=-2.2)
        optb.fit(x, y)

    with raises(ValueError):
        optb = OptimalBinning(min_n_bins=3, max_n_bins=2)
        optb.fit(x, y)

    with raises(ValueError):
        optb = OptimalBinning(min_bin_size=0.6)
        optb.fit(x, y)

    with raises(ValueError):
        optb = OptimalBinning(max_bin_size=-0.6)
        optb.fit(x, y)

    with raises(ValueError):
        optb = OptimalBinning(min_bin_size=0.5, max_bin_size=0.3)
        optb.fit(x, y)

    with raises(ValueError):
        optb = OptimalBinning(min_bin_n_nonevent=-2)
        optb.fit(x, y)

    with raises(ValueError):
        optb = OptimalBinning(max_bin_n_nonevent=-2)
        optb.fit(x, y)

    with raises(ValueError):
        optb = OptimalBinning(min_bin_n_nonevent=3, max_bin_n_nonevent=2)
        optb.fit(x, y)

    with raises(ValueError):
        optb = OptimalBinning(min_bin_n_event=-2)
        optb.fit(x, y)

    with raises(ValueError):
        optb = OptimalBinning(max_bin_n_event=-2)
        optb.fit(x, y)

    with raises(ValueError):
        optb = OptimalBinning(min_bin_n_event=3, max_bin_n_event=2)
        optb.fit(x, y)

    with raises(ValueError):
        optb = OptimalBinning(monotonic_trend="new_trend")
        optb.fit(x, y)

    with raises(ValueError):
        optb = OptimalBinning(min_event_rate_diff=1.1)
        optb.fit(x, y)

    with raises(ValueError):
        optb = OptimalBinning(max_pvalue=1.1)
        optb.fit(x, y)

    with raises(ValueError):
        optb = OptimalBinning(max_pvalue_policy="new_policy")
        optb.fit(x, y)

    with raises(ValueError):
        optb = OptimalBinning(gamma=-0.2)
        optb.fit(x, y)

    with raises(TypeError):
        optb = OptimalBinning(class_weight=[0, 1])
        optb.fit(x, y)

    with raises(ValueError):
        optb = OptimalBinning(class_weight="unbalanced")
        optb.fit(x, y)

    with raises(ValueError):
        optb = OptimalBinning(cat_cutoff=-0.2)
        optb.fit(x, y)

    with raises(TypeError):
        optb = OptimalBinning(user_splits={"a": [1, 2]})
        optb.fit(x, y)

    with raises(TypeError):
        optb = OptimalBinning(special_codes={1, 2, 3})
        optb.fit(x, y)

    with raises(ValueError):
        optb = OptimalBinning(split_digits=9)
        optb.fit(x, y)

    with raises(ValueError):
        optb = OptimalBinning(mip_solver="new_solver")
        optb.fit(x, y)

    with raises(ValueError):
        optb = OptimalBinning(time_limit=-2)
        optb.fit(x, y)

    with raises(TypeError):
        optb = OptimalBinning(verbose=1)
        optb.fit(x, y)


def test_numerical_default():
    optb = OptimalBinning()
    optb.fit(x, y)

    assert optb.status == "OPTIMAL"
    assert optb.splits == approx([11.42500019, 12.32999992, 13.09499979,
                                  13.70499992, 15.04500008, 16.92500019],
                                 rel=1e-6)

    optb.binning_table.build()
    assert optb.binning_table.iv == approx(5.04392547, rel=1e-6)

    optb.binning_table.analysis()
    assert optb.binning_table.gini == approx(0.87541620, rel=1e-6)
    assert optb.binning_table.js == approx(0.39378376, rel=1e-6)
    assert optb.binning_table.quality_score == approx(0.0, rel=1e-6)

    with raises(ValueError):
        optb.binning_table.plot(metric="new_metric")

    optb.binning_table.plot(metric="woe")
    optb.binning_table.plot(metric="event_rate")


def test_numerical_default_solvers():
    optb_mip_cbc = OptimalBinning(solver="mip", mip_solver="cbc")
    optb_mip_bop = OptimalBinning(solver="mip", mip_solver="bop")
    optb_cp = OptimalBinning(solver="cp")

    for optb in [optb_mip_bop, optb_mip_cbc, optb_cp]:
        optb.fit(x, y)
        assert optb.status == "OPTIMAL"
        assert optb.splits == approx([11.42500019, 12.32999992, 13.09499979,
                                      13.70499992, 15.04500008, 16.92500019],
                                     rel=1e-6)


def test_numerical_user_splits():
    user_splits = [11, 12, 13, 14, 15, 17]
    optb = OptimalBinning(user_splits=user_splits, max_pvalue=0.05)
    optb.fit(x, y)

    assert optb.status == "OPTIMAL"
    assert optb.splits == approx([13, 15, 17], rel=1e-6)

    optb.binning_table.build()
    assert optb.binning_table.iv == 4.819661314733627

    optb = OptimalBinning(user_splits=user_splits, max_pvalue=0.05,
                          max_pvalue_policy="all")
    optb.fit(x, y)
    optb.binning_table.build()
    assert optb.binning_table.iv == 4.819661314733627


def test_categorical_default_user_splits():
    df = pd.read_csv("data/test_categorical.csv", sep=",", engine="c")
    x = df.NAME_INCOME_TYPE.values
    y = df.TARGET.values

    optb = OptimalBinning(dtype="categorical", solver="mip", cat_cutoff=0.1,
                          verbose=True)
    optb.fit(x, y)

    assert optb.status == "OPTIMAL"
    assert optb.splits == [['Pensioner'], ['Working'],
                           ['Commercial associate'], ['State servant']]

    user_splits = [['Pensioner', 'Working'],
                   ['Commercial associate'], ['State servant']]

    optb = OptimalBinning(dtype="categorical", solver="mip", cat_cutoff=0.1,
                          user_splits=user_splits, verbose=True)
    optb.fit(x, y)

    assert optb.status == "OPTIMAL"
    assert optb.splits == [['State servant', 'Pensioner', 'Working',
                            'Commercial associate']]


def test_auto_modes():
    optb0 = OptimalBinning(monotonic_trend="auto")
    optb1 = OptimalBinning(monotonic_trend="auto_heuristic")
    optb2 = OptimalBinning(monotonic_trend="auto_asc_desc")

    for optb in [optb0, optb1, optb2]:
        optb.fit(x, y)
        assert optb.status == "OPTIMAL"
        assert optb.splits == approx([11.42500019, 12.32999992, 13.09499979,
                                      13.70499992, 15.04500008, 16.92500019],
                                     rel=1e-6)


def test_numerical_min_max_n_bins():
    optb_mip = OptimalBinning(solver="mip", min_n_bins=2, max_n_bins=5)
    optb_cp = OptimalBinning(solver="cp", min_n_bins=2, max_n_bins=5)

    for optb in [optb_mip, optb_cp]:
        optb.fit(x, y)
        assert optb.status == "OPTIMAL"
        assert 2 <= len(optb.splits + 1) <= 5


def test_outlier():
    with raises(ValueError):
        optb = OptimalBinning(outlier_detector="new_outlier")
        optb.fit(x, y)

    with raises(TypeError):
        optb = OptimalBinning(outlier_detector="range", outlier_params=[])
        optb.fit(x, y)

    optb = OptimalBinning(outlier_detector="zscore", verbose=True)
    optb.fit(x, y)
    assert optb.splits == approx([11.42500019, 12.32999992, 13.09499979,
                                  13.70499992, 15.04500008, 16.92500019],
                                 rel=1e-6)

    optb_eti = OptimalBinning(outlier_detector="range",
                              outlier_params={"interval_length": 0.9,
                                              "method": "ETI"})

    optb_hdi = OptimalBinning(outlier_detector="range",
                              outlier_params={"interval_length": 0.9,
                                              "method": "HDI"})

    for optb in [optb_eti, optb_hdi]:
        optb.fit(x, y)
        assert optb.splits == approx([11.42500019, 12.32999992, 13.09499979,
                                      13.70499992, 15.04500008, 16.92500019],
                                     rel=1e-6)


def test_numerical_regularization():
    optb_mip = OptimalBinning(solver="mip", gamma=4)
    optb_cp = OptimalBinning(solver="cp", gamma=4)
    optb_mip.fit(x, y)
    optb_cp.fit(x, y)

    assert len(optb_mip.splits) < 6
    assert len(optb_cp.splits) < 6


def test_numerical_default_transform():
    optb = OptimalBinning()
    with raises(NotFittedError):
        x_transform = optb.transform(x)

    optb.fit(x, y)

    x_transform = optb.transform([12, 14, 15, 21], metric="woe")
    assert x_transform == approx([-2.71097154, -0.15397917, -0.15397917,
                                  5.28332344], rel=1e-6)


def test_numerical_default_fit_transform():
    optb = OptimalBinning()

    x_transform = optb.fit_transform(x, y, metric="woe")
    assert x_transform[:5] == approx([5.28332344, 5.28332344, 5.28332344,
                                      -3.12517033, 5.28332344], rel=1e-6)


def test_categorical_transform():
    df = pd.read_csv("data/test_categorical.csv", sep=",", engine="c")
    x = df.NAME_INCOME_TYPE.values
    y = df.TARGET.values

    optb = OptimalBinning(dtype="categorical", solver="mip", cat_cutoff=0.1,
                          verbose=True)
    optb.fit(x, y)
    x_transform = optb.transform(["Pensioner", "Working",
                                  "Commercial associate", "State servant"])

    assert x_transform == approx([0.10793784, -0.00524477, -0.18017333,
                                  0.81450804], rel=1e-6)


def test_information():
    optb = OptimalBinning(solver="cp")

    with raises(NotFittedError):
        optb.information()

    optb.fit(x, y)

    with raises(ValueError):
        optb.information(print_level=-1)

    optb.information(print_level=0)
    optb.information(print_level=1)
    optb.information(print_level=2)

    optb = OptimalBinning(solver="mip")
    optb.fit(x, y)
    optb.information(print_level=2)


def test_verbose():
    optb = OptimalBinning(verbose=True)
    optb.fit(x, y)

    assert optb.status == "OPTIMAL"
