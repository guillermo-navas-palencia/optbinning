"""
ContinuousOptimalBinning testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import numpy as np
import pandas as pd

from pytest import approx, raises

from optbinning import ContinuousOptimalBinning
from sklearn.exceptions import NotFittedError
from tests.datasets import load_boston

data = load_boston()
df = pd.DataFrame(data.data, columns=data.feature_names)

variable = "LSTAT"
x = df[variable].values
y = data.target


def test_params():
    with raises(TypeError):
        optb = ContinuousOptimalBinning(name=1)
        optb.fit(x, y)

    with raises(ValueError):
        optb = ContinuousOptimalBinning(dtype="nominal")
        optb.fit(x, y)

    with raises(ValueError):
        optb = ContinuousOptimalBinning(prebinning_method="new_method")
        optb.fit(x, y)

    with raises(ValueError):
        optb = ContinuousOptimalBinning(max_n_prebins=-2)
        optb.fit(x, y)

    with raises(ValueError):
        optb = ContinuousOptimalBinning(min_prebin_size=0.6)
        optb.fit(x, y)

    with raises(ValueError):
        optb = ContinuousOptimalBinning(min_n_bins=-2)
        optb.fit(x, y)

    with raises(ValueError):
        optb = ContinuousOptimalBinning(max_n_bins=-2.2)
        optb.fit(x, y)

    with raises(ValueError):
        optb = ContinuousOptimalBinning(min_n_bins=3, max_n_bins=2)
        optb.fit(x, y)

    with raises(ValueError):
        optb = ContinuousOptimalBinning(min_bin_size=0.6)
        optb.fit(x, y)

    with raises(ValueError):
        optb = ContinuousOptimalBinning(max_bin_size=-0.6)
        optb.fit(x, y)

    with raises(ValueError):
        optb = ContinuousOptimalBinning(min_bin_size=0.5, max_bin_size=0.3)
        optb.fit(x, y)

    with raises(ValueError):
        optb = ContinuousOptimalBinning(monotonic_trend="new_trend")
        optb.fit(x, y)

    with raises(ValueError):
        optb = ContinuousOptimalBinning(min_mean_diff=-1.1)
        optb.fit(x, y)

    with raises(ValueError):
        optb = ContinuousOptimalBinning(max_pvalue=1.1)
        optb.fit(x, y)

    with raises(ValueError):
        optb = ContinuousOptimalBinning(max_pvalue_policy="new_policy")
        optb.fit(x, y)

    with raises(ValueError):
        optb = ContinuousOptimalBinning(cat_cutoff=-0.2)
        optb.fit(x, y)

    with raises(TypeError):
        optb = ContinuousOptimalBinning(user_splits={"a": [1, 2]})
        optb.fit(x, y)

    with raises(TypeError):
        optb = ContinuousOptimalBinning(special_codes={1, 2, 3})
        optb.fit(x, y)

    with raises(ValueError):
        optb = ContinuousOptimalBinning(split_digits=9)
        optb.fit(x, y)

    with raises(ValueError):
        optb = ContinuousOptimalBinning(time_limit=-2)
        optb.fit(x, y)

    with raises(TypeError):
        optb = ContinuousOptimalBinning(verbose=1)
        optb.fit(x, y)


def test_numerical_default():
    optb = ContinuousOptimalBinning()
    optb.fit(x, y)

    assert optb.status == "OPTIMAL"
    assert optb.splits == approx([4.6500001, 5.49499989, 6.86500001, 9.7249999,
                                  13.0999999, 14.4000001, 17.23999977,
                                  19.89999962, 23.31500053], rel=1e-6)

    optb.binning_table.build()
    optb.binning_table.analysis()
    optb.binning_table.plot(
        savefig="tests/results/test_continuous_binning.png")

    optb.binning_table.plot(
        add_special=False,
        savefig="tests/results/test_continuous_binning_no_special.png")

    optb.binning_table.plot(
        add_missing=False,
        savefig="tests/results/test_continuous_binning_no_missing.png")


def test_numerical_user_splits_fixed():
    user_splits = [4, 7, 7.1, 10, 16, 20, 23]

    with raises(ValueError):
        user_splits_fixed = [True, True, True, True, False, False, False]
        optb = ContinuousOptimalBinning(user_splits_fixed=user_splits_fixed)
        optb.fit(x, y)

    with raises(TypeError):
        user_splits_fixed = (False, False, False, False, False, True, False)
        optb = ContinuousOptimalBinning(user_splits=user_splits,
                                        user_splits_fixed=user_splits_fixed)
        optb.fit(x, y)

    with raises(ValueError):
        user_splits_fixed = [0, 0, 0, 0, 0, 1, 0]
        optb = ContinuousOptimalBinning(user_splits=user_splits,
                                        user_splits_fixed=user_splits_fixed)
        optb.fit(x, y)

    with raises(ValueError):
        user_splits_fixed = [False, False, False, False]
        optb = ContinuousOptimalBinning(user_splits=user_splits,
                                        user_splits_fixed=user_splits_fixed)
        optb.fit(x, y)

    user_splits_fixed = [True, True, True, True, False, False, False]
    optb = ContinuousOptimalBinning(user_splits=user_splits,
                                    user_splits_fixed=user_splits_fixed)
    optb.fit(x, y)

    assert optb.status == "INFEASIBLE"


def test_numerical_user_splits_non_unique():
    user_splits = [4, 7, 7, 10, 16, 20, 23]
    optb = ContinuousOptimalBinning(user_splits=user_splits)

    with raises(ValueError):
        optb.fit(x, y)


def test_categorical_user_splits():
    np.random.seed(0)
    n = 100000

    x = sum([[i] * n for i in [-1, 2, 3, 4, 7, 8, 9, 10]], [])
    y = list(np.random.binomial(1, 0.011665, n))
    y += list(np.zeros(n))
    y += list(np.random.binomial(1, 0.0133333, n))
    y += list(np.random.binomial(1, 0.166667, n))
    y += list(np.zeros(n))
    y += list(np.random.binomial(1, 0.0246041, n))
    y += list(np.zeros(n))
    y += list(np.random.binomial(1, 0.025641, n))

    user_splits = np.array([[2., 7., 9., 3., 10., 4.], [8], [-1]],
                           dtype=object)
    user_splits_fixed = [True, True, True]

    optb1 = ContinuousOptimalBinning(dtype="categorical",
                                     user_splits=user_splits)
    optb2 = ContinuousOptimalBinning(dtype="categorical",
                                     user_splits=user_splits,
                                     user_splits_fixed=user_splits_fixed)

    for optb in (optb1, optb2):
        optb.fit(x, y)
        optb.binning_table.build()
        assert optb.status == "OPTIMAL"


def test_numerical_max_pvalue():
    optb0 = ContinuousOptimalBinning(max_pvalue=0.05,
                                     max_pvalue_policy="consecutive")
    optb1 = ContinuousOptimalBinning(max_pvalue=0.05, max_pvalue_policy="all")

    for optb in [optb0, optb1]:
        optb.fit(x, y)
        assert optb.status == "OPTIMAL"

    assert optb0.splits == approx([4.6500001, 5.49499989, 7.68499994,
                                   9.7249999, 11.67499971, 14.4000001,
                                   17.239999, 23.315000], rel=1e-6)

    assert optb1.splits == approx([4.6500001, 5.49499989, 7.68499994,
                                   9.7249999, 11.67499971, 14.4000001,
                                   17.239999, 23.315000], rel=1e-6)


def test_min_mean_diff():
    min_mean_diff = 2

    optb = ContinuousOptimalBinning(
        monotonic_trend=None, min_mean_diff=min_mean_diff)
    optb.fit(x, y)

    mean = optb.binning_table.build()['Mean'].values[:-3]
    min_diff = np.absolute(mean[1:] - mean[:-1])
    assert np.all(min_diff >= min_mean_diff)


def test_auto_modes():
    x = df["INDUS"].values

    optb0 = ContinuousOptimalBinning(monotonic_trend="auto")
    optb1 = ContinuousOptimalBinning(monotonic_trend="auto_heuristic")
    optb2 = ContinuousOptimalBinning(monotonic_trend="auto_asc_desc")
    optb3 = ContinuousOptimalBinning(monotonic_trend="descending")

    for optb in [optb0, optb1, optb2, optb3]:
        optb.fit(x, y)
        assert optb.status == "OPTIMAL"

        optb.binning_table.build()
        optb.binning_table.analysis()

    assert optb0.binning_table.woe >= optb1.binning_table.woe
    assert optb2.splits == approx(optb3.splits, rel=1e-6)


def test_numerical_default_transform():
    optb = ContinuousOptimalBinning()
    with raises(NotFittedError):
        x_transform = optb.transform(x)

    optb.fit(x, y)

    x_transform = optb.transform([0.2, 4.1, 7.2, 26])
    assert x_transform == approx([39.718, 39.718, 25.56067416, 11.82978723],
                                 rel=1e-6)


def test_numerical_default_fit_transform():
    optb = ContinuousOptimalBinning()

    x_transform = optb.fit_transform(x, y)
    assert x_transform[:5] == approx([30.47142857, 25.56067416, 39.718, 39.718,
                                      30.47142857], rel=1e-6)


def test_verbose():
    optb = ContinuousOptimalBinning(verbose=True)
    optb.fit(x, y)

    assert optb.status == "OPTIMAL"
