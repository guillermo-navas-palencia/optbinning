"""
OptimalPWBinning testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2022

import pandas as pd

from pytest import approx, raises

from optbinning import OptimalPWBinning
from sklearn.datasets import load_breast_cancer
from sklearn.exceptions import NotFittedError


data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)

variable = "mean radius"
x = df[variable].values
y = data.target


def test_params():
    with raises(TypeError):
        optb = OptimalPWBinning(name=1)
        optb.fit(x, y)

    with raises(TypeError):
        optb = OptimalPWBinning(estimator=2)
        optb.fit(x, y)

    with raises(ValueError):
        optb = OptimalPWBinning(objective="new")
        optb.fit(x, y)

    with raises(ValueError):
        optb = OptimalPWBinning(degree=0.2)
        optb.fit(x, y)

    with raises(TypeError):
        optb = OptimalPWBinning(continuous=1)
        optb.fit(x, y)

    with raises(ValueError):
        optb = OptimalPWBinning(prebinning_method="new")
        optb.fit(x, y)

    with raises(ValueError):
        optb = OptimalPWBinning(min_prebin_size=0.9)
        optb.fit(x, y)

    with raises(ValueError):
        optb = OptimalPWBinning(min_n_bins=1.2)
        optb.fit(x, y)

    with raises(ValueError):
        optb = OptimalPWBinning(max_n_bins=1.2)
        optb.fit(x, y)

    with raises(ValueError):
        optb = OptimalPWBinning(min_n_bins=10, max_n_bins=5)
        optb.fit(x, y)

    with raises(ValueError):
        optb = OptimalPWBinning(min_bin_size=0.6)
        optb.fit(x, y)

    with raises(ValueError):
        optb = OptimalPWBinning(max_bin_size=1.1)
        optb.fit(x, y)

    with raises(ValueError):
        optb = OptimalPWBinning(min_bin_size=0.3, max_bin_size=0.2)
        optb.fit(x, y)

    with raises(ValueError):
        optb = OptimalPWBinning(monotonic_trend="new")
        optb.fit(x, y)

    with raises(ValueError):
        optb = OptimalPWBinning(monotonic_trend="convex", degree=2)
        optb.fit(x, y)

    with raises(ValueError):
        optb = OptimalPWBinning(n_subsamples=1001.2)
        optb.fit(x, y)

    with raises(ValueError):
        optb = OptimalPWBinning(max_pvalue=1.1)
        optb.fit(x, y)

    with raises(ValueError):
        optb = OptimalPWBinning(max_pvalue_policy="new_policy")
        optb.fit(x, y)

    with raises(ValueError):
        optb = OptimalPWBinning(outlier_detector="new_method")
        optb.fit(x, y)

    with raises(TypeError):
        optb = OptimalPWBinning(outlier_detector="range",
                                outlier_params="pass")
        optb.fit(x, y)

    with raises(TypeError):
        optb = OptimalPWBinning(user_splits={"a": [1, 2]})
        optb.fit(x, y)

    with raises(ValueError):
        optb = OptimalPWBinning(user_splits=None,
                                user_splits_fixed=[True, True])
        optb.fit(x, y)

    with raises(TypeError):
        optb = OptimalPWBinning(user_splits=[1, 2],
                                user_splits_fixed=(True, True))
        optb.fit(x, y)

    with raises(ValueError):
        optb = OptimalPWBinning(user_splits=[1, 2],
                                user_splits_fixed=[True, 1])
        optb.fit(x, y)

    with raises(ValueError):
        optb = OptimalPWBinning(user_splits=[1, 2],
                                user_splits_fixed=[True])
        optb.fit(x, y)

    with raises(TypeError):
        optb = OptimalPWBinning(special_codes={1, 2, 3})
        optb.fit(x, y)

    with raises(ValueError):
        optb = OptimalPWBinning(split_digits=9)
        optb.fit(x, y)

    with raises(ValueError):
        optb = OptimalPWBinning(solver=None)
        optb.fit(x, y)

    with raises(ValueError):
        optb = OptimalPWBinning(h_epsilon=0.9)
        optb.fit(x, y)

    with raises(ValueError):
        optb = OptimalPWBinning(quantile=0)
        optb.fit(x, y)

    with raises(ValueError):
        optb = OptimalPWBinning(regularization='l3')
        optb.fit(x, y)

    with raises(ValueError):
        optb = OptimalPWBinning(reg_l1=-0.5)
        optb.fit(x, y)

    with raises(ValueError):
        optb = OptimalPWBinning(reg_l2=-0.5)
        optb.fit(x, y)

    with raises(TypeError):
        optb = OptimalPWBinning(random_state='None')
        optb.fit(x, y)

    with raises(TypeError):
        optb = OptimalPWBinning(verbose=1)
        optb.fit(x, y)


def test_default():
    optb = OptimalPWBinning(name=variable)
    optb.fit(x, y)

    optb.binning_table.build()
    assert optb.binning_table.iv == approx(5.87474602, rel=1e-6)

    with raises(ValueError):
        optb.binning_table.plot(metric="new_metric")

    optb.binning_table.plot(
        metric="woe", savefig="tests/results/test_binning_piecewise.png")


def test_default_discontinuous():
    optb = OptimalPWBinning(name=variable, continuous=False)
    optb.fit(x, y)

    optb.binning_table.build()
    assert optb.binning_table.iv == approx(5.84465825, rel=1e-6)


def test_bounds_transform():
    optb = OptimalPWBinning(name=variable)
    optb.fit(x, y, lb=0.001, ub=0.999)

    x_transform_woe = optb.transform(x, metric="woe")
    assert x_transform_woe[:4] == approx(
        [3.99180564, 4.28245092, 4.17407503, -3.2565373], rel=1e-6)

    x_transform_event_rate = optb.transform(x, metric="event_rate")
    assert x_transform_event_rate[:4] == approx(
        [0.03015878, 0.02272502, 0.02526056, 0.97763604], rel=1e-6)


def test_bounds_fit_transform():
    optb = OptimalPWBinning(name=variable)

    x_transform_woe = optb.fit_transform(
        x, y, lb=0.001, ub=0.999, metric="woe")

    assert x_transform_woe[:4] == approx(
        [3.9918056, 4.2824509, 4.17407503, -3.25653732], rel=1e-6)
    x_transform_event_rate = optb.fit_transform(
        x, y, lb=0.001, ub=0.999, metric="event_rate")
    assert x_transform_event_rate[:4] == approx(
        [0.03015878, 0.02272502, 0.02526056, 0.97763604], rel=1e-6)


def test_solvers():
    for solver in ("auto", "ecos", "osqp"):
        optb = OptimalPWBinning(name=variable, solver=solver)
        optb.fit(x, y)

        optb.binning_table.build()
        assert optb.binning_table.iv == approx(5.87474602, rel=1e-6)


def test_user_splits():
    variable = "mean texture"
    x = df[variable].values

    user_splits = [14, 15, 16, 17, 20, 21, 22, 27]
    user_splits_fixed = [False, True, True, False, False, False, False, False]

    optb = OptimalPWBinning(name=variable, user_splits=user_splits,
                            user_splits_fixed=user_splits_fixed)

    optb.fit(x, y)


def test_information():
    optb = OptimalPWBinning()

    with raises(NotFittedError):
        optb.information()

    optb.fit(x, y)

    with raises(ValueError):
        optb.information(print_level=-1)

    optb.information(print_level=0)
    optb.information(print_level=1)
    optb.information(print_level=2)

    optb = OptimalPWBinning()
    optb.fit(x, y)
    optb.information(print_level=2)


def test_verbose():
    optb = OptimalPWBinning(verbose=True)
    optb.fit(x, y)

    assert optb.status == "OPTIMAL"
