"""
OptimalBinning2D testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2021

import pandas as pd

from pytest import approx, raises

from optbinning import OptimalBinning2D
from sklearn.datasets import load_breast_cancer
from sklearn.exceptions import NotFittedError


data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)

variable1 = "mean radius"
variable2 = "worst concavity"
x = df[variable1].values
y = df[variable2].values
z = data.target


def test_params():
    with raises(TypeError):
        optb = OptimalBinning2D(name_x=1)
        optb.fit(x, y, z)

    with raises(TypeError):
        optb = OptimalBinning2D(name_y=1)
        optb.fit(x, y, z)

    with raises(ValueError):
        optb = OptimalBinning2D(dtype_x="nominal")
        optb.fit(x, y, z)

    with raises(ValueError):
        optb = OptimalBinning2D(dtype_y="nominal")
        optb.fit(x, y, z)

    with raises(ValueError):
        optb = OptimalBinning2D(prebinning_method="new_method")
        optb.fit(x, y, z)

    with raises(ValueError):
        optb = OptimalBinning2D(strategy="new_strategy")
        optb.fit(x, y, z)

    with raises(ValueError):
        optb = OptimalBinning2D(solver="new_solver")
        optb.fit(x, y, z)

    with raises(ValueError):
        optb = OptimalBinning2D(divergence="new_divergence")
        optb.fit(x, y, z)

    with raises(ValueError):
        optb = OptimalBinning2D(max_n_prebins_x=-2)
        optb.fit(x, y, z)

    with raises(ValueError):
        optb = OptimalBinning2D(max_n_prebins_y=-2)
        optb.fit(x, y, z)

    with raises(ValueError):
        optb = OptimalBinning2D(min_prebin_size_x=0.6)
        optb.fit(x, y, z)

    with raises(ValueError):
        optb = OptimalBinning2D(min_prebin_size_y=0.6)
        optb.fit(x, y, z)

    with raises(ValueError):
        optb = OptimalBinning2D(min_n_bins=-2)
        optb.fit(x, y, z)

    with raises(ValueError):
        optb = OptimalBinning2D(max_n_bins=-2.2)
        optb.fit(x, y, z)

    with raises(ValueError):
        optb = OptimalBinning2D(min_n_bins=3, max_n_bins=2)
        optb.fit(x, y, z)

    with raises(ValueError):
        optb = OptimalBinning2D(min_bin_size=0.6)
        optb.fit(x, y, z)

    with raises(ValueError):
        optb = OptimalBinning2D(max_bin_size=-0.6)
        optb.fit(x, y, z)

    with raises(ValueError):
        optb = OptimalBinning2D(min_bin_size=0.5, max_bin_size=0.3)
        optb.fit(x, y, z)

    with raises(ValueError):
        optb = OptimalBinning2D(min_bin_n_nonevent=-2)
        optb.fit(x, y, z)

    with raises(ValueError):
        optb = OptimalBinning2D(max_bin_n_nonevent=-2)
        optb.fit(x, y, z)

    with raises(ValueError):
        optb = OptimalBinning2D(min_bin_n_nonevent=3, max_bin_n_nonevent=2)
        optb.fit(x, y, z)

    with raises(ValueError):
        optb = OptimalBinning2D(min_bin_n_event=-2)
        optb.fit(x, y, z)

    with raises(ValueError):
        optb = OptimalBinning2D(max_bin_n_event=-2)
        optb.fit(x, y, z)

    with raises(ValueError):
        optb = OptimalBinning2D(min_bin_n_event=3, max_bin_n_event=2)
        optb.fit(x, y, z)

    with raises(ValueError):
        optb = OptimalBinning2D(monotonic_trend_x="new_trend")
        optb.fit(x, y, z)

    with raises(ValueError):
        optb = OptimalBinning2D(monotonic_trend_y="new_trend")
        optb.fit(x, y, z)

    with raises(ValueError):
        optb = OptimalBinning2D(min_event_rate_diff_x=-1.1)
        optb.fit(x, y, z)

    with raises(ValueError):
        optb = OptimalBinning2D(min_event_rate_diff_y=1.1)
        optb.fit(x, y, z)

    with raises(ValueError):
        optb = OptimalBinning2D(gamma=-0.2)
        optb.fit(x, y, z)

    with raises(TypeError):
        optb = OptimalBinning2D(special_codes_x={1, 2, 3})
        optb.fit(x, y, z)

    with raises(TypeError):
        optb = OptimalBinning2D(special_codes_y={1, 2, 3})
        optb.fit(x, y, z)

    with raises(ValueError):
        optb = OptimalBinning2D(split_digits=9)
        optb.fit(x, y, z)

    with raises(ValueError):
        optb = OptimalBinning2D(n_jobs=1.2)
        optb.fit(x, y, z)

    with raises(ValueError):
        optb = OptimalBinning2D(time_limit=-2)
        optb.fit(x, y, z)

    with raises(TypeError):
        optb = OptimalBinning2D(verbose=1)
        optb.fit(x, y, z)


def test_numerical_default():
    optb = OptimalBinning2D()
    optb.fit(x, y, z)

    assert optb.status == "OPTIMAL"

    with raises(TypeError):
        optb.binning_table.build(show_bin_xy=1)

    optb.binning_table.build()
    assert optb.binning_table.iv == approx(7.63248244, rel=1e-6)

    optb.binning_table.analysis()
    assert optb.binning_table.gini == approx(0.96381005, rel=1e-6)
    assert optb.binning_table.js == approx(0.53356918, rel=1e-6)
    assert optb.binning_table.quality_score == approx(0.0, rel=1e-6)

    with raises(ValueError):
        optb.binning_table.plot(metric="new_metric")

    optb.binning_table.plot(
        metric="woe",
        savefig="tests/results/test_binning_2d_woe.png")

    optb.binning_table.plot(
        metric="event_rate",
        savefig="tests/results/test_binning_2d_event_rate.png")


def test_numerical_default_solvers():
    optb_mip = OptimalBinning2D(solver="mip")
    optb_cp = OptimalBinning2D(solver="cp")

    for optb in (optb_mip, optb_cp):
        optb.fit(x, y, z)
        assert optb.status == "OPTIMAL"

        optb.binning_table.build()
        assert optb.binning_table.iv == approx(7.63248244, rel=1e-6)


def test_numerical_strategy():
    optb = OptimalBinning2D(strategy="cart")
    optb.fit(x, y, z)

    assert optb.status == "OPTIMAL"

    optb.binning_table.build()
    assert optb.binning_table.iv == approx(6.63764258, rel=1e-6)


def test_numerical_monotonic_xy():
    optb_mip = OptimalBinning2D(solver="mip", monotonic_trend_x="descending",
                                monotonic_trend_y="descending")

    optb_cp = OptimalBinning2D(solver="cp", monotonic_trend_x="descending",
                               monotonic_trend_y="descending")

    for optb in [optb_mip, optb_cp]:
        optb.fit(x, y, z)
        assert optb.status == "OPTIMAL"

        optb.binning_table.build()
        assert optb.binning_table.iv == approx(7.59474677, rel=1e-6)

    optb_mip = OptimalBinning2D(solver="mip", monotonic_trend_x="ascending",
                                monotonic_trend_y="ascending")

    optb_cp = OptimalBinning2D(solver="cp", monotonic_trend_x="ascending",
                               monotonic_trend_y="ascending")

    for optb in [optb_mip, optb_cp]:
        optb.fit(x, y, z)
        assert optb.status == "OPTIMAL"

        optb.binning_table.build()
        assert optb.binning_table.iv == approx(0, rel=1e-6)


def test_numerical_min_max_n_bins():
    optb_mip = OptimalBinning2D(solver="mip", min_n_bins=2, max_n_bins=5)
    optb_cp = OptimalBinning2D(solver="cp", min_n_bins=2, max_n_bins=5)

    for optb in [optb_mip, optb_cp]:
        optb.fit(x, y, z)
        assert optb.status == "OPTIMAL"
        assert 2 <= len(optb.splits[0]) <= 5
        assert 2 <= len(optb.splits[1]) <= 5


def test_numerical_regularization():
    optb_mip = OptimalBinning2D(min_bin_size=0.05, solver="mip", gamma=600)
    optb_cp = OptimalBinning2D(min_bin_size=0.05, solver="cp", gamma=600)
    optb_mip.fit(x, y, z)
    optb_cp.fit(x, y, z)

    assert len(optb_mip.splits[0]) < 6
    assert len(optb_cp.splits[0]) < 6


def test_numerical_default_transform():
    optb = OptimalBinning2D()
    with raises(NotFittedError):
        z_transform = optb.transform(x, y)

    optb.fit(x, y, z)

    with raises(ValueError):
        z_transform = optb.transform(x, y, metric="new_metric")

    z_transform = optb.transform(x, y, metric="woe")
    assert z_transform[:5] == approx([5.37317977, 3.51688178, 5.37317977,
                                      0.52114951, 5.37317977], rel=1e-6)

    z_transform = optb.transform(x, y, metric="event_rate", check_input=True)
    assert z_transform[:5] == approx([0.00775194, 0.04761905, 0.00775194, 0.5,
                                      0.00775194], rel=1e-6)

    z_transform = optb.transform(x, y, metric="indices")
    assert z_transform[:5] == approx([12, 3, 12, 13, 12])

    z_transform = optb.transform(x, y, metric="bins")
    assert z_transform[0] == '[15.05, inf) $\\cup$ [0.32, inf)'


def test_numerical_default_fit_transform():
    optb = OptimalBinning2D()

    z_transform = optb.fit_transform(x, y, z, metric="woe")
    assert z_transform[:5] == approx([5.37317977, 3.51688178, 5.37317977,
                                      0.52114951, 5.37317977], rel=1e-6)


def test_numerical_categorical_transform():
    optb = OptimalBinning2D(dtype_x="numerical", dtype_y="categorical")
    optb.fit(x, y, z)
    z_transform = optb.fit_transform(x, y, z, metric="woe")

    assert z_transform[:5] == approx([5.28332344, 5.28332344, 5.28332344,
                                      -2.44333022,  5.28332344], rel=1e-6)


def test_categorical_categorical_transform():
    optb = OptimalBinning2D(dtype_x="categorical", dtype_y="categorical")
    optb.fit(x, y, z)
    z_transform = optb.fit_transform(x, y, z, metric="woe")

    assert z_transform[:5] == approx([2.86295531] * 5, rel=1e-6)


def test_information():
    optb = OptimalBinning2D(solver="cp")

    with raises(NotFittedError):
        optb.information()

    optb.fit(x, y, z)

    with raises(ValueError):
        optb.information(print_level=-1)

    optb.information(print_level=0)
    optb.information(print_level=1)
    optb.information(print_level=2)

    optb = OptimalBinning2D(solver="mip")
    optb.fit(x, y, z)
    optb.information(print_level=2)


def test_verbose():
    optb = OptimalBinning2D(verbose=True)
    optb.fit(x, y, z)

    assert optb.status == "OPTIMAL"
