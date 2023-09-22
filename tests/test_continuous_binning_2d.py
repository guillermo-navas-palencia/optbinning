"""
ContinuousOptimalBinning2D testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2022

import pandas as pd

from pytest import approx, raises

from optbinning import ContinuousOptimalBinning2D
from sklearn.exceptions import NotFittedError
from tests.datasets import load_boston

data = load_boston()
df = pd.DataFrame(data.data, columns=data.feature_names)

variable1 = "AGE"
variable2 = "INDUS"

x = df[variable1].values
y = df[variable2].values
z = data.target


def test_params():
    with raises(TypeError):
        optb = ContinuousOptimalBinning2D(name_x=1)
        optb.fit(x, y, z)

    with raises(TypeError):
        optb = ContinuousOptimalBinning2D(name_y=1)
        optb.fit(x, y, z)

    with raises(ValueError):
        optb = ContinuousOptimalBinning2D(dtype_x="nominal")
        optb.fit(x, y, z)

    with raises(ValueError):
        optb = ContinuousOptimalBinning2D(dtype_y="nominal")
        optb.fit(x, y, z)

    with raises(ValueError):
        optb = ContinuousOptimalBinning2D(prebinning_method="new_method")
        optb.fit(x, y, z)

    with raises(ValueError):
        optb = ContinuousOptimalBinning2D(strategy="new_strategy")
        optb.fit(x, y, z)

    with raises(ValueError):
        optb = ContinuousOptimalBinning2D(solver="new_solver")
        optb.fit(x, y, z)

    with raises(ValueError):
        optb = ContinuousOptimalBinning2D(max_n_prebins_x=-2)
        optb.fit(x, y, z)

    with raises(ValueError):
        optb = ContinuousOptimalBinning2D(max_n_prebins_y=-2)
        optb.fit(x, y, z)

    with raises(ValueError):
        optb = ContinuousOptimalBinning2D(min_prebin_size_x=0.6)
        optb.fit(x, y, z)

    with raises(ValueError):
        optb = ContinuousOptimalBinning2D(min_prebin_size_y=0.6)
        optb.fit(x, y, z)

    with raises(ValueError):
        optb = ContinuousOptimalBinning2D(min_n_bins=-2)
        optb.fit(x, y, z)

    with raises(ValueError):
        optb = ContinuousOptimalBinning2D(max_n_bins=-2.2)
        optb.fit(x, y, z)

    with raises(ValueError):
        optb = ContinuousOptimalBinning2D(min_n_bins=3, max_n_bins=2)
        optb.fit(x, y, z)

    with raises(ValueError):
        optb = ContinuousOptimalBinning2D(min_bin_size=0.6)
        optb.fit(x, y, z)

    with raises(ValueError):
        optb = ContinuousOptimalBinning2D(max_bin_size=-0.6)
        optb.fit(x, y, z)

    with raises(ValueError):
        optb = ContinuousOptimalBinning2D(min_bin_size=0.5, max_bin_size=0.3)
        optb.fit(x, y, z)

    with raises(ValueError):
        optb = ContinuousOptimalBinning2D(monotonic_trend_x="new_trend")
        optb.fit(x, y, z)

    with raises(ValueError):
        optb = ContinuousOptimalBinning2D(monotonic_trend_y="new_trend")
        optb.fit(x, y, z)

    with raises(ValueError):
        optb = ContinuousOptimalBinning2D(min_mean_diff_x='-1.1')
        optb.fit(x, y, z)

    with raises(ValueError):
        optb = ContinuousOptimalBinning2D(min_mean_diff_y='1.1')
        optb.fit(x, y, z)

    with raises(ValueError):
        optb = ContinuousOptimalBinning2D(gamma=-0.2)
        optb.fit(x, y, z)

    with raises(TypeError):
        optb = ContinuousOptimalBinning2D(special_codes_x={1, 2, 3})
        optb.fit(x, y, z)

    with raises(TypeError):
        optb = ContinuousOptimalBinning2D(special_codes_y={1, 2, 3})
        optb.fit(x, y, z)

    with raises(ValueError):
        optb = ContinuousOptimalBinning2D(split_digits=9)
        optb.fit(x, y, z)

    with raises(ValueError):
        optb = ContinuousOptimalBinning2D(n_jobs=1.2)
        optb.fit(x, y, z)

    with raises(ValueError):
        optb = ContinuousOptimalBinning2D(time_limit=-2)
        optb.fit(x, y, z)

    with raises(TypeError):
        optb = ContinuousOptimalBinning2D(verbose=1)
        optb.fit(x, y, z)


def test_numerical_default():
    optb = ContinuousOptimalBinning2D()
    optb.fit(x, y, z)

    assert optb.status == "OPTIMAL"

    with raises(TypeError):
        optb.binning_table.build(show_bin_xy=1)

    optb.binning_table.build()
    assert optb.binning_table.iv == approx(4.80825509, rel=1e-6)

    optb.binning_table.analysis()
    assert optb.binning_table.woe == approx(171.946019, rel=1e-6)

    optb.binning_table.plot(
        savefig="tests/results/test_continuous_binning_2d.png")


def test_numerical_default_solvers():
    optb_mip = ContinuousOptimalBinning2D(solver="mip")
    optb_cp = ContinuousOptimalBinning2D(solver="cp")

    for optb in (optb_mip, optb_cp):
        optb.fit(x, y, z)
        assert optb.status == "OPTIMAL"

        optb.binning_table.build()
        assert optb.binning_table.iv == approx(4.80825509, rel=1e-6)


def test_numerical_strategy():
    optb = ContinuousOptimalBinning2D(strategy="cart")
    optb.fit(x, y, z)

    assert optb.status == "OPTIMAL"

    optb.binning_table.build()
    assert optb.binning_table.iv == approx(4.61282007, rel=1e-6)


def test_numerical_monotonic_xy():
    optb_mip = ContinuousOptimalBinning2D(
        solver="mip",
        monotonic_trend_x="descending",
        monotonic_trend_y="descending")

    optb_cp = ContinuousOptimalBinning2D(
        solver="cp",
        monotonic_trend_x="descending",
        monotonic_trend_y="descending")

    for optb in [optb_mip, optb_cp]:
        optb.fit(x, y, z)
        assert optb.status == "OPTIMAL"

        optb.binning_table.build()
        assert optb.binning_table.iv == approx(4.5296802, rel=1e-6)

    optb_mip = ContinuousOptimalBinning2D(
        solver="mip",
        monotonic_trend_x="ascending",
        monotonic_trend_y="ascending")

    optb_cp = ContinuousOptimalBinning2D(
        solver="cp",
        monotonic_trend_x="ascending",
        monotonic_trend_y="ascending")

    for optb in [optb_mip, optb_cp]:
        optb.fit(x, y, z)
        assert optb.status == "OPTIMAL"

        optb.binning_table.build()
        assert optb.binning_table.iv == approx(0, rel=1e-6)


def test_numerical_min_max_n_bins():
    optb_mip = ContinuousOptimalBinning2D(
        solver="mip", min_n_bins=2, max_n_bins=5)
    optb_cp = ContinuousOptimalBinning2D(
        solver="cp", min_n_bins=2, max_n_bins=5)

    for optb in [optb_mip, optb_cp]:
        optb.fit(x, y, z)
        assert optb.status == "OPTIMAL"
        assert 2 <= len(optb.splits[0]) <= 5
        assert 2 <= len(optb.splits[1]) <= 5


def test_numerical_default_transform():
    optb = ContinuousOptimalBinning2D()
    with raises(NotFittedError):
        z_transform = optb.transform(x, y)

    optb.fit(x, y, z)

    with raises(ValueError):
        z_transform = optb.transform(x, y, metric="new_metric")

    z_transform = optb.transform(x, y, metric="mean", check_input=True)
    assert z_transform[:5] == approx([31.86, 19.425, 22.17959184, 31.86,
                                      31.86], rel=1e-6)

    # z_transform = optb.transform(x, y, metric="indices")
    # assert z_transform[:5] == approx([1, 15, 14, 1, 1])

    # z_transform = optb.transform(x, y, metric="bins")
    # assert z_transform[0] == '[37.25, 76.25) $\\cup$ (-inf, 3.99)'


def test_numerical_default_fit_transform():
    optb = ContinuousOptimalBinning2D()

    z_transform = optb.fit_transform(x, y, z, metric="mean")
    assert z_transform[:5] == approx([31.86, 19.425, 22.17959184, 31.86,
                                      31.86], rel=1e-6)


def test_numerical_categorical_transform():
    optb = ContinuousOptimalBinning2D(
        dtype_x="numerical", dtype_y="categorical")
    optb.fit(x, y, z)
    z_transform = optb.fit_transform(x, y, z, metric="mean")

    assert z_transform[:5] == approx([23.45581395, 31.67272727, 31.50344828,
                                      31.50344828, 31.50344828], rel=1e-6)


def test_categorical_categorical_transform():
    optb = ContinuousOptimalBinning2D(
        dtype_x="categorical", dtype_y="categorical")
    optb.fit(x, y, z)
    z_transform = optb.fit_transform(x, y, z, metric="mean")

    assert z_transform[:5] == approx([21.88918919, 24.04285714, 32.93571429,
                                      29.3, 32.93571429], rel=1e-6)



def test_information():
    optb = ContinuousOptimalBinning2D(solver="cp")

    with raises(NotFittedError):
        optb.information()

    optb.fit(x, y, z)

    with raises(ValueError):
        optb.information(print_level=-1)

    optb.information(print_level=0)
    optb.information(print_level=1)
    optb.information(print_level=2)

    optb = ContinuousOptimalBinning2D(solver="mip")
    optb.fit(x, y, z)
    optb.information(print_level=2)


def test_verbose():
    optb = ContinuousOptimalBinning2D(verbose=True)
    optb.fit(x, y, z)

    assert optb.status == "OPTIMAL"
