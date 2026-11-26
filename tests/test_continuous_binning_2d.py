"""
ContinuousOptimalBinning2D testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2022

import numpy as np
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

# 100 records with only x missing, 80 with only y missing, 10 with both
idx = np.random.RandomState(0).permutation(len(x))
only_x = np.zeros(len(x), dtype=bool)
only_x[idx[:100]] = True
only_y = np.zeros(len(x), dtype=bool)
only_y[idx[100:180]] = True
both = np.zeros(len(x), dtype=bool)
both[idx[180:190]] = True

x_nan = np.where(only_x | both, np.nan, x)
y_nan = np.where(only_y | both, np.nan, y)


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

    with raises(TypeError):
        optb = ContinuousOptimalBinning2D(split_missing=1)
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


def _missing_rows(table):
    # Rows of the bins of the records with only x or only y missing
    bin_x, bin_y = (table[column].map(lambda b: str(b) == "Missing")
                    for column in ("Bin x", "Bin y"))

    return table.index[bin_x & ~bin_y], table.index[bin_y & ~bin_x]


def test_split_missing_default():
    optb = ContinuousOptimalBinning2D()
    optb.fit(x_nan, y_nan, z)

    table = optb.binning_table.build()
    assert optb.binning_table.iv == approx(3.33076599, rel=1e-6)
    assert table["Count"].iloc[-2] == 190

    z_transform = optb.transform(x_nan, y_nan, metric="mean")
    assert z_transform[:5] == approx([29.67619048, 0, 28.5, 29.67619048, 0],
                                     rel=1e-6)
    assert np.all(z_transform[only_x | only_y | both] == 0)


def test_split_missing():
    optb = ContinuousOptimalBinning2D()
    optb.fit(x_nan, y_nan, z)
    table = optb.binning_table.build()

    optb_split = ContinuousOptimalBinning2D(split_missing=True)
    optb_split.fit(x_nan, y_nan, z)
    table_split = optb_split.binning_table.build()

    # The 2D bins do not change
    n_bins = len(optb.splits[0])
    assert optb_split.splits == optb.splits
    columns = ["Bin x", "Bin y", "Count", "Sum", "Mean"]
    pd.testing.assert_frame_equal(table_split[columns].iloc[:n_bins],
                                  table[columns].iloc[:n_bins])

    # Only x missing: bins on y. Only y missing: bins on x.
    rows_x, rows_y = _missing_rows(table_split)
    assert len(rows_x) == 3 and len(rows_y) == 2
    assert table_split.loc[rows_x, "Count"].sum() == 100
    assert table_split.loc[rows_y, "Count"].sum() == 80
    assert table_split["Count"].iloc[-2] == 10

    indices = optb_split.transform(x_nan, y_nan, metric="indices")

    for mask, values, rows in ((only_x, y, rows_x), (only_y, x, rows_y)):
        bins = indices[mask]
        assert set(bins) == set(rows)

        # bins are consecutive intervals of the variable that is present
        assert np.all(np.diff(bins[np.argsort(values[mask])]) >= 0)

        for i in rows:
            assert table_split.loc[i, "Count"] == np.count_nonzero(bins == i)
            assert table_split.loc[i, "Sum"] == approx(
                z[mask][bins == i].sum())
            assert table_split.loc[i, "Std"] == approx(
                z[mask][bins == i].std())

    optb_split.binning_table.analysis(print_output=False)
    optb_split.information(print_level=2)


def test_split_missing_transform():
    optb = ContinuousOptimalBinning2D()
    optb.fit(x_nan, y_nan, z)

    optb_split = ContinuousOptimalBinning2D(split_missing=True)
    optb_split.fit(x_nan, y_nan, z)

    table = optb_split.binning_table.build()
    labels = optb_split.binning_table.build(show_bin_xy=True)["Bin"].values
    clean = ~(only_x | only_y | both)
    one = only_x | only_y

    indices = optb_split.transform(x_nan, y_nan, metric="indices")

    for metric in ("mean", "indices", "bins"):
        z_transform = optb_split.transform(x_nan, y_nan, metric=metric)
        z_transform_default = optb.transform(x_nan, y_nan, metric=metric)

        assert np.all(z_transform[clean] == z_transform_default[clean])
        assert np.all(z_transform[both] == z_transform_default[both])

        if metric == "mean":
            assert z_transform[one] == approx(
                table["Mean"].values[indices[one]].astype(float))
        elif metric == "bins":
            assert np.all(z_transform[one] == labels[indices[one]])


def test_split_missing_constraints():
    # Bin sizes are counted in records of all the samples, as for the 2D
    # bins. Without constraints, the records with only x missing are split
    # into bins of 35, 27 and 38 records, and those with only y missing into
    # bins of 36 and 44 records. With min_bin_size=0.08 (41 records), each
    # group forms a single bin.
    optb = ContinuousOptimalBinning2D(split_missing=True, min_bin_size=0.08)
    optb.fit(x_nan, y_nan, z)
    assert optb.status == "OPTIMAL"

    table = optb.binning_table.build()
    rows = np.concatenate(_missing_rows(table))
    assert table.loc[rows, "Count"].tolist() == [100, 80]

    # max_bin_size=0.15 (76 records) with prebins of at least 41 records:
    # the 80 records with only y missing cannot be split, so they remain in
    # the missing bin.
    optb = ContinuousOptimalBinning2D(split_missing=True, max_bin_size=0.15,
                                      min_prebin_size_x=0.08,
                                      min_prebin_size_y=0.08)
    optb.fit(x_nan, y_nan, z)
    assert optb.status == "OPTIMAL"

    table = optb.binning_table.build()
    rows = np.concatenate(_missing_rows(table))
    assert table.loc[rows, "Count"].tolist() == [42, 58]
    assert table["Count"].iloc[-2] == 90

    # No bin can satisfy min_bin_size=0.25 (127 records)
    optb = ContinuousOptimalBinning2D(split_missing=True, min_bin_size=0.25)
    optb.fit(x_nan, y_nan, z)

    table = optb.binning_table.build()
    rows_x, rows_y = _missing_rows(table)
    assert not len(rows_x) and not len(rows_y)
    assert table["Count"].iloc[-2] == 190


def test_split_missing_small_subset():
    x_small = x.copy()
    x_small[idx[:10]] = np.nan

    optb = ContinuousOptimalBinning2D(split_missing=True)
    optb.fit(x_small, y, z)

    table = optb.binning_table.build()
    rows_x, rows_y = _missing_rows(table)
    assert len(rows_x) == 1 and not len(rows_y)
    assert table.loc[rows_x[0], "Bin y"] == "(-inf, inf)"
    assert table.loc[rows_x[0], "Count"] == 10
    assert table["Count"].iloc[-2] == 0


def test_split_missing_no_missing():
    optb = ContinuousOptimalBinning2D()
    optb.fit(x, y, z)

    optb_split = ContinuousOptimalBinning2D(split_missing=True)
    optb_split.fit(x, y, z)

    pd.testing.assert_frame_equal(optb_split.binning_table.build(),
                                  optb.binning_table.build())

    for metric in ("mean", "indices", "bins"):
        assert np.all(optb_split.transform(x, y, metric=metric) ==
                      optb.transform(x, y, metric=metric))
