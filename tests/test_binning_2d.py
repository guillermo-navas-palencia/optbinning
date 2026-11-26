"""
OptimalBinning2D testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2021

import numpy as np
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

# 120 records with only x missing, 100 with only y missing, 20 with both
idx = np.random.RandomState(0).permutation(len(x))
only_x = np.zeros(len(x), dtype=bool)
only_x[idx[:120]] = True
only_y = np.zeros(len(x), dtype=bool)
only_y[idx[120:220]] = True
both = np.zeros(len(x), dtype=bool)
both[idx[220:240]] = True

x_nan = np.where(only_x | both, np.nan, x)
y_nan = np.where(only_y | both, np.nan, y)


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

    with raises(TypeError):
        optb = OptimalBinning2D(split_missing=1)
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


def test_categorical_binning_table_labels():
    x_cat = pd.array(["a", "b", "c", "d"] * 50, dtype="string[pyarrow]")
    y_num = [float(i) for i in range(200)]
    z_bin = [int(c in ("a", "c")) for c in x_cat]

    optb = OptimalBinning2D(dtype_x="categorical", dtype_y="numerical")
    optb.fit(x_cat, y_num, z_bin)

    table = optb.binning_table.build(show_bin_xy=True)
    assert all(isinstance(label, str) for label in table["Bin"])
    assert not any("ArrowStringArray" in label for label in table["Bin"])


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


def _missing_rows(table):
    # Rows of the bins of the records with only x or only y missing
    bin_x, bin_y = (table[column].map(lambda b: str(b) == "Missing")
                    for column in ("Bin x", "Bin y"))

    return table.index[bin_x & ~bin_y], table.index[bin_y & ~bin_x]


def test_split_missing_default():
    optb = OptimalBinning2D()
    optb.fit(x_nan, y_nan, z)

    table = optb.binning_table.build()
    assert optb.binning_table.iv == approx(4.17362570, rel=1e-6)
    assert table["Count"].iloc[-2] == 240

    rows_x, rows_y = _missing_rows(table)
    assert not len(rows_x) and not len(rows_y)

    z_transform = optb.transform(x_nan, y_nan, metric="woe")
    assert z_transform[:5] == approx([4.92786875, 0, 4.92786875, 0.80883158,
                                      4.92786875], rel=1e-6)
    assert np.all(z_transform[only_x | only_y | both] == 0)


def test_split_missing():
    optb = OptimalBinning2D()
    optb.fit(x_nan, y_nan, z)
    table = optb.binning_table.build()

    optb_split = OptimalBinning2D(split_missing=True)
    optb_split.fit(x_nan, y_nan, z)
    table_split = optb_split.binning_table.build()

    # The 2D bins do not change
    n_bins = len(optb.splits[0])
    assert optb_split.splits == optb.splits
    columns = ["Bin x", "Bin y", "Count", "Non-event", "Event", "WoE"]
    pd.testing.assert_frame_equal(table_split[columns].iloc[:n_bins],
                                  table[columns].iloc[:n_bins])

    # Only x missing: bins on y. Only y missing: bins on x.
    rows_x, rows_y = _missing_rows(table_split)
    assert len(rows_x) == 2 and len(rows_y) == 2
    assert table_split.loc[rows_x, "Count"].sum() == 120
    assert table_split.loc[rows_y, "Count"].sum() == 100
    assert table_split["Count"].iloc[-2] == 20
    assert table_split.loc["Totals", "Count"] == len(x)

    indices = optb_split.transform(x_nan, y_nan, metric="indices")

    for mask, values, rows in ((only_x, y, rows_x), (only_y, x, rows_y)):
        bins = indices[mask]
        assert set(bins) == set(rows)

        # bins are consecutive intervals of the variable that is present
        assert np.all(np.diff(bins[np.argsort(values[mask])]) >= 0)

        for i in rows:
            assert table_split.loc[i, "Count"] == np.count_nonzero(bins == i)
            assert table_split.loc[i, "Event"] == z[mask][bins == i].sum()

    optb_split.binning_table.analysis(print_output=False)
    optb_split.information(print_level=2)


def test_split_missing_transform():
    optb = OptimalBinning2D()
    optb.fit(x_nan, y_nan, z)

    optb_split = OptimalBinning2D(split_missing=True)
    optb_split.fit(x_nan, y_nan, z)

    table = optb_split.binning_table.build()
    labels = optb_split.binning_table.build(show_bin_xy=True)["Bin"].values
    missing = len(table) - 2
    clean = ~(only_x | only_y | both)
    one = only_x | only_y

    indices = optb_split.transform(x_nan, y_nan, metric="indices",
                                   metric_missing="empirical")
    assert np.all(indices[both] == missing)

    for metric in ("woe", "event_rate", "indices", "bins"):
        z_transform = optb_split.transform(x_nan, y_nan, metric=metric)
        z_transform_default = optb.transform(x_nan, y_nan, metric=metric)

        # Records with both values or both missing are not affected
        assert np.all(z_transform[clean] == z_transform_default[clean])
        assert np.all(z_transform[both] == z_transform_default[both])

        # Records with one value missing take the value of their bin
        if metric == "woe":
            assert z_transform[one] == approx(
                table["WoE"].values[indices[one]].astype(float))
        elif metric == "event_rate":
            assert z_transform[one] == approx(
                table["Event rate"].values[indices[one]].astype(float))
        elif metric == "indices":
            assert np.all(z_transform[one] == indices[one])
        else:
            assert np.all(z_transform[one] == labels[indices[one]])
            assert np.all(np.char.startswith(
                z_transform[only_x].astype(str), "Missing $\\cup$ "))

    z_transform = optb_split.transform(x_nan, y_nan, metric="woe",
                                       metric_missing="empirical")
    assert z_transform[both] == approx(table["WoE"][missing])


def test_split_missing_constraints():
    # Bin sizes are counted in records of all the samples, as for the 2D
    # bins. Without constraints, the records with only y missing are split
    # into bins of 71 and 29 records. With min_bin_size=0.08 (46 records),
    # they form a single bin.
    optb = OptimalBinning2D(split_missing=True, min_bin_size=0.08)
    optb.fit(x_nan, y_nan, z)
    assert optb.status == "OPTIMAL"

    table = optb.binning_table.build()
    rows = np.concatenate(_missing_rows(table))
    assert table.loc[rows, "Count"].tolist() == [68, 52, 100]

    # max_bin_size=0.12 (69 records): the 100 records with only y missing
    # cannot be split into bins of at most 69 records, so they remain in the
    # missing bin.
    optb = OptimalBinning2D(split_missing=True, max_bin_size=0.12,
                            max_n_prebins_x=10, max_n_prebins_y=10)
    optb.fit(x_nan, y_nan, z)
    assert optb.status == "OPTIMAL"

    table = optb.binning_table.build()
    rows = np.concatenate(_missing_rows(table))
    assert table.loc[rows, "Count"].tolist() == [68, 52]
    assert table["Count"].iloc[-2] == 120

    # Without constraints, the first bin of the records with only x missing
    # has 4 non-events, and the last bin of the records with only y missing
    # has 1 event.
    optb = OptimalBinning2D(split_missing=True, min_bin_n_event=5,
                            min_bin_n_nonevent=5)
    optb.fit(x_nan, y_nan, z)
    assert optb.status == "OPTIMAL"

    table = optb.binning_table.build()
    rows = np.concatenate(_missing_rows(table))
    assert np.all(table.loc[rows, "Event"] >= 5)
    assert np.all(table.loc[rows, "Non-event"] >= 5)
    assert table.loc[rows, "Count"].sum() == 220

    # No bin can satisfy min_bin_size=0.25 (143 records)
    optb = OptimalBinning2D(split_missing=True, min_bin_size=0.25)
    optb.fit(x_nan, y_nan, z)

    table = optb.binning_table.build()
    rows_x, rows_y = _missing_rows(table)
    assert not len(rows_x) and not len(rows_y)
    assert table["Count"].iloc[-2] == 240


def test_split_missing_small_subset():
    # 10 records with x missing: too few records to split
    rows = np.concatenate([np.where(z == 0)[0][:5], np.where(z == 1)[0][:5]])
    x_small = x.copy()
    x_small[rows] = np.nan

    optb = OptimalBinning2D(split_missing=True)
    optb.fit(x_small, y, z)
    table = optb.binning_table.build()
    rows_x, rows_y = _missing_rows(table)
    assert len(rows_x) == 1 and not len(rows_y)
    assert table.loc[rows_x[0], "Bin y"] == "(-inf, inf)"
    assert table.loc[rows_x[0], "Count"] == 10
    assert table["Count"].iloc[-2] == 0

    # Only events: as the 2D bins, a bin needs events and non-events
    x_small = x.copy()
    x_small[np.where(z == 1)[0][:10]] = np.nan

    optb = OptimalBinning2D(split_missing=True)
    optb.fit(x_small, y, z)
    table = optb.binning_table.build()
    rows_x, rows_y = _missing_rows(table)
    assert not len(rows_x) and not len(rows_y)
    assert table["Count"].iloc[-2] == 10


def test_split_missing_no_missing():
    optb = OptimalBinning2D()
    optb.fit(x, y, z)

    optb_split = OptimalBinning2D(split_missing=True)
    optb_split.fit(x, y, z)

    pd.testing.assert_frame_equal(optb_split.binning_table.build(),
                                  optb.binning_table.build())

    for metric in ("woe", "indices", "bins"):
        assert np.all(optb_split.transform(x, y, metric=metric) ==
                      optb.transform(x, y, metric=metric))


def test_split_missing_special_codes():
    # Records with x missing and y special remain in the special and missing
    # bins as without split_missing.
    x_special = np.where(only_x, np.nan, x)
    y_special = np.where(np.isin(np.arange(len(x)), idx[:5]), -9, y)

    optb = OptimalBinning2D(special_codes_y=[-9], split_missing=True)
    optb.fit(x_special, y_special, z)
    table = optb.binning_table.build()
    rows_x, rows_y = _missing_rows(table)

    assert table.loc[rows_x, "Count"].sum() == 115
    assert table["Count"].iloc[-3] == 5
    assert table["Count"].iloc[-2] == 5

    z_transform = optb.transform(x_special, y_special, metric="bins")
    assert np.all(z_transform[idx[:5]] == "Missing")


def test_split_missing_categorical():
    y_cat = pd.qcut(y, 6, labels=list("abcdef")).astype(str).astype(object)
    missing_x = only_x | both

    optb = OptimalBinning2D(dtype_y="categorical", split_missing=True)
    optb.fit(x_nan, y_cat, z)
    table = optb.binning_table.build()
    rows_x, rows_y = _missing_rows(table)
    assert len(rows_x) > 1 and not len(rows_y)
    assert table.loc[rows_x, "Count"].sum() == np.count_nonzero(missing_x)

    # Each category of the records with x missing is in a single bin
    categories = np.concatenate(table.loc[rows_x, "Bin y"].values)
    assert sorted(categories) == sorted(set(y_cat[missing_x]))

    indices = optb.transform(x_nan, y_cat, metric="indices")
    labels = optb.transform(x_nan, y_cat, metric="bins")
    for i in rows_x:
        mask = missing_x & (indices == i)
        assert set(y_cat[mask]) <= set(table.loc[i, "Bin y"])
        assert np.all(labels[mask] == "Missing $\\cup$ {}".format(
            table.loc[i, "Bin y"]))


def test_split_missing_verbose():
    optb = OptimalBinning2D(split_missing=True, verbose=True)
    optb.fit(x_nan, y_nan, z)

    assert optb.status == "OPTIMAL"
