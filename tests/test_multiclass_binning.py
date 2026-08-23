"""
MulticlassOptimalBinning testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

from unittest import mock

import numpy as np
import pandas as pd

from pytest import approx, raises

from optbinning import MulticlassOptimalBinning
from sklearn.datasets import load_wine
from sklearn.exceptions import NotFittedError


data = load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)

variable = "ash"
x = df[variable].values
y = data.target


def test_params():
    with raises(TypeError):
        optb = MulticlassOptimalBinning(name=1)
        optb.fit(x, y)

    with raises(ValueError):
        optb = MulticlassOptimalBinning(prebinning_method="new_method")
        optb.fit(x, y)

    with raises(ValueError):
        optb = MulticlassOptimalBinning(solver="new_solver")
        optb.fit(x, y)

    with raises(ValueError):
        optb = MulticlassOptimalBinning(max_n_prebins=-2)
        optb.fit(x, y)

    with raises(ValueError):
        optb = MulticlassOptimalBinning(min_prebin_size=0.6)
        optb.fit(x, y)

    with raises(ValueError):
        optb = MulticlassOptimalBinning(min_n_bins=-2)
        optb.fit(x, y)

    with raises(ValueError):
        optb = MulticlassOptimalBinning(max_n_bins=-2.2)
        optb.fit(x, y)

    with raises(ValueError):
        optb = MulticlassOptimalBinning(min_n_bins=3, max_n_bins=2)
        optb.fit(x, y)

    with raises(ValueError):
        optb = MulticlassOptimalBinning(min_bin_size=0.6)
        optb.fit(x, y)

    with raises(ValueError):
        optb = MulticlassOptimalBinning(max_bin_size=-0.6)
        optb.fit(x, y)

    with raises(ValueError):
        optb = MulticlassOptimalBinning(min_bin_size=0.5, max_bin_size=0.3)
        optb.fit(x, y)

    with raises(ValueError):
        optb = MulticlassOptimalBinning(monotonic_trend=["new_trend", "auto"])
        optb.fit(x, y)

    with raises(ValueError):
        optb = MulticlassOptimalBinning(monotonic_trend="new_trend")
        optb.fit(x, y)

    with raises(ValueError):
        optb = MulticlassOptimalBinning(max_pvalue=1.1)
        optb.fit(x, y)

    with raises(ValueError):
        optb = MulticlassOptimalBinning(max_pvalue_policy="new_policy")
        optb.fit(x, y)

    with raises(TypeError):
        optb = MulticlassOptimalBinning(user_splits={"a": [1, 2]})
        optb.fit(x, y)

    with raises(TypeError):
        optb = MulticlassOptimalBinning(special_codes={1, 2, 3})
        optb.fit(x, y)

    with raises(ValueError):
        optb = MulticlassOptimalBinning(split_digits=9)
        optb.fit(x, y)

    with raises(ValueError):
        optb = MulticlassOptimalBinning(mip_solver="new_solver")
        optb.fit(x, y)

    with raises(ValueError):
        optb = MulticlassOptimalBinning(time_limit=-2)
        optb.fit(x, y)

    with raises(TypeError):
        optb = MulticlassOptimalBinning(verbose=1)
        optb.fit(x, y)


def test_numerical_default():
    optb = MulticlassOptimalBinning()
    optb.fit(x, y)

    assert optb.status == "OPTIMAL"
    assert optb.splits == approx([2.1450001, 2.245, 2.31499994, 2.6049999,
                                  2.6450001], rel=1e-6)

    optb.binning_table.build()
    optb.binning_table.analysis()
    assert optb.binning_table.js == approx(0.10989515, rel=1e-6)
    assert optb.binning_table.quality_score == approx(0.05279822, rel=1e-6)
    optb.binning_table.plot(
        savefig="tests/results/test_multiclass_binning.png")
    optb.binning_table.plot(
        add_special=False,
        savefig="tests/results/test_multiclass_binning_no_special.png")
    optb.binning_table.plot(
        add_missing=False,
        savefig="tests/results/test_multiclass_binning_no_missing.png")


def test_numerical_default_solvers():
    optb_mip_bop = MulticlassOptimalBinning(solver="mip", mip_solver="bop")
    optb_mip_bop.fit(x, y)

    optb_cp = MulticlassOptimalBinning(solver="cp")
    optb_cp.fit(x, y)

    for optb in [optb_mip_bop, optb_cp]:
        assert optb.status == "OPTIMAL"
        assert optb.splits == approx([2.1450001, 2.245, 2.31499994, 2.6049999,
                                      2.6450001], rel=1e-6)


def test_numerical_user_splits_fixed():
    user_splits = [2.1, 2.2, 2.3, 2.6, 2.9]

    with raises(ValueError):
        user_splits_fixed = [False, False, False, True, False]
        optb = MulticlassOptimalBinning(user_splits_fixed=user_splits_fixed)
        optb.fit(x, y)

    with raises(TypeError):
        user_splits_fixed = (False, False, False, True, False)
        optb = MulticlassOptimalBinning(user_splits=user_splits,
                                        user_splits_fixed=user_splits_fixed)
        optb.fit(x, y)

    with raises(ValueError):
        user_splits_fixed = [0, 0, 0, 1, 0]
        optb = MulticlassOptimalBinning(user_splits=user_splits,
                                        user_splits_fixed=user_splits_fixed)
        optb.fit(x, y)

    with raises(ValueError):
        user_splits_fixed = [False, False, False, False]
        optb = MulticlassOptimalBinning(user_splits=user_splits,
                                        user_splits_fixed=user_splits_fixed)
        optb.fit(x, y)

    user_splits_fixed = [False, False, False, True, True]

    with raises(ValueError):
        # pure pre-bins
        optb = MulticlassOptimalBinning(user_splits=user_splits,
                                        user_splits_fixed=user_splits_fixed)
        optb.fit(x, y)

    user_splits = [2.1, 2.2, 2.3, 2.6, 2.7]
    optb = MulticlassOptimalBinning(user_splits=user_splits,
                                    user_splits_fixed=user_splits_fixed)
    optb.fit(x, y)

    assert optb.status == "OPTIMAL"
    assert 2.7 in optb.splits


def test_numerical_user_splits_non_unique():
    user_splits = [2.1, 2.2, 2.2, 2.6, 2.9]
    optb = MulticlassOptimalBinning(user_splits=user_splits)

    with raises(ValueError):
        optb.fit(x, y)


def test_numerical_default_transform():
    optb = MulticlassOptimalBinning()
    with raises(NotFittedError):
        x_transform = optb.transform(x)

    optb.fit(x, y)

    x_transform = optb.transform([0.3, 2.1, 2.5, 3], metric="mean_woe")
    assert x_transform == approx([0.48973998, 0.48973998, -0.00074357,
                                  0.02189459], rel=1e-5)


def test_numerical_default_fit_transform():
    optb = MulticlassOptimalBinning()

    x_transform = optb.fit_transform(x, y, metric="mean_woe")
    assert x_transform[:5] == approx([-0.00074357, 0.48973998, 0.02189459,
                                      -0.00074357, 0.02189459], rel=1e-5)


def test_classes():
    optb = MulticlassOptimalBinning()
    optb.fit(x, y)

    assert optb.classes == approx([0, 1, 2])


def test_verbose():
    optb = MulticlassOptimalBinning(verbose=True)
    optb.fit(x, y)

    assert optb.status == "OPTIMAL"


def _plot_missing_marker_values(table, **plot_kwargs):
    # plot() shows/saves its figure internally, so intercept plt.savefig
    # to read the markers off the live Axes first. The "Bin missing"
    # markers (one per class) are always the last single-point lines.
    import matplotlib.pyplot as plt

    captured = {}

    def fake_savefig(*args, **kwargs):
        ax2 = plt.gcf().axes[-1]
        single_point_lines = [
            line for line in ax2.get_lines() if len(line.get_xdata()) == 1]
        n_classes = len(table.classes)
        captured["y"] = [
            line.get_ydata()[0] for line in single_point_lines[-n_classes:]]

    with mock.patch("matplotlib.pyplot.savefig", side_effect=fake_savefig):
        table.plot(savefig="tests/results/_tmp_plot.png", **plot_kwargs)

    return captured["y"]


def test_plot_missing_metric_matches_regardless_of_add_special():
    # See GH issue #376 / PR #377: pos_missing is a display position that
    # shifts with add_special=False, but was used to index metric_values
    # (event rate per class), whose entries stay at fixed trailing
    # positions. The missing bin's plotted per-class values must equal
    # metric_values[-1] regardless of add_special.
    x_local = x.copy()
    y_local = y.copy()

    # Special group: forced to class 0 only.
    x_local[:20] = -999
    y_local[:20] = 0

    # Missing group: forced to class 2 only, clearly different from the
    # special group above.
    x_local[20:40] = np.nan
    y_local[20:40] = 2

    optb = MulticlassOptimalBinning(name=variable, special_codes=[-999])
    optb.fit(x_local, y_local)

    table = optb.binning_table
    table.build()

    y_with_special = _plot_missing_marker_values(
        table, add_special=True, add_missing=True)
    y_without_special = _plot_missing_marker_values(
        table, add_special=False, add_missing=True)

    assert y_with_special == approx(list(table._event_rate[-1]), rel=1e-6)
    assert y_without_special == approx(
        list(table._event_rate[-1]), rel=1e-6)
