"""
MulticlassOptimalBinning testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

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


def test_transform_indices_and_bins():
    # transform() gathers per-sample values via np.digitize indices
    # instead of a per-bin masking loop (GH issue #388). Confirm
    # "indices"/"bins" are valid and every sample's bin actually
    # contains its value.
    optb = MulticlassOptimalBinning(name=variable)
    optb.fit(x, y)

    splits = optb.splits
    n_bins = len(splits) + 1

    indices = optb.transform(x, metric="indices")
    bins = optb.transform(x, metric="bins")

    assert ((indices >= 0) & (indices < n_bins)).all()

    bin_edges = np.concatenate([[-np.inf], splits, [np.inf]])
    for xi, bi, idx in zip(x, bins, indices):
        lo, hi = bin_edges[idx], bin_edges[idx + 1]
        assert lo <= xi < hi


def test_transform_no_splits():
    # Edge case: a single-bin solution has no splits at all, so
    # np.digitize is skipped and ``indices`` is built as a float array of
    # zeros (see transform_multiclass_target). The vectorized gather must
    # still handle this (it requires integer indices for fancy indexing).
    optb = MulticlassOptimalBinning(name=variable, max_n_bins=1)
    optb.fit(x, y)

    assert len(optb.splits) == 0

    for metric in ("mean_woe", "indices", "bins"):
        x_transform = optb.transform(x, metric=metric)
        assert len(np.unique(x_transform)) == 1


def test_verbose():
    optb = MulticlassOptimalBinning(verbose=True)
    optb.fit(x, y)

    assert optb.status == "OPTIMAL"
