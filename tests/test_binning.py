"""
OptimalBinning testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import numpy as np
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
        optb = OptimalBinning(divergence="new_divergence")
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
        optb = OptimalBinning(cat_unknown=list())
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

    optb.binning_table.plot(
        metric="woe", savefig="tests/results/test_binning.png")
    optb.binning_table.plot(
        metric="woe", add_special=False,
        savefig="tests/results/test_binning_no_special.png")
    optb.binning_table.plot(
        metric="woe", add_missing=False,
        savefig="tests/results/test_binning_no_missing.png")


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


def test_numerical_user_splits_non_unique():
    user_splits = [11, 12, 13, 14, 15, 15]
    optb = OptimalBinning(user_splits=user_splits, max_pvalue=0.05)

    with raises(ValueError):
        optb.fit(x, y)


def test_numerical_user_splits_fixed():
    user_splits = [11, 12, 13, 14, 15, 16, 17]

    with raises(ValueError):
        user_splits_fixed = [False, False, False, False, False, True, False]
        optb = OptimalBinning(user_splits_fixed=user_splits_fixed)
        optb.fit(x, y)

    with raises(TypeError):
        user_splits_fixed = (False, False, False, False, False, True, False)
        optb = OptimalBinning(user_splits=user_splits,
                              user_splits_fixed=user_splits_fixed)
        optb.fit(x, y)

    with raises(ValueError):
        user_splits_fixed = [0, 0, 0, 0, 0, 1, 0]
        optb = OptimalBinning(user_splits=user_splits,
                              user_splits_fixed=user_splits_fixed)
        optb.fit(x, y)

    with raises(ValueError):
        user_splits_fixed = [False, False, False, False]
        optb = OptimalBinning(user_splits=user_splits,
                              user_splits_fixed=user_splits_fixed)
        optb.fit(x, y)

    user_splits_fixed = [False, False, False, False, False, True, False]
    optb = OptimalBinning(user_splits=user_splits,
                          user_splits_fixed=user_splits_fixed)
    optb.fit(x, y)

    assert optb.status == "INFEASIBLE"

    user_splits = [11, 12, 13, 14, 15, 17]
    user_splits_fixed = [False, True, False, False, False, False]

    optb_mip = OptimalBinning(user_splits=user_splits,
                              user_splits_fixed=user_splits_fixed,
                              solver="mip")

    optb_cp = OptimalBinning(user_splits=user_splits,
                             user_splits_fixed=user_splits_fixed, solver="cp")

    for optb in (optb_mip, optb_cp):
        optb.fit(x, y)
        assert optb.status == "OPTIMAL"
        assert 12 in optb.splits

    optb2 = OptimalBinning()
    optb2.fit(x, y)

    optb.binning_table.build()
    optb2.binning_table.build()

    assert optb.binning_table.iv <= optb2.binning_table.iv


def test_categorical_default_user_splits():
    x = np.array([
        'Working', 'State servant', 'Working', 'Working', 'Working',
        'State servant', 'Commercial associate', 'State servant',
        'Pensioner', 'Working', 'Working', 'Pensioner', 'Working',
        'Working', 'Working', 'Working', 'Working', 'Working', 'Working',
        'State servant', 'Working', 'Commercial associate', 'Working',
        'Pensioner', 'Working', 'Working', 'Working', 'Working',
        'State servant', 'Working', 'Commercial associate', 'Working',
        'Working', 'Commercial associate', 'State servant', 'Working',
        'Commercial associate', 'Working', 'Pensioner', 'Working',
        'Commercial associate', 'Working', 'Working', 'Pensioner',
        'Working', 'Working', 'Pensioner', 'Working', 'State servant',
        'Working', 'State servant', 'Commercial associate', 'Working',
        'Commercial associate', 'Pensioner', 'Working', 'Pensioner',
        'Working', 'Working', 'Working', 'Commercial associate', 'Working',
        'Pensioner', 'Working', 'Commercial associate',
        'Commercial associate', 'State servant', 'Working',
        'Commercial associate', 'Commercial associate',
        'Commercial associate', 'Working', 'Working', 'Working',
        'Commercial associate', 'Working', 'Commercial associate',
        'Working', 'Working', 'Pensioner', 'Working', 'Pensioner',
        'Working', 'Working', 'Pensioner', 'Working', 'State servant',
        'Working', 'Working', 'Working', 'Working', 'Working',
        'Commercial associate', 'Commercial associate',
        'Commercial associate', 'Working', 'Commercial associate',
        'Working', 'Working', 'Pensioner'], dtype=object)

    y = np.array([
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])

    optb = OptimalBinning(dtype="categorical", solver="mip", cat_cutoff=0.1,
                          verbose=True)
    optb.fit(x, y)

    assert optb.status == "OPTIMAL"

    user_splits = np.array([
        ['Pensioner', 'Working'], ['Commercial associate'], ['State servant']
        ], dtype=object)

    optb = OptimalBinning(dtype="categorical", solver="mip", cat_cutoff=0.1,
                          user_splits=user_splits, verbose=True)
    optb.fit(x, y)

    assert optb.status == "OPTIMAL"


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

    optb1 = OptimalBinning(dtype="categorical", user_splits=user_splits)
    optb2 = OptimalBinning(dtype="categorical", user_splits=user_splits,
                           user_splits_fixed=user_splits_fixed)

    for optb in (optb1, optb2):
        optb.fit(x, y)
        optb.binning_table.build()
        assert optb.binning_table.iv == approx(0.09345086993827473, rel=1e-6)


def test_auto_modes():
    optb0 = OptimalBinning(monotonic_trend="auto")
    optb1 = OptimalBinning(monotonic_trend="auto_heuristic")
    optb2 = OptimalBinning(monotonic_trend="auto_asc_desc")
    optb3 = OptimalBinning(monotonic_trend="descending", verbose=True)

    for optb in [optb0, optb1, optb2, optb3]:
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


# def test_numerical_prebinning_kwargs():
#     optb_kwargs = OptimalBinning(solver="mip", prebinning_method="mdlp",
#                                  **{"max_candidates": 64})

#     optb_kwargs.fit(x, y)
#     optb_kwargs.binning_table.build()
#     assert optb_kwargs.binning_table.iv == approx(4.37337682, rel=1e-6)


def test_min_event_rate_diff():
    min_event_rate_diff = 0.01

    for solver, mip_solver in (('cp', 'bop'), ('mip', 'bop'), ('mip', 'cbc')):
        optb = OptimalBinning(solver=solver, mip_solver=mip_solver,
                              min_event_rate_diff=min_event_rate_diff)
        optb.fit(x, y)

        event_rate = optb.binning_table.build()['Event rate'].values[:-3]
        min_diff = np.absolute(event_rate[1:] - event_rate[:-1])
        assert np.all(min_diff >= min_event_rate_diff)


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
    x = np.array([
        'Working', 'State servant', 'Working', 'Working', 'Working',
        'State servant', 'Commercial associate', 'State servant',
        'Pensioner', 'Working', 'Working', 'Pensioner', 'Working',
        'Working', 'Working', 'Working', 'Working', 'Working', 'Working',
        'State servant', 'Working', 'Commercial associate', 'Working',
        'Pensioner', 'Working', 'Working', 'Working', 'Working',
        'State servant', 'Working', 'Commercial associate', 'Working',
        'Working', 'Commercial associate', 'State servant', 'Working',
        'Commercial associate', 'Working', 'Pensioner', 'Working',
        'Commercial associate', 'Working', 'Working', 'Pensioner',
        'Working', 'Working', 'Pensioner', 'Working', 'State servant',
        'Working', 'State servant', 'Commercial associate', 'Working',
        'Commercial associate', 'Pensioner', 'Working', 'Pensioner',
        'Working', 'Working', 'Working', 'Commercial associate', 'Working',
        'Pensioner', 'Working', 'Commercial associate',
        'Commercial associate', 'State servant', 'Working',
        'Commercial associate', 'Commercial associate',
        'Commercial associate', 'Working', 'Working', 'Working',
        'Commercial associate', 'Working', 'Commercial associate',
        'Working', 'Working', 'Pensioner', 'Working', 'Pensioner',
        'Working', 'Working', 'Pensioner', 'Working', 'State servant',
        'Working', 'Working', 'Working', 'Working', 'Working',
        'Commercial associate', 'Commercial associate',
        'Commercial associate', 'Working', 'Commercial associate',
        'Working', 'Working', 'Pensioner'], dtype=object)

    y = np.array([
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])

    # unknown category metric errors
    for cat_unknown, metric in ((1, "bins"), ("a", "indices"), ("b", "woe")):
        optb = OptimalBinning(dtype="categorical", solver="mip",
                              cat_cutoff=0.1, cat_unknown=cat_unknown)
        optb.fit(x, y)

        if metric == "bins":
            match = ("Invalid value for cat_unknown. cat_unknown must be "
                     "string if metric='bins'.")

        elif metric == "indices":
            match = ("Invalid value for cat_unknown. cat_unknown must be an "
                     "integer if metric='indices'.")

        elif metric in ("woe", "event_rate"):
            match = ("Invalid value for cat_unknown. cat_unknown must be "
                     "numeric if metric='{}'.".format(metric))

        with raises(ValueError, match=match):
            optb.transform(x=x, metric=metric)

    # general case
    optb = OptimalBinning(dtype="categorical", solver="mip", cat_cutoff=0.1)
    optb.fit(x, y)
    x_transform = optb.transform(["Pensioner", "Working",
                                  "Commercial associate", "State servant"])

    assert x_transform == approx([-0.26662866, 0.30873548, -0.55431074,
                                  0.30873548], rel=1e-6)

    # unknown category default case
    for metric, value in (("bins", "unknown"), ("indices", -1), ("woe", 0)):
        optb.fit(x, y)

        assert optb.transform(x=['new'], metric=metric)[0] == value


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


def test_numerical_transform_indices_and_bins():
    # transform()'s numerical branch gathers per-sample values via
    # np.digitize indices instead of a per-bin masking loop (GH issue
    # #388). Cross-check "indices"/"bins" against "woe"/"event_rate" and
    # the binning table itself, and confirm every sample's bin actually
    # contains its value.
    optb = OptimalBinning(name=variable, dtype="numerical")
    optb.fit(x, y)

    splits = optb.splits
    n_bins = len(splits) + 1

    indices = optb.transform(x, metric="indices")
    bins = optb.transform(x, metric="bins")
    woe = optb.transform(x, metric="woe")
    event_rate = optb.transform(x, metric="event_rate")

    assert ((indices >= 0) & (indices < n_bins)).all()

    table = optb.binning_table.build()
    table_woe = table["WoE"].values[:n_bins].astype(float)
    table_event_rate = table["Event rate"].values[:n_bins].astype(float)

    assert woe == approx(table_woe[indices], rel=1e-6)
    assert event_rate == approx(table_event_rate[indices], rel=1e-6)

    bin_edges = np.concatenate([[-np.inf], splits, [np.inf]])
    for xi, bi, idx in zip(x, bins, indices):
        lo, hi = bin_edges[idx], bin_edges[idx + 1]
        assert lo <= xi < hi
        assert bi == table["Bin"].values[idx]


def test_numerical_transform_no_splits():
    # Edge case: a single-bin solution has no splits at all, so
    # np.digitize is skipped and ``indices`` is built as a float array of
    # zeros (see transform_binary_target). The vectorized gather must
    # still handle this (it requires integer indices for fancy indexing).
    optb = OptimalBinning(name=variable, dtype="numerical", max_n_bins=1)
    optb.fit(x, y)

    assert len(optb.splits) == 0

    for metric in ("woe", "event_rate", "indices", "bins"):
        x_transform = optb.transform(x, metric=metric)
        assert len(np.unique(x_transform)) == 1


def test_categorical_transform_indices_and_bins():
    # transform()'s categorical branch resolves each sample's bin via
    # pd.factorize instead of looping over bins and testing membership
    # against the whole array each time (GH issue #388). Cross-check
    # "indices" against "woe"/"event_rate" and the binning table, and
    # exercise the edge cases the rewrite has to handle explicitly: a
    # category never seen during fit, and a missing (NaN) value.
    rng = np.random.RandomState(0)
    x_cat = rng.choice(np.array(['a', 'b', 'c', 'd', 'e']), size=500)
    y_cat = rng.randint(0, 2, 500)

    optb = OptimalBinning(name="x_cat", dtype="categorical")
    optb.fit(x_cat, y_cat)

    n_bins = len(optb.splits) + 1

    x_test = np.concatenate([
        x_cat[:50],
        np.array(['unseen_cat'], dtype=object),
        np.array([np.nan], dtype=object)])

    indices = optb.transform(x_test, metric="indices")
    woe = optb.transform(x_test, metric="woe")
    event_rate = optb.transform(x_test, metric="event_rate")

    table = optb.binning_table.build()
    table_woe = table["WoE"].values[:n_bins].astype(float)
    table_event_rate = table["Event rate"].values[:n_bins].astype(float)

    # The known samples (first 50) fall in a real bin.
    assert ((indices[:50] >= 0) & (indices[:50] < n_bins)).all()
    assert woe[:50] == approx(table_woe[indices[:50]], rel=1e-6)
    assert event_rate[:50] == approx(
        table_event_rate[indices[:50]], rel=1e-6)

    # Unseen category and missing value both fall back to cat_unknown /
    # the missing-value handling rather than crashing or landing in a
    # bin by accident.
    default_woe = optb.transform(['unseen_cat'], metric="woe")[0]
    assert woe[50] == approx(default_woe, rel=1e-6)
    assert not np.isnan(woe[51])  # missing value has its own default


def test_verbose():
    optb = OptimalBinning(verbose=True)
    optb.fit(x, y)

    assert optb.status == "OPTIMAL"
