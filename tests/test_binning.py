"""
OptimalBinning testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pytest import approx, raises, mark

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


def test_split_digits_negative():
    optb = OptimalBinning(split_digits=-1)
    optb.fit(x, y)

    assert optb.status == "OPTIMAL"
    assert np.all(np.mod(optb.splits, 10) == 0)


def test_binning_table_hhi_masked_bins():
    from optbinning.binning.binning_statistics import BinningTable

    table = BinningTable(
        name="test",
        dtype="numerical",
        special_codes=None,
        splits=np.array([1.0, 2.0]),
        n_nonevent=np.array([1, 0, 2, 0, 0]),
        n_event=np.array([2, 0, 3, 0, 0]),
    )

    table.build(add_totals=False)

    assert table._hhi == approx(0.53125)
    assert table._hhi_norm == approx(0.0625)


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


def test_dtype_autodetect():
    # Explicit dtype=None infers the variable type from the data.
    # See GH issue #316.
    x_cat = np.array([
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

    y_cat = np.array([
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])

    # automatic detection is opt-in
    optb = OptimalBinning(dtype=None)
    assert optb.dtype is None

    # numerical data (numpy array of floats) is inferred as "numerical"
    optb = OptimalBinning(dtype=None)
    optb.fit(x, y)
    assert optb._dtype == "numerical"
    assert optb.status == "OPTIMAL"

    # object-dtype string array is inferred as "categorical"
    optb = OptimalBinning(dtype=None, solver="mip", cat_cutoff=0.1)
    optb.fit(x_cat, y_cat)
    assert optb._dtype == "categorical"
    assert optb.status == "OPTIMAL"

    # plain numpy unicode string array (not dtype=object) is also
    # inferred as "categorical"
    x_cat_unicode = np.array(x_cat, dtype=str)
    optb = OptimalBinning(dtype=None, solver="mip", cat_cutoff=0.1)
    optb.fit(x_cat_unicode, y_cat)
    assert optb._dtype == "categorical"

    # pandas category dtype with *numeric* categories is inferred as
    # "categorical", not "numerical"
    x_coded = pd.Series(pd.Categorical(
        np.random.RandomState(0).choice([1, 2, 3], size=len(x_cat))))
    optb = OptimalBinning(dtype=None, solver="mip", cat_cutoff=0.1)
    optb.fit(x_coded, y_cat)
    assert optb._dtype == "categorical"

    # explicit dtype still overrides auto-detection
    optb = OptimalBinning(dtype="numerical")
    optb.fit(x, y)
    assert optb._dtype == "numerical"

    # invalid explicit dtype values are still rejected
    with raises(ValueError):
        optb = OptimalBinning(dtype="nominal")
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


def test_verbose():
    optb = OptimalBinning(verbose=True)
    optb.fit(x, y)

    assert optb.status == "OPTIMAL"


def test_to_json_read_json(tmp_path):
    # A binning object reloaded via read_json must reproduce the same
    # transform as the originally fitted object. See GH issue #387.
    optb = OptimalBinning(name=variable, dtype="numerical")
    optb.fit(x, y)

    path = str(tmp_path / "optb.json")
    optb.to_json(path)

    optb_loaded = OptimalBinning(name=variable, dtype="numerical")
    optb_loaded.read_json(path)

    assert optb_loaded.transform(x) == approx(optb.transform(x), rel=1e-6)
    assert (optb_loaded.transform(x, metric="bins") ==
            optb.transform(x, metric="bins")).all()


def test_to_json_read_json_categorical(tmp_path):
    # categories/cat_others are pandas/numpy array-likes and must be made
    # JSON-serializable before being written out, otherwise to_json raises
    # (e.g. "Object of type ndarray/ArrowStringArray is not JSON
    # serializable"). See GH issue #387.
    rng = np.random.RandomState(0)
    x_cat = rng.choice(np.array(['a', 'b', 'c', 'd', 'e']), size=500)
    y_cat = rng.randint(0, 2, 500)

    optb = OptimalBinning(name="x_cat", dtype="categorical")
    optb.fit(x_cat, y_cat)

    path = str(tmp_path / "optb_cat.json")
    optb.to_json(path)

    optb_loaded = OptimalBinning(name="x_cat", dtype="categorical")
    optb_loaded.read_json(path)

    assert optb_loaded.transform(x_cat) == approx(
        optb.transform(x_cat), rel=1e-6)


@mark.parametrize("groups", [None, [["a", "b"], ["c", "d"]],
                            [["a"], ["b", "c", "d"]]])
@mark.parametrize("as_array", [False, True])
@mark.parametrize("special_codes", [
    None, np.array(["special"]),
    {"flagged": np.array(["special", "unused"])}])
def test_categorical_json_roundtrip(tmp_path, groups, as_array, special_codes):
    # GH #317: export nested arrays and reload without constructor hints.
    values = np.repeat(["a", "b", "c", "d", "other", "special"], 100)
    target = np.concatenate([
        np.r_[np.zeros(100 - events), np.ones(events)]
        for events in [10, 20, 70, 80, 40, 30]])
    values = np.r_[values.astype(object), [np.nan] * 100]
    target = np.r_[target, np.tile([0, 1], 50)]
    user_splits = groups
    if as_array and groups is not None:
        user_splits = np.array(groups, dtype=object)
    fitted = OptimalBinning(
        name="category", dtype="categorical", user_splits=user_splits,
        special_codes=special_codes).fit(values, target)
    exported = fitted.to_dict()
    json.dumps(exported)
    path = str(tmp_path / "category.json")
    fitted.to_json(path)
    restored = OptimalBinning()
    restored.read_json(path)
    sample = np.array(["a", "b", "c", "d", "other", "special", np.nan,
                       "unknown"], dtype=object)
    for metric in ["woe", "event_rate", "indices"]:
        params = dict(metric=metric, metric_special="empirical",
                      metric_missing="empirical")
        actual = restored.transform(sample, **params)
        expected = fitted.transform(sample, **params)
        np.testing.assert_allclose(actual, expected)
    # Compare category membership rather than pandas/NumPy repr strings.
    assert [list(group) for group in restored.splits] == [
        list(group) for group in fitted.splits]
    pd.testing.assert_frame_equal(
        restored.binning_table.build().drop(columns="Bin"),
        fitted.binning_table.build().drop(columns="Bin"))
    assert restored.to_dict() == exported
    assert restored.name == fitted.name
    assert restored.dtype == "categorical"


def test_numerical_json_numpy_values(tmp_path):
    values = np.repeat(np.array([1, 2, 3, -999], dtype=np.int64), 100)
    target = np.concatenate([
        np.r_[np.zeros(100 - events), np.ones(events)]
        for events in [10, 30, 80, 20]])
    fitted = OptimalBinning(special_codes={"flagged": np.array([-999])})
    fitted.fit(values, target)
    path = str(tmp_path / "numerical.json")
    fitted.to_json(path)
    restored = OptimalBinning()
    restored.read_json(path)
    for metric in ["woe", "indices"]:
        params = dict(metric=metric, metric_special="empirical")
        np.testing.assert_allclose(restored.transform(values, **params),
                                   fitted.transform(values, **params))



def test_read_json_1_0_0_fixture():
    # Generated by the unmodified v1.0.0 writer (commit 271c906).
    # Keep this fixture independent of the current serialization code.
    path = Path(__file__).parent / "datasets/json/optimal_binning_1_0_0.json"
    loaded = OptimalBinning()
    loaded.read_json(str(path))
    sample = np.array(["a", "b", "c", "d", "other", "special", np.nan],
                      dtype=object)
    rates = np.array([0.15, 0.15, 0.75, 0.75, 0.4, 0.3, 0.5])
    expected = {
        "indices": [0, 0, 1, 1, 2, 3, 4],
        "event_rate": rates,
        "woe": np.log((1 - rates) / rates * 300 / 400)}
    for metric, values in expected.items():
        np.testing.assert_allclose(loaded.transform(
            sample, metric=metric, metric_special="empirical",
            metric_missing="empirical"), values)
    table = loaded.binning_table.build(add_totals=False)
    np.testing.assert_array_equal(table["Count"], [200, 200, 100, 100, 100])
    np.testing.assert_array_equal(table["Event"], [30, 150, 40, 30, 50])
    assert loaded.dtype == "categorical"
    assert loaded.special_codes == {"flagged": ["special"]}
    assert loaded.user_splits == [["a", "b"], ["c", "d"]]
    assert loaded.cat_unknown is None


def test_json_numpy_special_key(tmp_path):
    values = np.repeat([1., 2., 3., -999.], 100)
    target = np.concatenate([np.r_[np.zeros(100-n), np.ones(n)]
                             for n in [10, 30, 80, 20]])
    fitted = OptimalBinning(special_codes={np.int64(1): [-999]}).fit(
        values, target)
    path = str(tmp_path / "numpy_key.json")
    fitted.to_json(path)
    restored = OptimalBinning()
    restored.read_json(path)
    np.testing.assert_allclose(
        restored.transform(values, metric_special="empirical"),
        fitted.transform(values, metric_special="empirical"))


def test_json_mixed_cat_others(tmp_path):
    values = np.array(["a"] * 100 + ["b"] * 100 + [1] * 10 + ["1"] * 10,
                      dtype=object)
    target = np.concatenate([np.r_[np.zeros(100-n), np.ones(n)]
                             for n in [20, 80]] + [np.tile([0, 1], 10)])
    fitted = OptimalBinning(dtype="categorical", cat_cutoff=0.1).fit(
        values, target)
    path = str(tmp_path / "mixed_others.json")
    fitted.to_json(path)
    restored = OptimalBinning()
    restored.read_json(path)
    for metric in ["indices", "woe", "event_rate"]:
        np.testing.assert_allclose(restored.transform(values, metric=metric),
                                   fitted.transform(values, metric=metric))
    assert set(map(type, restored._cat_others)) == {int, str}


@mark.parametrize("unknown", [999, "unseen"])
def test_json_cat_unknown(tmp_path, unknown):
    values = np.repeat(["a", "b"], 100)
    target = np.r_[np.tile([0, 0, 0, 1], 25), np.tile([0, 1, 1, 1], 25)]
    fitted = OptimalBinning(dtype="categorical", cat_unknown=unknown).fit(
        values, target)
    path = str(tmp_path / "unknown.json")
    fitted.to_json(path)
    restored = OptimalBinning()
    restored.read_json(path)
    assert restored.cat_unknown == unknown
    metrics = (["bins"] if isinstance(unknown, str)
               else ["indices", "woe", "event_rate"])
    # This test covers the unknown replacement, not category label formatting.
    sample = ["new"] if isinstance(unknown, str) else ["a", "new"]
    for metric in metrics:
        np.testing.assert_array_equal(
            restored.transform(sample, metric=metric),
            fitted.transform(sample, metric=metric))
