"""
OptimalBinningSketch testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import numpy as np
import pandas as pd

from pytest import approx, raises

from optbinning import OptimalBinningSketch
from optbinning.exceptions import NotSolvedError
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)

variable = "mean radius"
x = df[variable].values
y = data.target


def test_params():
    with raises(TypeError):
        OptimalBinningSketch(name=1)

    with raises(ValueError):
        OptimalBinningSketch(dtype="nominal")

    with raises(ValueError):
        OptimalBinningSketch(sketch="new_sketch")

    with raises(ValueError):
        OptimalBinningSketch(eps=-1e-2)

    with raises(ValueError):
        OptimalBinningSketch(K=-3)

    with raises(ValueError):
        OptimalBinningSketch(solver="new_solver")

    with raises(ValueError):
        OptimalBinningSketch(divergence="new_divergence")

    with raises(ValueError):
        OptimalBinningSketch(max_n_prebins=-2)

    with raises(ValueError):
        OptimalBinningSketch(min_n_bins=-2)

    with raises(ValueError):
        OptimalBinningSketch(max_n_bins=-2.2)

    with raises(ValueError):
        OptimalBinningSketch(min_n_bins=3, max_n_bins=2)

    with raises(ValueError):
        OptimalBinningSketch(min_bin_size=0.6)

    with raises(ValueError):
        OptimalBinningSketch(max_bin_size=-0.6)

    with raises(ValueError):
        OptimalBinningSketch(min_bin_size=0.5, max_bin_size=0.3)

    with raises(ValueError):
        OptimalBinningSketch(min_bin_n_nonevent=-2)

    with raises(ValueError):
        OptimalBinningSketch(max_bin_n_nonevent=-2)

    with raises(ValueError):
        OptimalBinningSketch(min_bin_n_nonevent=3, max_bin_n_nonevent=2)

    with raises(ValueError):
        OptimalBinningSketch(min_bin_n_event=-2)

    with raises(ValueError):
        OptimalBinningSketch(max_bin_n_event=-2)

    with raises(ValueError):
        OptimalBinningSketch(min_bin_n_event=3, max_bin_n_event=2)

    with raises(ValueError):
        OptimalBinningSketch(monotonic_trend="new_trend")

    with raises(ValueError):
        OptimalBinningSketch(min_event_rate_diff=1.1)

    with raises(ValueError):
        OptimalBinningSketch(max_pvalue=1.1)

    with raises(ValueError):
        OptimalBinningSketch(max_pvalue_policy="new_policy")

    with raises(ValueError):
        OptimalBinningSketch(gamma=-0.2)

    with raises(ValueError):
        OptimalBinningSketch(cat_cutoff=-0.2)

    with raises(TypeError):
        OptimalBinningSketch(cat_heuristic=1)

    with raises(TypeError):
        OptimalBinningSketch(special_codes={1, 2, 3})

    with raises(ValueError):
        OptimalBinningSketch(split_digits=9)

    with raises(ValueError):
        OptimalBinningSketch(mip_solver="new_solver")

    with raises(ValueError):
        OptimalBinningSketch(time_limit=-2)

    with raises(TypeError):
        OptimalBinningSketch(verbose=1)


def test_numerical_default():
    optb = OptimalBinningSketch(sketch="gk", eps=1e-4)
    optb.add(x, y)
    optb.solve()

    assert optb.status == "OPTIMAL"

    optb.binning_table.build()
    assert optb.binning_table.iv == approx(5.04392547, rel=1e-2)

    optb.binning_table.analysis()
    assert optb.binning_table.gini == approx(0.87541620, rel=1e-2)
    assert optb.binning_table.js == approx(0.39378376, rel=1e-2)
    assert optb.binning_table.quality_score == approx(0.0, rel=1e-2)


def test_numerical_default_merge():
    optb1 = OptimalBinningSketch(sketch="gk", eps=1e-4)
    optb2 = OptimalBinningSketch(sketch="gk", eps=1e-4)

    x1, x2 = x[:200], x[200:]
    y1, y2 = y[:200], y[200:]

    optb1.add(x1, y1)
    optb2.add(x2, y2)
    optb1.merge(optb2)

    optb1.solve()

    assert optb1.status == "OPTIMAL"

    optb1.binning_table.build()
    assert optb1.binning_table.iv == approx(5.04392547, rel=1e-2)

    optb1.binning_table.analysis()
    assert optb1.binning_table.gini == approx(0.87541620, rel=1e-2)
    assert optb1.binning_table.js == approx(0.39378376, rel=1e-2)
    assert optb1.binning_table.quality_score == approx(0.0, rel=1e-2)


def test_numerical_default_tdigest():
    optb = OptimalBinningSketch(sketch="t-digest", eps=1e-4)
    optb.add(x, y)
    optb.solve()

    assert optb.status == "OPTIMAL"

    optb.binning_table.build()
    assert optb.binning_table.iv == approx(5.04392547, rel=1e-2)

    optb.binning_table.analysis()
    assert optb.binning_table.gini == approx(0.87541620, rel=1e-2)
    assert optb.binning_table.js == approx(0.39378376, rel=1e-2)
    assert optb.binning_table.quality_score == approx(0.0, rel=1e-2)


def test_numerical_default_tdigest_merge():
    optb1 = OptimalBinningSketch(sketch="t-digest", eps=1e-4)
    optb2 = OptimalBinningSketch(sketch="t-digest", eps=1e-4)

    x1, x2 = x[:200], x[200:]
    y1, y2 = y[:200], y[200:]

    optb1.add(x1, y1)
    optb2.add(x2, y2)
    optb1.merge(optb2)

    optb1.solve()

    assert optb1.status == "OPTIMAL"

    optb1.binning_table.build()
    assert optb1.binning_table.iv == approx(5.04392547, rel=1e-2)

    optb1.binning_table.analysis()
    assert optb1.binning_table.gini == approx(0.87541620, rel=1e-2)
    assert optb1.binning_table.js == approx(0.39378376, rel=1e-2)
    assert optb1.binning_table.quality_score == approx(0.0, rel=1e-2)


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

    optb = OptimalBinningSketch(dtype="categorical", solver="mip",
                                cat_cutoff=0.1, verbose=True)
    optb.add(x, y)
    optb.solve()

    assert optb.status == "OPTIMAL"


def test_information():
    optb = OptimalBinningSketch(solver="cp")

    with raises(NotSolvedError):
        optb.information()

    optb.add(x, y)
    optb.solve()

    with raises(ValueError):
        optb.information(print_level=-1)

    optb.information(print_level=0)
    optb.information(print_level=1)
    optb.information(print_level=2)

    optb = OptimalBinningSketch(solver="mip")
    optb.add(x, y)
    optb.solve()
    optb.information(print_level=2)


def test_verbose():
    optb = OptimalBinningSketch(verbose=True)
    optb.add(x, y)
    optb.solve()

    assert optb.status == "OPTIMAL"
