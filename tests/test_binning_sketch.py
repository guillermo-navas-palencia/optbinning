"""
OptimalBinningSketch testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import numpy as np
import pandas as pd

from pytest import approx, raises

from optbinning import OptimalBinningSketch
from sklearn.datasets import load_breast_cancer
from sklearn.exceptions import NotFittedError

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
