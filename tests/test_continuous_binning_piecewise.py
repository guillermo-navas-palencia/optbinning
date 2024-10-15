"""
ContinuousOptimalPWBinning testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2022

import numpy as np
import pandas as pd

from pytest import approx

from optbinning import ContinuousOptimalPWBinning
from tests.datasets import load_boston

data = load_boston()
df = pd.DataFrame(data.data, columns=data.feature_names)

variable = "LSTAT"
x = df[variable].values
y = data.target


def test_default():
    optb = ContinuousOptimalPWBinning(name=variable)
    optb.fit(x, y)

    optb.binning_table.build()
    optb.binning_table.plot(
        savefig="tests/results/test_continuous_binning_piecewise.png")


def test_transform():
    optb = ContinuousOptimalPWBinning(name=variable)
    optb.fit(x, y)

    x_transform = optb.transform(x)
    assert x_transform[:3] == approx(
        [31.46014643, 23.87619986, 37.31237732], rel=1e-6)


def test_fit_transform():
    optb = ContinuousOptimalPWBinning(name=variable)

    x_transform = optb.fit_transform(x, y)
    assert x_transform[:3] == approx(
        [31.46014643, 23.87619986, 37.31237732], rel=1e-6)


def test_special_codes():
    variable = "INDUS"
    x = df[variable].values

    x[:50] = -9
    x[50:100] = -8
    special_codes = {'special_-9': -9, 'special_-8': -8}

    optb = ContinuousOptimalPWBinning(
        name=variable, monotonic_trend="convex", special_codes=special_codes)
    optb.fit(x, y)

    x_transform = optb.transform([-9, -8], metric_special=1000)
    assert x_transform == approx([1000, 1000], rel=1e-6)

    x_transform = optb.transform([-9, -8], metric_special='empirical')
    assert x_transform == approx([20.502000, 24.116000], rel=1e-6)

    optb = ContinuousOptimalPWBinning(
        name=variable, monotonic_trend="convex", special_codes=[-9, -8])
    optb.fit(x, y)

    x_transform = optb.transform([-9, -8], metric_special=1000)
    assert x_transform == approx([1000, 1000], rel=1e-6)

    x_transform = optb.transform([-9, -8], metric_special='empirical')
    assert x_transform == approx([22.309, 22.309], rel=1e-6)

    x[45:50] = np.nan
    optb = ContinuousOptimalPWBinning(
        name=variable, monotonic_trend="convex", special_codes=special_codes)
    optb.fit(x, y)

    x_transform = optb.transform([np.nan], metric_missing='empirical')
    assert x_transform == approx([17.94], rel=1e-6)


def test_verbose():
    optb = ContinuousOptimalPWBinning(verbose=True)
    optb.fit(x, y)

    assert optb.status == "OPTIMAL"
