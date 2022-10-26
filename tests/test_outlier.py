"""
Outlier classes testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2022

import numpy as np
import pandas as pd

from pytest import approx, raises

from optbinning.binning.outlier import ModifiedZScoreDetector
from optbinning.binning.outlier import RangeDetector
from optbinning.binning.outlier import YQuantileDetector
from tests.datasets import load_boston

data = load_boston()
df = pd.DataFrame(data.data, columns=data.feature_names)

variable = "LSTAT"
x = df[variable].values
y = data.target


def test_range_params():
    with raises(ValueError):
        detector = RangeDetector(method="new")
        detector.fit(x)

    with raises(ValueError):
        detector = RangeDetector(interval_length=1.5)
        detector.fit(x)


def test_zscore_params():
    with raises(ValueError):
        detector = ModifiedZScoreDetector(threshold=-1.5)
        detector.fit(x)


def test_yquantile_params():
    with raises(ValueError):
        detector = YQuantileDetector(outlier_detector="new")
        detector.fit(x, y)

    with raises(TypeError):
        detector = YQuantileDetector(outlier_params=[])
        detector.fit(x, y)

    with raises(ValueError):
        detector = YQuantileDetector(n_bins=-1)
        detector.fit(x, y)

    with raises(ValueError):
        detector = YQuantileDetector(
            outlier_detector="range",
            outlier_params={"threshold": 3.7})

        detector.fit(x, y)


def test_range_default():
    detector = RangeDetector(method="ETI")
    detector.fit(x)
    assert np.count_nonzero(detector.get_support()) == 7

    detector = RangeDetector(method="HDI")
    detector.fit(x)
    assert np.count_nonzero(detector.get_support()) == 31


def test_zscore_default():
    detector = ModifiedZScoreDetector()
    detector.fit(x)

    mask = detector.get_support()
    assert np.count_nonzero(mask) == 2

    assert x[mask] == approx([37.97, 36.98])


def test_yquantile_default():
    detector = YQuantileDetector()
    detector.fit(x, y)
    mask = detector.get_support()

    assert x[mask] == approx(
        [7.56, 9.59, 7.26, 11.25, 14.79, 7.44, 9.53, 8.88])

    assert y[mask] == approx([39.8, 33.8, 43.1, 31, 30.7, 50, 50, 50])


def test_yquantile_outlier_params():
    detector = YQuantileDetector(n_bins=10, outlier_detector="range",
                                 outlier_params={'method': 'HDI'})

    detector.fit(x, y)
    assert np.count_nonzero(detector.get_support()) == 39
