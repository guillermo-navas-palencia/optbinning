"""Custom candidate providers preserve the ordinary refinement/solver path."""
import pickle

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.base import BaseEstimator, clone

from optbinning import OptimalBinning, ContinuousOptimalBinning


class FixedProvider(BaseEstimator):
    def __init__(self, cuts=(2.5, 5.5)):
        self.cuts = cuts

    def fit(self, x, y, sample_weight=None):
        self.x_ = x.copy()
        self.y_ = y.copy()
        self.weight_ = sample_weight.copy()
        self.splits_ = self.cuts
        return self


@pytest.mark.parametrize('estimator', [OptimalBinning, ContinuousOptimalBinning])
def test_provider_pipeline_equivalence(estimator):
    x = np.r_[np.repeat(np.arange(9.), 4), np.nan, -999.]
    y = np.tile([0., 1.], 19)
    if estimator is ContinuousOptimalBinning:
        y = y + np.arange(38) / 38
    weight = np.ones(38)
    weight[::3] = 2
    provider = FixedProvider()
    model = estimator(prebinning_estimator=provider, special_codes=[-999.],
                      monotonic_trend=None)
    assert clone(model).prebinning_estimator.cuts == provider.cuts
    model.fit(x, y, sample_weight=weight)
    baseline = estimator(user_splits=[2.5, 5.5], special_codes=[-999.],
                         monotonic_trend=None).fit(x, y, sample_weight=weight)
    assert not hasattr(provider, 'splits_')
    assert_allclose(model.prebinning_estimator_.x_, x[:36])
    assert_allclose(model.prebinning_estimator_.weight_, weight[:36])
    assert_allclose(model.splits, baseline.splits)
    assert_allclose(model.transform(x), baseline.transform(x))
    assert_allclose(pickle.loads(pickle.dumps(model)).transform(x),
                    model.transform(x))
    model.set_params(prebinning_estimator=None).fit(x, y)
    assert not hasattr(model, 'prebinning_estimator_')


@pytest.mark.parametrize('cuts', [[np.nan], [np.inf], [5, 2], [2, 2],
                                 [0], [9], [[2]], list(range(1, 9))])
def test_invalid_cuts(cuts):
    with pytest.raises(ValueError, match='Provider splits_'):
        OptimalBinning(prebinning_estimator=FixedProvider(cuts),
                       max_n_prebins=5).fit(np.repeat(np.arange(10.), 2),
                                           np.tile([0, 1], 10))


@pytest.mark.parametrize('kwargs', [dict(dtype='categorical'),
                                    dict(user_splits=[2]),
                                    dict(class_weight='balanced'),
                                    dict(max_depth=3)])
def test_unsupported_combinations(kwargs):
    with pytest.raises(ValueError, match='requires numerical'):
        OptimalBinning(prebinning_estimator=FixedProvider(), **kwargs).fit(
            np.repeat(np.arange(10.), 2), np.tile([0, 1], 10))


def test_constant_empty_candidates():
    model = OptimalBinning(prebinning_estimator=FixedProvider(())).fit(
        np.ones(20), np.tile([0, 1], 10))
    assert len(model.splits) == 0
