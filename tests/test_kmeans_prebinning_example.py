"""Exercise the standard k-means adapter used in the documentation."""
import pickle

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.base import clone

from examples.kmeans_prebinning import KMeansPrebins
from optbinning import OptimalBinning, ContinuousOptimalBinning


def test_kmeans_adapter_weights_and_interior_edges():
    x = np.repeat([0., 1., 10., 11.], 2)
    provider = KMeansPrebins(n_bins=2).fit(x)
    assert_allclose(provider.splits_, [5.5])
    # Weighted centers are .75 and 10.5, so their midpoint is 5.625.
    weights = np.repeat([.1, .3, .1, .1], 2)
    provider.fit(x, sample_weight=weights)
    assert_allclose(provider.splits_, [5.625])
    assert_allclose(provider.discretizer_.bin_edges_[0], [0., 5.625, 11.])


def test_kmeans_adapter_constant_feature():
    with pytest.warns(UserWarning, match='constant'):
        provider = KMeansPrebins(n_bins=2).fit(np.ones(20))
    assert len(provider.splits_) == 0


@pytest.mark.parametrize('estimator', [OptimalBinning, ContinuousOptimalBinning])
def test_kmeans_example_through_binning(estimator):
    x = np.repeat(np.arange(20.), 4)
    y = np.tile([0., 0., 1., 1.], 20)
    if estimator is ContinuousOptimalBinning:
        y = y + x / 20
    provider = KMeansPrebins(n_bins=5)
    model = estimator(prebinning_estimator=provider, max_n_prebins=5,
                      min_bin_size=.1, monotonic_trend=None).fit(x, y)
    assert model.status == 'OPTIMAL'
    assert not hasattr(provider, 'discretizer_')
    assert np.all(np.isin(model.splits, model.prebinning_estimator_.splits_))
    assert np.isfinite(model.transform(x)).all()
    assert clone(model).prebinning_estimator.n_bins == 5
    assert_allclose(pickle.loads(pickle.dumps(model)).transform(x),
                    model.transform(x))
