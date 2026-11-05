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


class MissingSplitsProvider(BaseEstimator):
    def fit(self, x, y, sample_weight=None):
        return self


class NonCloneableProvider:
    def fit(self, x, y, sample_weight=None):
        raise AssertionError("A non-cloneable provider must not be fitted.")


@pytest.mark.parametrize('estimator', [OptimalBinning, ContinuousOptimalBinning])
@pytest.mark.parametrize('provider,error,match', [
    (MissingSplitsProvider(), ValueError, 'must expose splits_'),
    (NonCloneableProvider(), TypeError, 'clone'),
    (object(), TypeError, 'must implement fit'),
])
def test_invalid_provider_contract(estimator, provider, error, match):
    with pytest.raises(error, match=match):
        estimator(prebinning_estimator=provider).fit(
            np.repeat(np.arange(10.), 2), np.tile([0, 1], 10))


@pytest.mark.parametrize('estimator', [OptimalBinning, ContinuousOptimalBinning])
def test_nested_parameters_and_information(estimator, capsys):
    model = estimator(prebinning_estimator=FixedProvider(),
                      monotonic_trend=None)
    model.set_params(prebinning_estimator__cuts=(1.5, 6.5))
    assert model.get_params()['prebinning_estimator__cuts'] == (1.5, 6.5)
    copied = clone(model)
    assert copied.prebinning_estimator is not model.prebinning_estimator
    copied.fit(np.repeat(np.arange(9.), 4), np.tile([0, 1], 18))
    assert_allclose(copied.prebinning_estimator_.splits_, [1.5, 6.5])
    copied.information(print_level=2)
    assert 'prebinning_estimator' in capsys.readouterr().out


@pytest.mark.parametrize('target', ['binary', 'continuous'])
@pytest.mark.parametrize('n_jobs', [1, 2])
def test_binning_process_clones_provider_per_variable(target, n_jobs):
    from optbinning import BinningProcess

    x = np.repeat(np.arange(9.), 4)
    X = np.column_stack([x, x[::-1]])
    y = np.tile([0, 1], 18)
    if target == 'continuous':
        y = y + np.arange(len(y)) / len(y)
    weights = np.tile([1., 2.], 18)
    provider = FixedProvider()
    params = {name: dict(prebinning_estimator=provider, monotonic_trend=None)
              for name in ['a', 'b']}
    process = BinningProcess(['a', 'b'], target_dtype=target, n_jobs=n_jobs,
                             binning_fit_params=params).fit(
                                 X, y, sample_weight=weights)
    fitted = [process.get_binned_variable(name) for name in ['a', 'b']]
    assert not hasattr(provider, 'splits_')
    assert fitted[0].prebinning_estimator_ is not fitted[1].prebinning_estimator_
    for i, model in enumerate(fitted):
        assert_allclose(model.prebinning_estimator_.x_, X[:, i])
        assert_allclose(model.prebinning_estimator_.weight_, weights)
        assert_allclose(process.transform(X)[:, i], model.transform(X[:, i]))


@pytest.mark.parametrize('estimator', [OptimalBinning, ContinuousOptimalBinning])
def test_custom_prebin_support_is_provider_policy(estimator):
    # The 10-row candidate survives even though built-in min_prebin_size is 20%.
    # The final 20% support constraint still forces it to merge.
    x = np.repeat([0., 1., 2.], [10, 40, 50])
    y = np.r_[np.tile([0., 1.], 5), np.tile([0., 1.], 20),
              np.tile([0., 1., 1., 1., 1.], 10)]
    model = estimator(prebinning_estimator=FixedProvider((.5, 1.5)),
                      monotonic_trend=None, min_prebin_size=.2,
                      min_bin_size=.2, min_n_bins=2, max_n_bins=2).fit(x, y)
    assert model.status == 'OPTIMAL'
    assert_allclose(model.prebinning_estimator_.splits_, [.5, 1.5])
    assert_allclose(model.splits, [1.5])
    assert_allclose(model.binning_table.build().loc[:1, 'Count'], [50, 50])


@pytest.mark.parametrize('estimator', [OptimalBinning, ContinuousOptimalBinning])
@pytest.mark.parametrize('scale', [.3, 1.7])
def test_fractional_weights_reach_provider_and_match_manual_cuts(estimator,
                                                               scale):
    x = np.r_[np.repeat(np.arange(9.), 4), np.nan, -999.]
    y = np.tile([0., 1.], 19)
    if estimator is ContinuousOptimalBinning:
        y = y + np.arange(len(y)) / len(y)
    weights = np.linspace(.5, 2., len(y)) * scale
    kwargs = dict(monotonic_trend=None, special_codes=[-999.])
    model = estimator(prebinning_estimator=FixedProvider(), **kwargs).fit(
        x, y, sample_weight=weights)
    manual = estimator(user_splits=[2.5, 5.5], **kwargs).fit(
        x, y, sample_weight=weights)
    assert_allclose(model.prebinning_estimator_.weight_, weights[:36])
    assert model.status == manual.status
    assert_allclose(model.splits, manual.splits)
    assert_allclose(model.transform(x), manual.transform(x))
