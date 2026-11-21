import numpy as np
import pytest

from optbinning import OptimalBinning


@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('solver', ['cp', 'mip'])
def test_existing_maximum_rounding_is_preserved(weighted, solver):
    # ceil(.551 * 100) permits 56 rows.
    x = np.r_[np.zeros(56), np.ones(44)]
    y = np.tile([0, 1], 50)
    weights = np.full(100, .3) if weighted else None
    model = OptimalBinning(solver=solver, user_splits=[.5],
                           user_splits_fixed=[True], max_bin_size=.551,
                           monotonic_trend=None).fit(x, y, sample_weight=weights)
    assert model.status == 'OPTIMAL'
    assert model.binning_table.build().loc[0, 'Count'] == 56


@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('kwargs', [dict(max_bin_size=.5), dict(min_n_bins=2),
                                   dict(min_bin_n_event=6)])
def test_existing_single_bin_shortcut_is_preserved(weighted, kwargs):
    weights = np.full(10, .3) if weighted else None
    model = OptimalBinning(**kwargs).fit(
        np.zeros(10), np.tile([0, 1], 5), sample_weight=weights)
    assert model.status == 'OPTIMAL'
    assert model.splits.size == 0
    assert model.binning_table.build().loc['Totals', 'Count'] == 10


@pytest.mark.parametrize('raw_support', [False, True])
@pytest.mark.parametrize('override', [1, .01, .2])
def test_cart_kwargs_override_defaults(raw_support, override):
    from sklearn.tree import DecisionTreeClassifier
    from optbinning.binning.prebinning import PreBinning

    x = np.arange(100.)
    y = np.r_[np.zeros(10), np.ones(10), np.tile([0, 1], 40)]
    weights = np.where(x < 20, 3., 1.) if raw_support else np.ones(100)
    kwargs = {'max_leaf_nodes': 6, 'random_state': 0}
    if raw_support:
        kwargs['min_samples_leaf'] = 25
    else:
        kwargs['min_weight_fraction_leaf'] = min(.5, 25 / weights.sum())
    kwargs['min_samples_leaf'] = override
    expected = DecisionTreeClassifier(**kwargs).fit(
        x[:, None], y, sample_weight=weights)
    actual = PreBinning('classification', 'cart', 6, 25,
                        raw_support=raw_support, min_samples_leaf=override,
                        random_state=0).fit(x, y, weights)
    cuts = np.unique(expected.tree_.threshold)
    np.testing.assert_array_equal(actual.splits, cuts[cuts != -2])
