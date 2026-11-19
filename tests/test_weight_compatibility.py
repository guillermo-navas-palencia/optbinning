"""Compatibility contracts for binary weight handling."""
import numpy as np
import pytest

from optbinning import OptimalBinning
from optbinning.binning.binning_statistics import bin_info


@pytest.mark.parametrize('weight,dtype', [(None, np.int64), (2, np.int64),
                                         (.25, np.float64)])
@pytest.mark.parametrize('solver', ['cp', 'mip'])
def test_count_dtypes_and_json(weight, dtype, solver, tmp_path):
    x = np.r_[np.repeat([0., 1.], 20), -99, -99, np.nan, np.nan]
    y = np.r_[np.tile([0, 1], 20), 0, 1, 0, 1]
    weights = None if weight is None else np.full(len(x), weight)
    model = OptimalBinning(solver=solver, user_splits=[.5],
                           user_splits_fixed=[True], special_codes=[-99],
                           monotonic_trend=None).fit(x, y, sample_weight=weights)
    assert model.binning_table.n_event.dtype == dtype
    assert model.binning_table.n_nonevent.dtype == dtype
    table = model.binning_table.build()
    for column in ['Count', 'Non-event', 'Event']:
        assert table[column].dtype == np.int64
    if weight is not None:
        assert table['Weighted count'].dtype == dtype
    path = str(tmp_path / 'counts.json')
    model.to_json(path)
    # read_json mutates the estimator; it does not return it.
    restored = OptimalBinning()
    restored.read_json(path)
    assert restored.binning_table.n_event.dtype == dtype
    np.testing.assert_array_equal(restored.binning_table.n_event,
                                  model.binning_table.n_event)


def test_fractional_special_mass_is_not_truncated():
    model = OptimalBinning(user_splits=[.5], special_codes=[-99],
                           monotonic_trend=None).fit(
        [0, 0, 1, 1, -99, -99], [0, 1, 0, 1, 0, 1],
        sample_weight=[1, 1, 1, 1, .2, .3])
    assert model.binning_table.n_event.dtype == np.float64
    assert model.binning_table.n_event[-2] == pytest.approx(.3)


@pytest.mark.parametrize('group', ['missing', 'special', 'others'])
def test_fractional_mass_only_outside_regular_bins(group):
    if group == 'others':
        model = OptimalBinning(dtype='categorical',
                               user_splits=[['a'], ['b']],
                               monotonic_trend=None)
        x = ['a', 'a', 'b', 'b', 'other', 'other']
        index = -3
    else:
        model = OptimalBinning(user_splits=[.5], special_codes={'code': [-99]},
                               monotonic_trend=None)
        x = [0, 0, 1, 1] + ([np.nan] * 2 if group == 'missing' else [-99] * 2)
        index = -1 if group == 'missing' else -2
    model.fit(x, [0, 1, 0, 1, 0, 1], sample_weight=[1, 1, 1, 1, .2, .3])
    assert model.binning_table.n_event[index] == pytest.approx(.3)
    assert model.binning_table.n_nonevent[index] == pytest.approx(.2)
    assert model.binning_table.n_event.dtype == np.float64


@pytest.mark.parametrize('weights', [np.ones(4), np.full(4, 2.),
                                    np.full(4, .25)])
def test_weighted_target_sums_keep_original_behavior(weights):
    from optbinning.binning.binning_statistics import target_info_samples

    ne, ev = target_info_samples(np.array([0, 1, 0, 1]), weights)
    expected = np.sum(weights[:2])
    assert ne.dtype == ev.dtype == expected.dtype
    assert ne == ev == expected


@pytest.mark.parametrize('fractional', [False, True])
def test_bin_info_explicit_fractional_flag(fractional):
    ne, ev = bin_info([True], np.array([2.25]), np.array([3.5]),
                      0., 0., 0., 0., 0., 0., [],
                      fractional_weights=fractional)
    assert ne.dtype == ev.dtype == (np.float64 if fractional else np.int64)
    assert ne[0] == (2.25 if fractional else 2)
    assert ev[0] == (3.5 if fractional else 3)


def test_shared_bin_info_retains_integer_default():
    # Scenario and sketch estimators share this helper but do not opt in to
    # fractional binary weights.
    ne, ev = bin_info([True], np.array([2.]), np.array([3.]),
                      0., 0., 0., 0., 0., 0., [])
    assert ne.dtype == ev.dtype == np.int64


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


def test_split_data_composes_and_returns_full_weights_once():
    from optbinning.binning.preprocessing import split_data

    x = np.array([0., 1., -99., np.nan])
    y = np.array([0, 1, 0, 1])
    weights = np.array([.2, .4, .6, .8])
    result = split_data('numerical', x, y, special_codes=[-99],
                        class_weight={0: 2, 1: 3}, sample_weight=weights)
    assert len(result) == 14
    np.testing.assert_allclose(result[-1], [.4, 1.2, 1.2, 2.4])
    np.testing.assert_allclose(result[9], [.4, 1.2])
    np.testing.assert_allclose(result[10], [2.4])
    np.testing.assert_allclose(result[11], [1.2])
    unweighted = split_data('numerical', x, y, special_codes=[-99])
    assert len(unweighted) == 14
    assert unweighted[-1] is None
    class_only = split_data('numerical', x, y, special_codes=[-99],
                            class_weight={0: 2, 1: 3})
    np.testing.assert_array_equal(class_only[-1], [2, 3, 2, 3])


@pytest.mark.parametrize('weight', [.3, np.array(.3), np.full(4, .3)])
def test_binary_scalar_and_vector_weights(weight):
    model = OptimalBinning(user_splits=[.5], monotonic_trend=None).fit(
        [0, 0, 1, 1], [0, 1, 0, 1], sample_weight=weight)
    assert model.binning_table.n_event.sum() == pytest.approx(.6)
    assert model.binning_table.n_nonevent.sum() == pytest.approx(.6)


@pytest.mark.parametrize('weight', [np.nan, np.inf, -1., 0.])
def test_binary_invalid_scalar_weight(weight):
    with pytest.raises(ValueError):
        OptimalBinning().fit([0, 1], [0, 1], sample_weight=weight)


@pytest.mark.parametrize('class_weight', [{0: 0., 1: 0.}, {0: np.inf, 1: 1.},
                                         {0: 1e308, 1: 1e308}])
def test_binary_rejects_invalid_composed_mass(class_weight):
    with np.errstate(over='ignore'):
        with pytest.raises(ValueError, match='Effective sample weights'):
            OptimalBinning(class_weight=class_weight).fit(
                [0, 1], [0, 1], sample_weight=[2., 2.])


@pytest.mark.parametrize('x_dtype', [np.float32, np.float64])
def test_binary_weights_keep_precision_and_do_not_mutate_input(x_dtype):
    x = np.array([0, 0, 1, 1, -99, np.nan], dtype=x_dtype)
    y = np.array([0, 1, 0, 1, 0, 1])
    weights = np.array([.123456789, .234567891, .345678912,
                        .456789123, .567891234, .678912345])
    before = weights.copy()
    model = OptimalBinning(user_splits=[.5], special_codes=[-99],
                           class_weight={0: 2, 1: 3},
                           monotonic_trend=None).fit(x, y, sample_weight=weights)
    np.testing.assert_array_equal(weights, before)
    assert model.binning_table.n_event.sum() == pytest.approx(
        3 * weights[y == 1].sum(), rel=1e-14)
    assert model.binning_table.n_nonevent.sum() == pytest.approx(
        2 * weights[y == 0].sum(), rel=1e-14)


@pytest.mark.parametrize('x_dtype', [np.float32, np.float64])
@pytest.mark.parametrize('weights', [None, .3, [.2, .3, .4, .5]])
def test_continuous_weights_retain_existing_conversion(x_dtype, weights,
                                                      monkeypatch):
    from sklearn.utils.validation import _check_sample_weight
    from optbinning import ContinuousOptimalBinning
    from optbinning.binning import continuous_binning

    original_split_data = continuous_binning.split_data
    x = np.array([0, 0, 1, 1], dtype=x_dtype)
    expected = _check_sample_weight(weights, x, dtype=x.dtype)

    def observe(*args, **kwargs):
        result = original_split_data(*args, **kwargs)
        np.testing.assert_array_equal(result[9], expected)
        assert result[9].dtype == expected.dtype
        assert len(result) == 14
        if weights is None:
            assert result[-1] is None
        else:
            np.testing.assert_array_equal(result[-1], expected)
        return result

    monkeypatch.setattr(continuous_binning, 'split_data', observe)
    model = ContinuousOptimalBinning(user_splits=[.5], monotonic_trend=None)
    if np.isscalar(weights) and weights is not None:
        with pytest.raises(TypeError, match='not iterable'):
            model.fit(x, [1., 2., 3., 4.], sample_weight=weights)
    else:
        model.fit(x, [1., 2., 3., 4.], sample_weight=weights)
