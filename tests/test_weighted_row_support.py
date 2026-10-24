"""Weights change risk estimates, not the observation support of a bin."""
import numpy as np
import pandas as pd
import pytest
from optbinning import OptimalBinning


@pytest.mark.parametrize('solver,mip', [('cp', 'bop'), ('mip', 'bop'),
                                      ('mip', 'cbc')])
@pytest.mark.parametrize('scale', [.1, 1., 10.])
def test_small_heavy_bin_fails_raw_support(solver, mip, scale):
    x = np.repeat([0., 1.], [90, 10])
    y = np.tile([0, 1], 50)
    weights = np.where(x == 0, 1., 20.) * scale
    model = OptimalBinning(
        solver=solver, mip_solver=mip, user_splits=[.5],
        user_splits_fixed=[True], min_bin_size=.2, monotonic_trend=None)
    model.fit(x, y, sample_weight=weights)
    assert model.status == 'INFEASIBLE'


@pytest.mark.parametrize('solver', ['cp', 'mip'])
def test_large_light_bin_fails_raw_maximum(solver):
    x = np.repeat([0., 1.], [90, 10])
    model = OptimalBinning(solver=solver, user_splits=[.5],
                           user_splits_fixed=[True], max_bin_size=.6,
                           monotonic_trend=None)
    model.fit(x, np.tile([0, 1], 50), sample_weight=np.where(x == 0, .1, 1.))
    assert model.status == 'INFEASIBLE'


def test_cart_row_floor_with_class_weights():
    rng = np.random.default_rng(3)
    x = np.arange(1000.)
    y = np.r_[rng.binomial(1, .02, 980), np.tile([0, 1], 10)]
    model = OptimalBinning(class_weight='balanced', min_prebin_size=.1,
                           monotonic_trend=None).fit(x, y)
    raw_ne, raw_e = model._raw_prebin_counts
    assert np.all(raw_ne + raw_e >= 100)
    assert model.binning_table.build().loc['Totals', 'Count'] == 1000


@pytest.mark.parametrize('categorical', [False, True])
def test_raw_weighted_table_special_others_json(categorical, tmp_path):
    x = np.array(['a', 'a', 'b', 'b', 'c', 's', 't', None], dtype=object)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    w = np.array([.2, .8, 1.2, 1.8, .5, 2.5, .4, .6])
    if categorical:
        model = OptimalBinning(dtype='categorical',
            user_splits=[['a'], ['b']], special_codes={'S': ['s'], 'T': ['t']},
            monotonic_trend=None)
    else:
        x = np.array([0, 0, 1, 1, 1, -99, -88, np.nan])
        model = OptimalBinning(user_splits=[.5],
            special_codes={'S': [-99], 'T': [-88]}, monotonic_trend=None)
    model.fit(x, y, sample_weight=w)
    table = model.binning_table.build()
    assert table.loc['Totals', 'Count'] == len(x)
    assert table.loc['Totals', 'Event'] == y.sum()
    assert table.loc['Totals', 'Weighted count'] == pytest.approx(w.sum())
    assert table.loc['Totals', 'Weighted event'] == pytest.approx(w[y == 1].sum())
    assert table.loc['Totals', 'Weighted event rate'] == pytest.approx(
        w[y == 1].sum() / w.sum())
    assert table.loc['Totals', 'Event rate'] == y.mean()
    path = tmp_path / 'weighted.json'
    model.to_json(str(path))
    restored = OptimalBinning()
    restored.read_json(str(path))
    pd.testing.assert_frame_equal(restored.binning_table.build(), table)
    np.testing.assert_allclose(restored.transform(x), model.transform(x))
    assert len(model.binning_table.build(add_totals=False)) == len(table) - 1


@pytest.mark.parametrize('kwargs', [dict(max_bin_size=.5), dict(min_n_bins=2),
                                   dict(min_bin_n_event=6)])
def test_single_bin_still_checks_support(kwargs):
    model = OptimalBinning(user_splits=[], **kwargs).fit(
        np.zeros(10), np.tile([0, 1], 5), sample_weight=np.full(10, .3))
    assert model.status == 'INFEASIBLE'


def test_weighted_refit_resets_table_schema():
    model = OptimalBinning(user_splits=[])
    x = np.arange(10); y = np.tile([0, 1], 5)
    model.fit(x, y, sample_weight=np.full(10, .3))
    assert 'Weighted count' in model.binning_table.build()
    model.fit(x, y)
    assert 'Weighted count' not in model.binning_table.build()


def test_legacy_json_remains_readable(tmp_path):
    model = OptimalBinning(user_splits=[]).fit(
        np.arange(10), np.tile([0, 1], 5))
    path = tmp_path / 'legacy.json'; model.to_json(str(path))
    restored = OptimalBinning(); restored.read_json(str(path))
    assert restored.binning_table.build().loc['Totals', 'Count'] == 10


@pytest.mark.parametrize('solver', ['cp', 'mip'])
def test_class_support_cannot_be_created_by_large_weights(solver):
    x = np.repeat([0., 1.], 10)
    y = np.r_[np.zeros(9), 1, np.tile([0, 1], 5)]
    weights = np.where(y == 1, 10., .3)
    model = OptimalBinning(solver=solver, user_splits=[.5],
        user_splits_fixed=[True], min_bin_n_event=2, monotonic_trend=None)
    model.fit(x, y, sample_weight=weights)
    assert model.status == 'INFEASIBLE'


def test_zero_weight_rows_remain_in_raw_table():
    weights = np.r_[0., np.full(19, .3)]
    model = OptimalBinning(user_splits=[]).fit(
        np.arange(20), np.tile([0, 1], 10), sample_weight=weights)
    table = model.binning_table.build()
    assert table.loc['Totals', 'Count'] == 20
    assert table.loc['Totals', 'Weighted count'] == pytest.approx(weights.sum())


def test_weighted_localsolver_rejected_before_optional_dependency():
    with pytest.raises(ValueError, match='Weighted binary binning requires'):
        OptimalBinning(solver='ls').fit(
            np.arange(10), np.tile([0, 1], 5), sample_weight=np.ones(10))


def test_fractional_row_bounds_with_no_feasible_integer_size():
    model = OptimalBinning(user_splits=[4.5], min_bin_size=.15,
        max_bin_size=.15, monotonic_trend=None).fit(
            np.arange(10), np.tile([0, 1], 5), sample_weight=np.full(10, .3))
    assert model.status == 'INFEASIBLE'


def test_single_categorical_group_preserves_raw_and_weighted_counts():
    model = OptimalBinning(dtype='categorical', user_splits=[['a', 'b']]).fit(
        ['a', 'a', 'b', 'b'], [0, 1, 0, 1], sample_weight=[.2, .3, .4, .5])
    table = model.binning_table.build()
    assert table.loc[0, 'Count'] == 4
    assert table.loc[0, 'Weighted count'] == pytest.approx(1.4)
    assert table.loc['Totals', 'Count'] == 4
