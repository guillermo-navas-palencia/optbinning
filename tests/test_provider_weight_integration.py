"""Custom prebinning preserves weighted statistics and raw-row support."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from optbinning import OptimalBinning


class Provider(BaseEstimator):
    def __init__(self, cuts=(.5,)):
        self.cuts = cuts

    def fit(self, x, y, sample_weight=None):
        self.splits_ = self.cuts
        self.weights_ = sample_weight.copy()
        return self


@pytest.mark.parametrize('solver,mip', [('cp', 'bop'), ('mip', 'bop'),
                                      ('mip', 'cbc')])
@pytest.mark.parametrize('scale', [.1, 1., 10.])
@pytest.mark.parametrize('bounds', [dict(min_bin_size=.2),
                                   dict(max_bin_size=.6)])
def test_custom_cuts_respect_raw_support(solver, mip, scale, bounds):
    x = np.repeat([0., 1.], [90, 10])
    y = np.tile([0, 1], 50)
    w = np.where(x == 0, 1., 20.) * scale
    model = OptimalBinning(prebinning_estimator=Provider(),
                           solver=solver, mip_solver=mip, min_n_bins=2,
                           monotonic_trend=None, **bounds).fit(
                               x, y, sample_weight=w)
    assert model.status == 'INFEASIBLE'
    np.testing.assert_allclose(model.prebinning_estimator_.weights_, w)


@pytest.mark.parametrize('solver', ['cp', 'mip'])
def test_custom_fractional_tables_json_and_zero_rows(solver, tmp_path):
    x = np.array([0., 0., 1., 1., 1., -99., np.nan])
    y = np.array([0, 1, 0, 1, 0, 1, 0])
    w = np.array([0., .8, 1.2, 1.8, .5, 2.5, .4])
    model = OptimalBinning(prebinning_estimator=Provider(), solver=solver,
                           special_codes=[-99.], monotonic_trend=None).fit(
                               x, y, sample_weight=w)
    table = model.binning_table.build()
    assert table.loc['Totals', 'Count'] == len(x)
    assert table.loc['Totals', 'Event'] == y.sum()
    assert table.loc['Totals', 'Weighted count'] == pytest.approx(w.sum())
    assert table.loc['Totals', 'Weighted event'] == pytest.approx(w[y == 1].sum())
    np.testing.assert_allclose(model.prebinning_estimator_.weights_, w[:5])
    path = str(tmp_path / 'custom.json')
    model.to_json(path)
    restored = OptimalBinning()
    restored.read_json(path)
    pd.testing.assert_frame_equal(restored.binning_table.build(), table)
    np.testing.assert_allclose(restored.transform(x), model.transform(x))


@pytest.mark.parametrize('solver', ['cp', 'mip'])
@pytest.mark.parametrize('class_weight', ['balanced', {0: .7, 1: 1.9}])
@pytest.mark.parametrize('with_sample_weight', [False, True])
def test_provider_receives_composed_clean_weights(solver, class_weight,
                                                with_sample_weight):
    from sklearn.utils.class_weight import compute_sample_weight

    x = np.r_[np.repeat([0., 1.], 12), np.nan, -99.]
    y = np.tile([0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0], 2)
    weights = np.linspace(.3, 1.7, len(x)) if with_sample_weight else None
    original = None if weights is None else weights.copy()
    effective = compute_sample_weight(class_weight, y)
    if weights is not None:
        effective *= weights
    kwargs = dict(solver=solver, monotonic_trend=None, special_codes=[-99.])
    model = OptimalBinning(prebinning_estimator=Provider(),
                           class_weight=class_weight, **kwargs).fit(
                               x, y, sample_weight=weights)
    manual = OptimalBinning(user_splits=[.5], **kwargs).fit(
        x, y, sample_weight=effective)

    np.testing.assert_allclose(model.prebinning_estimator_.weights_,
                               effective[:24])
    if weights is not None:
        np.testing.assert_array_equal(weights, original)
    assert model.status == manual.status
    np.testing.assert_allclose(model.splits, manual.splits)
    np.testing.assert_allclose(model.transform(x), manual.transform(x))
    pd.testing.assert_frame_equal(model.binning_table.build(),
                                  manual.binning_table.build())
