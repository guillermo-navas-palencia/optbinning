import numpy as np
import pytest
from numpy.testing import assert_allclose
from optbinning import BinningProcess, OptimalBinning, ContinuousOptimalBinning


@pytest.mark.parametrize('cls', [OptimalBinning, ContinuousOptimalBinning])
@pytest.mark.parametrize('categorical', [False, True])
def test_opt_in_dtype_roundtrip(tmp_path, cls, categorical):
    x = np.repeat([0., 1.], 60)
    if categorical:
        x = np.where(x == 0, 'low', 'high')
    y = np.r_[np.tile([0, 0, 0, 1], 15), np.tile([0, 1, 1, 1], 15)]
    if cls is ContinuousOptimalBinning:
        y = y * 3.1 + np.linspace(0, 1, 120)
    assert cls().dtype == 'numerical'
    fitted = cls(dtype=None).fit(x, y)
    path = str(tmp_path / 'model.json')
    fitted.to_json(path)
    restored = cls()
    restored.read_json(path)
    assert restored._dtype == ('categorical' if categorical else 'numerical')
    assert_allclose(restored.transform(x), fitted.transform(x))
    assert fitted.dtype is None


@pytest.mark.parametrize('cls', [OptimalBinning, ContinuousOptimalBinning])
def test_process_import_uses_fitted_dtype(cls):
    x = np.linspace(0, 1, 120)
    y = np.tile([0, 1, 1], 40)
    if cls is ContinuousOptimalBinning:
        y = y * 2.1 + x
    optb = cls(name='x', dtype=None).fit(x, y)
    process = BinningProcess(['x']).fit_from_dict({'x': optb})
    row = process.summary().iloc[0]
    assert row['dtype'] == 'numerical'
    assert row['n_bins'] == len(optb.splits) + 1
    assert_allclose(process.transform(x[:, None])[:, 0], optb.transform(x))
