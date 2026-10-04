import numpy as np
import pandas as pd
import pytest
from optbinning import BinningProcess, Scorecard
from optbinning.scorecard import ScorecardMonitoring
from optbinning.binning.binning_process import resolve_target_dtype
from sklearn.linear_model import LinearRegression


@pytest.mark.parametrize('target', [[1, np.nan], [1, np.inf], [[1], [2]],
                                    [], ['a', 'b'], [1+1j, 2+1j]])
def test_invalid_continuous_override(target):
    with pytest.raises((ValueError, TypeError)):
        resolve_target_dtype(target, 'continuous')


@pytest.mark.parametrize('target', [[0, 1, 2], [0.1, 0.7], [0, 0]])
def test_invalid_binary_override(target):
    with pytest.raises(ValueError):
        resolve_target_dtype(target, 'binary')


@pytest.mark.parametrize('outer,inner', [(None, 'continuous'),
                                        ('continuous', 'binary')])
def test_scorecard_override_precedence_and_monitoring(outer, inner):
    x = np.linspace(0, 1, 100)
    X = pd.DataFrame({'x': x})
    y = np.floor(x * 20) + 10
    model = Scorecard(BinningProcess(['x'], target_dtype=inner),
                      LinearRegression(), target_dtype=outer).fit(X, y)
    assert model._target_dtype == 'continuous'
    assert model.binning_process_._target_dtype == 'continuous'
    monitor = ScorecardMonitoring(model).fit(X, y, X, y)
    assert monitor._target_dtype == 'continuous'
    with pytest.raises(ValueError):
        ScorecardMonitoring(model).fit(X, np.full(100, np.nan), X, y)


def test_existing_positional_arguments():
    import inspect
    bp = list(inspect.signature(BinningProcess).parameters)
    sc = list(inspect.signature(Scorecard).parameters)
    assert bp[-3:] == ['n_jobs', 'verbose', 'target_dtype']
    assert sc[-2:] == ['verbose', 'target_dtype']
