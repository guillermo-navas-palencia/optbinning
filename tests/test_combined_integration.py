import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from optbinning import BinningProcess, Scorecard


def test_inferred_names_with_continuous_target_override():
    X = pd.DataFrame({'x': np.repeat([0., 1., 2.], 40)})
    y = np.tile(np.arange(40), 3) + X.x.to_numpy().astype(int) * 10
    process = BinningProcess(target_dtype='continuous')
    card = Scorecard(process, LinearRegression()).fit(X, y)
    assert card.binning_process_.variable_names is None
    assert list(card.binning_process_.get_feature_names_out()) == ['x']
    assert card.binning_process_._target_dtype == 'continuous'
    assert np.isfinite(card.predict(X)).all()
    assert np.isfinite(card.score(X)).all()
