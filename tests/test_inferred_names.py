import numpy as np
import pandas as pd
import pytest
from optbinning import BinningProcess, BinningProcessSketch
from sklearn.base import clone
from sklearn.exceptions import NotFittedError


@pytest.fixture
def sample():
    x = np.repeat([0., 1., 2.], 40)
    y = np.concatenate([np.r_[np.zeros(40-n), np.ones(n)] for n in [5, 20, 35]])
    return pd.DataFrame({'a': x, 'b': x + 1}), y


def test_inferred_names_survive_input_check_and_refit(sample):
    X, y = sample
    process = BinningProcess().fit(X, y, check_input=True)
    assert list(process.get_feature_names_out()) == ['a', 'b']
    assert process.variable_names is None
    assert clone(process).variable_names is None
    process.fit(X[['a']].rename(columns={'a': 'new'}), y)
    assert list(process.summary()['name']) == ['new']
    assert list(process.get_feature_names_out()) == ['new']
    assert list(process.transform(X.rename(columns={'a': 'new'})).columns) == ['new']


def test_explicit_names_snapshot_and_sketch(sample):
    X, y = sample
    names = ['a', 'b']
    process = BinningProcess(names).fit(X, y)
    names[0] = 'changed'
    assert list(process.get_support(names=True)) == ['a', 'b']
    sketch = BinningProcessSketch(['a', 'b'], selection_criteria={'iv': {'min': 0}})
    sketch.add(X, y)
    sketch.solve()
    assert list(sketch.summary()['name']) == ['a', 'b']


def test_categorical_names_preserved_with_input_check(sample):
    X, y = sample
    X['b'] = pd.Categorical(np.where(X['a'] == 0, 'low', 'high'))
    process = BinningProcess(categorical_variables=["b"]).fit(
        X, y, check_input=True)
    assert process.get_binned_variable('b').dtype == 'categorical'
    assert list(process.get_feature_names_out()) == ['a', 'b']


def test_failed_refit_does_not_reuse_old_fit(sample):
    X, y = sample
    process = BinningProcess().fit(X, y)
    with pytest.raises(TypeError):
        process.fit("invalid", y)
    with pytest.raises(NotFittedError):
        process.transform(X)


def test_inferred_names_pipeline_pandas_output(sample):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    X, y = sample
    pipeline = Pipeline([('binning', BinningProcess()),
                         ('classifier', LogisticRegression())])
    pipeline.set_output(transform='pandas')
    pipeline.fit(X, y)
    transformed = pipeline.named_steps['binning'].transform(X)
    assert list(transformed.columns) == list(X.columns)
    pd.testing.assert_index_equal(transformed.index, X.index)
    assert list(pipeline.named_steps['classifier'].feature_names_in_) == list(X.columns)
    assert pipeline.predict(X).shape == y.shape
