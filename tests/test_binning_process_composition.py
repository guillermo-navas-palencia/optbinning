"""Public sklearn composition contracts for BinningProcess.

No implementation changes or dependency on exact solver timings/objectives.
"""
import pickle

import numpy as np
import pandas as pd
import pytest
import sklearn
from numpy.testing import assert_allclose, assert_array_equal
from packaging.version import Version
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from optbinning import BinningProcess


@pytest.fixture
def credit_like_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data[:, :3], columns=['a', 'b', 'c'],
                     index=np.arange(len(data.target)) * 2 + 1)
    return X, data.target


def make_process():
    return BinningProcess(variable_names=['a', 'b', 'c'], max_n_prebins=5)


def test_clone_is_unfitted_and_does_not_share_state(credit_like_data):
    X, y = credit_like_data
    fitted = make_process().fit(X, y)
    copied = clone(fitted)
    assert copied.get_params() == fitted.get_params()
    with pytest.raises(NotFittedError):
        copied.get_feature_names_out()
    copied.fit(X, y)
    assert_allclose(copied.transform(X), fitted.transform(X))
    assert copied.get_binned_variable('a') is not fitted.get_binned_variable('a')


def test_pickle_preserves_transform_and_selected_names(credit_like_data):
    X, y = credit_like_data
    fitted = make_process().fit(X, y)
    restored = pickle.loads(pickle.dumps(fitted))
    assert_allclose(restored.transform(X), fitted.transform(X))
    assert_array_equal(restored.get_feature_names_out(),
                       fitted.get_feature_names_out())


def test_column_transformer_keeps_column_order_and_index(credit_like_data):
    X, y = credit_like_data
    model = ColumnTransformer([
        ('bins', make_process(), ['a', 'b', 'c']),
    ]).set_output(transform='pandas')
    actual = model.fit_transform(X, y)
    expected = make_process().fit_transform(X, y)
    assert_allclose(actual.to_numpy(), expected.to_numpy())
    pd.testing.assert_index_equal(actual.index, X.index)
    assert_array_equal(actual.columns, ['bins__a', 'bins__b', 'bins__c'])


def test_pipeline_grid_search_refits_without_mutating_template(credit_like_data):
    X, y = credit_like_data
    template = Pipeline([
        ('bins', make_process()),
        ('classifier', LogisticRegression(max_iter=300)),
    ])
    search = GridSearchCV(template, {'classifier__C': [0.1, 1.0]},
                          cv=StratifiedKFold(2, shuffle=True, random_state=7),
                          scoring='neg_log_loss', error_score='raise')
    search.fit(X, y)
    assert np.isfinite(search.cv_results_['mean_test_score']).all()
    probabilities = search.predict_proba(X)
    assert probabilities.shape == (len(X), 2)
    assert_allclose(probabilities.sum(axis=1), 1.0)
    with pytest.raises(NotFittedError):
        template.named_steps['bins'].get_feature_names_out()


@pytest.mark.xfail(
    strict=True, raises=NotFittedError,
    reason="GH #432: BinningProcess does not expose sklearn fitted-state "
           "metadata or a __sklearn_is_fitted__ hook.")
def test_check_is_fitted_recognizes_successful_fit(credit_like_data):
    X, y = credit_like_data
    process = make_process()
    with pytest.raises(NotFittedError):
        check_is_fitted(process)
    process.fit(X, y)
    check_is_fitted(process)


@pytest.mark.xfail(
    Version(sklearn.__version__) >= Version("1.8"),
    strict=True, raises=NotFittedError,
    reason="GH #432: sklearn >=1.8 enforces Pipeline fitted-state checks; "
           "its final BinningProcess is not recognized as fitted.")
def test_transformer_only_pipeline_after_fit(credit_like_data):
    X, y = credit_like_data
    model = Pipeline([('bins', make_process())])
    model.fit(X, y)
    assert_allclose(model.transform(X), model.named_steps['bins'].transform(X))
