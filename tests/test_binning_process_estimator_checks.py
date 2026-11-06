"""Run sklearn's general estimator contract checks on BinningProcess.

Generate every check from the estimator's unmodified tags. The explicit
registries below document existing failures tracked in GH #432, not a list of
checks to run. Each expected failure still executes; an unexpected pass or a
failure with a different exception type fails the suite. Remove an entry when
its underlying contract is resolved. Intentional support differences are kept
visible pending an API/tag decision, rather than labelled as solver defects.
"""

from functools import partial

import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import estimator_checks_generator

from optbinning import BinningProcess


# Shared failures reproduced with both automatic and continuous targets.
_SHARED_FAILURES = {
    "check_n_features_in_after_fitting": (
        AssertionError, "fit does not publish n_features_in_."),
    "check_n_features_in": (
        AssertionError, "fit does not publish n_features_in_."),
    "check_fit_check_is_fitted": (
        NotFittedError, "Successful fit is not recognized by check_is_fitted."),
    "check_sample_weights_pandas_series": (
        TypeError, "Inferred integer DataFrame column names are rejected by "
        "the per-variable name validation."),
    "check_sample_weights_not_an_array": (
        TypeError, "fit accepts only ndarray/DataFrame, not general array-like "
        "X; this support contract needs a decision."),
    "check_transformer_data_not_an_array": (
        TypeError, "fit accepts only ndarray/DataFrame, not general array-like "
        "X; this support contract needs a decision."),
    "check_estimators_empty_data_messages": (
        AssertionError, "Empty input is not consistently rejected by fit."),
    "check_estimators_nan_inf": (
        AssertionError, "Native missing-value handling disagrees with the "
        "default allow_nan tag; resolve tags without removing that support."),
    "check_estimator_sparse_tag": (
        AssertionError, "Sparse input is rejected without an explicit "
        "sparse-input diagnostic."),
    "check_estimator_sparse_array": (
        AssertionError, "Sparse input is rejected without an explicit "
        "sparse-input diagnostic."),
    "check_estimator_sparse_matrix": (
        AssertionError, "Sparse input is rejected without an explicit "
        "sparse-input diagnostic."),
    "check_fit2d_predict1d": (
        AssertionError, "One-dimensional transform input raises a shape "
        "unpacking error instead of a reshape diagnostic."),
}

# Target-specific failures: do not mark the corresponding continuous checks.
_DEFAULT_FAILURES = {
    "check_estimators_dtypes": (
        ValueError, "Generated binary labels reach empty bin statistics."),
    "check_sample_weights_shape": (
        ValueError, "Generated binary labels reach empty bin statistics "
        "before the weight-shape assertions."),
    "check_sample_weights_not_overwritten": (
        ValueError, "Generated binary labels reach empty bin statistics "
        "before the weight-mutation assertions."),
    "check_sample_weights_list": (
        ValueError, "The check generates multiclass targets, for which "
        "sample_weight is explicitly unsupported; support decision pending."),
    "check_sample_weight_equivalence_on_dense_data": (
        ValueError, "The check generates multiclass targets, for which "
        "sample_weight is explicitly unsupported; support decision pending."),
    "check_dtype_object": (
        ValueError, "Object X is treated as categorical, which the detected "
        "multiclass target does not support; support decision pending."),
    "check_fit2d_1sample": (
        AssertionError, "Degenerate input reaches empty bin statistics "
        "instead of a clear minimum-sample/class diagnostic."),
    "check_fit2d_1feature": (
        AssertionError, "Generated labels reach empty bin statistics "
        "instead of fitting the single feature."),
}

_CONTINUOUS_FAILURES = {
    "check_dtype_object": (
        AssertionError, "Invalid object entries raise an unhashable-dict "
        "error instead of a numeric-input diagnostic."),
}


def _estimator_cases(estimator):
    """Use sklearn's public generator, with per-check strict pytest marks."""
    known_failures = dict(_SHARED_FAILURES)
    known_failures.update(_CONTINUOUS_FAILURES
                          if estimator.target_dtype == "continuous"
                          else _DEFAULT_FAILURES)
    cases = []
    for instance, check in estimator_checks_generator(estimator):
        function = check
        while isinstance(function, partial):
            function = function.func
        name = function.__name__
        marks = []
        if name in known_failures:
            exception, reason = known_failures[name]
            marks.append(pytest.mark.xfail(
                strict=True, raises=exception, reason="GH #432: " + reason))
        cases.append(pytest.param(instance, check, marks=marks,
                                  id="{}-{}".format(instance, name)))
    return cases


@pytest.mark.parametrize("estimator, check", _estimator_cases(BinningProcess()))
def test_binning_process_estimator_contract(estimator, check):
    check(estimator)


# Integer-valued generated targets also have a valid continuous interpretation.
# Do not adapt the check data or override tags to force a passing result.
@pytest.mark.parametrize("estimator, check", _estimator_cases(
    BinningProcess(target_dtype="continuous")))
def test_continuous_binning_process_estimator_contract(estimator, check):
    check(estimator)
