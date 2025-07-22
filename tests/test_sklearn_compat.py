"""Tests for sklearn compat."""

from unittest import mock

import pytest

from optbinning import _sklearn_compat


@pytest.fixture()
def _cache_clear_version():
    """Clear the cache each time it's called so multiple tests can run."""
    _sklearn_compat._sklearn_version.cache_clear()
    yield
    _sklearn_compat._sklearn_version.cache_clear()


@pytest.fixture()
def _cache_clear_new_check_array_api():
    """Clear the cache each time it's called so multiple tests can run."""
    _sklearn_compat._new_check_array_api.cache_clear()
    yield
    _sklearn_compat._new_check_array_api.cache_clear()


@pytest.mark.parametrize(
    ("sklearn_version", "want"),
    [
        ("1.0.2", {"force_all_finite": True}),
        # post releases
        ("1.0.2.post", {"force_all_finite": True}),
        # dev releases
        ("1.0.dev0", {"force_all_finite": True}),
        ("1.6.0", {"ensure_all_finite": True}),
        ("1.7.0", {"ensure_all_finite": True}),
        # release candidate
        ("1.7.0rc1", {"ensure_all_finite": True}),
    ],
)
def test__check_array_ensure_finite_kwargs(
    sklearn_version, want, _cache_clear_version, _cache_clear_new_check_array_api
):
    with mock.patch.object(
        _sklearn_compat.sklearn, "__version__", new=sklearn_version
    ), mock.patch.object(
        _sklearn_compat, "_check_array", return_value=""
    ) as mock_check_array:
        _sklearn_compat.check_array(force_all_finite=True)
        kwargs = mock_check_array.call_args.kwargs
        assert kwargs == want

