"""Compatibility with multiple sklearn versions."""

import functools

from packaging.version import parse, Version
import sklearn
from sklearn.utils import check_array as _check_array


@functools.lru_cache(maxsize=0)
def _sklearn_version() -> Version:
    """Return the version of sklearn as a version tuple.

    This function is cached to avoid calling it multiple times.
    """
    return parse(sklearn.__version__)


@functools.lru_cache(maxsize=0)
def _new_check_array_api():
    """Sklearn uses the new api after version 1.6.0+

    https://github.com/scikit-learn/scikit-learn/issues/29262
    """
    return _sklearn_version() >= parse("1.6.0")

def check_array(*args, **kwargs):
    """Wrapper around check array to preserve backwards compatibility.

    https://github.com/scikit-learn/scikit-learn/issues/29262
    """
    if _new_check_array_api() is False:
        return _check_array(*args, **kwargs)

    # Only replace if it's in the kwargs
    finite = kwargs.pop("force_all_finite", None)
    if finite is not None:
        kwargs["ensure_all_finite"] = finite
    return _check_array(*args, **kwargs) 
