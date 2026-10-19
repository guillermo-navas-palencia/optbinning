"""Compare complete transform() calls against the OptBinning 1.0.0 helper.

Run from the repository root: PYTHONPATH=. python benchmarks/bench_transform.py
No timing thresholds are enforced. Each case checks output equivalence first.
Both paths use the same fitted model; only _apply_transform is swapped.
"""

from __future__ import annotations

import platform
import time
from unittest.mock import patch

import numpy as np
import pandas as pd

import optbinning
from optbinning import OptimalBinning
from optbinning.binning import transformations


# Reference body from v1.0.0 (271c906); annotations omitted for this benchmark.
def _reference_apply_transform(
    x, dtype, special_codes, metric, metric_special, metric_missing,
    metric_value, clean_mask, special_mask, missing_mask, indices,
    x_transform, x_clean, bins, n_bins, n_special, cat_unknown
):
    if dtype == "numerical":
        if metric == "bins":
            x_clean_transform = np.full(x_clean.shape, cat_unknown,
                                        dtype=object)
        else:
            x_clean_transform = np.full(x_clean.shape, cat_unknown)

        for i in range(n_bins):
            mask = (indices == i)
            x_clean_transform[mask] = metric_value[i]

        x_transform[clean_mask] = x_clean_transform
    else:
        x_p = pd.Series(x)
        for i in range(n_bins):
            mask = x_p.isin(bins[i])
            x_transform[mask] = metric_value[i]

    if special_codes:
        if isinstance(special_codes, dict):
            xt = pd.Series(x)
            for i, (k, s) in enumerate(special_codes.items()):
                sl = s if isinstance(s, (list, np.ndarray)) else [s]
                mask = xt.isin(sl).values
                if (metric_special == "empirical" or (metric == "indices" and
                    not isinstance(metric_special, int)) or
                        metric == "bins"):
                    x_transform[mask] = metric_value[n_bins + i]
                else:
                    x_transform[mask] = metric_special
        else:
            if (metric_special == "empirical" or
                (metric == "indices" and
                    not isinstance(metric_special, int)) or
                    metric == "bins"):
                x_transform[special_mask] = metric_value[n_bins]
            else:
                x_transform[special_mask] = metric_special

    if (metric_missing == "empirical" or
        (metric == "indices" and not isinstance(metric_missing, int)) or
            metric == "bins"):
        x_transform[missing_mask] = metric_value[n_bins + n_special]
    else:
        x_transform[missing_mask] = metric_missing

    return x_transform


def _time_transform(model, values, implementation, repeats=7):
    # Patch setup is outside timing; category lookup construction is inside.
    with patch.object(transformations, "_apply_transform", implementation):
        model.transform(values)
        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            model.transform(values)
            timings.append(time.perf_counter() - start)
    return np.median(timings)


def benchmark():
    print(f"OptBinning {optbinning.__version__}; Python {platform.python_version()}")
    print(f"NumPy {np.__version__}; pandas {pd.__version__}; {platform.platform()}")
    print("Full transform(), median of 7 runs after warm-up; same fitted model")
    print("dtype          rows     bins    old ms    new ms   speedup")
    current = transformations._apply_transform
    for dtype in ["numerical", "categorical"]:
        rng = np.random.default_rng(388)
        train = rng.normal(size=10000)
        target = rng.binomial(1, 1 / (1 + np.exp(-train)))
        if dtype == "categorical":
            train = np.array([f"c{i}" for i in np.digitize(
                train, np.linspace(-2, 2, 29))], dtype=object)
        model = OptimalBinning(dtype=dtype, max_n_bins=8).fit(train, target)
        for size in [10000, 100000, 1000000]:
            values = rng.choice(train, size=size)
            values[0] = np.nan
            if dtype == "categorical":
                values[1] = "unseen"
            # Validate all supported output metrics before timing WoE.
            for metric in ["woe", "event_rate", "indices", "bins"]:
                with patch.object(transformations, "_apply_transform",
                                  _reference_apply_transform):
                    expected = model.transform(values, metric=metric)
                actual = model.transform(values, metric=metric)
                if metric == "bins":
                    np.testing.assert_array_equal(actual, expected)
                else:
                    np.testing.assert_allclose(actual, expected)
            old = _time_transform(model, values, _reference_apply_transform)
            new = _time_transform(model, values, current)
            n_bins = len(model.splits) + (dtype == "numerical")
            print(f"{dtype:<12} {size:>8,} {n_bins:>8} "
                  f"{old*1000:>9.2f} {new*1000:>9.2f} {old/new:>8.2f}x",
                  flush=True)


if __name__ == "__main__":
    benchmark()
