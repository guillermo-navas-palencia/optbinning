"""
Standalone benchmark for OptimalBinning fit()/transform() (GH #388).

Not part of pytest/CI -- timing assertions are flaky across CI's OS/Python
matrix, so this is a script to run manually.

Usage: python benchmarks/bench_transform.py

Prints library/env versions, then two tables (fit(), transform()) comparing
"old" (pre-#388 implementation, kept only for comparison) against "new"
(current vectorized implementation) for numerical/categorical inputs at
10k/100k/1M rows. fit() is untouched by #388, so its table just runs the
current implementation twice as a noise baseline.
"""

import platform
import sys
import time

import numpy as np
import pandas as pd

import optbinning
from optbinning import OptimalBinning


SIZES = (10_000, 100_000, 1_000_000)


# ---------------------------------------------------------------------------
# Frozen reference implementation of transform() (pre-#388), for comparison
# only.
# ---------------------------------------------------------------------------

def _naive_numerical_apply(x_clean, indices, metric_value, n_bins,
                           cat_unknown):
    """What optbinning/binning/transformations.py::_apply_transform did
    for dtype="numerical" before #388."""
    x_clean_transform = np.full(x_clean.shape, cat_unknown)
    for i in range(n_bins):
        mask = (indices == i)
        x_clean_transform[mask] = metric_value[i]
    return x_clean_transform


def _naive_categorical_apply(x, bins, metric_value, n_bins, x_transform):
    """What optbinning/binning/transformations.py::_apply_transform did
    for the categorical branch before #388."""
    x_p = pd.Series(x)
    xt = x_transform.copy()
    for i in range(n_bins):
        mask = x_p.isin(bins[i])
        xt[mask] = metric_value[i]
    return xt


# ---------------------------------------------------------------------------
# Timing / printing helpers
# ---------------------------------------------------------------------------

def _time_it(fn, reps=5):
    fn()  # warm up (first call pays import/allocator warm-up costs)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def _print_versions():
    print("optbinning:", optbinning.__version__)
    print("python:    ", sys.version.split()[0])
    print("numpy:     ", np.__version__)
    print("pandas:    ", pd.__version__)
    print("platform:  ", platform.platform())


def _print_table(title, rows):
    print(f"\n=== {title} ===")
    header = (f"{'dtype':<12}{'n':>10}   {'old (ms)':>10}   "
              f"{'new (ms)':>10}   {'speedup':>8}")
    print(header)
    print("-" * len(header))
    for dtype, n, old_t, new_t in rows:
        speedup = old_t / new_t if new_t else float("nan")
        print(f"{dtype:<12}{n:>10,}   {old_t * 1000:>10.2f}   "
              f"{new_t * 1000:>10.2f}   {speedup:>7.2f}x")


# ---------------------------------------------------------------------------
# fit() table
# ---------------------------------------------------------------------------

def _bench_fit_one(dtype, n, n_cats=30):
    rng = np.random.RandomState(0)

    if dtype == "numerical":
        x = rng.randn(n)
    else:
        cats = np.array([f"cat_{i}" for i in range(n_cats)])
        x = rng.choice(cats, size=n).astype(object)

    y = rng.randint(0, 2, n)

    def do_fit():
        return OptimalBinning(name="x", dtype=dtype).fit(x, y)

    # fit() is not touched by #388 -- "old" and "new" both run today's
    # (only) implementation, timed independently, so the table shows
    # fit() is unaffected by this PR rather than claiming a speedup.
    old_t = _time_it(do_fit, reps=3)
    new_t = _time_it(do_fit, reps=3)
    return old_t, new_t


def bench_fit_table(sizes=SIZES):
    rows = []
    for dtype in ("numerical", "categorical"):
        for n in sizes:
            old_t, new_t = _bench_fit_one(dtype, n)
            rows.append((dtype, n, old_t, new_t))
    _print_table("fit()", rows)
    print("(fit() is unchanged by #388 -- both columns run today's only "
          "implementation, so speedup here is run-to-run noise, not a "
          "real improvement; expect it to hover near 1.00x.)")


# ---------------------------------------------------------------------------
# transform() table
# ---------------------------------------------------------------------------

def _bench_transform_numerical_one(n, n_bins=8):
    rng = np.random.RandomState(0)
    splits = np.sort(rng.randn(n_bins - 1))
    metric_value = rng.randn(n_bins)
    cat_unknown = 0.0

    x_clean = rng.randn(n)
    indices = np.digitize(x_clean, splits, right=False)

    old_t = _time_it(lambda: _naive_numerical_apply(
        x_clean, indices, metric_value, n_bins, cat_unknown))
    new_t = _time_it(lambda: metric_value[indices])
    return old_t, new_t


def _bench_transform_categorical_one(n, n_cats=30, n_bins=8):
    rng = np.random.RandomState(0)
    cats = np.array([f"cat_{i}" for i in range(n_cats)])
    bins = [np.asarray(b) for b in np.array_split(cats, n_bins)]
    metric_value = rng.randn(n_bins)
    cat_to_bin = {c: i for i, b in enumerate(bins) for c in b}

    x = rng.choice(cats, size=n).astype(object)
    x_transform = np.full(x.shape, 0.0)

    def new_apply():
        codes, uniques = pd.factorize(x)
        unique_bin_idx = np.fromiter(
            (cat_to_bin.get(u, -1) for u in uniques), dtype=int,
            count=len(uniques))
        code_to_bin = np.append(unique_bin_idx, -1)
        sample_bin_idx = code_to_bin[codes]
        known = sample_bin_idx >= 0
        xt = x_transform.copy()
        xt[known] = metric_value[sample_bin_idx[known]]
        return xt

    old_t = _time_it(lambda: _naive_categorical_apply(
        x, bins, metric_value, n_bins, x_transform))
    new_t = _time_it(new_apply)
    return old_t, new_t


def bench_transform_table(sizes=SIZES):
    rows = []
    for n in sizes:
        old_t, new_t = _bench_transform_numerical_one(n)
        rows.append(("numerical", n, old_t, new_t))
    for n in sizes:
        old_t, new_t = _bench_transform_categorical_one(n)
        rows.append(("categorical", n, old_t, new_t))
    _print_table("transform()", rows)


if __name__ == "__main__":
    _print_versions()
    bench_fit_table()
    bench_transform_table()
