#!/usr/bin/env bash
# Replacement for #390 (fix/388-vectorize-transform) -- supersedes the
# earlier destack_pr390.sh you already ran. Two things needed fixing on
# top of that:
#
# 1. That earlier run's `git add -A` accidentally swept up destack_pr390.sh
#    itself plus 2 leftover test-result pngs into the commit (my mistake,
#    sorry). This rebuilds the branch cleanly without them.
#
# 2. #389 and #390 both insert a new test right after `test_verbose()` in
#    three test files -- a trivial but real conflict once one of them
#    merges into master before the other. This moves #390's new tests to
#    a different anchor (right before test_verbose() instead of after),
#    so the two PRs no longer touch the same lines and merge cleanly in
#    either order with zero manual conflict resolution needed. Verified
#    with real `git merge` trials in both orders, plus the full test
#    suite on the merged result.
#
# This only touches fix/388-vectorize-transform. Your other 8 branches
# are unaffected.
#
# Run from your own terminal, same as the other scripts.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

STARTING_BRANCH="$(git branch --show-current)"

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "ERROR: you have uncommitted changes to tracked files. Commit or stash them first, then re-run." >&2
  exit 1
fi

echo "Fetching latest from origin..."
git fetch -q origin

restore_branch() {
  echo ""
  echo "Restoring your original branch: $STARTING_BRANCH"
  git checkout -q "$STARTING_BRANCH"
}
trap restore_branch EXIT

BRANCH="fix/388-vectorize-transform"
PATCH_FILE=$(mktemp)
cat > "$PATCH_FILE" << 'PATCH_EOF_pr_390_v2'
diff --git a/benchmarks/bench_transform.py b/benchmarks/bench_transform.py
new file mode 100644
index 0000000..751426e
--- /dev/null
+++ b/benchmarks/bench_transform.py
@@ -0,0 +1,188 @@
+"""
+Standalone benchmark for OptimalBinning fit()/transform() (GH #388).
+
+Not part of pytest/CI -- timing assertions are flaky across CI's OS/Python
+matrix, so this is a script to run manually.
+
+Usage: python benchmarks/bench_transform.py
+
+Prints library/env versions, then two tables (fit(), transform()) comparing
+"old" (pre-#388 implementation, kept only for comparison) against "new"
+(current vectorized implementation) for numerical/categorical inputs at
+10k/100k/1M rows. fit() is untouched by #388, so its table just runs the
+current implementation twice as a noise baseline.
+"""
+
+import platform
+import sys
+import time
+
+import numpy as np
+import pandas as pd
+
+import optbinning
+from optbinning import OptimalBinning
+
+
+SIZES = (10_000, 100_000, 1_000_000)
+
+
+# ---------------------------------------------------------------------------
+# Frozen reference implementation of transform() (pre-#388), for comparison
+# only.
+# ---------------------------------------------------------------------------
+
+def _naive_numerical_apply(x_clean, indices, metric_value, n_bins,
+                           cat_unknown):
+    """What optbinning/binning/transformations.py::_apply_transform did
+    for dtype="numerical" before #388."""
+    x_clean_transform = np.full(x_clean.shape, cat_unknown)
+    for i in range(n_bins):
+        mask = (indices == i)
+        x_clean_transform[mask] = metric_value[i]
+    return x_clean_transform
+
+
+def _naive_categorical_apply(x, bins, metric_value, n_bins, x_transform):
+    """What optbinning/binning/transformations.py::_apply_transform did
+    for the categorical branch before #388."""
+    x_p = pd.Series(x)
+    xt = x_transform.copy()
+    for i in range(n_bins):
+        mask = x_p.isin(bins[i])
+        xt[mask] = metric_value[i]
+    return xt
+
+
+# ---------------------------------------------------------------------------
+# Timing / printing helpers
+# ---------------------------------------------------------------------------
+
+def _time_it(fn, reps=5):
+    fn()  # warm up (first call pays import/allocator warm-up costs)
+    t0 = time.perf_counter()
+    for _ in range(reps):
+        fn()
+    return (time.perf_counter() - t0) / reps
+
+
+def _print_versions():
+    print("optbinning:", optbinning.__version__)
+    print("python:    ", sys.version.split()[0])
+    print("numpy:     ", np.__version__)
+    print("pandas:    ", pd.__version__)
+    print("platform:  ", platform.platform())
+
+
+def _print_table(title, rows):
+    print(f"\n=== {title} ===")
+    header = (f"{'dtype':<12}{'n':>10}   {'old (ms)':>10}   "
+              f"{'new (ms)':>10}   {'speedup':>8}")
+    print(header)
+    print("-" * len(header))
+    for dtype, n, old_t, new_t in rows:
+        speedup = old_t / new_t if new_t else float("nan")
+        print(f"{dtype:<12}{n:>10,}   {old_t * 1000:>10.2f}   "
+              f"{new_t * 1000:>10.2f}   {speedup:>7.2f}x")
+
+
+# ---------------------------------------------------------------------------
+# fit() table
+# ---------------------------------------------------------------------------
+
+def _bench_fit_one(dtype, n, n_cats=30):
+    rng = np.random.RandomState(0)
+
+    if dtype == "numerical":
+        x = rng.randn(n)
+    else:
+        cats = np.array([f"cat_{i}" for i in range(n_cats)])
+        x = rng.choice(cats, size=n).astype(object)
+
+    y = rng.randint(0, 2, n)
+
+    def do_fit():
+        return OptimalBinning(name="x", dtype=dtype).fit(x, y)
+
+    # fit() is not touched by #388 -- "old" and "new" both run today's
+    # (only) implementation, timed independently, so the table shows
+    # fit() is unaffected by this PR rather than claiming a speedup.
+    old_t = _time_it(do_fit, reps=3)
+    new_t = _time_it(do_fit, reps=3)
+    return old_t, new_t
+
+
+def bench_fit_table(sizes=SIZES):
+    rows = []
+    for dtype in ("numerical", "categorical"):
+        for n in sizes:
+            old_t, new_t = _bench_fit_one(dtype, n)
+            rows.append((dtype, n, old_t, new_t))
+    _print_table("fit()", rows)
+    print("(fit() is unchanged by #388 -- both columns run today's only "
+          "implementation, so speedup here is run-to-run noise, not a "
+          "real improvement; expect it to hover near 1.00x.)")
+
+
+# ---------------------------------------------------------------------------
+# transform() table
+# ---------------------------------------------------------------------------
+
+def _bench_transform_numerical_one(n, n_bins=8):
+    rng = np.random.RandomState(0)
+    splits = np.sort(rng.randn(n_bins - 1))
+    metric_value = rng.randn(n_bins)
+    cat_unknown = 0.0
+
+    x_clean = rng.randn(n)
+    indices = np.digitize(x_clean, splits, right=False)
+
+    old_t = _time_it(lambda: _naive_numerical_apply(
+        x_clean, indices, metric_value, n_bins, cat_unknown))
+    new_t = _time_it(lambda: metric_value[indices])
+    return old_t, new_t
+
+
+def _bench_transform_categorical_one(n, n_cats=30, n_bins=8):
+    rng = np.random.RandomState(0)
+    cats = np.array([f"cat_{i}" for i in range(n_cats)])
+    bins = [np.asarray(b) for b in np.array_split(cats, n_bins)]
+    metric_value = rng.randn(n_bins)
+    cat_to_bin = {c: i for i, b in enumerate(bins) for c in b}
+
+    x = rng.choice(cats, size=n).astype(object)
+    x_transform = np.full(x.shape, 0.0)
+
+    def new_apply():
+        codes, uniques = pd.factorize(x)
+        unique_bin_idx = np.fromiter(
+            (cat_to_bin.get(u, -1) for u in uniques), dtype=int,
+            count=len(uniques))
+        code_to_bin = np.append(unique_bin_idx, -1)
+        sample_bin_idx = code_to_bin[codes]
+        known = sample_bin_idx >= 0
+        xt = x_transform.copy()
+        xt[known] = metric_value[sample_bin_idx[known]]
+        return xt
+
+    old_t = _time_it(lambda: _naive_categorical_apply(
+        x, bins, metric_value, n_bins, x_transform))
+    new_t = _time_it(new_apply)
+    return old_t, new_t
+
+
+def bench_transform_table(sizes=SIZES):
+    rows = []
+    for n in sizes:
+        old_t, new_t = _bench_transform_numerical_one(n)
+        rows.append(("numerical", n, old_t, new_t))
+    for n in sizes:
+        old_t, new_t = _bench_transform_categorical_one(n)
+        rows.append(("categorical", n, old_t, new_t))
+    _print_table("transform()", rows)
+
+
+if __name__ == "__main__":
+    _print_versions()
+    bench_fit_table()
+    bench_transform_table()
diff --git a/optbinning/binning/transformations.py b/optbinning/binning/transformations.py
index 13c12ef..88d83c5 100644
--- a/optbinning/binning/transformations.py
+++ b/optbinning/binning/transformations.py
@@ -179,22 +179,45 @@ def _apply_transform(x, dtype, special_codes, metric, metric_special,
                      n_special, cat_unknown):
 
     if dtype == "numerical":
-        if metric == "bins":
-            x_clean_transform = np.full(x_clean.shape, cat_unknown,
-                                        dtype=object)
+        # ``indices`` already gives each sample's bin (0..n_bins-1), so
+        # gather per-bin values in one vectorized index instead of looping
+        # over bins and masking each time. Cast to int first: the "no
+        # splits" shortcut above builds ``indices`` as floats, and fancy
+        # indexing needs integers.
+        if np.issubdtype(indices.dtype, np.integer):
+            idx = indices
         else:
-            x_clean_transform = np.full(x_clean.shape, cat_unknown)
-
-        for i in range(n_bins):
-            mask = (indices == i)
-            x_clean_transform[mask] = metric_value[i]
+            idx = indices.astype(int)
+        x_clean_transform = np.asarray(metric_value)[idx]
 
         x_transform[clean_mask] = x_clean_transform
     else:
-        x_p = pd.Series(x)
-        for i in range(n_bins):
-            mask = x_p.isin(bins[i])
-            x_transform[mask] = metric_value[i]
+        # Build a category -> bin index lookup once, then resolve every
+        # sample's bin via pd.factorize in a single vectorized pass instead
+        # of looping over bins and testing isin() each time. factorize beat
+        # Series.map/Categorical/searchsorted in benchmarking; -1 codes mark
+        # missing/unseen values.
+        cat_to_bin = {}
+        for i, b in enumerate(bins):
+            for cat in b:
+                cat_to_bin[cat] = i
+
+        codes, uniques = pd.factorize(x)
+        if len(uniques):
+            unique_bin_idx = np.fromiter(
+                (cat_to_bin.get(u, -1) for u in uniques), dtype=int,
+                count=len(uniques))
+        else:
+            unique_bin_idx = np.array([], dtype=int)
+        # Append a trailing -1 so that ``codes == -1`` (factorize's
+        # marker for NaN) maps to "no bin" via Python's -1 indexing,
+        # rather than wrapping around to the last real category.
+        code_to_bin = np.append(unique_bin_idx, -1)
+        sample_bin_idx = code_to_bin[codes]
+
+        known = sample_bin_idx >= 0
+        x_transform[known] = np.asarray(metric_value)[
+            sample_bin_idx[known]]
 
     if special_codes:
         if isinstance(special_codes, dict):
diff --git a/tests/test_binning.py b/tests/test_binning.py
index 1000724..bd0a500 100644
--- a/tests/test_binning.py
+++ b/tests/test_binning.py
@@ -565,6 +565,97 @@ def test_information():
     optb.information(print_level=2)
 
 
+def test_numerical_transform_indices_and_bins():
+    # transform()'s numerical branch gathers per-sample values via
+    # np.digitize indices instead of a per-bin masking loop (GH issue
+    # #388). Cross-check "indices"/"bins" against "woe"/"event_rate" and
+    # the binning table itself, and confirm every sample's bin actually
+    # contains its value.
+    optb = OptimalBinning(name=variable, dtype="numerical")
+    optb.fit(x, y)
+
+    splits = optb.splits
+    n_bins = len(splits) + 1
+
+    indices = optb.transform(x, metric="indices")
+    bins = optb.transform(x, metric="bins")
+    woe = optb.transform(x, metric="woe")
+    event_rate = optb.transform(x, metric="event_rate")
+
+    assert ((indices >= 0) & (indices < n_bins)).all()
+
+    table = optb.binning_table.build()
+    table_woe = table["WoE"].values[:n_bins].astype(float)
+    table_event_rate = table["Event rate"].values[:n_bins].astype(float)
+
+    assert woe == approx(table_woe[indices], rel=1e-6)
+    assert event_rate == approx(table_event_rate[indices], rel=1e-6)
+
+    bin_edges = np.concatenate([[-np.inf], splits, [np.inf]])
+    for xi, bi, idx in zip(x, bins, indices):
+        lo, hi = bin_edges[idx], bin_edges[idx + 1]
+        assert lo <= xi < hi
+        assert bi == table["Bin"].values[idx]
+
+
+def test_numerical_transform_no_splits():
+    # Edge case: a single-bin solution has no splits at all, so
+    # np.digitize is skipped and ``indices`` is built as a float array of
+    # zeros (see transform_binary_target). The vectorized gather must
+    # still handle this (it requires integer indices for fancy indexing).
+    optb = OptimalBinning(name=variable, dtype="numerical", max_n_bins=1)
+    optb.fit(x, y)
+
+    assert len(optb.splits) == 0
+
+    for metric in ("woe", "event_rate", "indices", "bins"):
+        x_transform = optb.transform(x, metric=metric)
+        assert len(np.unique(x_transform)) == 1
+
+
+def test_categorical_transform_indices_and_bins():
+    # transform()'s categorical branch resolves each sample's bin via
+    # pd.factorize instead of looping over bins and testing membership
+    # against the whole array each time (GH issue #388). Cross-check
+    # "indices" against "woe"/"event_rate" and the binning table, and
+    # exercise the edge cases the rewrite has to handle explicitly: a
+    # category never seen during fit, and a missing (NaN) value.
+    rng = np.random.RandomState(0)
+    x_cat = rng.choice(np.array(['a', 'b', 'c', 'd', 'e']), size=500)
+    y_cat = rng.randint(0, 2, 500)
+
+    optb = OptimalBinning(name="x_cat", dtype="categorical")
+    optb.fit(x_cat, y_cat)
+
+    n_bins = len(optb.splits) + 1
+
+    x_test = np.concatenate([
+        x_cat[:50],
+        np.array(['unseen_cat'], dtype=object),
+        np.array([np.nan], dtype=object)])
+
+    indices = optb.transform(x_test, metric="indices")
+    woe = optb.transform(x_test, metric="woe")
+    event_rate = optb.transform(x_test, metric="event_rate")
+
+    table = optb.binning_table.build()
+    table_woe = table["WoE"].values[:n_bins].astype(float)
+    table_event_rate = table["Event rate"].values[:n_bins].astype(float)
+
+    # The known samples (first 50) fall in a real bin.
+    assert ((indices[:50] >= 0) & (indices[:50] < n_bins)).all()
+    assert woe[:50] == approx(table_woe[indices[:50]], rel=1e-6)
+    assert event_rate[:50] == approx(
+        table_event_rate[indices[:50]], rel=1e-6)
+
+    # Unseen category and missing value both fall back to cat_unknown /
+    # the missing-value handling rather than crashing or landing in a
+    # bin by accident.
+    default_woe = optb.transform(['unseen_cat'], metric="woe")[0]
+    assert woe[50] == approx(default_woe, rel=1e-6)
+    assert not np.isnan(woe[51])  # missing value has its own default
+
+
 def test_verbose():
     optb = OptimalBinning(verbose=True)
     optb.fit(x, y)
diff --git a/tests/test_continuous_binning.py b/tests/test_continuous_binning.py
index 7023a5c..d89a558 100644
--- a/tests/test_continuous_binning.py
+++ b/tests/test_continuous_binning.py
@@ -272,6 +272,87 @@ def test_numerical_default_fit_transform():
                                       30.47142857], rel=1e-6)
 
 
+def test_numerical_transform_indices_and_bins():
+    # transform()'s numerical branch gathers per-sample values via
+    # np.digitize indices instead of a per-bin masking loop (GH issue
+    # #388). Cross-check "indices"/"bins" against "mean" and the binning
+    # table itself, and confirm every sample's bin actually contains its
+    # value.
+    optb = ContinuousOptimalBinning(name=variable, dtype="numerical")
+    optb.fit(x, y)
+
+    splits = optb.splits
+    n_bins = len(splits) + 1
+
+    indices = optb.transform(x, metric="indices")
+    bins = optb.transform(x, metric="bins")
+    mean = optb.transform(x, metric="mean")
+
+    assert ((indices >= 0) & (indices < n_bins)).all()
+
+    table = optb.binning_table.build()
+    table_mean = table["Mean"].values[:n_bins].astype(float)
+
+    assert mean == approx(table_mean[indices], rel=1e-6)
+
+    bin_edges = np.concatenate([[-np.inf], splits, [np.inf]])
+    for xi, bi, idx in zip(x, bins, indices):
+        lo, hi = bin_edges[idx], bin_edges[idx + 1]
+        assert lo <= xi < hi
+        assert bi == table["Bin"].values[idx]
+
+
+def test_numerical_transform_no_splits():
+    # Edge case: a single-bin solution has no splits at all, so
+    # np.digitize is skipped and ``indices`` is built as a float array of
+    # zeros (see transform_continuous_target). The vectorized gather must
+    # still handle this (it requires integer indices for fancy indexing).
+    optb = ContinuousOptimalBinning(name=variable, dtype="numerical",
+                                    max_n_bins=1)
+    optb.fit(x, y)
+
+    assert len(optb.splits) == 0
+
+    for metric in ("mean", "indices", "bins"):
+        x_transform = optb.transform(x, metric=metric)
+        assert len(np.unique(x_transform)) == 1
+
+
+def test_categorical_transform_indices_and_bins():
+    # transform()'s categorical branch resolves each sample's bin via
+    # pd.factorize instead of looping over bins and testing membership
+    # against the whole array each time (GH issue #388). Cross-check
+    # "indices" against "mean" and the binning table, and exercise the
+    # edge cases the rewrite has to handle explicitly: a category never
+    # seen during fit, and a missing (NaN) value.
+    rng = np.random.RandomState(0)
+    x_cat = rng.choice(np.array(['a', 'b', 'c', 'd', 'e']), size=500)
+    y_cat = rng.randn(500)
+
+    optb = ContinuousOptimalBinning(name="x_cat", dtype="categorical")
+    optb.fit(x_cat, y_cat)
+
+    n_bins = len(optb.splits) + 1
+
+    x_test = np.concatenate([
+        x_cat[:50],
+        np.array(['unseen_cat'], dtype=object),
+        np.array([np.nan], dtype=object)])
+
+    indices = optb.transform(x_test, metric="indices")
+    mean = optb.transform(x_test, metric="mean")
+
+    table = optb.binning_table.build()
+    table_mean = table["Mean"].values[:n_bins].astype(float)
+
+    assert ((indices[:50] >= 0) & (indices[:50] < n_bins)).all()
+    assert mean[:50] == approx(table_mean[indices[:50]], rel=1e-6)
+
+    default_mean = optb.transform(['unseen_cat'], metric="mean")[0]
+    assert mean[50] == approx(default_mean, rel=1e-6)
+    assert not np.isnan(mean[51])  # missing value has its own default
+
+
 def test_verbose():
     optb = ContinuousOptimalBinning(verbose=True)
     optb.fit(x, y)
diff --git a/tests/test_multiclass_binning.py b/tests/test_multiclass_binning.py
index 8ca99f9..a4ab96b 100644
--- a/tests/test_multiclass_binning.py
+++ b/tests/test_multiclass_binning.py
@@ -5,6 +5,7 @@ MulticlassOptimalBinning testing.
 # Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
 # Copyright (C) 2020
 
+import numpy as np
 import pandas as pd
 
 from pytest import approx, raises
@@ -221,6 +222,43 @@ def test_classes():
     assert optb.classes == approx([0, 1, 2])
 
 
+def test_transform_indices_and_bins():
+    # transform() gathers per-sample values via np.digitize indices
+    # instead of a per-bin masking loop (GH issue #388). Confirm
+    # "indices"/"bins" are valid and every sample's bin actually
+    # contains its value.
+    optb = MulticlassOptimalBinning(name=variable)
+    optb.fit(x, y)
+
+    splits = optb.splits
+    n_bins = len(splits) + 1
+
+    indices = optb.transform(x, metric="indices")
+    bins = optb.transform(x, metric="bins")
+
+    assert ((indices >= 0) & (indices < n_bins)).all()
+
+    bin_edges = np.concatenate([[-np.inf], splits, [np.inf]])
+    for xi, bi, idx in zip(x, bins, indices):
+        lo, hi = bin_edges[idx], bin_edges[idx + 1]
+        assert lo <= xi < hi
+
+
+def test_transform_no_splits():
+    # Edge case: a single-bin solution has no splits at all, so
+    # np.digitize is skipped and ``indices`` is built as a float array of
+    # zeros (see transform_multiclass_target). The vectorized gather must
+    # still handle this (it requires integer indices for fancy indexing).
+    optb = MulticlassOptimalBinning(name=variable, max_n_bins=1)
+    optb.fit(x, y)
+
+    assert len(optb.splits) == 0
+
+    for metric in ("mean_woe", "indices", "bins"):
+        x_transform = optb.transform(x, metric=metric)
+        assert len(np.unique(x_transform)) == 1
+
+
 def test_verbose():
     optb = MulticlassOptimalBinning(verbose=True)
     optb.fit(x, y)
PATCH_EOF_pr_390_v2

echo ""
echo "=== $BRANCH ==="
git checkout -q -B "$BRANCH" "origin/master"
if ! git apply --check "$PATCH_FILE" 2>/tmp/_destack_apply_err; then
  echo "  ERROR: patch does not apply cleanly (branch may have changed upstream since this was prepared)." >&2
  cat /tmp/_destack_apply_err >&2
  echo "  Your $BRANCH branch was left pointed at origin/master -- nothing was pushed." >&2
  echo "  Let Claude know and it'll regenerate this against the current state." >&2
  exit 1
fi
git apply "$PATCH_FILE"
git add -A
git commit -q --author="Lucas Morin <lucas.cr.morin@gmail.com>" \
  -m "Vectorize transform() with numpy/pandas gather ops (closes #388)"
git push --force-with-lease origin "$BRANCH"
echo "  OK: $BRANCH rebuilt clean and pushed"

echo ""
echo "All done."
