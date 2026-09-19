#!/usr/bin/env bash
# Fixes the two real conflicts #408 (fix-224-scorecard-pvalues) had with
# #395 and #398, both in scorecard.py / test_scorecard.py -- by moving
# where #408 inserts its new code and its new test, same trick as the
# #390 destack: the content doesn't change, just where it lands, so it
# stops landing on the same lines #395/#398 also touch.
#
# Verified with real `git merge` trials: #395x#408 and #398x#408 both
# merge with zero conflicts now, in either order, plus the full test
# suite on the merged results.
#
# This only touches fix-224-scorecard-pvalues. No other branches are
# affected.
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

BRANCH="fix-224-scorecard-pvalues"
PATCH_FILE=$(mktemp)
cat > "$PATCH_FILE" << 'PATCH_EOF_pr_408_fix'
diff --git a/optbinning/scorecard/scorecard.py b/optbinning/scorecard/scorecard.py
index d3d960d..e300b44 100644
--- a/optbinning/scorecard/scorecard.py
+++ b/optbinning/scorecard/scorecard.py
@@ -14,10 +14,10 @@ import numpy as np
 import pandas as pd
 
 from scipy import stats
+from sklearn.linear_model import LogisticRegression
 
 from sklearn.base import BaseEstimator
 from sklearn.base import clone
-from sklearn.linear_model import LogisticRegression
 from sklearn.utils.multiclass import type_of_target
 
 from ..binning.base import Base
@@ -736,12 +736,6 @@ class Scorecard(Base, BaseEstimator):
             binning_table.loc[:, "Coefficient"] = c
             binning_table.loc[:, "Points"] = binning_table[bt_metric] * c
 
-            if pvalue_stats is not None:
-                se, z, pvalues = pvalue_stats
-                binning_table.loc[:, "Std. Error"] = se[i]
-                binning_table.loc[:, "Z-score"] = z[i]
-                binning_table.loc[:, "P-value"] = pvalues[i]
-
             nt = len(binning_table)
             if metric_special != 'empirical':
                 if isinstance(optb.special_codes, dict):
@@ -756,6 +750,13 @@ class Scorecard(Base, BaseEstimator):
 
             binning_table.index.names = ['Bin id']
             binning_table.reset_index(level=0, inplace=True)
+
+            if pvalue_stats is not None:
+                se, z, pvalues = pvalue_stats
+                binning_table.loc[:, "Std. Error"] = se[i]
+                binning_table.loc[:, "Z-score"] = z[i]
+                binning_table.loc[:, "P-value"] = pvalues[i]
+
             binning_tables.append(binning_table)
 
         df_scorecard = pd.concat(binning_tables)
diff --git a/tests/test_scorecard.py b/tests/test_scorecard.py
index 44b4382..82b2119 100644
--- a/tests/test_scorecard.py
+++ b/tests/test_scorecard.py
@@ -463,29 +463,6 @@ def test_verbose():
             scorecard.fit(X, y)
 
 
-def test_missing_metrics():
-    data = pd.DataFrame(
-        {'target': np.hstack(
-            (np.tile(np.array([0, 1]), 50),
-             np.array([0]*90 + [1]*10)
-             )
-         ),
-         'var': [np.nan] * 100 + ['A'] * 100}
-    )
-
-    binning_process = BinningProcess(['var'])
-    scaling_method_params = {'min': 0, 'max': 100}
-
-    scorecard = Scorecard(
-        binning_process=binning_process,
-        estimator=LogisticRegression(),
-        scaling_method="min_max",
-        scaling_method_params=scaling_method_params
-    ).fit(data, data.target)
-
-    assert scorecard.table()['Points'].iloc[-1] == approx(0, rel=1e-6)
-
-
 def test_pvalues():
     # Scorecard.table(style="detailed") exposes Wald-test p-values for
     # each explanatory variable when the estimator is a LogisticRegression
@@ -553,3 +530,26 @@ def test_pvalues():
         estimator=LinearRegression(),
         scaling_method=None).fit(X, y_cont)
     assert "P-value" not in scorecard_cont.table(style="detailed").columns
+
+
+def test_missing_metrics():
+    data = pd.DataFrame(
+        {'target': np.hstack(
+            (np.tile(np.array([0, 1]), 50),
+             np.array([0]*90 + [1]*10)
+             )
+         ),
+         'var': [np.nan] * 100 + ['A'] * 100}
+    )
+
+    binning_process = BinningProcess(['var'])
+    scaling_method_params = {'min': 0, 'max': 100}
+
+    scorecard = Scorecard(
+        binning_process=binning_process,
+        estimator=LogisticRegression(),
+        scaling_method="min_max",
+        scaling_method_params=scaling_method_params
+    ).fit(data, data.target)
+
+    assert scorecard.table()['Points'].iloc[-1] == approx(0, rel=1e-6)
PATCH_EOF_pr_408_fix

echo ""
echo "=== $BRANCH ==="
git checkout -q "$BRANCH"
git reset -q --hard "origin/$BRANCH"
if ! git apply --check "$PATCH_FILE" 2>/tmp/_pr408_apply_err; then
  echo "  ERROR: patch does not apply cleanly (branch may have changed upstream since this was prepared)." >&2
  cat /tmp/_pr408_apply_err >&2
  exit 1
fi
git apply "$PATCH_FILE"
git add -A -- ':!tests/results'
git commit -q --amend --no-edit
git push --force-with-lease origin "$BRANCH"
echo "  OK: $BRANCH conflicts resolved and pushed"

echo ""
echo "All done."
