CP performance across updates
=============================

The automated benchmark uses the released **OptBinning 1.0.0** revision
``271c906d5250f17633396f88d246d64b4c449380`` as its permanent reference.
It shows both cumulative changes from that release and paired comparisons
against the preceding update. Correctness changes and timing changes are
reported separately.

After the first successful collection, the generated comparison tables are
available as ``index.rst`` and ``index.html`` on this repository's
``benchmark-results`` branch. Source revisions, workflow and results belong to
the same repository; no contributor-fork checkpoints are required.
Full collection runs manually or after a successful documentation publication
notification; it does not run on PR updates or ordinary code pushes. The matching
``benchmark-documentation`` Actions artifact contains an offline HTML page and
compressed raw measurements. The results branch preserves these measurements
after the temporary Actions artifact expires.

Data and measurements
---------------------

* Construction: 112 deterministic synthetic aggregate cases covering binary,
  continuous, multiclass and two-dimensional CP formulations.
* Complete fitting: eight controls using the first feature of scikit-learn's
  bundled breast-cancer and diabetes datasets, numerical or converted to
  categorical values, with automatic or unrestricted shape selection.

Every comparison runs both revisions in the same pinned Python environment,
with seven paired repetitions, warm-ups and randomized execution order.
Construction and complete fitting have separate tables. A speedup above one
means faster execution. Panel p5 and p95 describe variation between cases,
not statistical confidence intervals. These are small runtime controls, not
an estimate of performance on every dataset.

Model protobufs and fit outputs are checked before assigning speedups.
Changed models remain visible with timings and changed-case counts, but no
equivalent-work speedup. Correctness is established by regression tests rather
than agreement with a defective old formulation.

Future updates
--------------

Each requested collection adds new code revisions and compares them with
1.0.0 and the preceding measured update (or an explicitly supplied reference). One record per measured revision
is kept; reruns replace it. Documentation-only updates reuse the previous code
record. The initial upstream reference is declared in
``benchmarks/cp/updates.json``; later code revisions are registered when requested.
Unmerged solver changes are not prerequisites. Each raw record retains dependency versions,
inputs, model hashes, outcomes, revisions and all timings.

The reporting protocol has its own fingerprint. Results from different
protocols or runners must not be subtracted to estimate incremental gains;
use the direct paired comparison. A manual refresh can remeasure the registered
updates. No noisy timing threshold blocks merges. PRs run normal correctness and lightweight harness tests only. The external
documentation publisher must send a ``documentation-published`` repository
dispatch with the published commit; until that integration is configured, use
manual collection. Benchmark failure leaves the last verified report available
and does not block documentation publication.
