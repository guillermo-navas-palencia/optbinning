CP performance benchmarks
=========================

We use `Airspeed Velocity (ASV) <https://asv.readthedocs.io/>`_ to track
runtime across code revisions, following the approach used by
`scikit-learn <https://scikit-learn.org/dev/developers/contributing.html#monitoring-performance>`_.
ASV handles repeated timings, environments, saved results and HTML reports.

The suite contains 112 deterministic CP model-construction cases and eight
complete-fit controls using scikit-learn's bundled breast-cancer and diabetes
datasets. Construction and fitting are timed separately, with seven repetitions.
Data preparation is outside the timers. Fits must finish with an optimal status.

Run a comparison
----------------

From a full repository checkout, install the benchmark tools and compare two
revisions::

    python -m pip install asv==0.6.5 virtualenv==20.35.4
    asv machine --yes
    asv continuous 271c906d5250f17633396f88d246d64b4c449380 HEAD
    asv publish
    asv preview

The fixed reference is OptBinning 1.0.0. Replace it with another commit to
compare successive updates. ASV creates a separate environment with the
versions in ``asv.conf.json``; its generated files are under ``build/asv/``.
The dashboard retains all recorded revisions. ``asv compare REF NEW`` compares
saved timings without rerunning the fits. Its ratio is new time / reference
time (below one is faster); speedup is the reciprocal.

Automated collection
--------------------

The **CP benchmark documentation** workflow runs manually or when the
repository receives a ``documentation-published`` event. It measures 1.0.0,
the selected revision, and an optional previous revision on the same runner.
It does not run on PR updates and does not change normal CI.

Results and the generated HTML dashboard are saved on this repository's
``benchmark-results`` branch and in an Actions artifact. Download the artifact
to browse the report locally. Failed runs retain diagnostic artifacts without
replacing the published results. The documentation publisher must send the
publication event with ``client_payload.ref``; until connected, runs are manual.

Read the results
----------------

Compare the same benchmark and environment. GitHub-hosted hardware can vary,
so small timing differences require confirmation. ASV reports runtime; it does
not establish equal partitions or predictive accuracy across revisions.
Correctness is assessed by the library's tests, including regression tests in
the relevant solver PRs. A correctness fix can legitimately change the work
being timed. Do not interpret such a ratio as an equivalent-problem speedup.
