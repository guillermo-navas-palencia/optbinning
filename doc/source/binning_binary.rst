Optimal binning with binary target
==================================

.. autoclass:: optbinning.OptimalBinning
   :members:
   :inherited-members:
   :show-inheritance:

Opt-in exact IV fast path
------------------------
``OptimalBinning(solver="auto", monotonic_trend=None)`` attempts an exact
dynamic program for the same integer-scaled IV objective as CP (scale 1e6).
The default ``solver="cp"`` is unchanged and CP presolve remains enabled.

The fast path accepts numerical features, 2–100 refined prebins, finite
positive integer event/non-event counts, bin-count bounds and row-support
bounds. Monotonicity, gamma, p-value constraints, minimum rate differences,
per-class count bounds, fixed cuts, categorical features and other divergences
fall back to CP with their original constraints. Fractional counts fall back.
With the usual automatic monotonicity setting, CP is generally selected.

DP uses an attempt threshold of min(0.1 seconds, time_limit/10), checked
between prefix states (not a hard deadline). On expiry CP is used with
its configured solver budget. As before, time_limit is not a hard whole-fit
wall-clock limit: pre-processing, model construction and dispatch add overhead.
``solver_backend_`` and ``information()`` report the actual backend. An
infeasible supported problem is reported as INFEASIBLE. Equal optimal integer
scores may yield different cutpoints: DP prefers fewer bins then its first
predecessor; exact cutpoint identity with CP is not guaranteed. Raw IV and the
integer-scaled objective are distinct. No global speedup is claimed.

The dynamic program stores prefix optima for each allowed bin count, maximizing
the sum of candidate interval scores in O(K m^2) time and O(K m) DP storage
(in addition to candidate-score storage), where m is the refined prebin count.
Run ``benchmarks/dp_fast_path/run.py`` and ``analyze.py`` for paired benchmarks.
Use ``--smoke`` for a small run; generated results are retained as CI artifacts.

The solver-only comparison of six versus unlimited final bins is available as
``python benchmarks/dp_fast_path/compare_bin_limits.py``. It disables the DP
count/time guards only in memory for benchmarking and records CP status and
objective bounds alongside timings. The performance workflow retains results
as artifacts; runtime is not a test assertion.
