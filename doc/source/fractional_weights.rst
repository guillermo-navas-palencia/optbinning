Fractional binary sample weights
================================

Binary ``OptimalBinning`` preserves fractional sample weights in fitting,
optimization, empirical transformations and JSON. Class weights multiply sample
weights once. ``class_weight="balanced"`` uses class frequencies, not weight
totals; the caller's weight array is not modified.

Observation support and weighted statistics
--------------------------------------------

``min_bin_size`` and ``max_bin_size`` refer to actual observation counts,
regardless of weights. Their denominator is the original input row count,
including missing and special rows; the bounds apply to regular optimized bins.
Both bounds retain the existing upward rounding convention. Explicit event/non-event
count constraints likewise refer to actual observations in each class.

For weighted binary fits, CART uses ``min_prebin_size`` as its default
actual-row minimum per leaf while retaining weighted impurity. Explicit
``prebinning_kwargs`` override CART defaults, as for other prebinning methods.
Unweighted fits retain their existing CART defaults. Other prebinning algorithms retain their
existing behavior; user-supplied cuts bypass CART's leaf-size constraint.
The existing single-bin shortcut also remains unchanged: it does not check
final-bin feasibility when prebinning leaves no choice of partition. Correcting
that shortcut across estimators is a separate change.
Weights still affect the objective, event rates, WoE, monotonicity and statistical
tests. This change does not add a separate weighted-mass size constraint.

For weighted fits, ``binning_table.build()`` returns raw ``Count``,
``Count (%)``, ``Non-event``, ``Event`` and ``Event rate`` columns, alongside
explicit ``Weighted count``, ``Weighted count (%)``, ``Weighted non-event``,
``Weighted event`` and ``Weighted event rate`` columns. WoE, IV and JS remain
weighted statistics, as do empirical transformations. Plots retain weighted
counts and rates, with explicit weighted labels. Unweighted tables retain their
existing columns. JSON roundtrips preserve both views; older JSON files remain
readable but their raw counts cannot be reconstructed from weighted masses.

.. code-block:: python

   model = OptimalBinning(solver="cp", min_bin_size=0.05)
   model.fit(x_train, y_train, sample_weight=exposure)
   table = model.binning_table.build()
   # At least 5% of input rows per regular bin, not 5% of exposure.
   print(table[["Count", "Weighted count", "Event rate", "Weighted event rate"]])

CP and MIP (BOP/CBC) are supported. Weighted LocalSolver fits are rejected rather
than silently retaining different size semantics. Fractional effective weights
with nonzero ``gamma`` remain unsupported and raise an explicit error. Pure-bin
support and exposure-specific calibration are outside this change.

Compatibility
-------------

Unweighted and integer-weighted counts retain integer dtypes; only fractional
weights require floating counts. Continuous and multiclass estimators retain
their existing prebinning behavior. Reusing a binary estimator resets its
pure-prebin flag, so a previous Hellinger/triangular fit cannot leave stale
class-support state for the next fit.
