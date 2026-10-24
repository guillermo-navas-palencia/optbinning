Fractional binary sample weights
================================

Binary OptimalBinning retains fractional sample weights as mass through
prebinning, optimization, tables and JSON, instead of truncating them to
integer counts. CP and MIP support constraints use effective mass.

Class weights multiply sample weights once, including CART prebinning. The
caller's weight array is unchanged. ``class_weight="balanced"`` uses class
frequencies, not exposure totals. Relative support uses the combined mass.

.. code-block:: python

   model = OptimalBinning(solver="cp", min_bin_size=0.05)
   model.fit(x_train, y_train, sample_weight=exposure)

Fractional mass constraints are implemented by excluding infeasible contiguous
candidates, so CP/BOP receive Boolean constraints without rounding mass. A
scale-aware floating-point tolerance handles summation at exact support bounds;
it does not intentionally relax substantive violations. Nonzero gamma with
fractional effective weights is explicitly rejected. CP and MIP (BOP/CBC) are
covered; optional LocalSolver is not certified by these tests.

This patch does not enable pure bins, change their default treatment or add
exposure-specific calibration. Exposure-weighted event rates estimate a
different quantity from unweighted rates; appropriate interpretation requires
the modeling use case. Pure-bin support remains a separate independent PR.

Tests cover mass conservation, empirical rates, missing/special values, JSON,
CP/MIP support bounds, class/sample weight composition and wide weight scales.
