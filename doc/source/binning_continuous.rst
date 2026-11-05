Optimal binning with continuous target
======================================

.. autoclass:: optbinning.ContinuousOptimalBinning
   :members:
   :inherited-members:
   :show-inheritance:

Custom numerical prebinning
--------------------------

CART remains the default for ``ContinuousOptimalBinning()``. To use k-means
instead, define ``KMeansPrebins`` from the example in :doc:`binning_binary`
and pass it explicitly::

    from optbinning import ContinuousOptimalBinning

    model = ContinuousOptimalBinning(
        prebinning_estimator=KMeansPrebins(n_bins=10),
        max_n_prebins=10,
        min_bin_size=0.05,
    )
    model.fit(x, y)

The same provider contract, cleaning/refinement pipeline and supported
combinations apply. ``min_bin_size`` constrains the final bins; k-means does
not enforce ``min_prebin_size``.
