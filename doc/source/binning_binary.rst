Optimal binning with binary target
==================================

.. autoclass:: optbinning.OptimalBinning
   :members:
   :inherited-members:
   :show-inheritance:

Custom numerical prebinning
--------------------------

CART remains the default: ``OptimalBinning()`` still uses
``prebinning_method="cart"`` and ``prebinning_estimator=None``. Existing
calls keep their current prebinning behavior. K-means is an explicit opt-in.

Use k-means instead of CART
~~~~~~~~~~~~~~~~~~~~~~~~~~
Define the small adapter below. It uses scikit-learn's standard
`KBinsDiscretizer <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html>`_
with ``strategy="kmeans"``, forwards sample weights, and exposes the interior
bin edges as candidate cuts. It uses only the predictor, not the target.

.. literalinclude:: ../../examples/kmeans_prebinning.py
   :language: python
   :end-before: def main():

With your training arrays ``x`` and ``y``, choose between the default and
k-means as follows::

    # Default: CART prebinning, unchanged.
    default_model = OptimalBinning()
    default_model.fit(x, y)

    # Opt in to k-means candidate cuts; keep the usual final solver.
    kmeans_model = OptimalBinning(
        prebinning_estimator=KMeansPrebins(n_bins=10),
        max_n_prebins=10,
        min_bin_size=0.05,
    )
    kmeans_model.fit(x, y)

The provider's ``n_bins`` controls the requested number of candidate bins;
``max_n_prebins`` is OptBinning's upper limit. The final solver may merge
candidates. K-means does not guarantee a minimum cluster size:
``min_bin_size=0.05`` constrains final bins, while ``min_prebin_size`` remains
a setting for built-in prebinners. Opt into this adapter with
``prebinning_estimator=KMeansPrebins(...)``, not
``prebinning_method="kmeans"``.

To run a complete example on generated data, execute
``python examples/kmeans_prebinning.py`` from the repository root.

Custom provider interface
~~~~~~~~~~~~~~~~~~~~~~~~~
A custom provider proposes candidate cuts; OptBinning still refines these cuts
and selects the final partition using its usual objective and constraints.
The same interface is available on ``ContinuousOptimalBinning``.
The provider must be sklearn-cloneable and implement
``fit(x, y, sample_weight=None)`` on one-dimensional arrays, exposing ``splits_``.

Only the clean numerical training rows reach the provider: missing values,
special codes and configured outliers are handled by the existing preprocessing
path. A cloned fitted provider is available as ``prebinning_estimator_``; the
original is not fitted. The provider must return strictly increasing finite
cuts inside the training range, with no more than ``max_n_prebins - 1`` cuts.
An empty cut vector represents one candidate bin.

``prebinning_method`` is ignored when a provider is supplied. The provider is
responsible for its own prebin support policy: ``min_prebin_size`` is a built-in
prebinner setting, not a custom-provider constraint. Final ``min_bin_size`` and
other optimization constraints still apply. Categorical inputs, ``user_splits``,
``class_weight`` and additional prebinning keyword arguments are rejected rather
than silently ignored. Providers must accept ``sample_weight`` even if the
caller did not supply weights. This API does not change the supervised
optimization objective or introduce a new unsupervised estimator.

Pickle preserves the fitted provider when its class is importable. JSON remains
an inference-only binning-table export; it does not serialize arbitrary provider
code or support reconstructing a refittable provider.

Provider policy and integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The provider owns the candidate-grid policy; OptBinning owns final binning.
For example, set a provider's minimum row fraction to 5% if its candidate
prebins must each contain at least 5% of the clean training rows. Setting
OptBinning's ``min_prebin_size=0.05`` alone does not constrain a custom
provider. OptBinning does not silently merge custom cuts to enforce that
setting. Its normal refinement can still remove cuts, and ``min_bin_size``
constrains the final partition.

Choose the provider's bin budget consistently with ``max_n_prebins``;
OptBinning rejects excess cuts rather than truncating them. Parameters can be
changed with ``set_params(prebinning_estimator__n_bins=...)``. In a
``BinningProcess``, pass the provider through the variable's
``binning_fit_params``; each variable fits its own clone.
