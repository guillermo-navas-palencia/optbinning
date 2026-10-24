Optimal binning with binary target
==================================

.. autoclass:: optbinning.OptimalBinning
   :members:
   :inherited-members:
   :show-inheritance:

Custom numerical prebinning
--------------------------

A custom provider proposes candidate cuts; OptBinning still refines these cuts
and selects the final partition using its usual objective and constraints.
The same interface is available on ``ContinuousOptimalBinning``.
The provider must be sklearn-cloneable and implement
``fit(x, y, sample_weight=None)`` on one-dimensional arrays, exposing ``splits_``.

For example, an unsupervised provider can ignore ``y`` while honoring weights::

    import numpy as np
    from sklearn.base import BaseEstimator
    from optbinning import OptimalBinning

    class WeightedQuantilePrebins(BaseEstimator):
        def __init__(self, n_bins=5):
            self.n_bins = n_bins

        def fit(self, x, y, sample_weight=None):
            weights = (np.ones(len(x)) if sample_weight is None
                       else np.asarray(sample_weight))
            order = np.argsort(x)
            values, weights = x[order], weights[order]
            keep = weights > 0
            values, weights = values[keep], weights[keep]
            if len(values) == 0:
                raise ValueError("Positive training weight is required.")
            cumulative = np.cumsum(weights) / np.sum(weights)
            cuts = np.interp(np.arange(1, self.n_bins) / self.n_bins,
                             cumulative, values)
            self.splits_ = np.unique(cuts[(cuts > values[0]) &
                                         (cuts < values[-1])])
            return self

    x = np.repeat(np.arange(20.), 4)
    y = np.tile([0, 1, 0, 1], 20)
    model = OptimalBinning(
        prebinning_estimator=WeightedQuantilePrebins(n_bins=5))
    model.fit(x, y)

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
