"""Use scikit-learn's standard k-means discretizer as a custom prebinner."""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import KBinsDiscretizer

from optbinning import OptimalBinning


class KMeansPrebins(BaseEstimator):
    """Adapt 1D input and bin_edges_ to OptBinning's provider interface.

    K-means uses only x, not y. It does not enforce minimum prebin sizes.
    OptBinning still applies its usual refinement and final-bin constraints.
    """

    def __init__(self, n_bins=5, random_state=0):
        self.n_bins = n_bins
        self.random_state = random_state

    def fit(self, x, y=None, sample_weight=None):
        self.discretizer_ = KBinsDiscretizer(
            n_bins=self.n_bins, encode='ordinal', strategy='kmeans',
            subsample=None, random_state=self.random_state)
        self.discretizer_.fit(np.asarray(x).reshape(-1, 1),
                              sample_weight=sample_weight)
        # Exclude the two outer edges; OptBinning needs interior cuts only.
        self.splits_ = self.discretizer_.bin_edges_[0][1:-1].copy()
        return self


def main():
    rng = np.random.default_rng(42)
    x = rng.normal(size=1000)
    y = rng.binomial(1, 1 / (1 + np.exp(-x)))
    model = OptimalBinning(
        prebinning_estimator=KMeansPrebins(n_bins=10),
        max_n_prebins=10, min_bin_size=.05, monotonic_trend='ascending')
    model.fit(x, y)
    print('Candidate cuts:', model.prebinning_estimator_.splits_)
    print('Final cuts:', model.splits)
    print(model.binning_table.build())


if __name__ == '__main__':
    main()
