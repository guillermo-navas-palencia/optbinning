"""CP construction and complete fitting; setup is outside measured time."""
import numpy as np

from optbinning.binning.cp import BinningCP
from optbinning.binning.continuous_cp import ContinuousBinningCP
from optbinning.binning.multiclass_cp import MulticlassBinningCP
from optbinning.binning.multidimensional.cp_2d import Binning2DCP


def construction_cases():
    """Yield (identity, unfitted model, build arguments); 112 cases total."""
    rng = np.random.default_rng(84)
    for target in ["binary", "continuous"]:
        for n in [8, 20]:
            ne = rng.integers(10, 100, n)
            e = rng.integers(10, 100, n)
            count = ne + e
            mean = rng.normal(size=n)
            for trend in [None, "ascending", "descending", "peak", "valley",
                          "convex", "concave"]:
                for gamma in [0, 0.1]:
                    key = [target, n, trend, gamma]
                    if target == "binary":
                        model = BinningCP(
                            trend, 2, 10, None, None, None, None, None, None,
                            0, None, "consecutive", gamma, None, 5)
                        args = ("iv", ne, e, None)
                    else:
                        model = ContinuousBinningCP(
                            trend, 2, 10, None, None, 0, None, "consecutive",
                            gamma, None, 5)
                        args = (count, count * mean,
                                count * (mean ** 2 + 1), None)
                    yield key, model, args

    rng = np.random.default_rng(91)
    for n in [5, 12]:
        events = rng.integers(5, 40, (n, 3))
        ne = events.sum(axis=1)[:, None] - events
        for trend in [None, "ascending", "descending", "peak", "valley"]:
            for gap in [0, 0.05]:
                model = MulticlassBinningCP(
                    [trend, None, None], 1, 4, None, None, gap, None,
                    "consecutive", None, 5)
                yield (["multiclass", n, trend, gap], model,
                       (ne, events, [None] * 3))

    # All rectangles of a 2 x 2 grid, including single cells and the full grid.
    rectangles = [(a, b, c, d) for a in range(2) for b in range(a + 1, 3)
                  for c in range(2) for d in range(c + 1, 3)]
    cells = [{2 * x + y for x in range(a, b) for y in range(c, d)}
             for a, b, c, d in rectangles]
    columns = [[i for i, group in enumerate(cells) if j in group]
               for j in range(4)]
    dx = {i: [j for j, s in enumerate(rectangles)
              if r[1] == s[0] and max(r[2], s[2]) < min(r[3], s[3])]
          for i, r in enumerate(rectangles)}
    dy = {i: [j for j, s in enumerate(rectangles)
              if r[3] == s[2] and max(r[0], s[0]) < min(r[1], s[1])]
          for i, r in enumerate(rectangles)}
    counts = np.array([20 * len(c) for c in cells])
    weights = rng.integers(1, 10000, len(rectangles))
    for target, values in [("binary", rng.uniform(0.1, 0.9, 9)),
                           ("continuous", rng.normal(size=9))]:
        for tx in [None, "ascending", "descending"]:
            for ty in [None, "ascending", "descending"]:
                for gamma in [0, 0.1]:
                    model = Binning2DCP(tx, ty, 1, 4, 0, 0, gamma, 1, 5)
                    args = (4, 9, columns, weights, dx, dy, values, counts)
                    yield ["2d", target, tx, ty, gamma], model, args


def _label(key):
    return "/".join(map(str, key))


class Construction:
    params = [[_label(key) for key, _, _ in construction_cases()]]
    param_names = ["case"]
    repeat = 7
    number = 1
    warmup_time = 0

    def setup(self, case):
        _, self.model, self.args = next(
            row for row in construction_cases() if _label(row[0]) == case)

    def time_build(self, case):
        self.model.build_model(*self.args)


class CompleteFit:
    params = [["binary", "continuous"], ["numerical", "categorical"],
              [None, "auto"]]
    param_names = ["target", "feature", "shape"]
    repeat = 7
    number = 1
    warmup_time = 0

    def setup(self, target, feature, shape):
        from sklearn.datasets import load_breast_cancer, load_diabetes
        from optbinning import OptimalBinning, ContinuousOptimalBinning

        data = load_breast_cancer() if target == "binary" else load_diabetes()
        self.x = data.data[:, 0].copy()
        self.y = data.target
        if feature == "categorical":
            cuts = np.unique(np.quantile(self.x, np.linspace(0, 1, 13)[1:-1]))
            self.x = np.array([f"c{i}" for i in np.digitize(self.x, cuts)])
        self.estimator = (OptimalBinning if target == "binary"
                          else ContinuousOptimalBinning)
        self.settings = dict(dtype=feature, monotonic_trend=shape,
                             max_n_prebins=20, max_n_bins=8, time_limit=5)
        self.model = self.estimator(**self.settings)
        # Warm up this fitting path before the timed fit.
        self.model.fit(self.x, self.y)

    def time_fit(self, target, feature, shape):
        self.model.fit(self.x, self.y)

    def teardown(self, target, feature, shape):
        # A time-limited fit must not look like a completed-fit speedup.
        assert self.model.status == "OPTIMAL", self.model.status
        metric = "event_rate" if target == "binary" else "mean"
        assert np.isfinite(self.model.transform(self.x, metric=metric)).all()
