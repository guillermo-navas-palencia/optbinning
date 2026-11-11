"""Deterministic CP construction cases (no downloaded or private data)."""
import hashlib

import numpy as np
from google.protobuf import text_format
from ortools.sat import cp_model_pb2

from optbinning.binning.cp import BinningCP
from optbinning.binning.continuous_cp import ContinuousBinningCP
from optbinning.binning.multiclass_cp import MulticlassBinningCP
from optbinning.binning.multidimensional.cp_2d import Binning2DCP


def canonical_hash(model):
    """Normalize only full linear-constraint sign reversal, not constraints."""
    proto = text_format.Parse(str(model.Proto()), cp_model_pb2.CpModelProto())
    for constraint in proto.constraints:
        if not constraint.HasField("linear"):
            continue
        linear = constraint.linear
        if next((v for v in linear.coeffs if v), 0) < 0:
            linear.coeffs[:] = [-v for v in linear.coeffs]

            def negate(value):
                if value == -(2 ** 63):
                    return 2 ** 63 - 1
                if value == 2 ** 63 - 1:
                    return -(2 ** 63)
                return -value

            linear.domain[:] = [negate(v) for v in reversed(linear.domain)]
    return hashlib.sha256(
        proto.SerializeToString(deterministic=True)).hexdigest()


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
