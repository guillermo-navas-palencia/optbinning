"""Restricted exact integer-IV partition solver; None requests CP fallback."""
import time
import numpy as np
from .model_data import model_data


def solve_iv(n_nonevent, n_event, min_n_bins, max_n_bins,
             min_bin_size, max_bin_size, time_limit):
    started = time.perf_counter()
    n = len(n_event)
    a, b = np.asarray(n_nonevent), np.asarray(n_event)
    if (n < 2 or n > 100 or a.shape != b.shape or a.ndim != 1
            or not np.isfinite(a).all() or not np.isfinite(b).all()
            or (a <= 0).any() or (b <= 0).any()
            or not np.equal(a, np.floor(a)).all()
            or not np.equal(b, np.floor(b)).all()
            or float(np.sum(a, dtype=float) + np.sum(b, dtype=float)) >= 2**53 or time_limit <= 0):
        return None
    deadline = started + min(.1, time_limit / 10)
    _, values, _, _ = model_data('iv', a, b, None, 'consecutive', 0, 1000000)
    low, high = min_n_bins or 1, min(max_n_bins or n, n)
    support = np.r_[0, np.cumsum(a + b)]
    # Python integers preserve the CP quantized objective without float sums.
    dp = [[None] * (n + 1) for _ in range(high + 1)]
    previous = [[None] * (n + 1) for _ in range(high + 1)]
    dp[0][0] = 0
    for k in range(1, high + 1):
        if time.perf_counter() > deadline:
            return None
        for stop in range(k, n + 1):
            if time.perf_counter() > deadline:
                return None
            for start in range(k - 1, stop):
                if dp[k-1][start] is None:
                    continue
                count = support[stop] - support[start]
                if ((min_bin_size is not None and count < min_bin_size)
                        or (max_bin_size is not None and count > max_bin_size)):
                    continue
                score = dp[k-1][start] + int(values[stop-1][start])
                if dp[k][stop] is None or score > dp[k][stop]:
                    dp[k][stop], previous[k][stop] = score, start
    if time.perf_counter() > deadline:
        return None
    candidates = [(dp[k][n], k) for k in range(low, high + 1)
                  if dp[k][n] is not None]
    solution = np.zeros(n, dtype=bool)
    if not candidates:
        solution[-1] = True
        return 'INFEASIBLE', solution, None, time.perf_counter() - started
    # Equal objective: fewer bins, then first predecessor encountered.
    score, k = max(candidates, key=lambda v: (v[0], -v[1]))
    stop = n
    while k:
        solution[stop-1] = True
        stop = previous[k][stop]
        k -= 1
    return 'OPTIMAL', solution, score, time.perf_counter() - started
