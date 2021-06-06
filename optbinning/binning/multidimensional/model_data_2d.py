"""
Model data for optimal binning 2D formulations.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import numpy as np

from scipy.special import xlogy


def _connected_rectangles(m, n, n_rectangles, monotonicity_x, monotonicity_y,
                          rows, cols, outer_h, outer_v):

    d_connected_x = None
    d_connected_y = None

    if monotonicity_x is not None:
        connected_x = {n * i + j: n * i + j + 1 for i in range(m)
                       for j in range(n - 1)}
        d_connected_x = {i: [] for i in range(n_rectangles)}

    if monotonicity_y is not None:
        connected_y = {n * i + j: n * i + j + n for i in range(m - 1)
                       for j in range(n)}
        d_connected_y = {i: [] for i in range(n_rectangles)}

    set_pos = set(range(n_rectangles))
    for i in range(n_rectangles):
        setr = set_pos - set().union(*(cols[r] for r in rows[i]))

        if monotonicity_x:
            hh = (connected_x[r] for r in outer_h[i] if r in connected_x)
            seth = set().union(*(cols[h] for h in hh))
            d_connected_x[i] = seth & setr

        if monotonicity_y:
            vv = (connected_y[r] for r in outer_v[i] if r in connected_y)
            setv = set().union(*(cols[v] for v in vv))
            d_connected_y[i] = setv & setr

    return d_connected_x, d_connected_y


def model_data(NE, E, monotonicity_x, monotonicity_y, scale, min_bin_size,
               max_bin_size, min_bin_n_event, max_bin_n_event,
               min_bin_n_nonevent, max_bin_n_nonevent):
    # Compute all rectangles and event and non-event records
    m, n = E.shape
    n_grid = m * n

    fe = np.ravel(E)
    fne = np.ravel(NE)

    n_fe = []
    n_fne = []

    rows = []
    cols = {c: [] for c in range(n_grid)}
    outer_h = []
    outer_v = []

    # Auxiliary checks
    is_min_bin_size = min_bin_size is not None
    is_max_bin_size = max_bin_size is not None
    is_min_bin_n_event = min_bin_n_event is not None
    is_max_bin_n_event = max_bin_n_event is not None
    is_min_bin_n_nonevent = min_bin_n_nonevent is not None
    is_max_bin_n_nonevent = max_bin_n_nonevent is not None

    # Cached rectangle shapes
    cached_rectangles = {}

    for k in range(1, m + 1):
        for l in range(1, n + 1):
            row = [n * ik + jl for ik in range(k) for jl in range(l)]
            cached_rectangles[(k, l)] = row

    n_rectangles = 0
    for i in range(m):
        for j in range(n):
            w = n - j
            h = m - i
            p = i * n + j
            for k in range(1, h + 1):
                for l in range(1, w + 1):
                    srow = cached_rectangles[(k, l)]
                    row = [p + r for r in srow]

                    sfe = fe[row].sum()
                    sfne = fne[row].sum()
                    
                    if sfe == 0 or sfne == 0:
                        continue

                    sn = sfe + sfne

                    if is_min_bin_size and sn < min_bin_size:
                        continue
                    elif is_max_bin_size and sn > max_bin_size:
                        continue
                    elif is_min_bin_n_event and sfe < min_bin_n_event:
                        continue
                    elif is_max_bin_n_event and sfe > max_bin_n_event:
                        continue
                    elif is_min_bin_n_nonevent and sfne < min_bin_n_nonevent:
                        continue
                    elif is_max_bin_n_nonevent and sfne > max_bin_n_nonevent:
                        continue

                    for r in row:
                        cols[r].append(n_rectangles)
    
                    if monotonicity_x is not None:
                        outer_h.append(
                            [row[_i * l + (l - 1)] for _i in range(k)])

                    if monotonicity_y is not None:
                        outer_v.append(
                            [row[(k - 1) * l + _j] for _j in range(l)])

                    rows.append(row)
                    n_fe.append(sfe)
                    n_fne.append(sfne)

                    n_rectangles += 1

    n_event = np.array(n_fe)
    n_nonevent = np.array(n_fne)

    # Connected rectangles
    if monotonicity_x is not None or monotonicity_y is not None:
        d_connected_x, d_connected_y = _connected_rectangles(
            m, n, n_rectangles, monotonicity_x, monotonicity_y, rows, cols,
            outer_h, outer_v)
    else:
        d_connected_x = None
        d_connected_y = None

    # Event and non-event rate
    n_records = n_event + n_nonevent

    # Event rate and Information value
    event_rate = np.zeros(n_rectangles)
    iv = np.zeros(n_rectangles)

    event_rate = n_event / n_records
    p = n_event / E.sum()
    q = n_nonevent / NE.sum()
    iv = xlogy(p - q, p / q)

    if scale is not None:
        V = (iv * scale).astype(int)
    else:
        V = iv

    return (n_grid, n_rectangles, rows, cols, V, d_connected_x,
            d_connected_y, iv, event_rate, n_event, n_nonevent, n_records)


def continuous_model_data(R, S, monotonicity_x, monotonicity_y, scale,
                          metric_norm="l1"):
    # Compute all rectangles, and sums and number of records
    m, n = R.shape
    n_grid = m * n

    # Compute mean
    M = np.zeros((m, n))
    mask_R = R > 0
    M[mask_R] = S[mask_R] / R[mask_R]

    MM = M.flatten()

    fs = np.ravel(S)
    fr = np.ravel(R)

    n_s = []
    n_r = []

    mean = []
    norm = []

    rows = []
    cols = {c: [] for c in range(n_grid)}
    outer_h = []
    outer_v = []

    n_rectangles = 0
    for i in range(m):
        for j in range(n):
            for k in range(i + 1, m + 1):
                for l in range(j + 1, n + 1):
                    row = [n * ik + jl for ik in range(i, k)
                           for jl in range(j, l)]
                    rows.append(row)

                    for r in row:
                        cols[r].append(n_rectangles)

                    if monotonicity_x is not None:
                        outer_h.append([n * ik + (l-1) for ik in range(i, k)])

                    if monotonicity_y is not None:
                        outer_v.append([n * (k-1) + jl for jl in range(j, l)])

                    s = sum(fs[row])
                    r = sum(fr[row])
                    n_s.append(s)
                    n_r.append(r)

                    u = s / r if r else 0
                    mean.append(u)

                    if metric_norm == "l1":
                        norm.append(np.absolute(MM[row] - u).sum())
                    else:
                        norm.append(np.square(MM[row] - u).sum())

                    n_rectangles += 1

    sums = np.array(n_s)
    n_records = np.array(n_r)
    mean = np.array(mean)
    norm = np.array(norm)

    # Connected rectangles
    if monotonicity_x is not None or monotonicity_y is not None:
        d_connected_x, d_connected_y = _connected_rectangles(
            m, n, n_rectangles, monotonicity_x, monotonicity_y, rows, cols,
            outer_h, outer_v)
    else:
        d_connected_x = None
        d_connected_y = None

    V = (norm * scale).astype(np.int64)

    return (n_grid, n_rectangles, cols, V, d_connected_x, d_connected_y,
            mean, sums, n_records)
