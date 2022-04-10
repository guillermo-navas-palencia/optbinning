"""
Model data for optimal binning 2D formulations using CART pruning.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2021

import numpy as np

from ..metrics import jeffrey
from ..metrics import jensen_shannon
from ..metrics import hellinger
from ..metrics import triangular
from .model_data_2d import _connected_rectangles


def refine_rectangles(rectangles):
    final_rectangles = []
    for rectangle in rectangles:
        rectangle_path = []
        for step in rectangle:
            split, threshold, feature = step[1:]
            path = (feature, split, threshold)
            rectangle_path.append(path)

        final_rectangles.append(rectangle_path)

    return final_rectangles


def get_rectangles(tree, feature_names):
    left = tree.tree_.children_left
    right = tree.tree_.children_right
    threshold = tree.tree_.threshold

    features = [feature_names[i] for i in tree.tree_.feature]

    # get ids of child nodes
    idx = np.argwhere(left == -1)[:, 0]

    def recurse(left, right, child, lineage=None):
        if lineage is None:
            lineage = [child]
        if child in left:
            parent = np.where(left == child)[0].item()
            split = '<='
        else:
            parent = np.where(right == child)[0].item()
            split = '>'

        lineage.append((parent, split, threshold[parent], features[parent]))

        if parent == 0:
            lineage.reverse()
            return lineage
        else:
            return recurse(left, right, parent, lineage)

    rectangles = []
    for child in idx:
        rectangle = []
        for node in recurse(left, right, child):
            if isinstance(node, tuple):
                rectangle.append(node)
        rectangles.append(rectangle)

    final_rectangles = refine_rectangles(rectangles)

    return final_rectangles


def get_auxiliary_matrices(tree, m, n, n_grid):
    cart_rectangles = get_rectangles(tree, [0, 1])

    A = np.empty((m, n), dtype=int)
    G = np.arange(n_grid).reshape((m, n)).astype(int)
    idx_x = np.arange(n)
    idx_y = np.arange(m)

    for i, rectangle in enumerate(cart_rectangles):
        mask_x = np.ones(n, dtype=bool)
        mask_y = np.ones(m, dtype=bool)

        for path in rectangle:
            if path[1] == "<=":
                if path[0] == 0:
                    mask_x &= idx_x <= path[2]
                else:
                    mask_y &= idx_y <= path[2]
            else:
                if path[0] == 0:
                    mask_x &= idx_x > path[2]
                else:
                    mask_y &= idx_y > path[2]

        mask = np.ix_(mask_y, mask_x)
        A[mask] = i

    return A, G


def model_data_cart(tree, divergence, NE, E, monotonicity_x, monotonicity_y,
                    scale, min_bin_size, max_bin_size, min_bin_n_event,
                    max_bin_n_event, min_bin_n_nonevent, max_bin_n_nonevent):

    m, n = E.shape
    n_grid = m * n

    fe = np.ravel(E)
    fne = np.ravel(NE)

    n_fe = []
    n_fne = []

    rows = []
    cols = {c: [] for c in range(n_grid)}
    outer_x = []
    outer_y = []

    # Auxiliary checks
    is_min_bin_size = min_bin_size is not None
    is_max_bin_size = max_bin_size is not None
    is_min_bin_n_event = min_bin_n_event is not None
    is_max_bin_n_event = max_bin_n_event is not None
    is_min_bin_n_nonevent = min_bin_n_nonevent is not None
    is_max_bin_n_nonevent = max_bin_n_nonevent is not None

    # Auxiliary matrices
    A, G = get_auxiliary_matrices(tree, m, n, n_grid)

    paths = set()

    n_rectangles = 0
    rectangles = []
    for i in range(m):
        for j in range(n):
            for k in range(i + 1, m + 1):
                for l in range(j + 1, n + 1):
                    path = tuple(sorted(set(A[i:k, j:l].ravel())))
                    if path not in paths:
                        if len(path) > 1:
                            M = np.isin(A, path)
                            sr = list(M.sum(axis=0)) + [0]
                            sc = list(M.sum(axis=1)) + [0]

                            if len(set(sr)) <= 2 and len(set(sc)) <= 2:
                                row = G[M].ravel()

                                sfe = fe[row].sum()
                                sfne = fne[row].sum()

                                if sfe == 0 or sfne == 0:
                                    continue

                                sn = sfe + sfne

                                if is_min_bin_size and sn < min_bin_size:
                                    continue
                                elif is_max_bin_size and sn > max_bin_size:
                                    continue
                                elif (is_min_bin_n_event and
                                        sfe < min_bin_n_event):
                                    continue
                                elif (is_max_bin_n_event and
                                        sfe > max_bin_n_event):
                                    continue
                                elif (is_min_bin_n_nonevent and
                                        sfne < min_bin_n_nonevent):
                                    continue
                                elif (is_max_bin_n_nonevent and
                                        sfne > max_bin_n_nonevent):
                                    continue

                                for r in row:
                                    cols[r].append(n_rectangles)

                                if monotonicity_x is not None:
                                    out = np.array([G[M]])
                                    outer_x.append(list(out[:, -1]))

                                if monotonicity_y is not None:
                                    out = np.array([G[M]])
                                    outer_y.append(list(out[-1, :]))

                                rows.append(row)
                                n_fe.append(sfe)
                                n_fne.append(sfne)

                                n_rectangles += 1

                                rectangles.append(path)
                        else:
                            rectangles.append(path)

                        paths.add(path)

    n_event = np.array(n_fe)
    n_nonevent = np.array(n_fne)

    # Connected rectangles
    if monotonicity_x is not None or monotonicity_y is not None:
        d_connected_x, d_connected_y = _connected_rectangles(
            m, n, n_rectangles, monotonicity_x, monotonicity_y, rows, cols,
            outer_x, outer_y)
    else:
        d_connected_x = None
        d_connected_y = None

    # Event and non-event rate
    n_records = n_event + n_nonevent

    # Event rate and Information value
    event_rate = n_event / n_records
    p = n_event / E.sum()
    q = n_nonevent / NE.sum()

    if divergence == "iv":
        iv = jeffrey(p, q)
    elif divergence == "js":
        iv = jensen_shannon(p, q)
    elif divergence == "hellinger":
        iv = hellinger(p, q)
    elif divergence == "triangular":
        iv = triangular(p, q)

    if scale is not None:
        c = (iv * scale).astype(int)
    else:
        c = iv

    return (n_grid, n_rectangles, rows, cols, c, d_connected_x, d_connected_y,
            event_rate, n_event, n_nonevent, n_records)


def continuous_model_data_cart(tree, R, S, SS, monotonicity_x, monotonicity_y,
                               scale, min_bin_size, max_bin_size):

    # Compute all rectangles and event and non-event records
    m, n = R.shape
    n_grid = m * n

    fr = np.ravel(R)
    fs = np.ravel(S)
    fss = np.ravel(SS)

    n_fr = []
    n_fs = []
    n_fss = []

    rows = []
    cols = {c: [] for c in range(n_grid)}
    outer_x = []
    outer_y = []

    # Auxiliary checks
    is_min_bin_size = min_bin_size is not None
    is_max_bin_size = max_bin_size is not None

    # Auxiliary matrices
    A, G = get_auxiliary_matrices(tree, m, n, n_grid)

    paths = set()

    n_rectangles = 0
    rectangles = []
    for i in range(m):
        for j in range(n):
            for k in range(i + 1, m + 1):
                for l in range(j + 1, n + 1):
                    path = tuple(sorted(set(A[i:k, j:l].ravel())))
                    if path not in paths:
                        if len(path) > 1:
                            M = np.isin(A, path)
                            sr = list(M.sum(axis=0)) + [0]
                            sc = list(M.sum(axis=1)) + [0]

                            if len(set(sr)) <= 2 and len(set(sc)) <= 2:
                                row = G[M].ravel()

                                sfr = fr[row].sum()
                                sfs = fs[row].sum()
                                sfss = fss[row].sum()

                                if sfr == 0:
                                    continue

                                if is_min_bin_size and sfr < min_bin_size:
                                    continue
                                elif is_max_bin_size and sfr > max_bin_size:
                                    continue

                                for r in row:
                                    cols[r].append(n_rectangles)

                                if monotonicity_x is not None:
                                    out = np.array([G[M]])
                                    outer_x.append(list(out[:, -1]))

                                if monotonicity_y is not None:
                                    out = np.array([G[M]])
                                    outer_y.append(list(out[-1, :]))

                                rows.append(row)
                                n_fr.append(sfr)
                                n_fs.append(sfs)
                                n_fss.append(sfss)

                                n_rectangles += 1

                                rectangles.append(path)
                        else:
                            rectangles.append(path)

                        paths.add(path)

    n_records = np.array(n_fr)
    sums = np.array(n_fs)
    ssums = np.array(n_fss)

    # Connected rectangles
    if monotonicity_x is not None or monotonicity_y is not None:
        d_connected_x, d_connected_y = _connected_rectangles(
            m, n, n_rectangles, monotonicity_x, monotonicity_y, rows, cols,
            outer_x, outer_y)
    else:
        d_connected_x = None
        d_connected_y = None

    # Mean and norm
    t_mean = S.sum() / R.sum()
    mean = sums / n_records
    stds = np.sqrt(ssums / n_records - mean ** 2)
    norm = np.absolute(mean - t_mean)

    if scale is not None:
        c = (norm * scale).astype(int)
    else:
        c = norm

    return (n_grid, n_rectangles, rows, cols, c, d_connected_x, d_connected_y,
            mean, n_records, sums, stds)
