"""
Model data for optimal binning formulations.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019


import numpy as np

from scipy import stats

from .metrics import jeffrey
from .metrics import jensen_shannon
from .metrics import hellinger
from .metrics import triangular


def test_proportions(e1, ne1, e2, ne2, zscore):
    n1 = e1 + ne1
    n2 = e2 + ne2
    p1 = e1 / n1
    p2 = e2 / n2
    p = (e1 + e2) / (n1 + n2)

    z = (p1 - p2) / np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return abs(z) < zscore


def find_pvalue_violation_indices(n, E, NE, max_pvalue, max_pvalue_policy):
    pvalue_violation_indices = []
    zscore = stats.norm.ppf(1.0 - max_pvalue / 2)

    if max_pvalue_policy == "all":
        for i in range(n - 1):
            for r in range(i + 1):
                ev = E[i][r]
                nev = NE[i][r]
                for j in range(i + 1, n):
                    for k in range(i + 1, j + 1):
                        ev2 = E[j][k]
                        nev2 = NE[j][k]
                        if test_proportions(ev, nev, ev2, nev2, zscore):
                            pvalue_violation_indices.append(([i, r], [j, k]))

    elif max_pvalue_policy == "consecutive":
        for i in range(n - 1):
            for r in range(i + 1):
                ev = E[i][r]
                nev = NE[i][r]
                for j in range(i + 1, n):
                    ev2 = E[j][i + 1]
                    nev2 = NE[j][i + 1]
                    if test_proportions(ev, nev, ev2, nev2, zscore):
                        pvalue_violation_indices.append(([i, r], [j, i+1]))

    return pvalue_violation_indices


def find_pvalue_violation_indices_continuous(n, U, S, R, max_pvalue,
                                             max_pvalue_policy):
    pvalue_violation_indices = []

    if max_pvalue_policy == "all":
        for i in range(n - 1):
            for t in range(i + 1):
                u = U[i][t]
                s = S[i][t]
                r = R[i][t]
                for j in range(i + 1, n):
                    for k in range(i + 1, j + 1):
                        u2 = U[j][k]
                        s2 = S[j][k]
                        r2 = R[j][k]
                        if stats.ttest_ind_from_stats(
                                u, s, r, u2, s2, r2, False)[1] > max_pvalue:
                            pvalue_violation_indices.append(([i, t], [j, k]))

    elif max_pvalue_policy == "consecutive":
        for i in range(n - 1):
            for k in range(i + 1):
                u = U[i][k]
                s = S[i][k]
                r = R[i][k]
                for j in range(i + 1, n):
                    u2 = U[j][i + 1]
                    s2 = S[j][i + 1]
                    r2 = R[j][i + 1]
                    if stats.ttest_ind_from_stats(
                            u, s, r, u2, s2, r2, False)[1] > max_pvalue:
                        pvalue_violation_indices.append(([i, k], [j, i+1]))

    return pvalue_violation_indices


def model_data(divergence, n_nonevent, n_event, max_pvalue, max_pvalue_policy,
               scale=None, return_nonevent_event=False):
    n = len(n_nonevent)

    t_n_event = n_event.sum()
    t_n_nonevent = n_nonevent.sum()

    D = []
    V = []

    E = []
    NE = []

    for i in range(1, n + 1):
        s_event = n_event[:i][::-1].cumsum()[::-1]
        s_nonevent = n_nonevent[:i][::-1].cumsum()[::-1]
        rate = s_event / (s_nonevent + s_event)

        p = s_event / t_n_event
        q = s_nonevent / t_n_nonevent

        if divergence == "iv":
            iv = jeffrey(p, q)
        elif divergence == "js":
            iv = jensen_shannon(p, q)
        elif divergence == "hellinger":
            iv = hellinger(p, q)
        elif divergence == "triangular":
            iv = triangular(p, q)

        if scale is not None:
            rate *= scale
            iv *= scale

            D.append(rate.astype(np.int64))
            V.append(iv.astype(np.int64))
        else:
            D.append(rate)
            V.append(iv)

        if max_pvalue is not None or return_nonevent_event:
            E.append(s_event)
            NE.append(s_nonevent)

    if max_pvalue is not None:
        pvalue_violation_indices = find_pvalue_violation_indices(
            n, E, NE, max_pvalue, max_pvalue_policy)
    else:
        pvalue_violation_indices = []

    if return_nonevent_event:
        return D, V, NE, E, pvalue_violation_indices

    return D, V, pvalue_violation_indices


def multiclass_model_data(n_nonevent, n_event, max_pvalue, max_pvalue_policy,
                          scale=None):

    n, n_classes = n_nonevent.shape

    DD = []
    PV = []
    VV = []

    for c in range(n_classes):
        t_n_event = n_event[:, c].sum()
        t_n_nonevent = n_nonevent[:, c].sum()

        D = []
        V = []

        E = []
        NE = []

        for i in range(1, n + 1):
            s_event = n_event[:i, c][::-1].cumsum()[::-1]
            s_nonevent = n_nonevent[:i, c][::-1].cumsum()[::-1]
            rate = s_event / (s_nonevent + s_event)

            p = s_event / t_n_event
            q = s_nonevent / t_n_nonevent
            iv = jeffrey(p, q)

            if scale is not None:
                rate *= scale
                iv *= scale

                rate = rate.astype(np.int64)
                iv = iv.astype(np.int64)

            D.append(rate)
            V.append(iv)

            if max_pvalue is not None:
                E.append(s_event)
                NE.append(s_nonevent)

        if max_pvalue is not None:
            pvalue_violation_indices = find_pvalue_violation_indices(
                n, E, NE, max_pvalue, max_pvalue_policy)
        else:
            pvalue_violation_indices = []

        DD.append(D)
        VV.append(V)
        PV.append(pvalue_violation_indices)

    return DD, VV, PV


def continuous_model_data(n_records, sums, ssums, max_pvalue,
                          max_pvalue_policy, scale=None):

    n = len(n_records)

    U = []
    UP = []
    S = []
    R = []
    V = []

    t_mean = sums.sum() / n_records.sum()

    for i in range(1, n + 1):
        s_n_records = n_records[:i][::-1].cumsum()[::-1]
        s_sums = sums[:i][::-1].cumsum()[::-1]
        s_ssums = ssums[:i][::-1].cumsum()[::-1]

        mean = s_sums / s_n_records
        std = np.sqrt(s_ssums / s_n_records - mean ** 2)
        norm = np.absolute(mean - t_mean)

        if scale is not None:
            mean_scaled = mean * scale
            norm_scaled = norm * scale

            mean_scaled = mean_scaled.astype(np.int64)
            norm_scaled = norm_scaled.astype(np.int64)

            U.append(mean_scaled)
            V.append(norm_scaled)
        else:
            U.append(mean)
            V.append(norm)

        if max_pvalue is not None:
            UP.append(mean)
            R.append(s_n_records)
            S.append(std)

    if max_pvalue is not None:
        pvalue_violation_indices = find_pvalue_violation_indices_continuous(
            n, UP, S, R, max_pvalue, max_pvalue_policy)
    else:
        pvalue_violation_indices = []

    return U, V, pvalue_violation_indices
