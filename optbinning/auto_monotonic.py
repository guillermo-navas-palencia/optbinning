"""
Automatic monotonic trend algorithm.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import numpy as np

from scipy.spatial import ConvexHull

from .metrics import jeffrey


def n_peaks_valleys(x):
    """
    Find number of peaks and valleys in an array of values.

    Parameters
    ----------
    x : array-like, shape = (n_samples)
        Data samples, where n_samples is the number of samples.

    Returns
    -------
    n_changes : number of changes (peaks and valleys).
    """
    diff_sign = np.sign(x[1:] - x[:-1])
    return np.count_nonzero(diff_sign[1:] != diff_sign[:-1])


def extreme_points_area(x):
    """
    Compute area within extreme points divided by total rectangular area.

    Parameters
    ----------
    x : array-like, shape = (n_samples)
        Data samples, where n_samples is the number of samples.

    Returns
    -------
    p_area : percentage of total area corresponding to are within
        extreme points.
    """
    n = len(x)

    pos_min = np.argmin(x)
    pos_max = np.argmax(x)

    x_iter = x[1:-1]
    if len(x_iter):
        xinit = 0
        xmin = pos_min
        xmax = pos_max
        xlast = n

        yinit = x[0]
        ymin = x[pos_min]
        ymax = x[pos_max]
        ylast = x[-1]

        triangle1 = np.array([[xinit, xmin, xmax],
                              [yinit, ymin, ymax], [1, 1, 1]])
        triangle2 = np.array([[xmin, xmax, xlast],
                              [ymin, ymax, ylast], [1, 1, 1]])

        area_1 = 0.5 * np.abs(np.linalg.det(triangle1))
        area_2 = 0.5 * np.abs(np.linalg.det(triangle2))
        sum_area = area_1 + area_2

        p_area = sum_area / ((ymax - ymin) * n)
    else:
        p_area = 0

    return p_area


def auto_monotonic_data(n_nonevent, n_event):
    n_prebins = len(n_nonevent)

    # Trend changes data
    n_records = n_nonevent + n_event
    event_rate = n_event / n_records
    n_trend_changes = n_peaks_valleys(event_rate)
    p_trend_changes = n_trend_changes / n_prebins

    # Linear regression coefficient sense
    lr_coef = np.polyfit(np.arange(n_prebins), event_rate, deg=1)[0]
    lr_sense = np.int(lr_coef > 0)

    # Breakpoints data
    pos_min = np.argmin(event_rate)
    pos_max = np.argmax(event_rate)
    min_event_rate = event_rate[pos_min]
    max_event_rate = event_rate[pos_max]

    n1 = n_prebins - 1
    p_bins_min_left = len(event_rate[:pos_min]) / n1
    p_bins_min_right = len(event_rate[pos_min+1:]) / n1

    p_bins_max_left = len(event_rate[:pos_max]) / n1
    p_bins_max_right = len(event_rate[pos_max+1:]) / n1

    total_records = np.sum(n_records)
    p_records_min_left = np.sum(n_records[:pos_min]) / total_records
    p_records_min_right = np.sum(n_records[pos_min+1:]) / total_records

    p_records_max_left = np.sum(n_records[:pos_max]) / total_records
    p_records_max_right = np.sum(n_records[pos_max+1:]) / total_records

    # Ratio triangular area approximation
    p_area = extreme_points_area(event_rate)

    # Convex hull of 2D
    points = np.zeros((n_prebins, 2))
    points[:, 0] = np.arange(n_prebins)
    points[:, 1] = event_rate

    rectangular_area = (max_event_rate - min_event_rate) * n_prebins

    if n_prebins > 2:
        hull = ConvexHull(points)
        p_convex_hull = hull.volume / rectangular_area
    else:
        p_convex_hull = 0

    # Pre-binning IV
    p_nev = n_nonevent / n_nonevent.sum()
    p_ev = n_event / n_event.sum()
    iv_prebins = jeffrey(p_nev, p_ev, return_sum=True)

    dict_data = {
        "n_prebins": n_prebins,
        "n_trend_changes": n_trend_changes,
        "p_trend_changes": p_trend_changes,
        "lr_sense": lr_sense,
        "pos_min": pos_min,
        "pos_max": pos_max,
        "p_bins_min_left": p_bins_min_left,
        "p_bins_min_right": p_bins_min_right,
        "p_bins_max_left": p_bins_max_left,
        "p_bins_max_right": p_bins_max_right,
        "p_records_min_left": p_records_min_left,
        "p_records_min_right": p_records_min_right,
        "p_records_max_left": p_records_max_left,
        "p_records_max_right": p_records_max_right,
        "p_area": p_area,
        "p_convex_hull": p_convex_hull,
        "iv_prebins": iv_prebins
    }

    return dict_data


def auto_monotonic_data_continuous(n_records, sums):
    n_prebins = len(n_records)

    # Trend changes data
    mean = sums / n_records
    n_trend_changes = n_peaks_valleys(mean)
    p_trend_changes = n_trend_changes / n_prebins

    # Linear regression coefficient sense
    lr_coef = np.polyfit(np.arange(n_prebins), mean, deg=1)[0]
    lr_sense = np.int(lr_coef > 0)

    # Breakpoints data
    pos_min = np.argmin(mean)
    pos_max = np.argmax(mean)
    min_event_rate = mean[pos_min]
    max_event_rate = mean[pos_max]

    n1 = n_prebins - 1
    p_bins_min_left = len(mean[:pos_min]) / n1
    p_bins_min_right = len(mean[pos_min+1:]) / n1

    p_bins_max_left = len(mean[:pos_max]) / n1
    p_bins_max_right = len(mean[pos_max+1:]) / n1

    total_records = np.sum(n_records)
    p_records_min_left = np.sum(n_records[:pos_min]) / total_records
    p_records_min_right = np.sum(n_records[pos_min+1:]) / total_records

    p_records_max_left = np.sum(n_records[:pos_max]) / total_records
    p_records_max_right = np.sum(n_records[pos_max+1:]) / total_records

    # Ratio triangular area approximation
    p_area = extreme_points_area(mean)

    # Convex hull of 2D
    points = np.zeros((n_prebins, 2))
    points[:, 0] = np.arange(n_prebins)
    points[:, 1] = mean

    rectangular_area = (max_event_rate - min_event_rate) * n_prebins

    if n_prebins > 2:
        hull = ConvexHull(points)
        p_convex_hull = hull.volume / rectangular_area
    else:
        p_convex_hull = 0

    dict_data = {
        "n_prebins": n_prebins,
        "n_trend_changes": n_trend_changes,
        "p_trend_changes": p_trend_changes,
        "lr_sense": lr_sense,
        "pos_min": pos_min,
        "pos_max": pos_max,
        "p_bins_min_left": p_bins_min_left,
        "p_bins_min_right": p_bins_min_right,
        "p_bins_max_left": p_bins_max_left,
        "p_bins_max_right": p_bins_max_right,
        "p_records_min_left": p_records_min_left,
        "p_records_min_right": p_records_min_right,
        "p_records_max_left": p_records_max_left,
        "p_records_max_right": p_records_max_right,
        "p_area": p_area,
        "p_convex_hull": p_convex_hull
    }

    return dict_data


def auto_monotonic_decision(lr_sense, p_records_min_left, p_records_min_right,
                            p_records_max_left, p_records_max_right, p_area,
                            p_convex_hull):

    if p_area <= 0.22145836800336838:
        if lr_sense == 0:
            if p_convex_hull <= 0.48331470787525177:
                if p_records_min_right <= 0.010740397498011589:
                    monotonic_trend = 1
                else:
                    if p_records_min_right <= 0.022145185619592667:
                        monotonic_trend = 3
                    else:
                        monotonic_trend = 1
            else:
                if p_records_max_right <= 0.6426683664321899:
                    monotonic_trend = 3
                else:
                    monotonic_trend = 1
        else:
            monotonic_trend = 0
    else:
        if p_records_min_right <= 0.06137961149215698:
            if p_convex_hull <= 0.23837491869926453:
                monotonic_trend = 1
            else:
                if p_records_max_left <= 0.10170064494013786:
                    if p_records_max_left <= 0.01817034650593996:
                        monotonic_trend = 3
                    else:
                        monotonic_trend = 1
                else:
                    monotonic_trend = 2
        else:
            if p_records_min_left <= 0.05336669087409973:
                if p_records_max_right <= 0.0695494469255209:
                    monotonic_trend = 0
                else:
                    if p_records_max_left <= 0.14705360680818558:
                        monotonic_trend = 0
                    else:
                        monotonic_trend = 2
            else:
                if p_records_min_left <= 0.8308950066566467:
                    monotonic_trend = 3
                else:
                    if p_records_max_right <= 0.1587613895535469:
                        monotonic_trend = 3
                    else:
                        monotonic_trend = 2

    if monotonic_trend == 0:
        return "ascending"
    elif monotonic_trend == 1:
        return "descending"
    elif monotonic_trend == 2:
        return "peak"
    elif monotonic_trend == 3:
        return "valley"


def auto_monotonic(n_nonevent, n_event):
    dict_data = auto_monotonic_data(n_nonevent, n_event)

    lr_sense = dict_data["lr_sense"]
    p_records_min_left = dict_data["p_records_min_left"]
    p_records_min_right = dict_data["p_records_min_right"]
    p_records_max_left = dict_data["p_records_max_left"]
    p_records_max_right = dict_data["p_records_max_right"]
    p_area = dict_data["p_area"]
    p_convex_hull = dict_data["p_convex_hull"]

    return auto_monotonic_decision(lr_sense, p_records_min_left,
                                   p_records_min_right, p_records_max_left,
                                   p_records_max_right, p_area, p_convex_hull)


def auto_monotonic_continuous(n_records, sums):
    dict_data = auto_monotonic_data_continuous(n_records, sums)

    lr_sense = dict_data["lr_sense"]
    p_records_min_left = dict_data["p_records_min_left"]
    p_records_min_right = dict_data["p_records_min_right"]
    p_records_max_left = dict_data["p_records_max_left"]
    p_records_max_right = dict_data["p_records_max_right"]
    p_area = dict_data["p_area"]
    p_convex_hull = dict_data["p_convex_hull"]

    return auto_monotonic_decision(lr_sense, p_records_min_left,
                                   p_records_min_right, p_records_max_left,
                                   p_records_max_right, p_area, p_convex_hull)
