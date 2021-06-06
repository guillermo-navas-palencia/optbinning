"""
Two-dimensional binning utils.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2021


def check_is_lp(gamma, monotonic_trend_x, monotonic_trend_y, min_n_bins,
                max_n_bins):

    is_no_gamma = gamma == 0
    is_no_monotonic = monotonic_trend_x is None and monotonic_trend_y is None
    is_no_bin_size = min_n_bins is None and max_n_bins is None

    is_lp = (is_no_gamma and is_no_monotonic and is_no_bin_size)

    return is_lp
