"""
Optimal binning algorithms default options.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019


optimal_binning_default_options = {
    "name": "",
    "dtype": "numerical",
    "prebinning_method": "cart",
    "solver": "cp",
    "max_n_prebins": 20,
    "min_prebin_size": 0.05,
    "min_n_bins": None,
    "max_n_bins": None,
    "min_bin_size": None,
    "max_bin_size": None,
    "min_bin_n_nonevent": None,
    "max_bin_n_nonevent": None,
    "min_bin_n_event": None,
    "max_bin_n_event": None,
    "monotonic_trend": "auto",
    "min_event_rate_diff": 0,
    "max_pvalue": None,
    "max_pvalue_policy": "consecutive",
    "class_weight": None,
    "cat_cutoff": None,
    "user_splits": None,
    "special_codes": None,
    "split_digits": None,
    "mip_solver": "bop",
    "time_limit": 100,
    "verbose": False
}


multiclass_optimal_binning_default_options = {
    "name": "",
    "prebinning_method": "cart",
    "solver": "cp",
    "max_n_prebins": 20,
    "min_prebin_size": 0.05,
    "min_n_bins": None,
    "max_n_bins": None,
    "min_bin_size": None,
    "max_bin_size": None,
    "monotonic_trend": "auto",
    "min_event_rate_diff": 0,
    "max_pvalue": None,
    "max_pvalue_policy": "consecutive",
    "user_splits": None,
    "special_codes": None,
    "split_digits": None,
    "mip_solver": "bop",
    "time_limit": 100,
    "verbose": False
}


continuous_optimal_binning_default_options = {
    "name": "",
    "dtype": "numerical",
    "prebinning_method": "cart",
    "max_n_prebins": 20,
    "min_prebin_size": 0.05,
    "min_n_bins": None,
    "max_n_bins": None,
    "min_bin_size": None,
    "max_bin_size": None,
    "monotonic_trend": "auto",
    "min_mean_diff": 0,
    "max_pvalue": None,
    "max_pvalue_policy": "consecutive",
    "cat_cutoff": None,
    "user_splits": None,
    "special_codes": None,
    "split_digits": None,
    "time_limit": 100,
    "verbose": False
}
