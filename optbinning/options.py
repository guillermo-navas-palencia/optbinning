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
    "divergence": "iv",
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
    "gamma": 0,
    "class_weight": None,
    "cat_cutoff": None,
    "user_splits": None,
    "user_splits_fixed": None,
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
    "max_pvalue": None,
    "max_pvalue_policy": "consecutive",
    "user_splits": None,
    "user_splits_fixed": None,
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
    "gamma": 0,
    "cat_cutoff": None,
    "user_splits": None,
    "user_splits_fixed": None,
    "special_codes": None,
    "split_digits": None,
    "time_limit": 100,
    "verbose": False
}


binning_process_default_options = {
    "max_n_prebins": 20,
    "min_prebin_size": 0.05,
    "min_n_bins": None,
    "max_n_bins": None,
    "min_bin_size": None,
    "max_bin_size": None,
    "max_pvalue": None,
    "max_pvalue_policy": "consecutive",
    "selection_criteria": None,
    "fixed_variables": None,
    "categorical_variables": None,
    "special_codes": None,
    "split_digits": None,
    "binning_fit_params": None,
    "binning_transform_params": None,
    "verbose": False
}


sboptimal_binning_default_options = {
    "name": "",
    "prebinning_method": "cart",
    "max_n_prebins": 20,
    "min_prebin_size": 0.05,
    "min_n_bins": None,
    "max_n_bins": None,
    "min_bin_size": None,
    "max_bin_size": None,
    "monotonic_trend": None,
    "min_event_rate_diff": 0,
    "max_pvalue": None,
    "max_pvalue_policy": "consecutive",
    "class_weight": None,
    "user_splits": None,
    "user_splits_fixed": None,
    "special_codes": None,
    "split_digits": None,
    "time_limit": 100,
    "verbose": False
}


optimal_binning_sketch_options = {
    "name": "",
    "dtype": "numerical",
    "sketch": "gk",
    "eps": 1e-4,
    "K": 25,
    "solver": "cp",
    "divergence": "iv",
    "max_n_prebins": 20,
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
    "gamma": 0,
    "cat_cutoff": None,
    "cat_heuristic": False,
    "special_codes": None,
    "split_digits": None,
    "mip_solver": "bop",
    "time_limit": 100,
    "verbose": False
}


binning_process_sketch_default_options = {
    "max_n_prebins": 20,
    "min_n_bins": None,
    "max_n_bins": None,
    "min_bin_size": None,
    "max_bin_size": None,
    "max_pvalue": None,
    "max_pvalue_policy": "consecutive",
    "selection_criteria": None,
    "categorical_variables": None,
    "special_codes": None,
    "split_digits": None,
    "binning_fit_params": None,
    "binning_transform_params": None,
    "verbose": False
}


optimal_pw_binning_options = {
    "name": "",
    "estimator": None,
    "objective": "l2",
    "degree": 1,
    "continuous": True,
    "prebinning_method": "cart",
    "max_n_prebins": 20,
    "min_prebin_size": 0.05,
    "min_n_bins": None,
    "max_n_bins": None,
    "min_bin_size": None,
    "max_bin_size": None,
    "monotonic_trend": "auto",
    "n_subsamples": None,
    "max_pvalue": None,
    "max_pvalue_policy": "consecutive",
    "outlier_detector": None,
    "outlier_params": None,
    "user_splits": None,
    "special_codes": None,
    "split_digits": None,
    "solver": "auto",
    "h_epsilon": 1.35,
    "quantile": 0.5,
    "regularization": None,
    "reg_l1": 1.0,
    "reg_l2": 1.0,
    "random_state": None,
    "verbose": False
}


scorecard_monitoring_default_options = {
    "scorecard": None,
    "psi_method": "cart",
    "psi_n_bins": 20,
    "psi_min_bin_size": 0.05,
    "show_digits": 2,
    "verbose": False
}


scorecard_default_options = {
    "binning_process": None,
    "estimator": None,
    "scaling_method": None,
    "scaling_method_params": None,
    "intercept_based": False,
    "reverse_scorecard": False,
    "rounding": False,
    "verbose": False
}


counterfactual_default_options = {
    "scorecard": None,
    "special_missing": False,
    "n_jobs": 1,
    "verbose": False
}


optimal_binning_2d_default_options = {
    "name_x": "",
    "name_y": "",
    "dtype_x": "numerical",
    "dtype_y": "numerical",
    "prebinning_method": "cart",
    "strategy": "grid",
    "solver": "cp",
    "divergence": "iv",
    "max_n_prebins_x": 5,
    "max_n_prebins_y": 5,
    "min_prebin_size_x": 0.05,
    "min_prebin_size_y": 0.05,
    "min_n_bins": None,
    "max_n_bins": None,
    "min_bin_size": None,
    "max_bin_size": None,
    "min_bin_n_nonevent": None,
    "max_bin_n_nonevent": None,
    "min_bin_n_event": None,
    "max_bin_n_event": None,
    "monotonic_trend_x": None,
    "monotonic_trend_y": None,
    "min_event_rate_diff_x": 0,
    "min_event_rate_diff_y": 0,
    "gamma": 0,
    "special_codes_x": None,
    "special_codes_y": None,
    "split_digits": None,
    "n_jobs": 1,
    "time_limit": 100,
    "verbose": False
}


continuous_optimal_binning_2d_default_options = {
    "name_x": "",
    "name_y": "",
    "dtype_x": "numerical",
    "dtype_y": "numerical",
    "prebinning_method": "cart",
    "strategy": "grid",
    "solver": "cp",
    "max_n_prebins_x": 5,
    "max_n_prebins_y": 5,
    "min_prebin_size_x": 0.05,
    "min_prebin_size_y": 0.05,
    "min_n_bins": None,
    "max_n_bins": None,
    "min_bin_size": None,
    "max_bin_size": None,
    "monotonic_trend_x": None,
    "monotonic_trend_y": None,
    "min_mean_diff_x": 0,
    "min_mean_diff_y": 0,
    "gamma": 0,
    "special_codes_x": None,
    "special_codes_y": None,
    "split_digits": None,
    "n_jobs": 1,
    "time_limit": 100,
    "verbose": False
}
