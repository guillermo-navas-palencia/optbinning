"""
Optimal piecewise binning information.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

from ...binning.binning_information import print_header
from ...binning.binning_information import print_optional_parameters
from ...binning.binning_information import print_name_status
from ...binning.binning_information import print_main_info


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


def print_prebinning_statistics(n_prebins):
    prebinning_stats = (
        "  Pre-binning statistics\n"
        "    Number of bins                {:>10}\n"
        ).format(n_prebins)

    print(prebinning_stats)


def print_solver_statistics(solver_type, solver):
    if isinstance(solver.stats, list):
        n_constraints = sum(info["n_constraints"] for info in solver.stats)
        n_variables = sum(info["n_variables"] for info in solver.stats)
    else:
        n_constraints = solver.stats["n_constraints"]
        n_variables = solver.stats["n_variables"]

    solver_stats = (
        "  Solver statistics\n"
        "    Type                          {:>10}\n"
        "    Number of variables           {:>10}\n"
        "    Number of constraints         {:>10}\n"
        ).format(solver_type, n_variables, n_constraints)

    print(solver_stats)


def print_timing(solver_type, solver, time_total, time_preprocessing,
                 time_estimator, time_prebinning, time_solver,
                 time_postprocessing):

    p_preprocessing = time_preprocessing / time_total
    p_estimator = time_estimator / time_total
    p_prebinning = time_prebinning / time_total
    p_solver = time_solver / time_total
    p_postprocessing = time_postprocessing / time_total

    time_stats = (
        "  Timing\n"
        "    Total time            {:>18.2f} sec\n"
        "    Pre-processing        {:>18.2f} sec   ({:>7.2%})\n"
        "    Estimator             {:>18.2f} sec   ({:>7.2%})\n"
        "    Pre-binning           {:>18.2f} sec   ({:>7.2%})\n"
        "    Solver                {:>18.2f} sec   ({:>7.2%})\n"
        "    Post-processing       {:>18.2f} sec   ({:>7.2%})\n"
        ).format(time_total, time_preprocessing, p_preprocessing,
                 time_estimator, p_estimator, time_prebinning, p_prebinning,
                 time_solver, p_solver, time_postprocessing, p_postprocessing)

    print(time_stats)


def retrieve_status(status):
    if isinstance(status, list):
        n_status = len(status)
        n_optimal = 0
        n_feasible = 0
        n_unbouded = 0
        for s in status:
            if "optimal" in s:
                n_optimal += 1
            elif "feasible" in s:
                n_feasible += 1
            elif "unbounded" in s:
                n_unbouded += 1
        if n_optimal == n_status:
            return "OPTIMAL"
        elif n_feasible == n_status:
            return "FEASIBLE"
        elif n_unbouded == n_status:
            return "UNBOUNDED"
        else:
            new_status = ""
            if n_optimal > 0:
                new_status += "OPTIMAL ({}/{})".format(n_optimal, n_status)
            if n_feasible > 0:
                new_status += "FEASIBLE ({}/{})".format(n_feasible, n_status)
            if n_unbouded > 0:
                new_status += "UNBOUNDED ({}/{})".format(n_unbouded, n_status)
        return new_status
    else:
        if "optimal" in status:
            return "OPTIMAL"
        elif "feasible" in status:
            return "FEASIBLE"
        elif "unbounded" in status:
            return "UNBOUNDED"


def print_binning_information(print_level, name, status, solver_type, solver,
                              time_total, time_preprocessing, time_estimator,
                              time_prebinning, time_solver,
                              time_postprocessing, n_prebins,
                              dict_user_options):

    print_header()

    if print_level == 2:
        dict_default_options = optimal_pw_binning_options

        print_optional_parameters(dict_default_options, dict_user_options)

    if print_level == 0:
        print_main_info(name, status, time_total)
    elif print_level >= 1:
        print_name_status(name, status)

        print_prebinning_statistics(n_prebins)

        if status in ("OPTIMAL", "FEASIBLE"):
            if solver is not None:
                print_solver_statistics(solver_type, solver)

            print_timing(solver_type, solver, time_total, time_preprocessing,
                         time_estimator, time_prebinning, time_solver,
                         time_postprocessing)
