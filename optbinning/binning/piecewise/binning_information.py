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
    "degree": 1,
    "continuity": True,
    "prebinning_method": "cart",
    "max_n_prebins": 20,
    "min_prebin_size": 0.05,
    "min_n_bins": None,
    "max_n_bins": None,
    "min_bin_size": None,
    "max_bin_size": None,
    "monotonic_trend": "auto",
    "n_subsamples": 10000,
    "max_pvalue": None,
    "max_pvalue_policy": "consecutive",
    "outlier_detector": None,
    "outlier_params": None,
    "user_splits": None,
    "special_codes": None,
    "split_digits": None,
    "solver": "clp",
    "time_limit": 100,
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
    n_constraints = solver.NumConstraints()
    n_variables = solver.NumVariables()
    iterations = solver.Iterations()
    objective = solver.Objective().Value()

    solver_stats = (
        "  Solver statistics\n"
        "    Type                          {:>10}\n"
        "    Number of variables           {:>10}\n"
        "    Number of constraints         {:>10}\n"
        "    Simplex iterations            {:>10}\n"
        "    Objective value               {:>10.4f}\n"
        ).format(solver_type, n_variables, n_constraints, iterations,
                 objective)

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
