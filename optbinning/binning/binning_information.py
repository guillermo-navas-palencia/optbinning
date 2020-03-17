"""
Optimal binning information.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import numpy as np

from .options import continuous_optimal_binning_default_options
from .options import multiclass_optimal_binning_default_options
from .options import optimal_binning_default_options

try:
    from localsolver import LSStatistics
    LOCALSOLVER_AVAILABLE = True
except ImportError:
    LOCALSOLVER_AVAILABLE = False


def print_header():
    header = (
        "optbinning (Version 0.4.0)\n"
        "Copyright (c) 2019-2020 Guillermo Navas-Palencia, Apache License 2.0"
        "\n")

    print(header)


def print_optional_parameters(dict_default_options, dict_user_options):
    option_format = "    {:<24} {:>15}   * {}\n"
    str_options = "  Begin options\n"
    for key, value in dict_default_options.items():
        user_value = dict_user_options[key]
        user_flag = "d" if value == user_value else "U"

        if user_value is None:
            user_value = "no"
        elif isinstance(user_value, (list, np.ndarray, dict)):
            user_value = "yes"

        str_options += option_format.format(key, str(user_value), user_flag)
    str_options += "  End options\n"
    print(str_options)


def print_prebinning_statistics(n_prebins, n_refinement):
    prebinning_stats = (
        "  Pre-binning statistics\n"
        "    Number of pre-bins            {:>10}\n"
        "    Number of refinements         {:>10}\n"
        ).format(n_prebins, n_refinement)

    print(prebinning_stats)


def print_solver_statistics(solver_type, solver):
    if solver_type == "cp":
        n_booleans = solver.NumBooleans()
        n_branches = solver.NumBranches()
        n_conflicts = solver.NumConflicts()
        objective = int(solver.ObjectiveValue())
        best_objective_bound = int(solver.BestObjectiveBound())

        solver_stats = (
            "  Solver statistics\n"
            "    Type                          {:>10}\n"
            "    Number of booleans            {:>10}\n"
            "    Number of branches            {:>10}\n"
            "    Number of conflicts           {:>10}\n"
            "    Objective value               {:>10}\n"
            "    Best objective bound          {:>10}\n"
            ).format(solver_type, n_booleans, n_branches, n_conflicts,
                     objective, best_objective_bound)
    elif solver_type == "mip":
        n_constraints = solver.NumConstraints()
        n_variables = solver.NumVariables()
        objective = solver.Objective().Value()
        best_bound = solver.Objective().BestBound()

        solver_stats = (
            "  Solver statistics\n"
            "    Type                          {:>10}\n"
            "    Number of variables           {:>10}\n"
            "    Number of constraints         {:>10}\n"
            "    Objective value               {:>10.4f}\n"
            "    Best objective bound          {:>10.4f}\n"
            ).format(solver_type, n_variables, n_constraints, objective,
                     best_bound)
    elif solver_type == "ls":
        if not LOCALSOLVER_AVAILABLE:
            raise ImportError('Cannot import localsolver. Install LocalSolver '
                              'or choose another solver, options are "cp" and '
                              '"mip".')

        n_iterations = LSStatistics.get_nb_iterations(solver.statistics)
        solver_stats = (
            "  Solver statistics\n"
            "    Type                          {:>10}\n"
            "    Number of iterations          {:>10}\n"
            ).format(solver_type, n_iterations)

    print(solver_stats)


def print_timing(solver_type, solver, time_total, time_preprocessing,
                 time_prebinning, time_solver, time_postprocessing):

    p_preprocessing = time_preprocessing / time_total
    p_prebinning = time_prebinning / time_total
    p_solver = time_solver / time_total
    p_postprocessing = time_postprocessing / time_total

    if solver_type == "cp" and solver is not None:
        time_optimizer = solver.WallTime()
        time_model_generation = time_solver - time_optimizer
        p_model_generation = time_model_generation / time_solver
        p_optimizer = time_optimizer / time_solver

        time_stats = (
            "  Timing\n"
            "    Total time            {:>18.2f} sec\n"
            "    Pre-processing        {:>18.2f} sec   ({:>7.2%})\n"
            "    Pre-binning           {:>18.2f} sec   ({:>7.2%})\n"
            "    Solver                {:>18.2f} sec   ({:>7.2%})\n"
            "      model generation    {:>18.2f} sec   ({:>7.2%})\n"
            "      optimizer           {:>18.2f} sec   ({:>7.2%})\n"
            "    Post-processing       {:>18.2f} sec   ({:>7.2%})\n"
            ).format(time_total, time_preprocessing, p_preprocessing,
                     time_prebinning, p_prebinning, time_solver, p_solver,
                     time_model_generation, p_model_generation, time_optimizer,
                     p_optimizer, time_postprocessing, p_postprocessing)
    else:
        time_stats = (
            "  Timing\n"
            "    Total time            {:>18.2f} sec\n"
            "    Pre-processing        {:>18.2f} sec   ({:>7.2%})\n"
            "    Pre-binning           {:>18.2f} sec   ({:>7.2%})\n"
            "    Solver                {:>18.2f} sec   ({:>7.2%})\n"
            "    Post-processing       {:>18.2f} sec   ({:>7.2%})\n"
            ).format(time_total, time_preprocessing, p_preprocessing,
                     time_prebinning, p_prebinning, time_solver, p_solver,
                     time_postprocessing, p_postprocessing)

    print(time_stats)


def print_name_status(name, status):
    if not name:
        name = "UNKNOWN"

    print("  Name    : {:<32}\n"
          "  Status  : {:<32}\n".format(name, status))


def print_main_info(name, status, time_total):
    print_name_status(name, status)

    print("  Time    : {:<7.4f} sec\n".format(round(time_total, 4)))


def print_binning_information(binning_type, print_level, name, status,
                              solver_type, solver, time_total,
                              time_preprocessing, time_prebinning, time_solver,
                              time_postprocessing, n_prebins, n_refinements,
                              dict_user_options):

    print_header()

    if print_level == 2:
        if binning_type == "optimalbinning":
            dict_default_options = optimal_binning_default_options
        elif binning_type == "multiclassoptimalbinning":
            dict_default_options = multiclass_optimal_binning_default_options
        elif binning_type == "continuousoptimalbinning":
            dict_default_options = continuous_optimal_binning_default_options

        print_optional_parameters(dict_default_options, dict_user_options)

    if print_level == 0:
        print_main_info(name, status, time_total)
    elif print_level >= 1:
        print_name_status(name, status)

        print_prebinning_statistics(n_prebins, n_refinements)

        if status in ("OPTIMAL", "FEASIBLE"):
            if solver is not None:
                print_solver_statistics(solver_type, solver)

            print_timing(solver_type, solver, time_total, time_preprocessing,
                         time_prebinning, time_solver, time_postprocessing)
