"""
General information routines.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2021

import numpy as np

from sklearn.base import BaseEstimator

try:
    from localsolver import LSStatistics
    LOCALSOLVER_AVAILABLE = True
except ImportError:
    LOCALSOLVER_AVAILABLE = False


def print_header():
    header = (
        "optbinning (Version 0.11.0)\n"
        "Copyright (c) 2019-2021 Guillermo Navas-Palencia, Apache License 2.0"
        "\n")

    print(header)


def print_optional_parameters(dict_default_options, dict_user_options):
    option_format = "    {:<24} {:>15}   * {}\n"
    str_options = "  Begin options\n"
    for key, value in dict_default_options.items():
        user_value = dict_user_options[key]

        if (isinstance(user_value, (list, np.ndarray, dict)) or
                value != user_value):
            user_flag = "U"
        else:
            user_flag = "d"

        if user_value is None:
            user_value = "no"
        elif isinstance(user_value, (list, np.ndarray, dict)):
            user_value = "yes"
        elif isinstance(user_value, BaseEstimator):
            user_value = "yes"

        str_options += option_format.format(key, str(user_value), user_flag)
    str_options += "  End options\n"
    print(str_options)


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

    elif solver_type == "lp":
        n_variables = solver.n_variables
        n_constraints = solver.n_constraints
        n_iterations = solver.n_iterations
        objective = solver.objective

        solver_stats = (
            "  Solver statistics\n"
            "    Type                          {:>10}\n"
            "    Number of variables           {:>10}\n"
            "    Number of constraints         {:>10}\n"
            "    Number of iterations          {:>10}\n"
            "    Objective value               {:>10.4f}\n"
            ).format(solver_type, n_variables, n_constraints, n_iterations,
                     objective)

    print(solver_stats)
