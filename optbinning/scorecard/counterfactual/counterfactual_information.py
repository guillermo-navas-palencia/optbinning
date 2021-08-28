"""
Counterfactual information.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2021

from ...information import print_header
from ...information import print_optional_parameters
from ...information import print_solver_statistics
from ...options import counterfactual_default_options


def print_status(status):
    print("  Status  : {:<32}\n".format(status))


def print_main_info(status, time_total):
    print_status(status)

    print("  Time    : {:<7.4f} sec\n".format(time_total))


def print_objectives(objectives):
    str_objectives = "  Objectives\n"

    for objname, objexp in objectives.items():
        objval = objexp.solution_value()
        if objname in ("diversity_features", "diversity_values"):
            objval = abs(objval)

        str_objectives += "    {:<18}            {:>10.4f}\n".format(
            objname, objval)

    print(str_objectives)


def print_timing(time_total, time_fit, time_solver, time_postprocessing):
    p_fit = time_fit / time_total
    p_solver = time_solver / time_total
    p_postprocessing = time_postprocessing / time_solver

    time_stats = (
        "  Timing\n"
        "    Total time            {:>18.2f} sec\n"
        "    Fit                   {:>18.2f} sec   ({:>7.2%})\n"
        "    Solver                {:>18.2f} sec   ({:>7.2%})\n"
        "    Post-processing       {:>18.2f} sec   ({:>7.2%})\n"
        ).format(time_total, time_fit, p_fit, time_solver, p_solver,
                 time_postprocessing, p_postprocessing)

    print(time_stats)


def print_counterfactual_information(print_level, status, solver, objectives,
                                     time_total, time_fit, time_solver,
                                     time_postprocessing, dict_user_options):

    print_header()

    if print_level == 2:
        dict_default_options = counterfactual_default_options
        print_optional_parameters(dict_default_options, dict_user_options)

    if print_level == 0:
        print_main_info(status, time_total)
    elif print_level >= 1:
        print_status(status)

        if status in ("OPTIMAL", "FEASIBLE"):
            if solver is not None:
                print_solver_statistics("mip", solver)
                print_objectives(objectives)

        print_timing(time_total, time_fit, time_solver, time_postprocessing)
