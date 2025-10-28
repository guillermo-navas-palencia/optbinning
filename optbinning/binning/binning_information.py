"""
Optimal binning information.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

from typing import Any

from ..information import print_header
from ..information import print_optional_parameters
from ..information import print_solver_statistics
from ..options import continuous_optimal_binning_default_options
from ..options import multiclass_optimal_binning_default_options
from ..options import optimal_binning_default_options
from ..options import sboptimal_binning_default_options
from ..options import continuous_optimal_binning_2d_default_options
from ..options import optimal_binning_2d_default_options


def print_prebinning_statistics(n_prebins: int, n_refinement: int) -> None:
    prebinning_stats = (
        "  Pre-binning statistics\n"
        "    Number of pre-bins            {:>10}\n"
        "    Number of refinements         {:>10}\n"
        ).format(n_prebins, n_refinement)

    print(prebinning_stats)


def print_timing(
    solver_type: str,
    solver: object,
    time_total: float,
    time_preprocessing: float,
    time_prebinning: float,
    time_solver: float,
    time_optimizer: float,
    time_postprocessing: float
) -> None:

    p_preprocessing = time_preprocessing / time_total
    p_prebinning = time_prebinning / time_total
    p_solver = time_solver / time_total
    p_postprocessing = time_postprocessing / time_total

    if solver_type == "cp" and solver is not None:
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


def print_name_status(name: str, status: str) -> None:
    if not name:
        name = "UNKNOWN"

    print("  Name    : {:<32}\n"
          "  Status  : {:<32}\n".format(name, status))


def print_main_info(name: str, status: str, time_total: float) -> None:
    print_name_status(name, status)

    print("  Time    : {:<7.4f} sec\n".format(time_total))


def print_binning_information(
    binning_type: str,
    print_level: int,
    name: str,
    status: str,
    solver_type: str,
    solver: object,
    time_total: float,
    time_preprocessing: float,
    time_prebinning: float,
    time_solver: float,
    time_optimizer: float,
    time_postprocessing: float,
    n_prebins: int,
    n_refinements: int,
    dict_user_options: dict[str, Any]
) -> None:

    print_header()

    if print_level == 2:
        if binning_type == "optimalbinning":
            d_default_options = optimal_binning_default_options
        elif binning_type == "multiclassoptimalbinning":
            d_default_options = multiclass_optimal_binning_default_options
        elif binning_type == "continuousoptimalbinning":
            d_default_options = continuous_optimal_binning_default_options
        elif binning_type == "sboptimalbinning":
            d_default_options = sboptimal_binning_default_options
        elif binning_type == "optimalbinning2d":
            d_default_options = optimal_binning_2d_default_options
        elif binning_type == "continuousoptimalbinning2d":
            d_default_options = continuous_optimal_binning_2d_default_options

        print_optional_parameters(d_default_options, dict_user_options)

    if print_level == 0:
        print_main_info(name, status, time_total)
    elif print_level >= 1:
        print_name_status(name, status)

        print_prebinning_statistics(n_prebins, n_refinements)

        if status in ("OPTIMAL", "FEASIBLE"):
            if solver is not None:
                print_solver_statistics(solver_type, solver)

            print_timing(solver_type, solver, time_total, time_preprocessing,
                         time_prebinning, time_solver, time_optimizer,
                         time_postprocessing)
