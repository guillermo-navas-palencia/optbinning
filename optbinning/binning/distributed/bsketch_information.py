"""
Binning sketch information.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

from ...binning.binning_information import print_main_info
from ...binning.binning_information import print_name_status
from ...binning.binning_information import print_prebinning_statistics
from ...information import print_header
from ...information import print_optional_parameters
from ...information import print_solver_statistics
from ...options import optimal_binning_sketch_options


def print_timing(
    solver_type: str,
    solver: object,
    time_total: float,
    time_prebinning: float,
    time_solver: float,
    time_optimizer: float,
    time_postprocessing: float
) -> None:

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
            "    Pre-binning           {:>18.2f} sec   ({:>7.2%})\n"
            "    Solver                {:>18.2f} sec   ({:>7.2%})\n"
            "      model generation    {:>18.2f} sec   ({:>7.2%})\n"
            "      optimizer           {:>18.2f} sec   ({:>7.2%})\n"
            "    Post-processing       {:>18.2f} sec   ({:>7.2%})\n"
            ).format(time_total, time_prebinning, p_prebinning, time_solver,
                     p_solver, time_model_generation, p_model_generation,
                     time_optimizer, p_optimizer, time_postprocessing,
                     p_postprocessing)
    else:
        time_stats = (
            "  Timing\n"
            "    Total time            {:>18.2f} sec\n"
            "    Pre-binning           {:>18.2f} sec   ({:>7.2%})\n"
            "    Solver                {:>18.2f} sec   ({:>7.2%})\n"
            "    Post-processing       {:>18.2f} sec   ({:>7.2%})\n"
            ).format(time_total, time_prebinning, p_prebinning, time_solver,
                     p_solver, time_postprocessing, p_postprocessing)

    print(time_stats)


def print_streaming_timing(
    memory_usage: float,
    n_records: int,
    n_add: int,
    time_add: float,
    n_solve: int,
    time_solve: float
) -> None:
    r_add = time_add / n_add
    r_solve = time_solve / n_solve

    records_stats = (
        "  Streaming statistics\n"
        "    Sketch memory usage   {:>18.5f} MB\n"
        "    Processed records     {:>18}\n"
        "    Add operations        {:>18}\n"
        "    Solve operations      {:>18}\n"
        ).format(memory_usage, n_records, n_add, n_solve)

    time_stats = (
        "  Streaming timing\n"
        "    Time add              {:>18.2f} sec   ({:6.4f} sec / add)\n"
        "    Time solve            {:>18.2f} sec   ({:6.4f} sec / solve)\n"
        ).format(time_add, r_add, time_solve, r_solve)

    print(records_stats)
    print(time_stats)


def print_binning_information(
    binning_type: str,
    print_level: int,
    name: str,
    status: str,
    solver_type: str,
    solver: object,
    time_total: float,
    time_prebinning: float,
    time_solver: float,
    time_optimizer: float,
    time_postprocessing: float,
    n_prebins: int,
    n_refinements: int,
    n_records: int,
    n_add: int,
    time_add: float,
    n_solve: int,
    time_solve: float,
    memory_usage: float,
    dict_user_options: dict
) -> None:

    print_header()

    if print_level == 2:
        if binning_type == "optimalbinningsketch":
            dict_default_options = optimal_binning_sketch_options

        print_optional_parameters(dict_default_options, dict_user_options)

    if print_level == 0:
        print_main_info(name, status, time_total)
    elif print_level >= 1:
        print_name_status(name, status)

        print_prebinning_statistics(n_prebins, n_refinements)

        if status in ("OPTIMAL", "FEASIBLE"):
            if solver is not None:
                print_solver_statistics(solver_type, solver)

            print_timing(solver_type, solver, time_total, time_prebinning,
                         time_solver, time_optimizer, time_postprocessing)

        print_streaming_timing(memory_usage, n_records, n_add, time_add,
                               n_solve, time_solve)
