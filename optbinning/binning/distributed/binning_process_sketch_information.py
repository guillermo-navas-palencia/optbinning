"""
Binning process sketch information.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2021

from ...binning.binning_information import print_header
from ...binning.binning_information import print_optional_parameters


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


def print_main_info(n_records, n_variables, time_add, time_solve):
    print("  Number of records   : {}".format(n_records))
    print("  Number of variables : {}".format(n_variables))
    print("  Time add            : {:<10.4f} sec".format(time_add))
    print("  Time solve          : {:<10.4f} sec\n".format(time_solve))


def print_binning_process_sketch_statistics(
        n_records, n_variables, target_dtype, n_numerical, n_categorical,
        n_selected, n_add, time_add, n_solve, time_solve):

    r_add = time_add / n_add
    r_solve = time_solve / n_solve

    stats = (
        "  Statistics\n"
        "    Number of records             {:>10}\n"
        "    Number of variables           {:>10}\n"
        "    Target type                   {:>10}\n\n"
        "    Number of numerical           {:>10}\n"
        "    Number of categorical         {:>10}\n"
        "    Number of selected            {:>10}\n"
        ).format(n_records, n_variables, target_dtype, n_numerical,
                 n_categorical, n_selected)

    records_stats = (
        "  Streaming statistics\n"
        "    Add operations        {:>18}\n"
        "    Solve operations      {:>18}\n"
        ).format(n_add, n_solve)

    time_stats = (
        "  Streaming timing\n"
        "    Time add              {:>18.2f} sec   ({:6.4f} sec / add)\n"
        "    Time solve            {:>18.2f} sec   ({:6.4f} sec / solve)\n"
        ).format(time_add, r_add, time_solve, r_solve)

    print(stats)
    print(records_stats)
    print(time_stats)


def print_binning_process_sketch_information(
        print_level, n_records, n_variables, target_dtype, n_numerical,
        n_categorical, n_selected, n_add, time_add, n_solve, time_solve,
        dict_user_options):

    print_header()

    if print_level == 2:
        dict_default_options = binning_process_sketch_default_options
        print_optional_parameters(dict_default_options, dict_user_options)

    if print_level == 0:
        print_main_info(n_records, n_variables, time_add, time_solve)
    elif print_level >= 1:
        print_binning_process_sketch_statistics(
            n_records, n_variables, target_dtype, n_numerical, n_categorical,
            n_selected, n_add, time_add, n_solve, time_solve)
