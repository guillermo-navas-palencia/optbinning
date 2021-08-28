"""
Binning process information.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

from ..information import print_header
from ..information import print_optional_parameters
from ..options import binning_process_default_options


def print_main_info(n_records, n_variables, time_total):
    print("  Number of records   : {}".format(n_records))
    print("  Number of variables : {}".format(n_variables))
    print("  Time                : {:<10.4f} sec\n".format(time_total))


def print_binning_process_statistics(n_records, n_variables, target_dtype,
                                     n_numerical, n_categorical, n_selected,
                                     time_total):
    stats = (
        "  Statistics\n"
        "    Number of records             {:>10}\n"
        "    Number of variables           {:>10}\n"
        "    Target type                   {:>10}\n\n"
        "    Number of numerical           {:>10}\n"
        "    Number of categorical         {:>10}\n"
        "    Number of selected            {:>10}\n\n"
        "  Time                            {:>10.4f} sec\n"
        ).format(n_records, n_variables, target_dtype, n_numerical,
                 n_categorical, n_selected, time_total)

    print(stats)


def print_binning_process_information(print_level, n_records, n_variables,
                                      target_dtype, n_numerical, n_categorical,
                                      n_selected, time_total,
                                      dict_user_options):
    print_header()

    if print_level == 2:
        dict_default_options = binning_process_default_options
        print_optional_parameters(dict_default_options, dict_user_options)

    if print_level == 0:
        print_main_info(n_records, n_variables, time_total)
    elif print_level >= 1:
        print_binning_process_statistics(n_records, n_variables, target_dtype,
                                         n_numerical, n_categorical,
                                         n_selected, time_total)
