"""
Binning process information.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

from typing import Any

from ..information import print_header
from ..information import print_optional_parameters
from ..options import binning_process_default_options


def print_main_info(
    n_records: int,
    n_variables: int,
    time_total: float
) -> None:
    print("  Number of records   : {}".format(n_records))
    print("  Number of variables : {}".format(n_variables))
    print("  Time                : {:<10.4f} sec\n".format(time_total))


def print_binning_process_statistics(
    n_records: int,
    n_variables: int,
    target_dtype: str,
    n_numerical: int,
    n_categorical: int,
    n_selected: int,
    time_total: float
) -> None:
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


def print_binning_process_information(
    print_level: int,
    n_records: int,
    n_variables: int,
    target_dtype: str,
    n_numerical: int,
    n_categorical: int,
    n_selected: int,
    time_total: float,
    dict_user_options: dict[str, Any]
) -> None:
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
