"""
Monitoring information.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

from ..binning.binning_information import print_header
from ..binning.binning_information import print_optional_parameters
from ..options import scorecard_monitoring_default_options


def print_main_info(n_records_a, n_records_e, n_variables, time_total):
    print("  Number of records A : {}".format(n_records_a))
    print("  Number of records E : {}".format(n_records_e))
    print("  Number of variables : {}".format(n_variables))
    print("  Time                : {:<7.4f} sec\n".format(time_total))


def print_monitoring_statistics(n_records_a, n_records_e, n_variables,
                                target_dtype, time_total, time_system,
                                time_variables):

    stats = (
        "  Statistics\n"
        "    Number of records Actual      {:>10}\n"
        "    Number of records Expected    {:>10}\n"
        "    Number of scorecard variables {:>10}\n"
        "    Target type                   {:>10}\n"
        ).format(n_records_a, n_records_e, n_variables, target_dtype)

    print(stats)

    p_system = time_system / time_total
    p_variables = time_variables / time_total

    time_stats = (
        "  Timing\n"
        "    Total time            {:>18.2f} sec\n"
        "    System stability      {:>18.2f} sec   ({:>7.2%})\n"
        "    Variables stability   {:>18.2f} sec   ({:>7.2%})\n"
        ).format(time_total, time_system, p_system, time_variables,
                 p_variables)

    print(time_stats)


def print_monitoring_information(print_level, n_records_a, n_records_e,
                                 n_variables, target_dtype, time_total,
                                 time_system, time_variables,
                                 dict_user_options):

    print_header()

    if print_level == 2:
        dict_default_options = scorecard_monitoring_default_options
        print_optional_parameters(dict_default_options, dict_user_options)

    if print_level == 0:
        print_main_info(n_records_a, n_records_e, n_variables, time_total)
    elif print_level >= 1:
        print_monitoring_statistics(n_records_a, n_records_e, n_variables,
                                    target_dtype, time_total, time_system,
                                    time_variables)
