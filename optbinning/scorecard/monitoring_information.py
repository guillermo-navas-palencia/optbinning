"""
Monitoring information.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020


from ..binning.binning_information import print_header
from ..binning.binning_information import print_optional_parameters


scorecard_monitoring_default_options = {
    "target": "",
    "scorecard": None,
    "psi_method": "cart",
    "psi_n_bins": 20,
    "psi_min_bin_size": 0.05,
    "show_digits": 2,
    "verbose": False
}


def print_main_info(n_records_a, n_records_e, n_variables, time_total):
    print("  Number of records A : {}".format(n_records_a))
    print("  Number of records E : {}".format(n_records_e))
    print("  Number of variables : {}".format(n_variables))
    print("  Time                : {:<7.4f} sec\n".format(time_total))


def print_monitoring_statistics():
    pass


def print_monitoring_information():
    pass
