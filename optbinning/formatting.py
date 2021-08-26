"""
Printing utilities.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import numbers
import textwrap

import pandas as pd


def dataframe_to_string(df, tab=None):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas.DataFrame.")

    if tab is not None:
        if not isinstance(tab, numbers.Integral) or tab < 0:
            raise ValueError("tab must be a positive integer; got {}."
                             .format(tab))

    df_string = textwrap.dedent(df.to_string(index=False))

    if tab is None:
        return df_string
    else:
        return textwrap.indent(df_string, " " * tab)
