"""
Binning tables for optimal continuous binning.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020


class PWBinningTable:
    def __init__(self, name, splits, coef, n_nonevent, n_event):
        self.name = name
        self.splits = splits
        self.coef = coef
        self.n_nonevent = n_nonevent
        self.n_event = n_event

    def build(self, show_digits=2, add_totals=True):
        pass

    def plot(self, metric="woe", add_special=True, add_missing=True, 
             savefig=None):
        pass

    def analysis(self, print_output=True):
        pass


class PWContinuousBinningTable:
    def __init__(self, name, splits, coef, n_records, sums, min_target,
                 max_target, n_zeros):

        self.name = name
        self.splits = splits
        self.coef = coef
        self.n_records = n_records
        self.sums = sums
        self.min_target = min_target
        self.max_target = max_target
        self.n_zeros = n_zeros

    def build(self, show_digits=2, add_totals=True):
        pass

    def plot(self, add_special=True, add_missing=True, savefig=None):
        pass

    def analysis(self, print_output=True):
        pass
