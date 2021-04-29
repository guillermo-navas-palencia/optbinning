"""
Mixed-integer programming formulation for counterfactual explanations.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2021

import numpy as np

from ortools.linear_solver import pywraplp


class CFMIP:
    def __init__(self):
        pass

    def build_model(self):
        if self.method == "weighted":
            if self.n_cf == 1:
                self.weighted_model()
            else:
                self.multi_weighted_model()
        elif self.method == "hierarchical":
            if self.n_cf == 1:
                self.hierarchical_model()
            else:
                self.multi_hierarchical_model()            

    def weighted_model(self):
        pass

    def hierarchical_model(self):
        pass

    def multi_weighted_model(self):
        pass

    def multi_hierarchical_model(self):
        pass

    def solve(self):
        pass

    def decision_variables(self):
        pass

    def add_constraint_proximity(self):
        pass

    def add_constraint_closeness(self):
        pass

    def add_constraint_max_changes(self):
        pass
