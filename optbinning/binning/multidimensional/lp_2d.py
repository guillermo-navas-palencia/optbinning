"""
Generalized assigment problem: solve constrained optimal 2D binning problem.
Linear programming implementation.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2021

import itertools
import warnings

import numpy as np

from scipy.optimize import linprog
from scipy.optimize import OptimizeWarning
from scipy.sparse import csr_matrix


warnings.filterwarnings("ignore", category=OptimizeWarning)


class LPStats:
    def __init__(self, n_variables, n_constraints, n_iterations, objective):
        self.n_variables = n_variables
        self.n_constraints = n_constraints
        self.n_iterations = n_iterations
        self.objective = objective


class Binning2DLP:
    def __init__(self):
        self.solver_ = None

    def solve(self, n_grid, n_rectangles, dcols, c):
        rows = list(itertools.chain(*([k] * len(v) for k, v in dcols.items())))
        cols = list(itertools.chain(*(dcols.values())))
        data = np.ones(len(rows), dtype=np.int8)

        A = csr_matrix((data, (rows, cols)), shape=(n_grid, n_rectangles))
        b = np.ones(n_grid, dtype=np.int8)

        res = linprog(-c, A_eq=A, b_eq=b, method='highs-ds')

        # map status
        if res.status == 0:
            status_name = "OPTIMAL"
        elif res.status == 1:
            status_name = "MAX_ITERATIONS"
        elif res.status == 2:
            status_name = "INFEASIBLE"
        elif res.status == 3:
            status_name = "UNBOUNDED"
        elif res.status == 4:
            status_name = "ABNORMAL"

        # problem statistics and objective
        self.solver_ = LPStats(n_rectangles, n_grid, res.nit, -res.fun)

        solution = res.x.astype(bool)

        return status_name, solution
