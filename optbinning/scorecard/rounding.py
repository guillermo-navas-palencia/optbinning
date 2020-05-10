"""
Rounding strategy.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import numpy as np

from ortools.linear_solver import pywraplp


class RoundingMIP:
    def __init__(self):
        self.solver_ = None

        self._nb = None
        self._nn = None
        self._p = None

    def build_model(self, df_scorecard):
        # Parameters
        points = []
        mins = []
        maxs = []
        for variable in df_scorecard.Variable.unique():
            mask = df_scorecard.Variable == variable
            p = df_scorecard[mask].Points.values
            mins.append(p.min())
            maxs.append(p.max())
            points.append(p)

        nb = len(points)
        nn = [len(p) for p in points]

        min_point = np.rint(np.sum(mins))
        max_point = np.rint(np.sum(maxs))

        min_p = np.min(mins)
        max_p = np.max(maxs)

        # Initialize solver
        solver = pywraplp.Solver(
                'RoundingMIP', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

        # Decision variables
        p = {}
        tp = {}
        tm = {}
        min_b = {}
        max_b = {}
        for i in range(nb):
            min_b[i] = solver.IntVar(min_p, max_p, "min_b[{}]".format(i))
            max_b[i] = solver.IntVar(min_p, max_p, "max_b[{}]".format(i))
            for j in range(nn[i]):
                p[i, j] = solver.IntVar(min_p, max_p, "p[{}, {}]".format(i, j))
                tp[i, j] = solver.NumVar(0, np.inf, "tp[{}, {}]".format(i, j))
                tm[i, j] = solver.NumVar(0, np.inf, "tm[{}, {}]".format(i, j))

        # Objective function
        solver.Minimize(solver.Sum([solver.Sum([tp[i, j] + tm[i, j]
                        for j in range(nn[i])]) for i in range(nb)]))

        # Constraints
        for i in range(nb):
            for j in range(nn[i]):
                solver.Add(tp[i, j] - tm[i, j] == points[i][j] - p[i, j])

                # Max score constraint for each variable
                solver.Add(max_b[i] >= p[i, j])

                # Min score constraints for each variable
                solver.Add(min_b[i] <= p[i, j])

        # Sum of minimum/maximum point by variable must be min_point/max_point
        solver.Add(solver.Sum([min_b[i] for i in range(nb)]) == min_point)
        solver.Add(solver.Sum([max_b[i] for i in range(nb)]) == max_point)

        self.solver_ = solver
        self._nb = nb
        self._nn = nn
        self._p = p

    def solve(self):
        status = self.solver_.Solve()

        if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            if status == pywraplp.Solver.OPTIMAL:
                status_name = "OPTIMAL"
            else:
                status_name = "FEASIBLE"

            # compute solution
            solution = []
            for i in range(self._nb):
                for j in range(self._nn[i]):
                    solution.append(self._p[i, j].solution_value())
        else:
            if status == pywraplp.Solver.ABNORMAL:
                status_name = "ABNORMAL"
            elif status == pywraplp.Solver.INFEASIBLE:
                status_name = "INFEASIBLE"
            elif status == pywraplp.Solver.UNBOUNDED:
                status_name = "UNBOUNDED"
            else:
                status_name = "UNKNOWN"

            solution = None

        return status_name, solution
