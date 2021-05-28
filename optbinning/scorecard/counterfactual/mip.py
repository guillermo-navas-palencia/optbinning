"""
Mixed-integer programming formulation for a single counterfactual explanations.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2021

import numpy as np

from ortools.linear_solver import pywraplp

from .utils import logistic_pw


class CFMIP:
    def __init__(self, method, objectives, max_changes, non_actionable,
                 hard_constraints, soft_constraints, priority_tol, n_jobs,
                 time_limit):

        self.method = method
        self.objectives = objectives
        self.max_changes = max_changes
        self.non_actionable = non_actionable
        self.hard_constraints = hard_constraints
        self.soft_constraints = soft_constraints

        self.priority_tol = priority_tol

        self.n_jobs = n_jobs
        self.time_limit = time_limit

    def build_model(self, scorecard, x, y, outcome_type, intercept, coef,
                    min_p, max_p, wrange, F, mu, nbins, metric):

        # Parameters
        p = len(coef)

        # Parameters - probability outcome requires piecewise approximation
        if outcome_type == "probability":
            b_pw, c_pw = logistic_pw(min_p, max_p, 15)
        else:
            b_pw, c_pw = None, None

        if self.method == "weighted":
            self.weighted_model(
                p, nbins, metric, x, y, outcome_type, intercept, coef, min_p,
                max_p, wrange, F, mu, b_pw, c_pw)
        elif self.method == "hierarchical":
            self.hierarchical_model(
                p, nbins, metric, x, y, outcome_type, intercept, coef, min_p,
                max_p, wrange, F, mu, b_pw, c_pw)

    def weighted_model(self, p, nbins, metric, x, y, outcome_type, intercept,
                       coef, min_p, max_p, wrange, F, mu, b_pw, c_pw):

        # Initialize solver
        solver = pywraplp.Solver(
            'CFMIP', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

        # Decision variables
        (x_p, t_p, t_m, m_p, m_m, a, z, q_p, q_m, h,
         s, f) = self.decision_variables(solver, outcome_type, p, nbins, b_pw)

        # Objective functions
        weights = {**self.objectives, **self.soft_constraints}
        names = weights.keys()

        objs = self.compute_objectives(solver, names, p, wrange, t_p, t_m, m_p,
                                       m_m, q_p, q_m)

        solver.Minimize(solver.Sum([weights[name] * objs[name]
                                    for name in names]))

        # Constraint: proximity
        if "proximity" in self.objectives:
            self.add_constraint_proximity(
                solver, p, x, nbins, metric, x_p, t_p, t_m, z)

        # Constraint: closeness
        if "closeness" in self.objectives:
            self.add_constraint_closeness(solver, p, F, mu, x_p, m_p, m_m)

        # Constraint: max changes
        self.add_constraint_max_changes(solver, p, nbins, a, z)

        # Constraint: actionable features
        self.add_constraint_actionable(solver, p, a)

        # Constraints applicable depending on outcome type
        if outcome_type == "binary":
            self.add_constraint_opposite(solver, p, y, intercept, coef, x_p)

        elif outcome_type in ("probability", "continuous"):
            self.add_constraint_min_max_diff_outcome(
                solver, p, y, intercept, coef, outcome_type, b_pw, c_pw, x_p,
                q_p, q_m, h, s, f)

        self.solver_ = solver
        self._p = p
        self._objectives = objs
        self._nbins = nbins
        self._z = z

    def hierarchical_model(self, p, nbins, metric, x, y, outcome_type,
                           intercept, coef, min_p, max_p, wrange, F, mu, b_pw,
                           c_pw):

        # Objective functions
        weights = {**self.objectives, **self.soft_constraints}

        # descending priority
        names = [k for k, v in sorted(
            weights.items(), key=lambda item: item[1], reverse=True)]

        # store previous objectives
        pre_objs = {}

        for name in names:
            # Initialize solver
            solver = pywraplp.Solver(
                'CFMIP', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

            # Decision variables
            (x_p, t_p, t_m, m_p, m_m, a, z, q_p, q_m, h,
             s, f) = self.decision_variables(
             solver, outcome_type, p, nbins, b_pw)

            # Objective functions
            objs = self.compute_objectives(solver, names, p, wrange, t_p, t_m,
                                           m_p, m_m, q_p, q_m)

            solver.Minimize(objs[name])

            # Constraint: maximum relative degradation of previous objectives
            if pre_objs:
                d_p = {}
                d_m = {}

                for i, (pobj_name, pobj_val) in enumerate(pre_objs.items()):
                    # Auxiliary decision variablaes
                    d_p[i] = solver.NumVar(0, np.inf, "d_p[{}]".format(i))
                    d_m[i] = solver.NumVar(0, np.inf, "d_m[{}]".format(i))

                    # Constraint: maximum relative degradation
                    absval = abs(pobj_val)
                    solver.Add(d_p[i] - d_m[i] == objs[pobj_name] - pobj_val)
                    solver.Add(d_p[i] + d_m[i] <= self.priority_tol * absval)

            # Constraint: proximity
            if "proximity" in self.objectives:
                self.add_constraint_proximity(
                    solver, p, x, nbins, metric, x_p, t_p, t_m, z)

            # Constraint: closeness
            if "closeness" in self.objectives:
                self.add_constraint_closeness(solver, p, F, mu, x_p, m_p, m_m)

            # Constraint: max changes
            self.add_constraint_max_changes(solver, p, nbins, a, z)

            # Constraint: actionable features
            self.add_constraint_actionable(solver, p, a)

            # Constraints applicable depending on outcome type
            if outcome_type == "binary":
                self.add_constraint_opposite(
                    solver, p, y, intercept, coef, x_p)

            elif outcome_type in ("probability", "continuous"):
                self.add_constraint_min_max_diff_outcome(
                    solver, p, y, intercept, coef, outcome_type, b_pw, c_pw,
                    x_p, q_p, q_m, h, s, f)

            self.solver_ = solver
            self._p = p
            self._objectives = objs
            self._nbins = nbins
            self._z = z

            # Only solve #objs - 1
            if name != names[-1]:
                status_name, solution = self.solve()

                if status_name not in ("OPTIMAL", "FEASIBLE"):
                    raise Exception("{} problem in hierarchical model."
                                    .format(status_name))
                else:
                    pre_objs[name] = self._objectives[name].solution_value()

    def solve(self):
        self.solver_.SetTimeLimit(self.time_limit * 1000)
        self.solver_.SetNumThreads(self.n_jobs)
        status = self.solver_.Solve()

        if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            if status == pywraplp.Solver.OPTIMAL:
                status_name = "OPTIMAL"
            else:
                status_name = "FEASIBLE"

            solution = np.array([np.array([self._z[i, j].solution_value()
                                           for j in range(self._nbins[i])]
                                          ).astype(bool)
                                 for i in range(self._p)], dtype=object)
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

    def decision_variables(self, solver, outcome_type, p, nbins, b_pw):
        x_p = {}
        a = {}
        z = {}

        t_p = {}
        t_m = {}
        m_p = {}
        m_m = {}
        q_p = None
        q_m = None
        h = None
        s = None
        f = None

        for i in range(p):
            x_p[i] = solver.NumVar(-np.inf, np.inf, "x_p[{}]".format(i))
            a[i] = solver.IntVar(0, nbins[i], "a[{}]".format(i))
            for j in range(nbins[i]):
                z[i, j] = solver.BoolVar("z[{}, {}]".format(i, j))

            if "proximity" in self.objectives:
                t_p[i] = solver.NumVar(0, np.inf, "t_p[{}]".format(i))
                t_m[i] = solver.NumVar(0, np.inf, "t_m[{}]".format(i))

            if "closeness" in self.objectives:
                m_p[i] = solver.NumVar(0, np.inf, "m_p[{}]".format(i))
                m_m[i] = solver.NumVar(0, np.inf, "m_m[{}]".format(i))

        if outcome_type in ("probability", "continuous"):
            q_p = solver.NumVar(0, np.inf, "q_p")
            q_m = solver.NumVar(0, np.inf, "q_m")

            if outcome_type == "probability":
                f = solver.NumVar(0, 1, "f")
                h = {}
                s = {}
                for r in range(len(b_pw)):
                    h[r] = solver.NumVar(-np.inf, np.inf, "h[{}]".format(r))
                    s[r] = solver.BoolVar("s[{}]".format(r))

        return x_p, t_p, t_m, m_p, m_m, a, z, q_p, q_m, h, s, f

    def add_constraint_proximity(self, solver, p, x, nbins, metric, x_p, t_p,
                                 t_m, z):
        for i in range(p):
            # Absolute value
            solver.Add(t_p[i] - t_m[i] == x[i] - x_p[i])

            # Compute x'
            solver.Add(x_p[i] == x[i] + solver.Sum([
                (metric[i][j] - x[i]) * z[i, j] for j in range(nbins[i])]))

    def add_constraint_closeness(self, solver, p, F, mu, x_p, m_p, m_m):
        # Mahalanobis distance l1-norm
        for i in range(p):
            solver.Add(m_p[i] - m_m[i] == solver.Sum(
                [F[j, i] * (x_p[j] - mu[j]) for j in range(i, p)]))

    def add_constraint_max_changes(self, solver, p, nbins, a, z):
        for i in range(p):
            solver.Add(a[i] == solver.Sum([z[i, j] for j in range(nbins[i])]))
            solver.Add(a[i] <= 1)

        solver.Add(solver.Sum([a[i] for i in range(p)]) <= self.max_changes)

    def add_constraint_actionable(self, solver, p, a):
        for i in self.non_actionable:
            solver.Add(a[i] == 0)

    def add_constraint_opposite(self, solver, p, y, intercept, coef, x_p):
        activation = intercept + solver.Sum(
            [coef[i] * x_p[i] for i in range(p)])

        if y == 1:
            solver.Add(activation >= 1e-6)
        else:
            solver.Add(activation <= 0)

    def add_constraint_min_max_diff_outcome(self, solver, p, y, intercept,
                                            coef, outcome_type, b_pw, c_pw,
                                            x_p, q_p, q_m, h, s, f):

        activation = intercept + solver.Sum(
            [coef[i] * x_p[i] for i in range(p)])

        if outcome_type == "probability":
            nbins_pw = len(b_pw)

            solver.Add(
                solver.Sum([h[r] for r in range(nbins_pw)]) == activation)

            solver.Add(solver.Sum([c_pw[r][1] * h[r] + c_pw[r][0] * s[r]
                                   for r in range(nbins_pw)]) == f)

            solver.Add(solver.Sum([s[r] for r in range(nbins_pw)]) == 1)

            for r in range(nbins_pw):
                solver.Add(b_pw[r][0] * s[r] <= h[r])
                solver.Add(h[r] <= b_pw[r][1] * s[r])

            if "min_outcome" in self.hard_constraints:
                solver.Add(f >= y)

            if "max_outcome" in self.hard_constraints:
                solver.Add(f <= y)

            if "diff_outcome" in self.soft_constraints:
                solver.Add(q_p - q_m == f - y)
        else:
            if "min_outcome" in self.hard_constraints:
                solver.Add(activation >= y)

            if "max_outcome" in self.hard_constraints:
                solver.Add(activation <= y)

            if "diff_outcome" in self.soft_constraints:
                solver.Add(q_p - q_m == activation - y)

    def obj_proximity(self, solver, p, wrange, t_p, t_m):
        # Absolute distance
        return solver.Sum([wrange[i] * (t_p[i] + t_m[i]) for i in range(p)])

    def obj_closeness(self, solver, p, m_p, m_m):
        # Mahalanobis distance
        return solver.Sum([(m_p[i] + m_m[i]) for i in range(p)])

    def obj_diff_outcome(self, solver, q_p, q_m):
        # Absolute diff outcome
        return q_p + q_m

    def compute_objectives(self, solver, names, p, wrange, t_p, t_m, m_p, m_m,
                           q_p, q_m):
        objs = {}
        for name in names:
            if name == "proximity":
                objs[name] = self.obj_proximity(solver, p, wrange, t_p, t_m)
            elif name == "closeness":
                objs[name] = self.obj_closeness(solver, p, m_p, m_m)
            elif name == "diff_outcome":
                objs[name] = self.obj_diff_outcome(solver, q_p, q_m)

        return objs
