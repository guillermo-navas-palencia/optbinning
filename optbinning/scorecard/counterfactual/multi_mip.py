"""
Mixed-integer programming formulation for multiple counterfactual explanations.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2021

import numpy as np

from ortools.linear_solver import pywraplp

from .mip import CFMIP


class MCFMIP(CFMIP):
    def __init__(self, K, method, objectives, max_changes, non_actionable,
                 hard_constraints, soft_constraints, priority_tol, n_jobs,
                 time_limit):

        self.K = K
        self.method = method
        self.objectives = objectives
        self.max_changes = max_changes
        self.non_actionable = non_actionable
        self.hard_constraints = hard_constraints
        self.soft_constraints = soft_constraints

        self.priority_tol = priority_tol

        self.n_jobs = n_jobs
        self.time_limit = time_limit

    def weighted_model(self, p, nbins, metric, x, y, outcome_type, intercept,
                       coef, min_p, max_p, wrange, F, mu, b_pw, c_pw):

        # Initialize solver
        solver = pywraplp.Solver(
            'CFMIP', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

        # Decision variables
        (x_p, t_p, t_m, m_p, m_m, a, z, q_p, q_m,
         h, s, f, u, d) = self.decision_variables(solver, outcome_type, p,
                                                  nbins, b_pw)

        # Objective functions
        weights = {**self.objectives, **self.soft_constraints}
        names = weights.keys()

        objs = self.compute_objectives(solver, names, p, nbins, wrange, t_p,
                                       t_m, m_p, m_m, q_p, q_m, u, d)

        solver.Minimize(solver.Sum([weights[name] * objs[name]
                                    for name in names]))

        # Constraints
        for k in range(self.K):
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

            # Diversity constraints
            for l in range(k + 1, self.K):
                for i in range(p):
                    hki = a[k, i]
                    hli = a[l, i]

                    solver.Add(u[k, l, i] <= hki + hli)
                    solver.Add(u[k, l, i] >= hki - hli)
                    solver.Add(u[k, l, i] >= hli - hki)
                    solver.Add(u[k, l, i] <= 2 - hki - hli)

                    for j in range(nbins[i]):
                        solver.Add(d[k, l, i, j] <= z[k, i, j] + z[l, i, j])
                        solver.Add(d[k, l, i, j] >= z[k, i, j] - z[l, i, j])
                        solver.Add(d[k, l, i, j] >= -z[k, i, j] + z[l, i, j])
                        solver.Add(d[k, l, i, j] <= 2 - z[k, i, j]-z[l, i, j])

                    if "diversity_values" in self.hard_constraints:
                        solver.Add(solver.Sum(
                            [d[k, l, i, j] for j in range(nbins[i])]
                            ) >= a[k, i] + a[l, i] - 1)

                if "diversity_features" in self.hard_constraints:
                    solver.Add(solver.Sum([u[k, l, i] for i in range(p)]) >= 1)

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
            (x_p, t_p, t_m, m_p, m_m, a, z, q_p, q_m,
             h, s, f, u, d) = self.decision_variables(solver, outcome_type, p,
                                                      nbins, b_pw)

            # Objective functions
            objs = self.compute_objectives(solver, names, p, nbins, wrange,
                                           t_p, t_m, m_p, m_m, q_p, q_m, u, d)

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

            # Constraints
            for k in range(self.K):
                # Constraint: proximity
                if "proximity" in self.objectives:
                    self.add_constraint_proximity(
                        solver, p, x, nbins, metric, x_p, t_p, t_m, z)

                # Constraint: closeness
                if "closeness" in self.objectives:
                    self.add_constraint_closeness(
                        solver, p, F, mu, x_p, m_p, m_m)

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
                        solver, p, y, intercept, coef, outcome_type, b_pw,
                        c_pw, x_p, q_p, q_m, h, s, f)

                # Diversity constraints
                for l in range(k + 1, self.K):
                    for i in range(p):
                        hki = a[k, i]
                        hli = a[l, i]

                        solver.Add(u[k, l, i] <= hki + hli)
                        solver.Add(u[k, l, i] >= hki - hli)
                        solver.Add(u[k, l, i] >= hli - hki)
                        solver.Add(u[k, l, i] <= 2 - hki - hli)

                        for j in range(nbins[i]):
                            solver.Add(d[k, l, i, j] <= z[k, i, j]+z[l, i, j])
                            solver.Add(d[k, l, i, j] >= z[k, i, j]-z[l, i, j])
                            solver.Add(d[k, l, i, j] >= -z[k, i, j]+z[l, i, j])
                            solver.Add(d[k, l, i, j] <= (
                                2-z[k, i, j]-z[l, i, j]))

                        if "diversity_values" in self.hard_constraints:
                            solver.Add(solver.Sum(
                                [d[k, l, i, j] for j in range(nbins[i])]
                                ) >= a[k, i] + a[l, i] - 1)

                    if "diversity_features" in self.hard_constraints:
                        solver.Add(
                            solver.Sum([u[k, l, i] for i in range(p)]) >= 1)

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

            solution = np.array(
                [np.array(
                    [np.array([self._z[k, i, j].solution_value()
                               for j in range(self._nbins[i])]).astype(bool)
                     for i in range(self._p)], dtype=object)
                 for k in range(self.K)], dtype=object)
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
        u = {}
        d = {}

        t_p = {}
        t_m = {}
        m_p = {}
        m_m = {}
        q_p = {}
        q_m = {}
        h = {}
        s = {}
        f = {}

        for k in range(self.K):
            for i in range(p):
                x_p[k, i] = solver.NumVar(-np.inf, np.inf,
                                          "x_p[{}, {}]".format(k, i))
                a[k, i] = solver.IntVar(0, nbins[i], "a[{}, {}]".format(k, i))
                for j in range(nbins[i]):
                    z[k, i, j] = solver.BoolVar("z[{},{},{}]".format(k, i, j))

                if "proximity" in self.objectives:
                    t_p[k, i] = solver.NumVar(0, np.inf,
                                              "t_p[{}, {}]".format(k, i))
                    t_m[k, i] = solver.NumVar(0, np.inf,
                                              "t_m[{}, {}]".format(k, i))

                if "closeness" in self.objectives:
                    m_p[k, i] = solver.NumVar(0, np.inf,
                                              "m_p[{}, {}]".format(k, i))
                    m_m[k, i] = solver.NumVar(0, np.inf,
                                              "m_m[{}, {}]".format(k, i))

            if outcome_type in ("probability", "continuous"):
                q_p[k] = solver.NumVar(0, np.inf, "q_p[{}]".format(k))
                q_m[k] = solver.NumVar(0, np.inf, "q_m[{}".format(k))

                if outcome_type == "probability":
                    f[k] = solver.NumVar(0, 1, "f[{}]".format(k))
                    for r in range(len(b_pw)):
                        h[k, r] = solver.NumVar(-np.inf, np.inf,
                                                "h[{}, {}]".format(k, r))
                        s[k, r] = solver.BoolVar("s[{}, {}]".format(k, r))

            for l in range(k + 1, self.K):
                for i in range(p):
                    u[k, l, i] = solver.BoolVar(
                        "u[{}, {}, {}]".format(k, l, i))

                    for j in range(nbins[i]):
                        d[k, l, i, j] = solver.BoolVar(
                            "d[{}, {}, {}, {}]".format(k, l, i, j))

        return x_p, t_p, t_m, m_p, m_m, a, z, q_p, q_m, h, s, f, u, d

    def add_constraint_proximity(self, solver, p, x, nbins, metric, x_p, t_p,
                                 t_m, z):
        for k in range(self.K):
            for i in range(p):
                # Absolute value
                solver.Add(t_p[k, i] - t_m[k, i] == x[i] - x_p[k, i])

                # Compute x'
                solver.Add(x_p[k, i] == x[i] + solver.Sum(
                    [(metric[i][j] - x[i]) * z[k, i, j]
                     for j in range(nbins[i])]))

    def add_constraint_closeness(self, solver, p, F, mu, x_p, m_p, m_m):
        # Mahalanobis distance l1-norm
        for k in range(self.K):
            for i in range(p):
                solver.Add(m_p[k, i] - m_m[k, i] == solver.Sum(
                    [F[j, i] * (x_p[k, j] - mu[j]) for j in range(i, p)]))

    def add_constraint_max_changes(self, solver, p, nbins, a, z):
        for k in range(self.K):
            for i in range(p):
                solver.Add(a[k, i] == solver.Sum(
                    [z[k, i, j] for j in range(nbins[i])]))
                solver.Add(a[k, i] <= 1)

            solver.Add(
                solver.Sum([a[k, i] for i in range(p)]) <= self.max_changes)

    def add_constraint_actionable(self, solver, p, a):
        for k in range(self.K):
            for i in self.non_actionable:
                solver.Add(a[k, i] == 0)

    def add_constraint_opposite(self, solver, p, y, intercept, coef, x_p):
        for k in range(self.K):
            activation = intercept + solver.Sum(
                [coef[i] * x_p[k, i] for i in range(p)])

            if y == 1:
                solver.Add(activation >= 1e-6)
            else:
                solver.Add(activation <= 0)

    def add_constraint_min_max_diff_outcome(self, solver, p, y, intercept,
                                            coef, outcome_type, b_pw, c_pw,
                                            x_p, q_p, q_m, h, s, f):

        for k in range(self.K):
            activation = intercept + solver.Sum(
                [coef[i] * x_p[k, i] for i in range(p)])

            if outcome_type == "probability":
                nbins_pw = len(b_pw)

                solver.Add(
                    solver.Sum(
                        [h[k, r] for r in range(nbins_pw)]) == activation)

                solver.Add(solver.Sum(
                    [c_pw[r][1] * h[k, r] + c_pw[r][0] * s[k, r]
                     for r in range(nbins_pw)]) == f[k])

                solver.Add(solver.Sum([s[k, r] for r in range(nbins_pw)]) == 1)

                for r in range(nbins_pw):
                    solver.Add(b_pw[r][0] * s[k, r] <= h[k, r])
                    solver.Add(h[k, r] <= b_pw[r][1] * s[k, r])

                if "min_outcome" in self.hard_constraints:
                    solver.Add(f[k] >= y)

                if "max_outcome" in self.hard_constraints:
                    solver.Add(f[k] <= y)

                if "diff_outcome" in self.soft_constraints:
                    solver.Add(q_p[k] - q_m[k] == f[k] - y)
            else:
                if "min_outcome" in self.hard_constraints:
                    solver.Add(activation >= y)

                if "max_outcome" in self.hard_constraints:
                    solver.Add(activation <= y)

                if "diff_outcome" in self.soft_constraints:
                    solver.Add(q_p[k] - q_m[k] == activation - y)

    def obj_proximity(self, solver, p, wrange, t_p, t_m):
        # Absolute distance
        return solver.Sum([solver.Sum([wrange[i] * (t_p[k, i] + t_m[k, i])
                                       for i in range(p)])
                           for k in range(self.K)])

    def obj_closeness(self, solver, p, m_p, m_m):
        # Mahalanobis distance
        return solver.Sum([solver.Sum([(m_p[k, i] + m_m[k, i])
                                       for i in range(p)])
                           for k in range(self.K)])

    def obj_diff_outcome(self, solver, q_p, q_m):
        # Absolute diff outcome
        return solver.Sum([q_p[k] + q_m[k] for k in range(self.K)])

    def obj_diversity_features(self, solver, p, u):
        # Add description
        return solver.Sum([solver.Sum([solver.Sum([u[k, l, i]
                                                   for i in range(p)])
                                       for l in range(k + 1, self.K)])
                           for k in range(self.K)])

    def obj_diversity_values(self, solver, p, nbins, d):
        # Add description
        return solver.Sum(
            [solver.Sum([solver.Sum([solver.Sum([d[k, l, i, j]
                                                 for j in range(nbins[i])])
                                     for i in range(p)])
             for l in range(k + 1, self.K)]) for k in range(self.K)])

    def compute_objectives(self, solver, names, p, nbins, wrange, t_p, t_m,
                           m_p, m_m, q_p, q_m, u, d):
        objs = {}
        for name in names:
            if name == "proximity":
                objs[name] = self.obj_proximity(solver, p, wrange, t_p, t_m)
            elif name == "closeness":
                objs[name] = self.obj_closeness(solver, p, m_p, m_m)
            elif name == "diff_outcome":
                objs[name] = self.obj_diff_outcome(solver, q_p, q_m)
            elif name == "diversity_features":
                objs[name] = -self.obj_diversity_features(solver, p, u)
            elif name == "diversity_values":
                objs[name] = -self.obj_diversity_values(solver, p, nbins, d)

        return objs
