"""
Counterfactual explanations for scorecard models.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2021

import numbers

import numpy as np
import pandas as pd

from ...logging import Logger
from ..scorecard import Scorecard
from .base import BaseCounterfactual
from .mip import CFMIP
from .model_data import model_data
from .multi_mip import MCFMIP
from .problem_data import problem_data


OBJECTIVES = ("proximity", "closeness")


HARD_CONSTRAINTS = {
    "binary": ("diversity_features", "diversity_values"),
    "probability": ("diversity_features", "diversity_values",
                    "min_outcome", "max_outcome"),
    "continuous": ("diversity_features", "diversity_values",
                   "min_outcome", "max_outcome")
}


SOFT_CONSTRAINTS = {
    "binary": ("diversity_features", "diversity_values"),
    "probability": ("diversity_features", "diversity_values", "diff_outcome"),
    "continuous": ("diversity_features", "diversity_values", "diff_outcome")
}


def _check_parameters(scorecard, special_missing, priority_tol, time_limit,
                      verbose):
    # Check scorecard
    if not isinstance(scorecard, Scorecard):
        raise TypeError()

    scorecard._check_is_fitted()

    if not isinstance(special_missing, bool):
        raise TypeError("special_missing must be a boolean; got {}."
                        .format(special_missing))

    if (not isinstance(priority_tol, numbers.Number) or
            not 0 <= priority_tol <= 1):
        raise ValueError()

    if not isinstance(time_limit, numbers.Number) or time_limit < 0:
        raise ValueError("time_limit must be a positive value in seconds; "
                         "got {}.".format(time_limit))

    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a boolean; got {}.".format(verbose))


def _check_generate_params(query, y, outcome_type, n_cf, method, objectives,
                           max_changes, actionable_features, hard_constraints,
                           soft_constraints, variable_names, target_dtype):

    # Check query
    if not isinstance(query, (dict, pd.DataFrame)):
        raise TypeError()

    # Check target
    if not isinstance(y, numbers.Number):
        raise TypeError()

    # Check target and outcome type
    if target_dtype == "binary":
        if outcome_type not in ("binary", "probability"):
            raise ValueError()
        elif outcome_type == "binary" and y not in [0, 1]:
            raise ValueError()
        elif outcome_type == "probability" and not 0 <= y <= 1:
            raise ValueError()
    elif target_dtype == "continuous":
        if outcome_type != "continuous":
            raise ValueError()

    # Check number of counterfactuals
    if not isinstance(n_cf, numbers.Integral) or n_cf <= 0:
        raise ValueError()

    # Check actionable features
    if actionable_features is not None:
        if not isinstance(actionable_features, (list, np.ndarray)):
            raise TypeError()

        for av in actionable_features:
            if av not in variable_names:
                raise ValueError()

    # Check method and constraints
    _check_objectives_method_constraints(
        method, objectives, hard_constraints, soft_constraints,
        outcome_type)


def _check_objectives_method_constraints(method, objectives, hard_constraints,
                                         soft_constraints, outcome_type):

    # Check types
    if method not in ("weighted", "hierarchical"):
        raise ValueError()

    if objectives is not None:
        if not isinstance(objectives, dict):
            raise TypeError()

        if not len(objectives):
            raise ValueError()

        for obj, value in objectives.items():
            if obj not in OBJECTIVES:
                raise ValueError()
            elif not isinstance(value, numbers.Number) or value <= 0:
                raise ValueError()

    if hard_constraints is not None:
        if not isinstance(hard_constraints, (list, tuple, np.ndarray)):
            raise TypeError()

        if len(hard_constraints) != len(set(hard_constraints)):
            raise ValueError()

        for hc in set(hard_constraints):
            if hc not in HARD_CONSTRAINTS[outcome_type]:
                raise ValueError()

    if soft_constraints is not None:
        if not isinstance(soft_constraints, dict):
            raise TypeError()

        if len(soft_constraints) != len(set(soft_constraints)):
            raise ValueError()

        for sc, value in soft_constraints.items():
            if sc not in SOFT_CONSTRAINTS[outcome_type]:
                raise ValueError()
            elif not isinstance(value, numbers.Number) or value <= 0:
                raise ValueError()

    # Check combination of hard and soft constraints for outcome type
    # probability and continuous. Al least one of:
    # - min_outcome
    # - max_outcome
    # - diff_outcome
    # must be included.
    if outcome_type in ("probability", "continuous"):
        if hard_constraints is None and soft_constraints is None:
            raise ValueError()

        # check number of suitable constraints
        _scons = ("min_outcome", "max_outcome", "diff_outcome")
        _hard = list(hard_constraints) if hard_constraints is not None else []
        _soft = list(soft_constraints) if soft_constraints is not None else []

        _hard_soft = np.array(_hard + _soft)
        _selected = np.array([c in _scons for c in _hard_soft])
        n_selected = np.count_nonzero(_selected)

        if n_selected == 0:
            raise ValueError('If outcome_type={}, at least one of the '
                             'hard_constraints "min_outcome", "max_outcome" '
                             'or the soft_constraint "diff_outcome" must be '
                             'selected.'.format(outcome_type))


class Counterfactual(BaseCounterfactual):
    def __init__(self, scorecard, special_missing=False, priority_tol=0.1,
                 n_jobs=1, time_limit=10, verbose=True):

        self.scorecard = scorecard
        self.special_missing = special_missing
        self.priority_tol = priority_tol

        self.n_jobs = n_jobs
        self.time_limit = time_limit

        self.verbose = verbose

        # auxiliary
        self._cfs = None
        self._status = None

        # logger
        self._class_logger = Logger(__name__)
        self._logger = self._class_logger.logger

        # flags
        self._is_fitted = False
        self._is_generated = False

    def fit(self, df):
        """"""
        if not isinstance(df, pd.DataFrame):
            raise TypeError()

        # Scorecard selected variables
        self._variable_names = self.scorecard.binning_process_.get_support(
            names=True)

        for v in self._variable_names:
            if v not in df.columns:
                raise ValueError()

        # Problem data
        intercept, coef, min_p, max_p, wrange, F, mu = problem_data(
            self.scorecard, df[self._variable_names])

        self._intercept = intercept
        self._coef = coef
        self._min_p = min_p
        self._max_p = max_p
        self._wrange = wrange
        self._F = F
        self._mu = mu

        self._is_fitted = True

    def generate(self, query, y, outcome_type, n_cf, method="weighted",
                 objectives=None, max_changes=None, actionable_features=None,
                 hard_constraints=None, soft_constraints=None):

        self._check_is_fitted()

        # Check parameters
        _check_generate_params(
            query, y, outcome_type, n_cf, method, objectives, max_changes,
            actionable_features, hard_constraints, soft_constraints,
            self._variable_names, self.scorecard._target_dtype)

        # Transform query using scorecard binning process
        x, query = self._transform_query(query)

        # Set default objectives
        if objectives is None:
            if method == "weighted":
                _objectives = dict(zip(OBJECTIVES, (1, 1)))
            else:
                _objectives = dict(zip(OBJECTIVES, (2, 1)))
        else:
            _objectives = objectives

        # Clean constraints given the number of counterfactuals
        _hard_constraints, _soft_constraints = self._prepare_constraints(
            outcome_type, n_cf, hard_constraints, soft_constraints)

        # Indices of non actionable features
        non_actionable = self._non_actionable_indices(actionable_features)

        # Optimization problem
        if n_cf == 1:
            optimizer = CFMIP(method, _objectives, max_changes, non_actionable,
                              _hard_constraints, _soft_constraints,
                              self.priority_tol, self.n_jobs, self.time_limit)
        else:
            optimizer = MCFMIP(n_cf, method, _objectives, max_changes,
                               non_actionable, _hard_constraints,
                               _soft_constraints, self.priority_tol,
                               self.n_jobs, self.time_limit)

        # Problem data. Indices is required to construct counterfactual
        nbins, metric, indices = model_data(
            self.scorecard, x, self.special_missing)

        optimizer.build_model(self.scorecard, x, y, outcome_type,
                              self._intercept, self._coef, self._min_p,
                              self._max_p, self._wrange, self._F, self._mu,
                              nbins, metric)

        # Optimization
        status, solution = optimizer.solve()

        self._status = status
        self._solution = solution

        self._optimizer = optimizer

        # Post-processing
        if status in ("OPTIMAL", "FEASIBLE"):
            cfs = []
            sc = self.scorecard.table()

            if n_cf == 1:
                new_indices, new_query, score = self._get_counterfactual(
                    query, sc, x, nbins, metric, indices, solution)

                cfs.append({"outcome_type": outcome_type,
                            "query": new_query,
                            "score": score,
                            "features": new_indices.keys()})
            else:
                for k in range(n_cf):
                    new_indices, new_query, score = self._get_counterfactual(
                        query, sc, x, nbins, metric, indices, solution[k])

                    cfs.append({"outcome_type": outcome_type,
                                "query": new_query,
                                "score": score,
                                "features": new_indices.keys()})
        else:
            cfs = None

        self._cfs = cfs

        self._is_generated = True

        return self

    def _get_counterfactual(self, query, sc, x, nbins, metric, indices,
                            solution):
        new_indices = {}
        score = 0
        for i, v in enumerate(self._variable_names):
            new_index = np.array(indices[i])[solution[i]]
            if len(new_index):
                new_indices[v] = new_index

            new_metric = x[i] + np.sum(
                [(metric[i][j] - x[i]) * solution[i][j]
                 for j in range(nbins[i])])

            score += self._coef[i] * new_metric

        score += self._intercept

        new_query = query.copy()
        for v, index in new_indices.items():
            new_query[v] = sc[sc["Variable"] == v]["Bin"][index].values

        return new_indices, new_query, score

    def display(self, only_changes=False, show_outcome=False):
        """
        Parameters
        ----------
        only_changes : boolean (default=False)

        show_outcome : boolean (default=False)

        Returns
        -------
        counterfactuals : pandas.DataFrame
        """
        self._check_is_generated()

        if self._cfs is None:
            raise Exception("No feasible counterfactuals were found.")

        cf_queries = []

        for cf in self._cfs:
            cf_query = cf["query"].copy()

            if only_changes:
                cf_features = cf["features"]
                for v in cf_query.columns:
                    if v not in cf_features:
                        cf_query[v] = "-"

            if show_outcome:
                outcome_type = cf["outcome_type"]

                if outcome_type == "continuous":
                    cf_query["outcome"] = cf["score"]
                else:
                    cf_score = cf["score"]
                    cf_query["outcome"] = 1.0 / (1.0 + np.exp(-cf_score))

            cf_queries.append(cf_query)

        return pd.concat(cf_queries)

    def _transform_query(self, query):
        if isinstance(query, dict):
            query = pd.DataFrame.from_dict(query, orient="index").T

        x = self.scorecard.binning_process_.transform(
            query[self._variable_names]).values.ravel()

        return x, query

    def _prepare_constraints(self, outcome_type, n_cf, hard_constraints,
                             soft_constraints):

        # Remove diversity_features and diversity_values if n_cf == 1.
        hard_cons = {} if hard_constraints is None else hard_constraints
        soft_cons = {} if soft_constraints is None else soft_constraints

        return hard_cons, soft_cons

    def _non_actionable_indices(self, actionable_features):
        non_actionable = []

        if actionable_features is not None:
            for i, av in enumerate(self._variable_names):
                if av not in actionable_features:
                    non_actionable.append(i)

        return non_actionable

    @property
    def status(self):
        return self._status
