"""
Counterfactual explanations for scorecard models.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2021

import numbers

import pandas as pd

from ...logging import Logger
from ..scorecard import Scorecard
from .base import BaseCounterfactual
from .mip import CFMIP


OBJECTIVES = ("proximity", "closeness")


HARD_CONSTRAINTS = {
    "binary": ("diversity_features", "diversity_values"),
    "probability": ("diversity_features", "diversity_values",
                    "min_outcome", "max_outcome")
    "continuous": ("diversity_features", "diversity_values",
                   "min_outcome", "max_outcome")
}


SOFT_CONSTRAINTS = {
    "binary": ("diversity_features", "diversity_values"),
    "probability": ("diversity_features", "diversity_values", "diff_outcome")
    "continuous": ("diversity_features", "diversity_values", "diff_outcome")
}


def _check_parameters(scorecard, solver, priority_tol, time_limit, verbose):
    # Check scorecard
    if not isinstance(scorecard, Scorecard):
        raise TypeError()

    scorecard._check_is_fitted()

    if solver not in ("cp", "mip"):
        raise ValueError()

    if (not isinstance(priority_tol, numbers.Number) or
            not 0 <= priority_tol <= 1):
        raise ValueError()

    if not isinstance(time_limit, numbers.Number) or time_limit < 0:
        raise ValueError("time_limit must be a positive value in seconds; "
                         "got {}.".format(time_limit))

    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a boolean; got {}.".format(verbose))    


def _check_generate_params(query, y, outcome_type, n_cf, max_changes,
                           actionable_features, objectives, method, weights,
                           priority, hard_constraints, soft_constraints,
                           variable_names, target_dtype):

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


def _check_method_constraints(method, objectives, hard_constraints,
                              soft_constraints, outcome_type):
    
    # Check types
    if method not in ("weighted", "hierarchical"):
        raise ValueError()

    if objectives is not None:
        if not isinstance(objectives, dict):
            raise TypeError()

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
            raise ValueError('If outcome_type={}, one of the hard_constraints '
                             '"min_outcome", "max_outcome" or the '
                             'soft_constraint "diff_outcome" must be '
                             'selected.'.format(outcome_type))
        elif n_selected > 1:
            raise ValueError("Only one of the constraints in {} can be "
                             "selected; got {}."
                             .format(_scons, _hard_soft[_selected]))


class Counterfactual(BaseCounterfactual):
    def __init__(self, scorecard, solver="mip", priority_tol=0.1,
                 time_limit=10, verbose=True):
        self.scorecard = scorecard
        self.solver = solver
        self.priority_tol = priority_tol
        self.time_limit = time_limit
        self.verbose = verbose

        # auxiliary

        # logger
        self._class_logger = Logger(__name__)
        self._logger = self._class_logger.logger

        # flags
        self._is_fitted = False

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

        # Check parameters
        _check_generate_params(
            query, y, outcome_type, n_cf, method, objectives, max_changes,
            actionable_features, hard_constraints, soft_constraints,
            self._variable_names, self.scorecard._target_dtype)

        # Transform query using scorecard binning process
        x = self._transform_query(query)

        # Set default objectives
        if objectives is None:
            if method == "weighted":
                _objectives = dict(zip(OBJECTIVES, (1, 1)))
            else:
                _objectives = dict(zip(OBJECTIVES, (2, 1)))
        else:
            _objectives = objectives

        # Optimization problem
        if self.solver == "cp":
            pass
        elif self.solver == "mip":
            optimizer = CFMIP()

        optimizer.build_model()

        status, solution = optimizer.solve()

        self._status = status
        self._solution = solution

        self._optimizer = optimizer

    def display(self):
        pass
