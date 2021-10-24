"""
Counterfactual explanations for scorecard models.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2021

import numbers
import time

import numpy as np
import pandas as pd

from ...information import solver_statistics
from ...logging import Logger
from ..scorecard import Scorecard
from .base import BaseCounterfactual
from .counterfactual_information import print_counterfactual_information
from .mip import CFMIP
from .model_data import model_data
from .multi_mip import MCFMIP
from .problem_data import problem_data


logger = Logger(__name__).logger


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


def _check_parameters(scorecard, special_missing, n_jobs, verbose):
    # Check scorecard
    if not isinstance(scorecard, Scorecard):
        raise TypeError("scorecard must be a Scorecard instance.")

    scorecard._check_is_fitted()

    if not isinstance(special_missing, bool):
        raise TypeError("special_missing must be a boolean; got {}."
                        .format(special_missing))

    if not isinstance(n_jobs, numbers.Integral) or n_jobs <= 0:
        raise ValueError("n_jobs must be a positive integer; got {}."
                         .format(n_jobs))

    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a boolean; got {}.".format(verbose))


def _check_generate_params(query, y, outcome_type, n_cf, method, objectives,
                           max_changes, actionable_features, hard_constraints,
                           soft_constraints, variable_names, target_dtype):

    # Check query
    if not isinstance(query, (dict, pd.DataFrame)):
        raise TypeError("query must be a dict or a pandas.DataFrame.")

    # Check target
    if not isinstance(y, numbers.Number):
        raise TypeError("y must be numeric.")

    # Check target and outcome type
    if target_dtype == "binary":
        if outcome_type not in ("binary", "probability"):
            raise ValueError("outcome_type must either binary or probability "
                             "if target_dtype=binary; got {}."
                             .format(outcome_type))
        elif outcome_type == "binary" and y not in [0, 1]:
            raise ValueError("y must be either 0 or 1 if outcome_type=binary; "
                             "got {}.".format(y))
        elif outcome_type == "probability" and not 0 <= y <= 1:
            raise ValueError("y must be in [0, 1] if outcome_type=probability "
                             "; got {}.".format(y))
    elif target_dtype == "continuous":
        if outcome_type != "continuous":
            raise ValueError("outcome_type must be continuous if "
                             "target_dtype=continuous; got {}."
                             .format(outcome_type))

    # Check number of counterfactuals
    if not isinstance(n_cf, numbers.Integral) or n_cf <= 0:
        raise ValueError("n_cf must be a positive integer; got {}."
                         .format(n_cf))

    if max_changes is not None:
        if not isinstance(max_changes, numbers.Integral) or max_changes <= 0:
            raise ValueError("max_changes must be a positive integer; got {}."
                             .format(max_changes))

    # Check actionable features
    if actionable_features is not None:
        if not isinstance(actionable_features, (list, np.ndarray)):
            raise TypeError("actionable_features must be either a list or "
                            "a numpy.ndarray.")

        for av in actionable_features:
            if av not in variable_names:
                raise ValueError("actionable feature {} is not in {}."
                                 .format(av, variable_names))

    # Check method and constraints
    _check_objectives_method_constraints(
        method, objectives, hard_constraints, soft_constraints,
        outcome_type)


def _check_objectives_method_constraints(method, objectives, hard_constraints,
                                         soft_constraints, outcome_type):

    # Check types
    if method not in ("weighted", "hierarchical"):
        raise ValueError('Invalid value for method. Allowed string values are '
                         '"weighted" and "hierarchical".')

    if objectives is not None:
        if not isinstance(objectives, dict):
            raise TypeError("objectives must be a dict.")

        if not len(objectives):
            raise ValueError("objectives cannot be empty.")

        for obj, value in objectives.items():
            if obj not in OBJECTIVES:
                raise ValueError("objective names must be in {}; got {}."
                                 .format(OBJECTIVES, obj))
            elif not isinstance(value, numbers.Number) or value <= 0:
                raise ValueError("objective values must be positive; got {}."
                                 .format({obj, value}))

    if hard_constraints is not None:
        if not isinstance(hard_constraints, (list, tuple, np.ndarray)):
            raise TypeError("hard_constraints must a list, tuple or "
                            "numpy.ndarray.")

        if len(hard_constraints) != len(set(hard_constraints)):
            raise ValueError("hard_constraints cannot be repeated.")

        for hc in hard_constraints:
            if hc not in HARD_CONSTRAINTS[outcome_type]:
                raise ValueError(
                    "Invalid hard constraint for outcome_type={}. Allowed "
                    "strings values are {}.".format(
                        outcome_type, HARD_CONSTRAINTS[outcome_type]))

    if soft_constraints is not None:
        if not isinstance(soft_constraints, dict):
            raise TypeError("soft_constraints must be a dict.")

        for sc, value in soft_constraints.items():
            if sc not in SOFT_CONSTRAINTS[outcome_type]:
                raise ValueError(
                    "Invalid soft constraint for outcome_type={}. Allowed "
                    "string values are {}.".format(
                        outcome_type, SOFT_CONSTRAINTS[outcome_type]))
            elif not isinstance(value, numbers.Number) or value <= 0:
                raise ValueError("soft constraint values must be positive; "
                                 "got {}.".format({sc, value}))

    # Check combination of hard and soft constraints for outcome type
    # probability and continuous. Al least one of:
    #   [min_outcome, max_outcome, diff_outcome]
    # must be included.
    if outcome_type in ("probability", "continuous"):
        if hard_constraints is None and soft_constraints is None:
            raise ValueError("If outcome_type is either probability or "
                             "continuous, at least one hard constraint or"
                             "soft constraint must be provided.")

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
    """Optimal counterfactual explanations given a scorecard model.

    Parameters
    ----------
    scorecard : object
        A ``Scorecard`` instance.

    special_missing : bool (default=False)
        Whether the special and missing bin are considered as valid
        counterfactual values.

    n_jobs : int, optional (default=1)
        Number of cores to run the optimization solver.

    verbose : bool (default=False)
        Enable verbose output.
    """
    def __init__(self, scorecard, special_missing=False, n_jobs=1,
                 verbose=False):
        self.scorecard = scorecard
        self.special_missing = special_missing
        self.n_jobs = n_jobs

        self.verbose = verbose

        # auxiliary
        self._cfs = None

        # info
        self._optimizer = None
        self._status = None

        # timing
        self._time_fit = None
        self._time_solver = None
        self._time_postprocessing = None

        # flags
        self._is_fitted = False
        self._is_generated = False

    def fit(self, X):
        """Fit counterfactual. Compute problem data to generate counterfactual
        explanations.

        Parameters
        ----------
        X : pandas.DataFrame (n_samples, n_features)
            Training vector, where n_samples is the number of samples.

        Returns
        -------
        self : Counterfactual
            Fitted counterfactual.
        """
        time_init = time.perf_counter()

        if self.verbose:
            logger.info("Counterfactual fit started.")
            logger.info("Options: check parameters.")

        _check_parameters(**self.get_params(deep=False))

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas.DataFrame.")

        # Scorecard selected variables
        self._variable_names = self.scorecard.binning_process_.get_support(
            names=True)

        for v in self._variable_names:
            if v not in X.columns:
                raise ValueError("Variable {} not in X. X must include {}."
                                 .format(v, self._variable_names))

        if self.verbose:
            logger.info("Compute optimization problem data.")

        # Problem data
        intercept, coef, min_p, max_p, wrange, F, mu = problem_data(
            self.scorecard, X[self._variable_names])

        self._intercept = intercept
        self._coef = coef
        self._min_p = min_p
        self._max_p = max_p
        self._wrange = wrange
        self._F = F
        self._mu = mu

        self._time_fit = time.perf_counter() - time_init

        if self.verbose:
            logger.info("Counterfactual fit terminated. Time: {:.4f}s"
                        .format(self._time_fit))

        self._is_fitted = True
        self._is_generated = False

        return self

    def information(self, print_level=1):
        """Print overview information about the options settings and
        statistics.

        Parameters
        ----------
        print_level : int (default=1)
            Level of details.
        """
        self._check_is_generated()

        if not isinstance(print_level, numbers.Integral) or print_level < 0:
            raise ValueError("print_level must be an integer >= 0; got {}."
                             .format(print_level))

        if self._optimizer is not None:
            solver, _ = solver_statistics("mip", self._optimizer.solver_)
            objectives = self._optimizer._objectives
            time_solver = self._time_solver
        else:
            solver = None
            objectives = None
            time_solver = 0

        time_total = self._time_fit + time_solver + self._time_postprocessing

        dict_user_options = self.get_params(deep=False)

        print_counterfactual_information(
            print_level, self._status, solver, objectives, time_total,
            self._time_fit, time_solver, self._time_postprocessing,
            dict_user_options)

    def generate(self, query, y, outcome_type, n_cf, method="weighted",
                 objectives=None, max_changes=None, actionable_features=None,
                 hard_constraints=None, soft_constraints=None,
                 priority_tol=0.1, time_limit=10):
        """Generate counterfactual explanations given objectives and
        constraints.

        Parameters
        ----------
        query : dict or pandas.DataFrame
            Input data points for which a single or multiple counterfactual
            explanations are to be generated.

        y : int or float
            Desired outcome.

        outcome_type : str
            Desired outcome type. Supported outcome types are "binary",
            "probability" and "continuous".

        n_cf : int
            Number of counterfactuals to be generated.

        method : str (default="weighted")
            Multi-objective optimization method. Supported methods are
            "weighted" and "hierarchical".

        objectives : dict or None (default=None)
            Objectives with their corresponding weights or priorities,
            depending on the method.

        max_changes : int or None (default=None)
            Maximum number of features to be changed. If None, the maximum
            number of changes is half of the number of features.

        actionable_features : array-like or None (default=None)
            List of actionable features. If None. all features are suitable to
            be changed.

        hard_constraints : array-like or None (default=None)
            Constraint to be enforced when solving the underlying optimization
            problem.

        soft_constraints : dict or None (default=None)
            Constraints to be moved to the objective function as a penalization
            term.

        priority_tol : float, optional (default=0.1)
            Relative tolerance when solving the multi-objective optimization
            problem with ``method="hierarchical"``.

        time_limit : int (default=10)
            The maximum time in seconds to run the optimization solver.

        Returns
        -------
        self : Counterfactual
            Generated counterfactuals.
        """
        time_init = time.perf_counter()

        self._check_is_fitted()

        if self.verbose:
            logger.info("Counterfactual generation started.")
            logger.info("Options: check parameters.")

        # Check parameters
        _check_generate_params(
            query, y, outcome_type, n_cf, method, objectives, max_changes,
            actionable_features, hard_constraints, soft_constraints,
            self._variable_names, self.scorecard._target_dtype)

        # Check priority tolerance
        if (not isinstance(priority_tol, numbers.Number) or
                not 0 <= priority_tol <= 1):
            raise ValueError("priority_tol must be in [0, 1]; got {}."
                             .format(priority_tol))

        # Check time limit
        if not isinstance(time_limit, numbers.Number) or time_limit <= 0:
            raise ValueError("time_limit must be a positive value in seconds; "
                             "got {}.".format(time_limit))

        # Transform query using scorecard binning process
        x, query = self._transform_query(query)

        if self.verbose:
            logger.info("Options: check objectives and constraints.")

        # Set default objectives
        if objectives is None:
            if method == "weighted":
                _objectives = dict(zip(OBJECTIVES, (1, 1)))
            else:
                _objectives = dict(zip(OBJECTIVES, (2, 1)))
        else:
            _objectives = objectives

        # Set max changes
        if max_changes is None:
            _max_changes = len(self._variable_names) // 2
        else:
            _max_changes = max_changes

        # Clean constraints given the number of counterfactuals
        _hard_constraints, _soft_constraints = self._prepare_constraints(
            outcome_type, n_cf, hard_constraints, soft_constraints)

        # Indices of non actionable features
        non_actionable = self._non_actionable_indices(actionable_features)

        # Optimization problem
        if self.verbose:
            logger.info("Optimizer started.")

        time_solver = time.perf_counter()

        if n_cf == 1:
            optimizer = CFMIP(method, _objectives, _max_changes,
                              non_actionable, _hard_constraints,
                              _soft_constraints, priority_tol, self.n_jobs,
                              time_limit)
        else:
            optimizer = MCFMIP(n_cf, method, _objectives, _max_changes,
                               non_actionable, _hard_constraints,
                               _soft_constraints, priority_tol, self.n_jobs,
                               time_limit)

        # Problem data. Indices is required to construct counterfactual
        if self.verbose:
            logger.info("Optimizer: build model...")

        nbins, metric, indices = model_data(
            self.scorecard, x, self.special_missing)

        optimizer.build_model(self.scorecard, x, y, outcome_type,
                              self._intercept, self._coef, self._min_p,
                              self._max_p, self._wrange, self._F, self._mu,
                              nbins, metric)

        # Optimization
        if self.verbose:
            logger.info("Optimizer: solve...")

        status, solution = optimizer.solve()

        self._status = status
        self._optimizer = optimizer

        self._time_solver = time.perf_counter() - time_solver

        if self.verbose:
            logger.info("Optimizer terminated. Time: {:.4f}s"
                        .format(self._time_solver))

        # Post-processing
        if self.verbose:
            logger.info("Post-processing started.")

        time_postprocessing = time.perf_counter()

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

        self._time_postprocessing = time.perf_counter() - time_postprocessing

        if self.verbose:
            logger.info("Post-processing terminated. Time: {:.4f}s"
                        .format(self._time_postprocessing))

        self._time_total = time.perf_counter() - time_init

        if self.verbose:
            logger.info("Counterfactual generation terminated. Status: {}. "
                        "Time: {:.4f}s".format(self._status, self._time_total))

        # Completed successfully
        self._is_generated = True

        return self

    def display(self, show_only_changes=False, show_outcome=False):
        """Display the generatedcounterfactual explanations.

        Parameters
        ----------
        show_only_changes : boolean (default=False)
            Whether to show only changes on feature values.

        show_outcome : boolean (default=False)
            Whether to add a column with the scorecard outcome. If
            ``outcome_type`` is "binary" or "probability", the estimated
            probability of the counterfactual is added.

        Returns
        -------
        counterfactuals : pandas.DataFrame
            Counterfactual explanations.
        """
        self._check_is_generated()
        self._check_counterfactual_is_found()

        if not isinstance(show_only_changes, bool):
            raise TypeError("show_only_changes must be a boolean; got {}."
                            .format(show_only_changes))

        if not isinstance(show_outcome, bool):
            raise TypeError("show_outcome must be a boolean; got {}."
                            .format(show_outcome))

        cf_queries = []
        for cf in self._cfs:
            cf_query = cf["query"].copy()

            if show_only_changes:
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

    def _transform_query(self, query):
        if isinstance(query, dict):
            query = pd.DataFrame.from_dict(query, orient="index").T

        x = self.scorecard.binning_process_.transform(
            query[self._variable_names]).values.ravel()

        return x, query

    def _prepare_constraints(self, outcome_type, n_cf, hard_constraints,
                             soft_constraints):
        # Remove diversity_features and diversity_values if n_cf == 1.
        diversity_constraints = ["diversity_features", "diversity_values"]

        if hard_constraints is None:
            hard_cons = {}
        elif n_cf == 1:
            hard_cons = [c for c in hard_constraints
                         if c not in diversity_constraints]
        else:
            hard_cons = hard_constraints

        if soft_constraints is None:
            soft_cons = {}
        elif n_cf == 1:
            soft_cons = {c: v for c, v in soft_constraints.items()
                         if c not in diversity_constraints}
        else:
            soft_cons = soft_constraints

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
        """The status of the underlying optimization solver.

        Returns
        -------
        status : str
        """
        self._check_is_generated()

        return self._status
