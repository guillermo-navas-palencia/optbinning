"""
Scorecard development.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Kasper De Blieck <kasperdeblieck@hotmail.com>
# Copyright (C) 2020

import numbers
import pickle
import time

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.utils.multiclass import type_of_target

from ..binning.base import Base
from ..binning.binning_process import BinningProcess
from ..logging import Logger
from .rounding import RoundingMIP
from .scorecard_information import print_scorecard_information


logger = Logger(__name__).logger


def _check_parameters(binning_process, estimator, scaling_method,
                      scaling_method_params, intercept_based,
                      reverse_scorecard, rounding, verbose):

    if not isinstance(binning_process, BinningProcess):
        raise TypeError("binning_process must be a BinningProcess instance.")

    if not (hasattr(estimator, "fit") and hasattr(estimator, "predict")):
        raise TypeError("estimator must be an object with methods fit and "
                        "predict.")

    if scaling_method is not None:
        if scaling_method not in ("pdo_odds", "min_max"):
            raise ValueError('Invalid value for scaling_method. Allowed '
                             'string values are "pd_odds" and "min_max".')

        if scaling_method_params is None:
            raise ValueError("scaling_method_params cannot be None if "
                             "scaling_method is provided.")

        if not isinstance(scaling_method_params, dict):
            raise TypeError("scaling_method_params must be a dict.")

    if not isinstance(intercept_based, bool):
        raise TypeError("intercept_based must be a boolean; got {}."
                        .format(intercept_based))

    if not isinstance(reverse_scorecard, bool):
        raise TypeError("reverse_scorecard must be a boolean; got {}."
                        .format(reverse_scorecard))

    if not isinstance(rounding, bool):
        raise TypeError("rounding must be a boolean; got {}.".format(rounding))

    if rounding and scaling_method is None:
        raise ValueError("rounding is only applied if scaling method is "
                         "not None.")

    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a boolean; got {}.".format(verbose))


def _check_scorecard_scaling(scaling_method, scaling_method_params,
                             rounding, target_type):
    if scaling_method is not None:
        if scaling_method == "pdo_odds":
            default_keys = ("pdo", "odds", "scorecard_points")

            if target_type != "binary":
                raise ValueError('scaling_method "pd_odds" is not supported '
                                 'for a continuous target.')

        elif scaling_method == "min_max":
            default_keys = ("min", "max")

        if set(scaling_method_params.keys()) != set(default_keys):
            raise ValueError("scaling_method_params must be {} given "
                             "scaling_method = {}."
                             .format(default_keys, scaling_method))

        if scaling_method == "pdo_odds":
            for param in default_keys:
                value = scaling_method_params[param]
                if not isinstance(value, numbers.Number) or value <= 0:
                    raise ValueError("{} must be a positive number; got {}."
                                     .format(param, value))

        elif scaling_method == "min_max":
            for param in default_keys:
                value = scaling_method_params[param]
                if not isinstance(value, numbers.Number):
                    raise ValueError("{} must be numeric; got {}."
                                     .format(param, value))

            if scaling_method_params["min"] > scaling_method_params["max"]:
                raise ValueError("min must be <= max; got {} <= {}."
                                 .format(scaling_method_params["min"],
                                         scaling_method_params["max"]))

            if rounding:
                score_min = scaling_method_params['min']
                if int(score_min) != score_min:
                    raise ValueError("min must be an integer if rounding=True"
                                     "; got {}.".format(score_min))

                score_max = scaling_method_params['max']
                if int(score_max) != score_max:
                    raise ValueError("max must be an integer if rounding=True"
                                     "; got {}.".format(score_max))


def _compute_scorecard_points(points, binning_tables, method, method_data,
                              intercept, reverse_scorecard):
    """Apply scaling method to scorecard."""
    n = len(binning_tables)

    sense = -1 if reverse_scorecard else 1

    if method == "pdo_odds":
        pdo = method_data["pdo"]
        odds = method_data["odds"]
        scorecard_points = method_data["scorecard_points"]

        factor = pdo / np.log(2)
        offset = scorecard_points - factor * np.log(odds)

        new_points = -(sense * points + intercept / n) * factor + offset / n
    elif method == "min_max":
        a = method_data["min"]
        b = method_data["max"]

        min_p = np.sum([np.min(bt.Points) for bt in binning_tables])
        max_p = np.sum([np.max(bt.Points) for bt in binning_tables])

        smin = intercept + min_p
        smax = intercept + max_p

        slope = sense * (a - b) / (smax - smin)
        if reverse_scorecard:
            shift = a - slope * smin
        else:
            shift = b - slope * smin

        base_points = shift + slope * intercept
        new_points = base_points / n + slope * points

    return new_points


def _compute_intercept_based(df_scorecard):
    """Compute an intercept-based scorecard.

    All points within a variable are adjusted so that the lowest point is zero.
    """
    scaled_points = np.zeros(df_scorecard.shape[0])
    selected_variables = df_scorecard.Variable.unique()
    intercept = 0
    for variable in selected_variables:
        mask = df_scorecard.Variable == variable
        points = df_scorecard[mask].Points.values
        min_point = np.min(points)
        scaled_points[mask] = points - min_point
        intercept += min_point

    return scaled_points, intercept


class Scorecard(Base, BaseEstimator):
    """Scorecard development given a binary or continuous target dtype.

    Parameters
    ----------
    binning_process : object
        A ``BinningProcess`` instance.

    estimator : object
        A supervised learning estimator with a ``fit`` and ``predict`` method
        that provides information about feature coefficients through a
        ``coef_`` attribute. For binary classification, the estimator must
        include a ``predict_proba`` method.

    scaling_method : str or None (default=None)
        The scaling method to control the range of the scores. Supported
        methods are "pdo_odds" and "min_max". Method "pdo_odds" is only
        applicable for binary classification. If None, no scaling is applied.

    scaling_method_params : dict or None (default=None)
        Dictionary with scaling method parameters. If
        ``scaling_method="pdo_odds"`` parameters required are: "pdo", "odds",
        and "scorecard_points". If ``scaling_method="min_max"`` parameters
        required are "min" and "max". If ``scaling_method=None``, this
        parameter is not used.

    intercept_based : bool (default=False)
        Build a intercept-based scorecard. A intercept-based scorecard modifies
        the original scorecard by setting the smallest point for each variable
        to zero and updating the intercept accordingly.

    reverse_scorecard: bool (default=False)
        Whether to change the sense of the relationship between predictions and
        scorecard points to ascending/descending.

    rounding : bool (default=False)
        Whether to round scorecard points. If ``scaling_method="min_max"`` a
        mixed-integer programming problem is solved to guarantee the
        minimum/maximum score after rounding. Otherwise, the scorecard points
        are round to the nearest integer.

    verbose : bool (default=False)
        Enable verbose output.

    Attributes
    ----------
    binning_process_ : object
        The external binning process.

    estimator_ : object
        The external estimator fit on the reduced dataset.

    intercept_ : float
        The intercept if ``intercept_based=True``.
    """
    def __init__(self, binning_process, estimator, scaling_method=None,
                 scaling_method_params=None, intercept_based=False,
                 reverse_scorecard=False, rounding=False, verbose=False):

        self.binning_process = binning_process
        self.estimator = estimator
        self.scaling_method = scaling_method
        self.scaling_method_params = scaling_method_params
        self.intercept_based = intercept_based
        self.reverse_scorecard = reverse_scorecard
        self.rounding = rounding
        self.verbose = verbose

        # attributes
        self.binning_process_ = None
        self.estimator_ = None
        self.intercept_ = 0

        self._metric_special = None
        self._metric_missing = None

        # auxiliary
        self._target_dtype = None

        # timing
        self._time_total = None
        self._time_binning_process = None
        self._time_estimator = None
        self._time_build_scorecard = None
        self._time_rounding = None

        self._is_fitted = False

    def fit(self, X, y, sample_weight=None, metric_special=0, metric_missing=0,
            show_digits=2, check_input=False):
        """Fit scorecard.

        Parameters
        ----------
        X : pandas.DataFrame (n_samples, n_features)
            Training vector, where n_samples is the number of samples.

        y : array-like of shape (n_samples,)
            Target vector relative to x.

        sample_weight : array-like of shape (n_samples,) (default=None)
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.
            This option is only available for a binary target.

        metric_special : float or str (default=0)
            The metric value to transform special codes in the input vector.
            Supported metrics are "empirical" to use the empirical WoE or
            event rate, and any numerical value.

        metric_missing : float or str (default=0)
            The metric value to transform missing values in the input vector.
            Supported metrics are "empirical" to use the empirical WoE or
            event rate and any numerical value.

        check_input : bool (default=False)
            Whether to check input arrays.

        show_digits : int, optional (default=2)
            The number of significant digits of the bin column.

        Returns
        -------
        self : Scorecard
            Fitted scorecard.
        """
        return self._fit(X, y, sample_weight, metric_special, metric_missing,
                         show_digits, check_input)

    def information(self, print_level=1):
        """Print overview information about the options settings and
        statistics.

        Parameters
        ----------
        print_level : int (default=1)
            Level of details.
        """
        self._check_is_fitted()

        if not isinstance(print_level, numbers.Integral) or print_level < 0:
            raise ValueError("print_level must be an integer >= 0; got {}."
                             .format(print_level))

        n_numerical = list(
            self.binning_process_._variable_dtypes.values()).count("numerical")
        n_categorical = self.binning_process_._n_variables - n_numerical
        n_selected = np.count_nonzero(self.binning_process_._support)

        dict_user_options = self.get_params(deep=False)

        print_scorecard_information(
            print_level, self.binning_process_._n_samples,
            self.binning_process_._n_variables, self._target_dtype,
            n_numerical, n_categorical, n_selected, self._time_total,
            self._time_binning_process, self._time_estimator,
            self._time_build_scorecard, self._time_rounding, dict_user_options)

    def predict(self, X):
        """Predict using the fitted underlying estimator and the reduced
        dataset.

        Parameters
        ----------
        X : pandas.DataFrame (n_samples, n_features)
            Training vector, where n_samples is the number of samples.

        Returns
        -------
        pred: array of shape (n_samples)
            The predicted target values.
        """
        self._check_is_fitted()

        X_t = X[self.binning_process_.variable_names]
        X_t = self.binning_process_.transform(
            X, metric_special=self._metric_special,
            metric_missing=self._metric_missing)
        return self.estimator_.predict(X_t)

    def predict_proba(self, X):
        """Predict class probabilities using the fitted underlying estimator
        and the reduced dataset.

        Parameters
        ----------
        X : pandas.DataFrame (n_samples, n_features)
            Training vector, where n_samples is the number of samples.

        Returns
        -------
        p: array of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        self._check_is_fitted()

        X_t = X[self.binning_process_.variable_names]
        X_t = self.binning_process_.transform(
            X, metric_special=self._metric_special,
            metric_missing=self._metric_missing)
        return self.estimator_.predict_proba(X_t)

    def score(self, X):
        """Score of the dataset.

        Parameters
        ----------
        X : pandas.DataFrame (n_samples, n_features)
            Training vector, where n_samples is the number of samples.

        Returns
        -------
        score: array of shape (n_samples)
            The score of the input samples.
        """
        self._check_is_fitted()

        X_t = X[self.binning_process_.variable_names]
        X_t = self.binning_process_.transform(
            X_t, metric="indices", metric_special="empirical",
            metric_missing="empirical")

        score_ = np.zeros(X_t.shape[0])
        selected_variables = self.binning_process_.get_support(names=True)

        for variable in selected_variables:
            mask = self._df_scorecard.Variable == variable
            points = self._df_scorecard[mask].Points.values
            score_ += points[X_t[variable]]

        return score_ + self.intercept_

    def table(self, style="summary"):
        """Scorecard table.

        Parameters
        ----------
        style : str, optional (default="summary")
            Scorecard's style. Supported styles are "summary" and "detailed".
            Summary only includes columns variable, bin description and points.
            Detailed contained additional columns with bin information and
            estimator coefficients.

        Returns
        -------
        table : pandas.DataFrame
            The scorecard table.
        """
        self._check_is_fitted()

        if style not in ("summary", "detailed"):
            raise ValueError('Invalid value for style. Allowed string '
                             'values are "summary" and "detailed".')

        if style == "summary":
            columns = ["Variable", "Bin", "Points"]
        elif style == "detailed":
            main_columns = ["Variable", "Bin id", "Bin"]
            columns = self._df_scorecard.columns
            rest_columns = [col for col in columns if col not in main_columns]
            columns = main_columns + rest_columns

        return self._df_scorecard[columns]

    @classmethod
    def load(cls, path):
        """Load scorecard from pickle file.

        Parameters
        ----------
        path : str
            Pickle file path.

        Example
        -------
        >>> from optbinning import Scorecard
        >>> scorecard = Scorecard.load("my_scorecard.pkl")
        """
        if not isinstance(path, str):
            raise TypeError("path must be a string.")

        with open(path, "rb") as f:
            return pickle.load(f)

    def save(self, path):
        """Save scorecard to pickle file.

        Parameters
        ----------
        path : str
            Pickle file path.
        """
        if not isinstance(path, str):
            raise TypeError("path must be a string.")

        with open(path, "wb") as f:
            pickle.dump(self, f)

    def _fit(self, X, y, sample_weight, metric_special, metric_missing,
             show_digits, check_input):

        # Store the metrics for missing and special bins for predictions
        self._metric_special = metric_special
        self._metric_missing = metric_missing

        time_init = time.perf_counter()

        if self.verbose:
            logger.info("Scorecard building process started.")
            logger.info("Options: check parameters.")

        _check_parameters(**self.get_params(deep=False))

        # Check X dtype
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas.DataFrame.")

        # Target type and metric
        self._target_dtype = type_of_target(y)

        if self._target_dtype not in ("binary", "continuous"):
            raise ValueError("Target type {} is not supported."
                             .format(self._target_dtype))

        _check_scorecard_scaling(self.scaling_method,
                                 self.scaling_method_params,
                                 self.rounding,
                                 self._target_dtype)

        # Check sample weight
        if sample_weight is not None and self._target_dtype != "binary":
            raise ValueError("Target type {} does not support sample weight."
                             .format(self._target_dtype))

        if self._target_dtype == "binary":
            metric = "woe"
            bt_metric = "WoE"
        elif self._target_dtype == "continuous":
            metric = "mean"
            bt_metric = "Mean"

        if self.verbose:
            logger.info("Dataset: {} target.".format(self._target_dtype))

        # Fit binning process
        if self.verbose:
            logger.info("Binning process started.")

        time_binning_process = time.perf_counter()
        self.binning_process_ = clone(self.binning_process)
        # Suppress binning process verbosity
        self.binning_process_.set_params(verbose=False)

        X_t = self.binning_process_.fit_transform(
            X[self.binning_process.variable_names], y, sample_weight, metric,
            metric_special, metric_missing, show_digits, check_input)

        self._time_binning_process = time.perf_counter() - time_binning_process

        if self.verbose:
            logger.info("Binning process terminated. Time: {:.4f}s"
                        .format(self._time_binning_process))

        # Fit estimator
        time_estimator = time.perf_counter()
        if self.verbose:
            logger.info("Fitting estimator.")

        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X_t, y, sample_weight)

        self._time_estimator = time.perf_counter() - time_estimator

        if self.verbose:
            logger.info("Fitting terminated. Time {:.4f}s"
                        .format(self._time_estimator))

        # Get coefs
        intercept = 0
        if hasattr(self.estimator_, 'coef_'):
            coefs = self.estimator_.coef_
            if hasattr(self.estimator_, 'intercept_'):
                intercept = self.estimator_.intercept_
        else:
            raise RuntimeError('The classifier does not expose '
                               '"coef_" attribute.')

        # Build scorecard
        time_build_scorecard = time.perf_counter()

        if self.verbose:
            logger.info("Scorecard table building started.")

        selected_variables = self.binning_process_.get_support(names=True)
        binning_tables = []
        for i, variable in enumerate(selected_variables):
            optb = self.binning_process_.get_binned_variable(variable)
            binning_table = optb.binning_table.build(
                show_digits=show_digits, add_totals=False)
            c = coefs.ravel()[i]
            binning_table.loc[:, "Variable"] = variable
            binning_table.loc[:, "Coefficient"] = c
            binning_table.loc[:, "Points"] = binning_table[bt_metric] * c

            nt = len(binning_table)
            if metric_special != 'empirical':
                if isinstance(optb.special_codes, dict):
                    n_specials = len(optb.special_codes)
                else:
                    n_specials = 1

                binning_table.loc[
                    nt-1-n_specials:nt-2, "Points"] = metric_special * c
            elif metric_missing != 'empirical':
                binning_table.loc[nt-1, "Points"] = metric_missing * c

            binning_table.index.names = ['Bin id']
            binning_table.reset_index(level=0, inplace=True)
            binning_tables.append(binning_table)

        df_scorecard = pd.concat(binning_tables)
        df_scorecard.reset_index()

        # Apply score points
        if self.scaling_method is not None:
            points = df_scorecard["Points"]
            scaled_points = _compute_scorecard_points(
                points, binning_tables, self.scaling_method,
                self.scaling_method_params, intercept, self.reverse_scorecard)

            df_scorecard.loc[:, "Points"] = scaled_points

        if self.intercept_based:
            scaled_points, self.intercept_ = _compute_intercept_based(
                df_scorecard)
            df_scorecard.loc[:, "Points"] = scaled_points

        time_rounding = time.perf_counter()
        if self.rounding:
            points = df_scorecard["Points"]
            if self.scaling_method in ("pdo_odds", None):
                round_points = np.rint(points)

                if self.intercept_based:
                    self.intercept_ = np.rint(self.intercept_)
            elif self.scaling_method == "min_max":
                round_mip = RoundingMIP()
                round_mip.build_model(df_scorecard)
                status, round_points = round_mip.solve()

                if status not in ("OPTIMAL", "FEASIBLE"):
                    if self.verbose:
                        logger.warning("MIP rounding failed, method nearest "
                                       "integer used instead.")
                    # Back-up method
                    round_points = np.rint(points)

                if self.intercept_based:
                    self.intercept_ = np.rint(self.intercept_)

            df_scorecard.loc[:, "Points"] = round_points
        self._time_rounding = time.perf_counter() - time_rounding

        self._df_scorecard = df_scorecard

        self._time_build_scorecard = time.perf_counter() - time_build_scorecard
        self._time_total = time.perf_counter() - time_init

        if self.verbose:
            logger.info("Scorecard table terminated. Time: {:.4f}s"
                        .format(self._time_build_scorecard))
            logger.info("Scorecard building process terminated. Time: {:.4f}s"
                        .format(self._time_total))

        # Completed successfully
        self._is_fitted = True

        return self
