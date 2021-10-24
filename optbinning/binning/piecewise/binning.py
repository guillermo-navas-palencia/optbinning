"""
Optimal piecewise binning algorithm for binary target.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import time

from sklearn.linear_model import LogisticRegression

from ...binning.binning_statistics import target_info
from ...logging import Logger
from .base import _check_parameters
from .base import BasePWBinning
from .binning_statistics import PWBinningTable
from .metrics import binary_metrics
from .transformations import transform_binary_target


logger = Logger(__name__).logger


class OptimalPWBinning(BasePWBinning):
    """Optimal Piecewise binning of a numerical variable with respect to a
    binary target.

    Parameters
    ----------
    name : str, optional (default="")
        The variable name.

    estimator : object or None (default=None)
        An esimator to compute probability estimates. If None, it uses
        `sklearn.linear_model.LogisticRegression
        <https://scikit-learn.org/stable/modules/generated/
        sklearn.linear_model.LogisticRegression.html>`_. The estimator must be
        an object with method `predict_proba`.

    objective : str, optional (default="l2")
        The objective function. Supported objectives are "l2", "l1", "huber"
        and "quantile". Note that "l1", "huber" and "quantile" are robust
        objective functions.

    degree : int (default=1)
        The degree of the polynomials.

        * degree = 0: piecewise constant functions.
        * degree = 1: piecewise linear functions.
        * degree > 1: piecewise polynomial functions.

    continuous : bool (default=True)
        Whether to fit a continuous or discontinuous piecewise regression.

    prebinning_method : str, optional (default="cart")
        The pre-binning method. Supported methods are "cart" for a CART
        decision tree, "quantile" to generate prebins with approximately same
        frequency and "uniform" to generate prebins with equal width. Method
        "cart" uses `sklearn.tree.DecistionTreeClassifier
        <https://scikit-learn.org/stable/modules/generated/sklearn.tree.
        DecisionTreeClassifier.html>`_.

    max_n_prebins : int (default=20)
        The maximum number of bins after pre-binning (prebins).

    min_prebin_size : float (default=0.05)
        The fraction of mininum number of records for each prebin.

    min_n_bins : int or None, optional (default=None)
        The minimum number of bins. If None, then ``min_n_bins`` is
        a value in ``[0, max_n_prebins]``.

    max_n_bins : int or None, optional (default=None)
        The maximum number of bins. If None, then ``max_n_bins`` is
        a value in ``[0, max_n_prebins]``.

    min_bin_size : float or None, optional (default=None)
        The fraction of minimum number of records for each bin. If None,
        ``min_bin_size = min_prebin_size``.

    max_bin_size : float or None, optional (default=None)
        The fraction of maximum number of records for each bin. If None,
        ``max_bin_size = 1.0``.

    monotonic_trend : str or None, optional (default="auto")
        The **event rate** monotonic trend. Supported trends are “auto”,
        "auto_heuristic" and "auto_asc_desc" to automatically determine the
        trend maximizing IV using a machine learning classifier, "ascending",
        "descending", "concave", "convex", "peak" and "peak_heuristic" to allow
        a peak change point, and "valley" and "valley_heuristic" to allow a
        valley change point. Trends "auto_heuristic", "peak_heuristic" and
        "valley_heuristic" use a heuristic to determine the change point,
        and are significantly faster for large size instances (``max_n_prebins
        > 20``). Trend "auto_asc_desc" is used to automatically select the best
        monotonic trend between "ascending" and "descending". If None, then the
        monotonic constraint is disabled.

    n_subsamples : int or None (default=None)
        Number of subsamples to fit the piecewise regression algorithm. If
        None, all values are considered.

    max_pvalue : float or None, optional (default=0.05)
        The maximum p-value among bins. The Z-test is used to detect bins
        not satisfying the p-value constraint. Option supported by solvers
        "cp" and "mip".

    max_pvalue_policy : str, optional (default="consecutive")
        The method to determine bins not satisfying the p-value constraint.
        Supported methods are "consecutive" to compare consecutive bins and
        "all" to compare all bins.

    outlier_detector : str or None, optional (default=None)
        The outlier detection method. Supported methods are "range" to use
        the interquartile range based method or "zcore" to use the modified
        Z-score method.

    outlier_params : dict or None, optional (default=None)
        Dictionary of parameters to pass to the outlier detection method.

    user_splits : array-like or None, optional (default=None)
        The list of pre-binning split points when ``dtype`` is "numerical" or
        the list of prebins when ``dtype`` is "categorical".

    user_splits_fixed : array-like or None (default=None)
        The list of pre-binning split points that must be fixed.

    special_codes : array-like or None, optional (default=None)
        List of special codes. Use special codes to specify the data values
        that must be treated separately.

    split_digits : int or None, optional (default=None)
        The significant digits of the split points. If ``split_digits`` is set
        to 0, the split points are integers. If None, then all significant
        digits in the split points are considered.

    solver : str, optional (default="auto")
        The optimizer to solve the underlying mathematical optimization
        problem. Supported solvers are `"ecos"
        <https://github.com/embotech/ecos>`_, `"osqp"
        <https://github.com/oxfordcontrol/osqp>`_, "direct", to choose the
        direct solver, and "auto", to choose the most appropriate solver for
        the problem.

    h_epsilon: float (default=1.35)
        The parameter h_epsilon used when ``objective="huber"``, controls the
        number of samples that should be classified as outliers.

    quantile : float (default=0.5)
        The parameter quantile is the q-th quantile to be used when
        ``objective="quantile"``.

    regularization: str or None (default=None)
        Type of regularization. Supported regularization are "l1" (Lasso) and
        "l2" (Ridge). If None, no regularization is applied.

    reg_l1 : float (default=1.0)
        L1 regularization term. Increasing this value will smooth the
        regression model. Only applicable if ``regularization="l1"``.

    reg_l2 : float (default=1.0)
        L2 regularization term. Increasing this value will smooth the
        regression model. Only applicable if ``regularization="l2"``.

    random_state : int, RandomState instance or None, (default=None)
        If ``n_subsamples < n_samples``, controls the shuffling applied to the
        data before applying the split.

    verbose : bool (default=False)
        Enable verbose output.
    """
    def __init__(self, name="", estimator=None, objective="l2", degree=1,
                 continuous=True, prebinning_method="cart", max_n_prebins=20,
                 min_prebin_size=0.05, min_n_bins=None, max_n_bins=None,
                 min_bin_size=None, max_bin_size=None, monotonic_trend="auto",
                 n_subsamples=None, max_pvalue=None,
                 max_pvalue_policy="consecutive", outlier_detector=None,
                 outlier_params=None, user_splits=None, user_splits_fixed=None,
                 special_codes=None, split_digits=None, solver="auto",
                 h_epsilon=1.35, quantile=0.5, regularization=None, reg_l1=1.0,
                 reg_l2=1.0, random_state=None, verbose=False):

        super().__init__(name, estimator, objective, degree, continuous,
                         prebinning_method, max_n_prebins, min_prebin_size,
                         min_n_bins, max_n_bins, min_bin_size, max_bin_size,
                         monotonic_trend, n_subsamples, max_pvalue,
                         max_pvalue_policy, outlier_detector, outlier_params,
                         user_splits, user_splits_fixed, special_codes,
                         split_digits, solver, h_epsilon, quantile,
                         regularization, reg_l1, reg_l2, random_state, verbose)

        self._problem_type = "classification"

        self._n_nonevent_special = None
        self._n_nonevent_missing = None
        self._n_event_special = None
        self._n_event_missing = None
        self._t_n_nonevent = None
        self._t_n_event = None

    def fit_transform(self, x, y, metric="woe", metric_special=0,
                      metric_missing=0, lb=None, ub=None, check_input=False):
        """Fit the optimal piecewise binning according to the given training
        data, then transform it.

        Parameters
        ----------
        x : array-like, shape = (n_samples,)
            Training vector, where n_samples is the number of samples.

        y : array-like, shape = (n_samples,)
            Target vector relative to x.

        metric : str (default="woe")
            The metric used to transform the input vector. Supported metrics
            are "woe" to choose the Weight of Evidence and "event_rate" to
            choose the event rate.

        metric_special : float or str (default=0)
            The metric value to transform special codes in the input vector.
            Supported metrics are "empirical" to use the empirical WoE or
            event rate, and any numerical value.

        metric_missing : float or str (default=0)
            The metric value to transform missing values in the input vector.
            Supported metrics are "empirical" to use the empirical WoE or
            event rate and any numerical value.

        lb : float or None (default=None)
            Avoid values below the lower bound lb.

        ub : float or None (default=None)
            Avoid values above the upper bound ub.

        check_input : bool (default=False)
            Whether to check input arrays.

        Returns
        -------
        x_new : numpy array, shape = (n_samples,)
            Transformed array.
        """
        return self.fit(x, y, lb, ub, check_input).transform(
            x, metric, metric_special, metric_missing, lb, ub, check_input)

    def transform(self, x, metric="woe", metric_special=0, metric_missing=0,
                  lb=None, ub=None, check_input=False):
        """Transform given data to Weight of Evidence (WoE) or event rate using
        bins from the fitted optimal piecewise binning.

        Parameters
        ----------
        x : array-like, shape = (n_samples,)
            Training vector, where n_samples is the number of samples.

        metric : str (default="woe")
            The metric used to transform the input vector. Supported metrics
            are "woe" to choose the Weight of Evidence and "event_rate" to
            choose the event rate.

        metric_special : float or str (default=0)
            The metric value to transform special codes in the input vector.
            Supported metrics are "empirical" to use the empirical WoE or
            event rate, and any numerical value.

        metric_missing : float or str (default=0)
            The metric value to transform missing values in the input vector.
            Supported metrics are "empirical" to use the empirical WoE or
            event rate and any numerical value.

        lb : float or None (default=None)
            Avoid values below the lower bound lb.

        ub : float or None (default=None)
            Avoid values above the upper bound ub.

        check_input : bool (default=False)
            Whether to check input arrays.

        Returns
        -------
        x_new : numpy array, shape = (n_samples,)
            Transformed array.
        """
        self._check_is_fitted()

        return transform_binary_target(
            self._optb.splits, x, self._c, lb, ub, self._t_n_nonevent,
            self._t_n_event, self._n_nonevent_special, self._n_event_special,
            self._n_nonevent_missing, self._n_event_missing,
            self.special_codes, metric, metric_special, metric_missing,
            check_input)

    def _fit(self, x, y, lb, ub, check_input):
        time_init = time.perf_counter()

        if self.verbose:
            logger.info("Optimal piecewise binning started.")
            logger.info("Options: check parameters.")

        _check_parameters(**self.get_params(deep=False),
                          problem_type=self._problem_type)

        # Pre-processing
        if self.verbose:
            logger.info("Pre-processing started.")

        self._n_samples = len(x)

        if self.verbose:
            logger.info("Pre-processing: number of samples: {}"
                        .format(self._n_samples))

        time_preprocessing = time.perf_counter()

        [x_clean, y_clean, x_missing, y_missing, x_special, y_special,
         _, _, _, _, _, _, _] = self._fit_preprocessing(x, y, check_input)

        self._time_preprocessing = time.perf_counter() - time_preprocessing

        if self.verbose:
            n_clean = len(x_clean)
            n_missing = len(x_missing)
            n_special = len(x_special)

            logger.info("Pre-processing: number of clean samples: {}"
                        .format(n_clean))

            logger.info("Pre-processing: number of missing samples: {}"
                        .format(n_missing))

            logger.info("Pre-processing: number of special samples: {}"
                        .format(n_special))

            if self.outlier_detector is not None:
                n_outlier = self._n_samples-(n_clean + n_missing + n_special)
                logger.info("Pre-processing: number of outlier samples: {}"
                            .format(n_outlier))

            logger.info("Pre-processing terminated. Time: {:.4f}s"
                        .format(self._time_preprocessing))

        # Pre-binning
        # Fit estimator and compute event_rate = P[Y=1, X=x]
        time_estimator = time.perf_counter()

        if self.estimator is None:
            self.estimator = LogisticRegression()

            if self.verbose:
                logger.info("Pre-binning: set logistic regression as an "
                            "estimator.")

        if self.verbose:
            logger.info("Pre-binning: estimator fitting started.")

        self.estimator.fit(x_clean.reshape(-1, 1), y_clean)
        event_rate = self.estimator.predict_proba(x_clean.reshape(-1, 1))[:, 1]

        self._time_estimator = time.perf_counter() - time_estimator

        if self.verbose:
            logger.info("Pre-binning: estimator terminated. Time {:.4f}s."
                        .format(self._time_estimator))

        # Fit optimal binning algorithm for continuous target. Use optimal
        # split points to compute optimal piecewise functions
        self._fit_binning(x_clean, y_clean, event_rate, lb, ub)

        # Post-processing
        if self.verbose:
            logger.info("Post-processing started.")
            logger.info("Post-processing: compute binning information.")

        time_postprocessing = time.perf_counter()

        # Compute n_nonevent and n_event for special and missing
        special_target_info = target_info(y_special)
        self._n_nonevent_special = special_target_info[0]
        self._n_event_special = special_target_info[1]

        missing_target_info = target_info(y_missing)
        self._n_nonevent_missing = missing_target_info[0]
        self._n_event_missing = missing_target_info[1]

        bt = self._optb.binning_table.build(add_totals=False)
        n_nonevent = bt["Non-event"].values
        n_event = bt["Event"].values
        n_nonevent[self._n_bins] = self._n_nonevent_special
        n_nonevent[self._n_bins + 1] = self._n_nonevent_missing
        n_event[self._n_bins] = self._n_event_special
        n_event[self._n_bins + 1] = self._n_event_missing

        self._t_n_nonevent = n_nonevent.sum()
        self._t_n_event = n_event.sum()

        # Compute metrics
        if self.verbose:
            logger.info("Post-processing: compute performance metrics.")

        d_metrics = binary_metrics(
            x_clean, y_clean, self._optb.splits, self._c, self._t_n_nonevent,
            self._t_n_event, self._n_nonevent_special, self._n_event_special,
            self._n_nonevent_missing, self._n_event_missing,
            self.special_codes)

        # Binning table
        self._binning_table = PWBinningTable(
            self.name, self._optb.splits, self._c, n_nonevent, n_event,
            x_clean.min(), x_clean.max(), d_metrics)

        self._time_postprocessing = time.perf_counter() - time_postprocessing

        if self.verbose:
            logger.info("Post-processing terminated. Time: {:.4f}s"
                        .format(self._time_postprocessing))

        self._time_total = time.perf_counter() - time_init

        if self.verbose:
            logger.info("Optimal piecewise binning terminated. Status: {}. "
                        "Time: {:.4f}s".format(self._status, self._time_total))

        # Completed successfully
        self._is_fitted = True

        return self
