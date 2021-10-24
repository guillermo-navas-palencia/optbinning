"""
Optimal piecewise binning for continuous target.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import time

import numpy as np

from ...logging import Logger
from .base import _check_parameters
from .base import BasePWBinning
from .binning_statistics import PWContinuousBinningTable
from .metrics import continuous_metrics
from .transformations import transform_continuous_target


logger = Logger(__name__).logger


class ContinuousOptimalPWBinning(BasePWBinning):
    """Optimal Piecewise binning of a numerical variable with respect to a
    binary target.

    Parameters
    ----------
    name : str, optional (default="")
        The variable name.

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
        The monotonic trend. Supported trends are “auto”, "auto_heuristic" and
        "auto_asc_desc" to automatically determine the trend maximizing IV
        using a machine learning classifier, "ascending", "descending",
        "concave", "convex", "peak" and "peak_heuristic" to allow a peak change
        point, and "valley" and "valley_heuristic" to allow a valley change
        point. Trends "auto_heuristic", "peak_heuristic" and "valley_heuristic"
        use a heuristic to determine the change point, and are significantly
        faster for large size instances (``max_n_prebins > 20``). Trend
        "auto_asc_desc" is used to automatically select the best monotonic
        trend between "ascending" and "descending". If None, then the
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
    def __init__(self, name="", objective="l2", degree=1,
                 continuous=True, prebinning_method="cart", max_n_prebins=20,
                 min_prebin_size=0.05, min_n_bins=None, max_n_bins=None,
                 min_bin_size=None, max_bin_size=None, monotonic_trend="auto",
                 n_subsamples=None, max_pvalue=None,
                 max_pvalue_policy="consecutive", outlier_detector=None,
                 outlier_params=None, user_splits=None, user_splits_fixed=None,
                 special_codes=None, split_digits=None, solver="auto",
                 h_epsilon=1.35, quantile=0.5, regularization=None, reg_l1=1.0,
                 reg_l2=1.0, random_state=None, verbose=False):

        super().__init__(name, None, objective, degree, continuous,
                         prebinning_method, max_n_prebins, min_prebin_size,
                         min_n_bins, max_n_bins, min_bin_size, max_bin_size,
                         monotonic_trend, n_subsamples, max_pvalue,
                         max_pvalue_policy, outlier_detector, outlier_params,
                         user_splits, user_splits_fixed, special_codes,
                         split_digits, solver, h_epsilon, quantile,
                         regularization, reg_l1, reg_l2, random_state, verbose)

        self._problem_type = "regression"

        self._n_records_missing = None
        self._n_records_special = None
        self._sum_special = None
        self._sum_missing = None
        self._std_special = None
        self._std_missing = None
        self._min_target_missing = None
        self._min_target_special = None
        self._max_target_missing = None
        self._max_target_special = None
        self._n_zeros_missing = None
        self._n_zeros_special = None

    def fit_transform(self, x, y, metric_special=0, metric_missing=0,
                      lb=None, ub=None, check_input=False):
        """Fit the optimal piecewise binning according to the given training
        data, then transform it.

        Parameters
        ----------
        x : array-like, shape = (n_samples,)
            Training vector, where n_samples is the number of samples.

        y : array-like, shape = (n_samples,)
            Target vector relative to x.

        metric_special : float or str (default=0)
            The metric value to transform special codes in the input vector.
            Supported metrics are "empirical" to use the empirical mean and any
            numerical value.

        metric_missing : float or str (default=0)
            The metric value to transform missing values in the input vector.
            Supported metrics are "empirical" to use the empirical mean and any
            numerical value.

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
        return self.fit(x, y, check_input).transform(
            x, metric_special, metric_missing, lb, ub, check_input)

    def transform(self, x, metric_special=0, metric_missing=0,
                  lb=None, ub=None, check_input=False):
        """Transform given data using bins from the fitted optimal piecewise
        binning.

        Parameters
        ----------
        x : array-like, shape = (n_samples,)
            Training vector, where n_samples is the number of samples.

        metric_special : float or str (default=0)
            The metric value to transform special codes in the input vector.
            Supported metrics are "empirical" to use the empirical mean and any
            numerical value.

        metric_missing : float or str (default=0)
            The metric value to transform missing values in the input vector.
            Supported metrics are "empirical" to use the empirical mean and any
            numerical value.

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

        return transform_continuous_target(
            self._optb.splits, x, self._c, lb, ub, self._n_records_special,
            self._sum_special, self._n_records_missing, self._sum_missing,
            self.special_codes, metric_special, metric_missing, check_input)

    def _fit(self, x, y, lb, ub, check_input):
        time_init = time.perf_counter()

        if self.verbose:
            logger.info("Optimal piecewise binning started.")
            logger.info("Options: check parameters.")

        _check_parameters(**self.get_params(deep=False), estimator=None,
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
        self._time_estimator = 0

        # Fit optimal binning algorithm for continuous target. Use optimal
        # split points to compute optimal piecewise functions
        self._fit_binning(x_clean, y_clean, y_clean, lb, ub)

        # Post-processing
        if self.verbose:
            logger.info("Post-processing started.")
            logger.info("Post-processing: compute binning information.")

        time_postprocessing = time.perf_counter()

        # Compute n_records and sum for special and missing
        self._n_records_special = len(y_special)
        self._sum_special = np.sum(y_special)
        self._n_zeros_special = np.count_nonzero(y_special == 0)
        if len(y_special):
            self._std_special = np.std(y_special)
            self._min_target_special = np.min(y_special)
            self._max_target_special = np.max(y_special)

        self._n_records_missing = len(y_missing)
        self._sum_missing = np.sum(y_missing)
        self._n_zeros_missing = np.count_nonzero(y_missing == 0)
        if len(y_missing):
            self._std_missing = np.std(y_missing)
            self._min_target_missing = np.min(y_missing)
            self._max_target_missing = np.max(y_missing)

        bt = self._optb.binning_table.build(add_totals=False)
        n_records = bt["Count"].values
        sums = bt["Sum"].values
        stds = bt["Std"].values
        min_target = bt["Min"].values
        max_target = bt["Max"].values
        n_zeros = bt["Zeros count"].values

        n_records[self._n_bins] = self._n_records_special
        n_records[self._n_bins + 1] = self._n_records_missing
        sums[self._n_bins] = self._sum_special
        sums[self._n_bins + 1] = self._sum_missing
        stds[self._n_bins] = self._std_special
        stds[self._n_bins + 1] = self._std_missing
        min_target[self._n_bins] = self._min_target_special
        min_target[self._n_bins + 1] = self._min_target_missing
        max_target[self._n_bins] = self._max_target_special
        max_target[self._n_bins + 1] = self._max_target_missing
        n_zeros[self._n_bins] = self._n_zeros_special
        n_zeros[self._n_bins + 1] = self._n_zeros_missing

        # Compute metrics
        if self.verbose:
            logger.info("Post-processing: compute performance metrics.")

        d_metrics = continuous_metrics(
            x_clean, y_clean, self._optb.splits, self._c, lb, ub,
            self._n_records_special, self._sum_special,
            self._n_records_missing, self._sum_missing, self.special_codes)

        # Binning table
        self._binning_table = PWContinuousBinningTable(
            self.name, self._optb.splits, self._c, n_records, sums, stds,
            min_target, max_target, n_zeros, lb, ub, x_clean.min(),
            x_clean.max(), d_metrics)

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
