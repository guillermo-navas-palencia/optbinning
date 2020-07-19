Release Notes
=============

Version 0.7.0 (2020-07-19)
--------------------------

New features:

   - Batch and streaming optimal binning.
   - New parameter ``divergence`` to select the divergence measure to maximize.

Tutorials:

   - Tutorial: optimal binning sketch with binary target
   - Tutorial: optimal binning sketch with binary target using PySpark

Bugfixes:

   - Catch error from Qhull library used by scipy.spatial.ConvexHull.


Version 0.6.1 (2020-06-07)
--------------------------

New features:

   - Options ``add_special`` and ``add_missing`` in all binning table plots.
   - Prebinning methods' parameters are accessible via ``**prebinning_kwargs``.
   - Add support MDLP algorithm for binary target.

Bugfixes:

   - Fix bug in solution when the status is not feasible or optimal for LocalSolver, ``solver="ls"``.
   - Fix several bugs for categorical variables with ``user_splits`` and ``user_splits_fixed``.
   - Fix bug in binning process when passing ``user_splits`` and ``user_splits_fixed`` via parameter ``binning_fit_params``.


Version 0.6.0 (2020-05-24)
--------------------------

New features:

   - Scorecard development supporting binary and continuous target.
   - Plotting functions: ``plot_auc_roc``, ``plot_cap`` and ``plot_ks``.
   - Optimal binning classes introduce ``sample_weight`` parameter in methods ``fit`` and ``fit_transform``.
   - Optimal binning classes introduce two options for parameter ``metric`` in methods ``fit_transform`` and ``transform``: ``metric="bins"`` and ``metric="indices"``.


Tutorials:

   - Tutorial: optimal binning with binary target - large scale.
   - Tutorial: Scorecard with binary target.
   - Tutorial: Scorecard with continuous target.


Version 0.5.0 (2020-04-13)
--------------------------

New features:

   - Scenario-based stochastic optimal binning.
   - New parameter ``user_split_fixed`` to force user-defined split points.

Tutorials:
   
   - Tutorial: Telco customer churn.
   - Tutorial: optimal binning with binary target under uncertainty.

Bugfixes:

   - Fix monotonic trend for non-auto mode in ``MulticlassOptimalBinning``.


Version 0.4.0 (2020-03-22)
--------------------------

New features:

   - New ``monotonic_trend`` auto modes options: "auto_heuristic" and "auto_asc_desc".
   - New ``monotonic_trend`` options: "peak_heuristic" and "valley_heuristic". These options produce a remarkable speedup for large size instances.
   - Minimum Description Length Principle (MDLP) discretization algorithm.

Improvements:

   - ``BinningProcess`` now supports ``pandas.DataFrame`` as input X.
   - New unit test added.


Version 0.3.1 (2020-03-17)
--------------------------

Bugfixes:

   - Fix setup.py packages using find_packages.


Version 0.3.0 (2020-03-13)
--------------------------

New features:

   - Class ``OptBinning`` introduces a new constraint to reduce dominating bins, using parameter ``gamma``.
   - Metrics HHI, HHI regularized and Cramer's V added to ``binning_table.analysis`` method. Updated quality score.
   - Added column min/max target and zeros count to ``ContinuousOptimalBinning`` binning table.
   - Binning algorithms support univariate outlier detection methods.

Tutorials:

   - Tutorial: optimal binning with binary target. New section: Reduction of dominating bins.
   - Enhance binning process tutorials.


Version 0.2.0 (2020-02-02)
--------------------------

New features:

   - Binning process to support optimal binning of all variables in dataset.
   - Added ``print_output`` option to ``binning_table.analysis`` method.


Improvements:

   - New unit tests added.

Tutorials:

   - Tutorial: Binning process with Scikit-learn pipelines.
   - Tutorial: FICO Explainable Machine Learning Challenge using binning process.   

Bugfixes:

   - Fix ``OptBinning.information`` print level default option.
   - Avoid numpy.digitize if no splits.
   - Compute Gini in ``binning_table.build`` method.


Version 0.1.1 (2020-01-24)
--------------------------

Bugfixes:

   * Fix a bug in ``OptimalBinning.fit_transform`` when calling ``tranform`` internally.
   * Replace np.int by np.int64 in ``model_data.py`` functions to guarantee 64-bit integer on Windows.
   * Fix a bug in ``_chech_metric_special_missing``.


Version 0.1.0 (2020-01-22)
--------------------------

* First release of OptBinning.
