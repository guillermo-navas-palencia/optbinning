Release Notes
=============
Version 0.20.? (2025-02-?)
---------------------------

New features:

   - Added a `.transform()` method to the Scorecard class. (`Discussion 345 <https://github.com/guillermo-navas-palencia/optbinning/discussions/345>`_).



Version 0.20.0 (2024-10-29)
---------------------------

Improvements:

   - Axis labels are cut off when saving the table plot to file (`Issue 303 <https://github.com/guillermo-navas-palencia/optbinning/issues/303>`_).
   - Packaging improvements (`Issue 335 <https://github.com/guillermo-navas-palencia/optbinning/issues/335>`_).

Bugfixes:

   - Legend missing in ScorecardMonitoring.psi_plot() (`Issue 327 <https://github.com/guillermo-navas-palencia/optbinning/issues/327>`_).


Version 0.19.0 (2024-01-16)
---------------------------

Improvements:

   - Adjust plot size (`Issue 244 <https://github.com/guillermo-navas-palencia/optbinning/issues/244>`_).
   - Save optimal binning object in JSON format (`Issue 96 <https://github.com/guillermo-navas-palencia/optbinning/issues/96>`_).
   - Plot IV/WoE metric in binning table plot for binary and continuous target.

Bugfixes:

   - Keep pandas.DataFrame index in transform method (`Issue 286 <https://github.com/guillermo-navas-palencia/optbinning/issues/286>`_).
   - Fix BinningProcess's binning_transform_params="bins" (`Issue 266 <https://github.com/guillermo-navas-palencia/optbinning/issues/266>`_).


Version 0.18.0 (2023-09-22)
---------------------------

Bugfixes:

   - Fix numpy array object (`Issue 229 <https://github.com/guillermo-navas-palencia/optbinning/issues/229>`_).
   - Fix ``show_bin_labels`` (`Issue 262 <https://github.com/guillermo-navas-palencia/optbinning/issues/262>`_).
   - Fix ``special_codes_y`` (`Issue 263 <https://github.com/guillermo-navas-palencia/optbinning/issues/263>`_).


Version 0.17.3 (2023-02-12)
---------------------------

Improvements:

   - Implement ``sample_weight`` check in Scorecard class (`Issue 228 <https://github.com/guillermo-navas-palencia/optbinning/issues/228>`_).

Bugfixes:

   - Fix ``metric_missing`` ignored in Scorecard class (`Issue 226 <https://github.com/guillermo-navas-palencia/optbinning/issues/226>`_).

Dependencies:

   - Update RoPWR required version.


Version 0.17.2 (2022-12-15)
---------------------------

Improvements:

   - Modify max-pvalue and min_diff constraints for CP and MIP formulation to avoid suboptimal solutions.

Bugfixes:

   - Use keyword arguments in ``compute_class_weight`` (`Issue 222 <https://github.com/guillermo-navas-palencia/optbinning/issues/222>`_).
   - Remove preprocessing step when monotonic trend in (ascending, descending) for scenario-based binning (`Issue 216 <https://github.com/guillermo-navas-palencia/optbinning/issues/216>`_).

Dependencies:

   - Update scikit-learn and ortools required versions.


Version 0.17.1 (2022-11-20)
---------------------------

New features:

   - Add parameter ``cat_unknown`` to assign values to the unobserved categories during training.

Improvements:

   - Add method ``decision_function`` to ``Scorecard`` (`Issue 198 <https://github.com/guillermo-navas-palencia/optbinning/issues/198>`_).


Version 0.17.0 (2022-11-06)
---------------------------

New features:

   - Optimize formulation of minimum difference constraints for all optimal binning classes and support these constraints regardless of the monotonic trend (`Issue 201 <https://github.com/guillermo-navas-palencia/optbinning/issues/201>`_).

   - Implementation of sample weight for ``ContinuousOptimalBinning`` (`Issue 131 <https://github.com/guillermo-navas-palencia/optbinning/issues/131>`_).


Bugfixes:

   - Fix ``ContinuousOptimalBinning`` prebinning step when no prebinning splits were generated (`Issue 205 <https://github.com/guillermo-navas-palencia/optbinning/issues/205>`_).


Version 0.16.1 (2022-10-28)
---------------------------

New features:

   - Outlier detector ``YQuantileDetector`` for continuous target (`Issue 203 <https://github.com/guillermo-navas-palencia/optbinning/issues/203>`_).

Improvements:

   - Add support to solver SCS and HIGHS for optimal piecewise binning classes.
   - Unit testing outlier detector methods.

Bugfixes:

   - Pass ``lb`` and ``ub`` as keyword arguments to RoPWR fit method (required since ropwr>=0.4.0).


Version 0.16.0 (2022-10-24)
---------------------------

New features:

   - Treatment of special codes separately for optimal piecewise binning classes (`Issue 191 <https://github.com/guillermo-navas-palencia/optbinning/issues/191>`_).

Improvements:

   - Allow plot ``style="actual"`` for stochastic optimal binning.
   - Unit testing optimal piecewise binning classes (`Issue 93 <https://github.com/guillermo-navas-palencia/optbinning/issues/93>`_).
   - Unit testing add macOS Monterey 12.

Bugfixes:

   - Fix sample weight for ``BinningProcess`` when ``n_jobs != 1`` (`Issue 190 <https://github.com/guillermo-navas-palencia/optbinning/issues/190>`_).
   - Fix transform method for optimal binning 2D when dtype is categorical (`Issue 197 <https://github.com/guillermo-navas-palencia/optbinning/issues/197>`_).
   - Fix ``max_pvalue`` default value in documentation (`Issue 199 <https://github.com/guillermo-navas-palencia/optbinning/issues/199>`_).


Version 0.15.1 (2022-09-06)
---------------------------

New features:

   - New parameter ``show_bin_labels`` for binning tables (`Issue 180 <https://github.com/guillermo-navas-palencia/optbinning/issues/180>`_).


Version 0.15.0 (2022-07-20)
---------------------------

New features:

   - Optimal binning 2D support to categorical variables for binary and continuous target.

Improvements:

   - Integer intercept if ``rounding=True`` (`Issue 165 <https://github.com/guillermo-navas-palencia/optbinning/issues/165>`_).
   - Parameter ``show_digits`` applies to scorecard table bin column (`Issue 170 <https://github.com/guillermo-navas-palencia/optbinning/issues/170>`_).

Bugfixes:

   - Fix ``Scorecard.score`` method when there are special and missing bins. (`Issue 179 <https://github.com/guillermo-navas-palencia/optbinning/pull/179>`_).
   - Fix x and y axis labels in ``OptimalBinning2D`` plots, x and y were interchanged.


Version 0.14.1 (2022-04-11)
---------------------------

Bugfixes:

   - Fix new setup function.


Version 0.14.0 (2022-04-10)
---------------------------

New features:

   - Optimal binning 2D with continuous target.

Improvements:

   - Set tdigest and pympler dependencies as optional. This change avoids accumulation-tree issues faced by several users. Remove dill dependency.
   - New continuous binning objective function leading to improvements in regression metrics.

Bugfixes:

   - Fix binning 2D minimum difference constraints.

Tutorials:

   - Tutorial: optimal binning 2D with continuous target


Version 0.13.1 (2022-02-18)
---------------------------

Bugfixes:

   - Fix binning process summary update (`Issue 151 <https://github.com/guillermo-navas-palencia/optbinning/issues/151>`_).

   - Fix pandas 1.4.0 (python > 3.8) slicing issue with method at (`Pull 148 <https://github.com/guillermo-navas-palencia/optbinning/pull/148>`_).

   - Fix minor typos (`Pull 147 <https://github.com/guillermo-navas-palencia/optbinning/pull/147>`_).

   - Fix binning plot for multiple special values.

Version 0.13.0 (2021-11-24)
---------------------------

New features:

   - Treatment of special codes separately for optbinning classes (`Issue 115 <https://github.com/guillermo-navas-palencia/optbinning/issues/115>`_).

Bugfixes:

   - Various bug fixes for the ``OptimalBinning2D`` class. See `Issue 138 <https://github.com/guillermo-navas-palencia/optbinning/issues/138>`_, for instance.

Tutorials:

   - Tutorial: optimal binning 2D with binary target


Version 0.12.2 (2021-10-03)
---------------------------

Improvements:

   - Do not store optimization solver instance as class attribute.
   - Do not store logger as a class attribute.


Version 0.12.1 (2021-09-12)
---------------------------

New features:

   - Binning process supports ``sample_weight`` for binary target. `Issue 124 <https://github.com/guillermo-navas-palencia/optbinning/issues/124>`_

   - Binning process can fix variables not satisfying selection criteria. `Issue 123 <https://github.com/guillermo-navas-palencia/optbinning/issues/123>`_


Version 0.12.0 (2021-08-28)
---------------------------

New features:

   - Optimal binning 2D with binary target.

Improvements:

   - Update bin string format in binning tables.
   - Simplify logic when ``style="actual"`` in binning table plots.


API changes:

   - Scorecard fit method arguments changed to the usual ``(X, y)``: `Issue 111 <https://github.com/guillermo-navas-palencia/optbinning/issues/111>`_


Version 0.11.0 (2021-05-28)
---------------------------

New features:

   - Counterfactual explanations for scorecard modelling.

Improvements:

   - Replace pickle by dill in save and load methods.

Bugfixes:

   - Parallel binning uses joblib: `Issue 103 <https://github.com/guillermo-navas-palencia/optbinning/issues/103>`_
   - Fix custom  ``metric_special`` and ``metric_missing`` in binning_transform_params.


Version 0.10.0 (2021-04-27)
---------------------------

New features:

   - Batch and streaming binning process.

Improvements:

   - Improve LocalSolver formulation for optimal binning with a binary target.

Bugfixes:

   - Fix MulticlassOptimalBinning when no prebins: `Issue 94 <https://github.com/guillermo-navas-palencia/optbinning/issues/94>`_
   - Fix metric_missing and metric_special defined for fitting, but not for predictions or scorecard points: `Issue 100 <https://github.com/guillermo-navas-palencia/optbinning/issues/100>`_


Version 0.9.2 (2021-03-12)
--------------------------

New features:

   - Binning process can update binned variables with new optimal binning object using method ``update_binned_variable``.

Improvements:
   
   - Prevent large divisions to avoid overflow issues with int32 during Gini calculation.

Tutorials:

   - Tutorial: FICO Explainable Machine Learning Challenge - updating binning


Version 0.9.1 (2021-02-14)
--------------------------

New features:

   - Binning process can be constructed using OptimalBinning objects previously fitted. Method ``fit_from_dict``.
   - Binning process can process large datasets directly on disk. Allowed file formats are csv and parquet. Methods ``fit_disk``, ``fit_transform_disk`` and ``transform_disk``.

Bugfixes:

   - Fix saving all OptBinning classes: `Issue 77 <https://github.com/guillermo-navas-palencia/optbinning/issues/77>`_


Version 0.9.0 (2021-01-14)
--------------------------

New features:

   - Optimal piecewise polynomial binning.
   - New plotting option for binning table for binary and continuous target. Parameter ``style`` allows to represent the binning plot with the actual scale, i.e., actual bin widths.

Improvements:

   - Improve computation of p-values and binning table analysis for ``ContinuousOptimalBinning``.

Tutorials:
   
   - Tutorial: optimal piecewise binning with binary target
   - Tutorial: optimal piecewise binning with continuous target

Bugfixes:

   - Fix sample weights bug: `Issue 64 <https://github.com/guillermo-navas-palencia/optbinning/issues/64>`_


Version 0.8.0 (2020-09-18)
--------------------------

New features:

   - Scorecard monitoring supporting binning and continuous target.
   - OptimalBinning computes the Kolmogorov-Smirnov statistic.
   - Optimal binning classes show optimal monotonic trend information in the binning table analysis method.
   - ContinuousBinningTable adds method ``analysis``.
   - Scorecard incorporates methods ``load`` and ``save`` to serialize and deserialize a scorecard using pickle module.
   - BinningProcess class supports multiprocessing via parameter ``n_jobs``.

Tutorials:

   - Tutorial: Scorecard monitoring


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
