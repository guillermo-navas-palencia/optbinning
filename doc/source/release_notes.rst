Release Notes
=============

Version 0.2.0 (2020-02-02)
--------------------------

New additions:

   - Binning process to support optimal binning of all variables in dataset.
   - Add ``print_output`` option to ``binning_table.analysis`` method.
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
