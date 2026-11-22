Binning tables
==============

Accessing significance tests
----------------------------

After building and analyzing a binning table, use ``df_tests`` to retrieve
its adjacent-bin significance results without parsing printed output::

    table = optb.binning_table
    table.build()
    table.analysis(print_output=False)
    tests = table.df_tests
    p_values = tests["p-value"]

This property is available for binary, continuous and multiclass targets.
It returns a copy of the results from the most recent successful analysis.
Calling it before ``analysis()`` raises ``NotFittedError``. With fewer than
two regular bins, the result is an empty DataFrame with the usual columns.
Special, missing and other-category bins are not compared.

The results are the same tests already reported by ``analysis()``. Binary
analysis supports chi-square and Fisher tests and includes Bayesian A/B
probabilities; continuous analysis uses Welch's t-test; multiclass analysis
uses the chi-square test. These are bin-comparison results, not p-values
for fitted scorecard coefficients.

Binning table: binary target
----------------------------

.. autoclass:: optbinning.binning.binning_statistics.BinningTable
   :members:
   :inherited-members:
   :show-inheritance:

Binning table: continuous target
--------------------------------

.. autoclass:: optbinning.binning.binning_statistics.ContinuousBinningTable
   :members:
   :inherited-members:
   :show-inheritance:

Binning table: multiclass target
--------------------------------

.. autoclass:: optbinning.binning.binning_statistics.MulticlassBinningTable
   :members:
   :inherited-members:
   :show-inheritance:   