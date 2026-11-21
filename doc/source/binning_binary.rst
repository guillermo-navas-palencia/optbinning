Optimal binning with binary target
==================================

.. autoclass:: optbinning.OptimalBinning
   :members:
   :inherited-members:
   :show-inheritance:

Effective weight validation
---------------------------

Sample weights (including scalar weights) are combined with class weights once.
The resulting masses must be finite and nonnegative, with a finite positive
total. Weight arrays must match the input length; column-shaped arrays are
rejected rather than broadcast. Fractional masses remain masses rather than
integer borrower counts and must not be passed as counts to Bayesian WoE.
See :doc:`fractional_weights` for the distinction between raw-row support
constraints and weighted fitting/statistics, and the current solver limitations.
