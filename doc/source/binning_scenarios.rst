Stochastic optimal binning
==========================

Introduction
------------
The data used when performing optimal binning is generally assumed to be known accurately and being fully representative of past, present, and future data. This confidence might produce misleading results, especially with data representing future events such as product demand, churn rate, or probability of default.

Stochastic programming is a framework for explicitly incorporating uncertainty. Stochastic programming uses random variables to account for data variability and optimizes the expected value of the objective function. Optbinning implements the stochastic programming approach using the two-stage scenario-based formulation (also known as extensive form or deterministic equivalent), obtaining a deterministic mixed-integer linear programming formulation. The scenario-based formulation guarantees the nonanticipativity constraint and a solution that must be feasible for each scenario, leading to a more **robust** solution.


Scenario-based optimal binning
------------------------------

.. autoclass:: optbinning.binning.uncertainty.SBOptimalBinning
   :members:
   :inherited-members:
   :show-inheritance:

