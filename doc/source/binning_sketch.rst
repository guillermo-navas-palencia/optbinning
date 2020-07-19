Optimal binning sketch with binary target
=========================================

Introduction
------------

The optimal binning is the constrained discretization of a numerical feature into bins given a binary target, maximizing a statistic such as Jeffrey's divergence or Gini. Binning is a data preprocessing technique commonly used in binary classification, but the current list of existing binning algorithms supporting constraints lacks a method to handle streaming data. The new class OptimalBinningSketch implements a new scalable, memory-efficient and robust algorithm for performing optimal binning in the streaming settings. Algorithmic details are discussed in http://gnpalencia.org/blog/2020/binning_data_streams/.


Algorithms
----------

OptimalBinningSketch
""""""""""""""""""""

.. autoclass:: optbinning.binning.distributed.OptimalBinningSketch
   :members:
   :inherited-members:
   :show-inheritance:


GK: Greenwald-Khanna's algorithm
""""""""""""""""""""""""""""""""

.. autoclass:: optbinning.binning.distributed.GK
   :members:
   :inherited-members:
   :show-inheritance:


Binning sketch: numerical variable - binary target
""""""""""""""""""""""""""""""""""""""""""""""""""

.. autoclass:: optbinning.binning.distributed.BSketch
   :members:
   :inherited-members:
   :show-inheritance:


Binning sketch: categorical variable - binary target
""""""""""""""""""""""""""""""""""""""""""""""""""""

.. autoclass:: optbinning.binning.distributed.BCatSketch
   :members:
   :inherited-members:
   :show-inheritance:   