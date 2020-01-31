.. optbinning documentation master file, created by
   sphinx-quickstart on Thu Dec 19 10:54:06 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: _images/logo.svg
  :width: 550
  :alt: Alternative text

|

OptBinning: The Python Optimal Binning library
==============================================

The optimal binning is the optimal discretization of a variable into bins given a discrete or continuous numeric target. **OptBinning** is a library
written in Python implementing a **rigorous** and **flexible** mathematical programming formulation to solving the optimal binning problem for a binary, continuous and multiclass target type, incorporating constraints not previously addressed.

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   installation
   tutorials
   release_notes

.. toctree::
   :maxdepth: 1
   :caption: Optimal binning algorithms

   binning_binary
   binning_continuous
   binning_multiclass
   binning_process
   binning_tables
   binning_utilities
