Release Notes
=============


Version 0.1.1 (2020-01-24)
--------------------------

Bugfixes:

* Fix a bug in ``OptimalBinning.fit_transform`` when calling ``tranform`` internally.
* Replace np.int by np.int64 in ``model_data.py`` functions to guarantee 64-bit integer on Windows.
* Fix a bug in ``_chech_metric_special_missing``.


Version 0.1.0 (2020-01-22)
--------------------------

* First release of OptBinning.