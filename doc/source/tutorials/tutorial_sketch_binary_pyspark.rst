Tutorial: optimal binning sketch with binary target using PySpark
=================================================================

In this example, we use PySpark mapPartitions function to compute the optimal
binning of a single variable from a large dataset in a distributed fashion. The dataset is split into 4 partitions.

.. code::

  from pyspark.sql import SparkSession

  spark.conf.set("spark.sql.execution.arrow.enabled", "true")

  df = spark.read.csv("data/kaggle/HomeCreditDefaultRisk/application_train.csv",
                      sep=",", header=True, inferSchema=True)

  n_partitions = 4
  df = df.repartition(n_partitions)


We prepare the MapReduce structure

.. code ::

  import pandas as pd
  from optbinning import OptimalBinningSketch

  variable = "EXT_SOURCE_3"
  target = "TARGET"
  columns = [variable, target]


  def add(partition):
      df_pandas = pd.DataFrame.from_records(partition, columns=columns)
      x = df_pandas[variable]
      y = df_pandas[target]
      optbsketch = OptimalBinningSketch(eps=0.001)
      optbsketch.add(x, y)
      
      return [optbsketch]

  def merge(optbsketch, other_optbsketch):
      optbsketch.merge(other_optbsketch)
      
      return optbsketch

Finally, with the required columns, we use mapPartitions and method
treeReduce to aggregate the ``OptimalBinningSketch`` instance of each partition.

.. code ::

  optbsketch = df.select(columns).rdd.mapPartitions(lambda partition: add(partition)
                                                   ).treeReduce(merge)