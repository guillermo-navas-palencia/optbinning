"""
Counterfactual model data.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2021


def model_data(scorecard, x):
    s_vars = scorecard.binning_process_.get_support(names=True)
    n_vars = len(s_vars)

    sc = scorecard.table(style="detailed")
    metric_name = "WoE" if scorecard._target_dtype == "binary" else "Mean"

    # Number of bins and metric
    nbins = []
    metric = []

    for i, v in enumerate(s_vars):
        metric_i = sc[sc.Variable == v][metric_name].values[:-2]
        metric_i = [mi for mi in metric_i if mi != x[i]]
        
        metric.append(metric_i)
        nbins.append(len(metric_i))

    return nbins, metric
