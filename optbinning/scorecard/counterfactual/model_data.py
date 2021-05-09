"""
Counterfactual model data.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2021


def model_data(scorecard, x, special_missing):
    s_vars = scorecard.binning_process_.get_support(names=True)

    sc = scorecard.table(style="detailed")
    metric_name = "WoE" if scorecard._target_dtype == "binary" else "Mean"

    # Number of bins, metric and indices
    nbins = []
    metric = []
    indices = []
    for i, v in enumerate(s_vars):
        metric_i = sc[sc.Variable == v][metric_name].values

        if not special_missing:
            metric_i = metric_i[:-2]

        _metric = []
        _indices = []
        for j, m in enumerate(metric_i):
            if m != x[i]:
                _indices.append(j)
                _metric.append(m)

        metric.append(_metric)
        nbins.append(len(_metric))
        indices.append(_indices)

    return nbins, metric, indices
