"""
Binning sketch plots.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import matplotlib.pyplot as plt
import numpy as np


def plot_progress_divergence(df, divergence):
    n = len(df)
    n_records = df.n_records
    div = df.divergence

    mv_div_mean = div.rolling(n, min_periods=1).mean()
    mv_div_std = div.rolling(n, min_periods=1).std()
    mv_div_std /= np.sqrt(np.arange(1, n+1))

    div_low = np.maximum(0, div - mv_div_std * 1.959963984540054)
    div_high = div + mv_div_std * 1.959963984540054

    plt.plot(n_records, div, label="divergence: {}".format(divergence),
             marker="o")
    plt.plot(n_records, mv_div_mean, linestyle="-.", color="green",
             label="moving mean")
    plt.fill_between(n_records, div_low, div_high, alpha=0.2, color="green",
                     label="Standard error")

    plt.xlabel("Processed records", fontsize=12)
    plt.ylabel("Divergence: {}".format(divergence), fontsize=12)
    plt.legend(fontsize=12)

    plt.show()
