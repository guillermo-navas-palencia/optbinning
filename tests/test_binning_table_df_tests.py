"""Numerical and edge-case coverage for exposed analysis results."""

import numpy as np
import pandas as pd
from pytest import mark, raises
from scipy import stats
from sklearn.exceptions import NotFittedError

from optbinning import (OptimalBinning, ContinuousOptimalBinning,
                        MulticlassOptimalBinning)


@mark.parametrize("kind", ["binary", "continuous", "multiclass"])
@mark.parametrize("n_bins", [1, 3])
def test_analysis_results(kind, n_bins):
    x = np.repeat([0., 1., 2.], 60)
    if kind == "binary":
        cls = OptimalBinning
        y = np.concatenate([np.r_[np.zeros(60 - n), np.ones(n)]
                            for n in [10, 30, 50]])
    elif kind == "continuous":
        cls = ContinuousOptimalBinning
        y = x + np.tile(np.linspace(-0.4, 0.4, 60), 3)
    else:
        cls = MulticlassOptimalBinning
        y = np.concatenate([np.repeat([0, 1, 2], counts)
                            for counts in [[30, 20, 10], [10, 30, 20],
                                           [20, 10, 30]]])
    model = cls(user_splits=[0.5, 1.5], max_n_bins=n_bins,
                monotonic_trend=None).fit(x, y)
    table = model.binning_table
    with raises(NotFittedError):
        table.df_tests
    built = table.build(add_totals=False)
    table.analysis(print_output=False)
    result = table.df_tests
    assert len(result) == n_bins - 1
    assert list(result.columns[:4]) == [
        "Bin A", "Bin B", "t-statistic", "p-value"]
    for i, row in result.iterrows():
        assert row["Bin A"] == i
        assert row["Bin B"] == i + 1
        if kind == "binary":
            obs = np.array([table.n_nonevent[i:i+2], table.n_event[i:i+2]])
            statistic, pvalue, _, _ = stats.chi2_contingency(
                obs, correction=False)
        elif kind == "multiclass":
            statistic, pvalue, _, _ = stats.chi2_contingency(
                table.n_event[i:i+2], correction=False)
        else:
            statistic, pvalue = stats.ttest_ind_from_stats(
                built.loc[i, "Mean"], table.stds[i], table.n_records[i],
                built.loc[i+1, "Mean"], table.stds[i+1], table.n_records[i+1],
                equal_var=False)
        np.testing.assert_allclose([row["t-statistic"], row["p-value"]],
                                   [statistic, pvalue])
    snapshot = result.copy()
    result["p-value"] = -1
    pd.testing.assert_frame_equal(table.df_tests, snapshot)
    if kind == "binary":
        table.analysis(pvalue_test="fisher", print_output=False)
        assert "t-statistic" not in table.df_tests
        assert "odd ratio" in table.df_tests
        for i, row in table.df_tests.iterrows():
            expected = stats.fisher_exact(np.array([
                table.n_nonevent[i:i+2], table.n_event[i:i+2]]))
            np.testing.assert_allclose([row["odd ratio"], row["p-value"]],
                                       expected)
