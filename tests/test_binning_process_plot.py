"""Composition of existing binning plots without changing their appearance."""

from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytest import fixture, mark, raises
from sklearn.datasets import load_iris
from sklearn.exceptions import NotFittedError

from optbinning import BinningProcess


@fixture(params=["binary", "continuous", "multiclass"])
def process(request):
    data = load_iris()
    X = pd.DataFrame(data.data[:, :3], columns=data.feature_names[:3])
    if request.param == "binary":
        y = (data.target == 0).astype(int)
    elif request.param == "continuous":
        y = data.data[:, 3] + np.linspace(0, 0.01, len(X))
    else:
        y = data.target
    return BinningProcess(list(X.columns), max_n_bins=4).fit(X, y)


def test_grid(process, tmp_path):
    with patch.object(plt, "show") as show:
        fig, axes = process.plot()
        show.assert_not_called()
    assert axes.shape == (2, 2)
    names = process.get_support(names=True)
    assert [ax.get_title() for ax in axes.flat if ax.get_visible()] == list(names)
    assert not axes[1, 1].get_visible()
    assert len(fig.axes) == 7  # Four primary axes and three metric axes.
    fig.canvas.draw()
    assert all(ax.get_legend() is None for ax in fig.axes)
    assert len(fig.legends) == 1
    fig.savefig(tmp_path / "grid.png")
    plt.close(fig)


def test_single_variable_and_order(process):
    names = list(process.get_support(names=True))
    fig, axes = process.plot(variable_names=[names[-1]])
    assert axes.shape == (1, 1)
    assert axes[0, 0].get_title() == names[-1]
    plt.close(fig)
    fig, axes = process.plot(variable_names=names[::-1], ncols=3,
                             add_special=False, add_missing=False)
    assert [ax.get_title() for ax in axes.flat] == names[::-1]
    plt.close(fig)


def test_supplied_axes_ownership(process, tmp_path):
    table = process.get_binned_variable(process.variable_names[0]).binning_table
    table.build()
    fig, ax = plt.subplots()
    unrelated, _ = plt.subplots()  # Must not draw into the current figure.
    with patch.object(plt, "show") as show, patch.object(plt, "close") as close:
        assert table.plot(ax=ax, savefig=str(tmp_path / "panel.png")) is ax
        show.assert_not_called()
        close.assert_not_called()
    assert len(fig.axes) == 2
    assert len(unrelated.axes) == 1
    assert ax.get_title() == table.name
    assert len(ax.patches) > 0
    assert fig.axes[1].get_legend() is not None
    plt.close(fig)
    plt.close(unrelated)
    with patch.object(plt, "show") as show:
        assert table.plot() is None
        show.assert_called_once()
    plt.close("all")


def test_validation(process):
    figures = plt.get_fignums()
    with raises(TypeError, match="share_legend"):
        process.plot(share_legend="yes")
    with raises(TypeError, match="share_metric"):
        process.plot(share_metric="yes")
    for ncols in [0, -1, 1.5, True]:
        with raises(ValueError):
            process.plot(ncols=ncols)
    for names in [[], [process.variable_names[0]] * 2, ["not_a_feature"]]:
        with raises(ValueError):
            process.plot(variable_names=names)
    with raises(TypeError):
        process.plot(variable_names="not_a_list")
    with raises(TypeError):
        process.plot(add_missing="yes")
    assert plt.get_fignums() == figures


def test_not_fitted():
    with raises(NotFittedError):
        BinningProcess(["x"]).plot()


def test_selection_and_large_grid():
    values = np.repeat([0., 1., 2.], 60)
    y = np.concatenate([np.r_[np.zeros(60-n), np.ones(n)]
                        for n in [10, 30, 50]])
    names = [f"x{i}" for i in range(26)]
    X = pd.DataFrame({name: values + i for i, name in enumerate(names)})
    process = BinningProcess(
        names, selection_criteria={"iv": {"strategy": "highest", "top": 2}}
    ).fit(X, y)
    fig, axes = process.plot()
    assert axes.shape == (1, 2)
    assert sum(ax.get_visible() for ax in axes.flat) == 2
    plt.close(fig)
    for count, shape in [(4, (2, 2)), (5, (2, 3)), (8, (3, 3)),
                         (9, (3, 3)), (10, (3, 4)), (26, (5, 6))]:
        fig, axes = process.plot(variable_names=names[:count])
        assert axes.shape == shape
        assert sum(ax.get_visible() for ax in axes.flat) == count
        visible_names = [ax.get_title() for ax in axes.flat
                         if ax.get_visible()]
        assert visible_names == names[:count]
        plt.close(fig)
    fig, axes = process.plot(variable_names=names, ncols=5)
    assert axes.shape == (6, 5)
    assert sum(ax.get_visible() for ax in axes.flat) == 26
    assert axes.flat[25].get_title() == "x25"
    plt.close(fig)


@mark.parametrize("target_kind", ["binary", "continuous"])
def test_mixed_numerical_categorical_grid(target_kind, tmp_path):
    rng = np.random.default_rng(315)
    numeric = rng.normal(size=400)
    category = rng.choice(["low", "medium", "high"], size=400)
    effect = pd.Series(category).map({"low": -1, "medium": 0, "high": 1})
    signal = numeric + effect.to_numpy()
    if target_kind == "binary":
        y = rng.binomial(1, 1 / (1 + np.exp(-signal)))
    else:
        y = signal + rng.normal(scale=0.3, size=400)
    X = pd.DataFrame({"numeric": numeric, "category": category})
    X.loc[:9, "numeric"] = np.nan
    X.loc[10:19, "category"] = None
    process = BinningProcess(
        ["numeric", "category"], categorical_variables=["category"],
        max_n_bins=4).fit(X, y)
    fig, axes = process.plot(show_bin_labels=True)
    assert axes.shape == (1, 2)
    assert [ax.get_title() for ax in axes.flat] == ["numeric", "category"]
    assert process.get_binned_variable("numeric").dtype == "numerical"
    assert process.get_binned_variable("category").dtype == "categorical"
    fig.canvas.draw()
    assert len(axes[0, 0].patches) > 0
    assert len(axes[0, 1].patches) > 0
    labels = " ".join(label.get_text() for label in axes[0, 1].get_xticklabels())
    assert "low" in labels and "high" in labels
    fig.savefig(tmp_path / f"mixed_{target_kind}.png")
    plt.close(fig)


def test_shared_metric_axes(process):
    fig, axes = process.plot(share_metric=False)
    metrics = fig.axes[axes.size:]
    limits = [ax.get_ylim() for ax in metrics]
    expected = (min(low for low, high in limits),
                max(high for low, high in limits))
    counts = [ax.get_ylim() for ax in axes.flat]
    assert not metrics[0].get_shared_y_axes().joined(metrics[0], metrics[1])
    plt.close(fig)

    fig, axes = process.plot()
    metrics = fig.axes[axes.size:]
    for metric in metrics:
        np.testing.assert_allclose(metric.get_ylim(), expected)
        assert metrics[0].get_shared_y_axes().joined(metrics[0], metric)
    assert [ax.get_ylim() for ax in axes.flat] == counts
    assert not axes.flat[0].get_shared_y_axes().joined(
        axes.flat[0], axes.flat[1])
    # Later adjustments propagate to metric axes, including extreme limits.
    metrics[-1].set_ylim(-100, 100)
    for metric in metrics:
        assert metric.get_ylim() == (-100, 100)
    assert [ax.get_ylim() for ax in axes.flat] == counts
    plt.close(fig)


def test_shared_legend(process):
    fig, axes = process.plot(share_legend=False)
    legends = [ax.get_legend() for ax in fig.axes
               if ax.get_legend() is not None]
    expected = [text.get_text() for text in legends[0].get_texts()]
    assert len(legends) == 3
    assert not fig.legends
    plt.close(fig)

    fig, axes = process.plot()
    assert all(ax.get_legend() is None for ax in fig.axes)
    assert len(fig.legends) == 1
    assert [text.get_text() for text in fig.legends[0].get_texts()] == expected
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_box = fig.legends[0].get_window_extent(renderer)
    assert fig.bbox.contains(legend_box.x0, legend_box.y0)
    assert fig.bbox.contains(legend_box.x1, legend_box.y1)
    assert all(not legend_box.overlaps(ax.get_window_extent(renderer))
               for ax in axes.flat if ax.get_visible())
    plt.close(fig)
