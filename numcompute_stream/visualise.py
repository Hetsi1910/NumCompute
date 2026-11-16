"""Lightweight matplotlib plotting for streaming experiments.

All functions return the ``matplotlib.figure.Figure`` so they work in scripts,
notebooks and pipeline logs. Each accepts ``save_path`` (write an image file)
and ``show`` (call ``plt.show()`` for inline/interactive display); neither is
required. In a notebook assign the result (``_ = plot_...``) to avoid the
figure being displayed twice.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "plot_metric_over_time",
    "compare_models",
    "plot_predictions_vs_ground_truth",
    "plot_confusion_matrix",
    "plot_feature_importance",
    "plot_stream_history",
]


def _series(values, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    return arr


def _finish(fig, save_path, show, close: bool = False):
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    if close:
        plt.close(fig)
    return fig


def plot_metric_over_time(
    metric_values,
    title: str = "Metric over time",
    ylabel: str = "Metric",
    xlabel: str = "Chunk",
    label: Optional[str] = None,
    ax=None,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Plot a metric (e.g. accuracy) across chunks.

    Parameters
    ----------
    metric_values : array_like, shape (n_chunks,)
        One value per chunk; NaN values leave a gap in the line.
    title, ylabel, xlabel : str, optional
        Text labels.
    label : str, optional
        Legend entry.
    ax : matplotlib Axes, optional
        Draw on an existing axes instead of a new figure.
    save_path : str, optional
        If given, the figure is written there (png/pdf/...).
    show : bool, optional
        Call ``plt.show()``. Default False.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If ``metric_values`` is empty or not 1-D.
    """
    y = _series(metric_values, "metric_values")
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure
    ax.plot(np.arange(len(y)), y, marker="o", markersize=3, linewidth=1.5, label=label)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(alpha=0.3)
    if label:
        ax.legend()
    return _finish(fig, save_path, show)


def compare_models(
    metric1,
    metric2,
    labels: Sequence[str] = ("Model 1", "Model 2"),
    title: str = "Model comparison",
    ylabel: str = "Accuracy",
    xlabel: str = "Chunk",
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Compare two models' streaming metric curves (e.g. different ``buffer_size``).

    Parameters
    ----------
    metric1, metric2 : array_like, 1-D
        Metric per chunk of each model (lengths may differ).
    labels : sequence of 2 str, optional
        Legend names.
    title, ylabel, xlabel : str, optional
    save_path, show : optional
        See module docstring.

    Returns
    -------
    matplotlib.figure.Figure
        The legend also shows each curve's mean (NaN ignored).

    Raises
    ------
    ValueError
        If a series is empty/not 1-D or ``labels`` does not have 2 entries.
    """
    a, b = _series(metric1, "metric1"), _series(metric2, "metric2")
    if len(labels) != 2:
        raise ValueError(f"labels must have exactly 2 entries, got {len(labels)}")
    fig, ax = plt.subplots(figsize=(7, 4))
    for series, lab in ((a, labels[0]), (b, labels[1])):
        mean = np.nanmean(series) if np.isfinite(series).any() else float("nan")
        ax.plot(np.arange(len(series)), series, marker="o", markersize=3, label=f"{lab} (mean {mean:.3f})")
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(alpha=0.3)
    ax.legend()
    return _finish(fig, save_path, show)


def plot_predictions_vs_ground_truth(
    y_true,
    y_pred,
    title: str = "Predictions vs ground truth (latest chunk)",
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Visualise predicted vs actual labels for a chunk.

    Parameters
    ----------
    y_true, y_pred : array_like, shape (n,)
        Labels (numbers or strings), same length, non-empty.
    title : str, optional
    save_path, show : optional

    Returns
    -------
    matplotlib.figure.Figure
        Top panel: true (line) and predicted (dots) class per sample, errors
        marked with red crosses. Bottom panel: bar chart of correct vs wrong
        per class. The accuracy is shown in the title.

    Raises
    ------
    ValueError
        If the inputs are empty, not 1-D, or differ in length.
    """
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    if yt.ndim != 1 or yp.ndim != 1:
        raise ValueError(f"y_true and y_pred must be 1-D, got shapes {yt.shape} and {yp.shape}")
    if yt.shape != yp.shape:
        raise ValueError(f"y_true and y_pred differ in length: {yt.shape[0]} vs {yp.shape[0]}")
    if yt.size == 0:
        raise ValueError("y_true and y_pred must not be empty")
    classes = np.union1d(yt, yp)
    ti, pi = np.searchsorted(classes, yt), np.searchsorted(classes, yp)
    wrong = ti != pi
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={"height_ratios": [2, 1]})
    x = np.arange(yt.size)
    ax1.step(x, ti, where="mid", color="tab:blue", alpha=0.6, label="ground truth")
    ax1.scatter(x, pi, s=14, color="tab:orange", label="prediction", zorder=3)
    ax1.scatter(x[wrong], pi[wrong], marker="x", s=40, color="red", label="error", zorder=4)
    ax1.set_yticks(range(classes.size))
    ax1.set_yticklabels([str(c) for c in classes])
    ax1.set(xlabel="Sample in chunk", ylabel="Class", title=f"{title} - accuracy {1 - wrong.mean():.3f}")
    ax1.legend(loc="upper right", fontsize=8)
    correct = np.bincount(ti[~wrong], minlength=classes.size)
    errors = np.bincount(ti[wrong], minlength=classes.size)
    pos = np.arange(classes.size)
    ax2.bar(pos, correct, label="correct", color="tab:green")
    ax2.bar(pos, errors, bottom=correct, label="wrong", color="tab:red")
    ax2.set_xticks(pos)
    ax2.set_xticklabels([str(c) for c in classes])
    ax2.set(xlabel="True class", ylabel="Samples")
    ax2.legend(fontsize=8)
    return _finish(fig, save_path, show)


def plot_confusion_matrix(
    cm,
    labels: Optional[Sequence] = None,
    title: str = "Confusion matrix",
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Heat-map of a (k, k) confusion matrix with counts printed in the cells.

    Raises
    ------
    ValueError
        If ``cm`` is not a non-empty square matrix or ``labels`` has the wrong length.
    """
    m = np.asarray(cm)
    if m.ndim != 2 or m.shape[0] != m.shape[1] or m.size == 0:
        raise ValueError(f"cm must be a non-empty square matrix, got shape {m.shape}")
    if labels is not None and len(labels) != m.shape[0]:
        raise ValueError(f"labels needs {m.shape[0]} entries, got {len(labels)}")
    names = [str(i) for i in range(m.shape[0])] if labels is None else [str(v) for v in labels]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(m, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(names)),
        yticks=range(len(names)),
        xticklabels=names,
        yticklabels=names,
        xlabel="Predicted",
        ylabel="True",
        title=title,
    )
    thresh = m.max() / 2.0 if m.max() > 0 else 0
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, str(int(m[i, j])), ha="center", va="center", color="white" if m[i, j] > thresh else "black")
    return _finish(fig, save_path, show)


def plot_feature_importance(
    importances,
    feature_names: Optional[Sequence[str]] = None,
    title: str = "Feature importance",
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Horizontal bar chart of feature importances (e.g. ``tree.feature_importances_``)."""
    imp = _series(importances, "importances")
    if feature_names is not None and len(feature_names) != imp.size:
        raise ValueError(f"feature_names needs {imp.size} entries, got {len(feature_names)}")
    names = [f"x[{i}]" for i in range(imp.size)] if feature_names is None else list(feature_names)
    order = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(6, 0.4 * imp.size + 1.5))
    ax.barh([names[i] for i in order], imp[order], color="tab:purple")
    ax.set(title=title, xlabel="Importance")
    return _finish(fig, save_path, show)


def plot_stream_history(
    history: Sequence[Dict[str, Any]],
    title: str = "Stream run",
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Four-panel overview of a :class:`~numcompute_stream.stream.StreamTrainer` log.

    Panels: chunk vs cumulative accuracy (retrain events marked), buffer fill,
    memory footprint, fit time per chunk.

    Raises
    ------
    ValueError
        If ``history`` is empty.
    """
    if not history:
        raise ValueError("history is empty; run the trainer on at least one chunk first")
    x = np.arange(len(history))
    get = lambda k: np.array([r.get(k, np.nan) for r in history], dtype=float)  # noqa: E731
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    ax = axes[0, 0]
    ax.plot(x, get("chunk_accuracy"), marker="o", markersize=3, label="chunk accuracy", alpha=0.7)
    ax.plot(x, get("cumulative_accuracy"), linewidth=2, label="cumulative accuracy")
    for i in np.flatnonzero(get("retrained") == 1):
        ax.axvline(i, color="grey", linestyle=":", alpha=0.6)
    ax.set(title="Accuracy (dotted = retrain)", xlabel="Chunk", ylabel="Accuracy", ylim=(0, 1.02))
    ax.legend(fontsize=8)
    axes[0, 1].plot(x, get("buffer_fill"), color="tab:orange")
    axes[0, 1].set(title="Buffer fill", xlabel="Chunk", ylabel="Samples")
    axes[1, 0].plot(x, get("memory_bytes") / 1024.0, color="tab:green")
    axes[1, 0].set(title="Model memory footprint", xlabel="Chunk", ylabel="KiB")
    axes[1, 1].bar(x, get("fit_time") * 1000.0, color="tab:red")
    axes[1, 1].set(title="Fit time per chunk", xlabel="Chunk", ylabel="ms")
    for a in axes.ravel():
        a.grid(alpha=0.3)
    fig.suptitle(title)
    return _finish(fig, save_path, show)
