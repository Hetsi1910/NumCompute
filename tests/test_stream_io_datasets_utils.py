"""Unit tests for numcompute_stream.io, datasets, utils and visualise."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from numcompute_stream import visualise as vis
from numcompute_stream.datasets import make_drift_stream
from numcompute_stream.io import (
    iter_array_chunks,
    iter_csv_xy_chunks,
    load_csv_xy,
    read_feature_names,
    split_xy,
    write_csv_xy,
)
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.utils import Timer, check_X_y, check_labels, check_positive_int, estimate_nbytes, is_streaming


# ------------------------------------------------------------------ io
@pytest.fixture
def csv_path(tmp_path):
    X, y = make_drift_stream(95, seed=1, missing_rate=0.05)
    p = tmp_path / "s.csv"
    write_csv_xy(p, X, y, header=[f"a{i}" for i in range(X.shape[1])] + ["cls"])
    return p, X, y


def test_csv_roundtrip_load(csv_path):
    p, X, y = csv_path
    X2, y2 = load_csv_xy(p)
    assert np.allclose(X2, X, equal_nan=True, atol=1e-5) and np.array_equal(y2, y) and y2.dtype.kind == "i"


def test_csv_chunk_sizes_and_content(csv_path):
    p, X, y = csv_path
    chunks = list(iter_csv_xy_chunks(p, 40))
    assert [len(c[1]) for c in chunks] == [40, 40, 15]
    assert np.array_equal(np.concatenate([c[1] for c in chunks]), y)


def test_feature_names_exclude_target(csv_path):
    assert read_feature_names(csv_path[0])[:2] == ["a0", "a1"] and "cls" not in read_feature_names(csv_path[0])


def test_split_xy_target_column_variants():
    data = np.arange(12.0).reshape(3, 4)
    X, y = split_xy(data, 0)
    assert X.shape == (3, 3) and y.tolist() == [0, 4, 8]
    with pytest.raises(ValueError):
        split_xy(data, 9)
    with pytest.raises(ValueError):
        split_xy(np.arange(3.0))


def test_iter_array_chunks_order_shuffle_and_errors():
    X, y = np.arange(10.0).reshape(-1, 1), np.arange(10)
    assert [len(b) for _, b in iter_array_chunks(X, y, 4)] == [4, 4, 2]
    assert next(iter_array_chunks(X, y, 4))[1].tolist() == [0, 1, 2, 3]
    s = np.concatenate([b for _, b in iter_array_chunks(X, y, 3, shuffle=True, seed=0)])
    assert sorted(s.tolist()) == list(range(10)) and s.tolist() != list(range(10))
    with pytest.raises(ValueError):
        list(iter_array_chunks(X, y[:5], 4))
    with pytest.raises(ValueError):
        list(iter_array_chunks(X, y, 0))


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_csv_xy("does_not_exist.csv")


def test_write_csv_validation(tmp_path):
    with pytest.raises(ValueError):
        write_csv_xy(tmp_path / "x.csv", np.ones((2, 2)), [0])
    with pytest.raises(ValueError):
        write_csv_xy(tmp_path / "x.csv", np.ones((2, 2)), [0, 1], header=["only"])


# ------------------------------------------------------------------ datasets
def test_dataset_reproducible_and_shapes():
    a, b = make_drift_stream(200, seed=4), make_drift_stream(200, seed=4)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert a[0].shape == (200, 6) and set(a[1]) <= {0, 1, 2}


def test_dataset_missing_rate_and_validation():
    X, _ = make_drift_stream(1000, missing_rate=0.1, seed=0)
    assert 0.07 < np.isnan(X).mean() < 0.13
    for kw in ({"n_features": 3}, {"n_classes": 1}, {"label_noise": 2.0}, {"missing_rate": 1.0}, {"drift_at": 1.5}):
        with pytest.raises(ValueError):
            make_drift_stream(50, **kw)


# ------------------------------------------------------------------ utils
def test_check_positive_int():
    assert check_positive_int(3, "k") == 3
    with pytest.raises(TypeError):
        check_positive_int(True, "k")
    with pytest.raises(ValueError):
        check_positive_int(0, "k")


def test_check_labels_and_X_y():
    assert check_labels(np.array([[1], [2]])).tolist() == [1, 2]
    with pytest.raises(ValueError):
        check_labels(np.zeros((2, 2)))
    X, y = check_X_y([[1, 2]], [0])
    assert X.dtype == float and y.shape == (1,)


def test_is_streaming_and_estimate_nbytes_and_timer():
    assert is_streaming(DecisionTreeClassifier()) and not is_streaming(object())
    small = estimate_nbytes(DecisionTreeClassifier())
    big = estimate_nbytes({"a": np.zeros(10_000)})
    assert big > 80_000 > small
    cyc = []
    cyc.append(cyc)
    assert estimate_nbytes(cyc) > 0  # cycle-safe
    with Timer() as t:
        pass
    assert t.elapsed >= 0


# ------------------------------------------------------------------ visualise
@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def test_plot_metric_over_time_saves_file_and_handles_nan(tmp_path):
    out = tmp_path / "m.png"
    fig = vis.plot_metric_over_time([np.nan, 0.5, 0.7], "t", "acc", save_path=str(out))
    assert out.exists() and fig.axes[0].get_title() == "t" and fig.axes[0].get_ylabel() == "acc"


def test_compare_models_two_lines_and_label_check(tmp_path):
    fig = vis.compare_models([0.1, 0.2, 0.3], [0.2, 0.4], labels=("A", "B"))
    assert len(fig.axes[0].lines) == 2
    with pytest.raises(ValueError, match="2 entries"):
        vis.compare_models([1], [2], labels=("A",))


def test_plot_predictions_vs_ground_truth_and_errors():
    fig = vis.plot_predictions_vs_ground_truth([0, 1, 2, 1], [0, 2, 2, 1])
    assert "0.750" in fig.axes[0].get_title()
    vis.plot_predictions_vs_ground_truth(["a", "b"], ["a", "a"])
    with pytest.raises(ValueError):
        vis.plot_predictions_vs_ground_truth([], [])
    with pytest.raises(ValueError, match="length"):
        vis.plot_predictions_vs_ground_truth([0, 1], [0])


def test_plot_empty_input_raises():
    with pytest.raises(ValueError, match="empty"):
        vis.plot_metric_over_time([])
    with pytest.raises(ValueError):
        vis.plot_metric_over_time(np.zeros((2, 2)))


def test_confusion_importance_and_history_plots(tmp_path):
    vis.plot_confusion_matrix([[3, 1], [0, 4]], labels=["a", "b"])
    with pytest.raises(ValueError):
        vis.plot_confusion_matrix(np.zeros((2, 3)))
    vis.plot_feature_importance([0.2, 0.8], ["x", "y"])
    with pytest.raises(ValueError):
        vis.plot_feature_importance([0.2, 0.8], ["x"])
    hist = [
        {
            "chunk_accuracy": 0.5,
            "cumulative_accuracy": 0.5,
            "retrained": True,
            "buffer_fill": 0,
            "memory_bytes": 1024,
            "fit_time": 0.01,
        }
    ] * 3
    vis.plot_stream_history(hist, save_path=str(tmp_path / "h.png"))
    assert (tmp_path / "h.png").exists()
    with pytest.raises(ValueError):
        vis.plot_stream_history([])
