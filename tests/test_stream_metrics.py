"""Unit tests for numcompute_stream.metrics (streaming metrics)."""

import numpy as np
import pytest

import numcompute.metrics as batch
from numcompute_stream.metrics import AUC, F1, Accuracy, ConfusionMatrix, Precision, Recall


@pytest.fixture
def yy():
    rng = np.random.default_rng(1)
    yt = rng.integers(0, 3, 600)
    yp = np.where(rng.random(600) < 0.7, yt, rng.integers(0, 3, 600))
    return yt, yp


def _stream(metric, yt, yp, parts=7):
    for a, b in zip(np.array_split(yt, parts), np.array_split(yp, parts)):
        metric.update(a, b)
    return metric


def test_accuracy_equals_batch(yy):
    assert _stream(Accuracy(), *yy).result() == pytest.approx(batch.accuracy(*yy))


@pytest.mark.parametrize("cls,fn", [(Precision, batch.precision), (Recall, batch.recall), (F1, batch.f1)])
@pytest.mark.parametrize("avg", ["macro", "micro", "weighted"])
def test_prf_equal_batch(cls, fn, avg, yy):
    got = _stream(cls(average=avg), *yy).result()
    assert got == pytest.approx(fn(*yy, average=avg))


def test_binary_average_matches_batch():
    yt = np.array([0, 1, 1, 0, 1, 0, 1, 1])
    yp = np.array([0, 1, 0, 0, 1, 1, 1, 1])
    assert _stream(Precision(average="binary"), yt, yp, 2).result() == pytest.approx(batch.precision(yt, yp))


def test_per_class_average_none(yy):
    r = _stream(Recall(average=None), *yy).result()
    assert r.shape == (3,)


def test_confusion_matrix_accumulates_equal_batch(yy):
    cm = _stream(ConfusionMatrix(), *yy)
    assert np.array_equal(cm.result(), batch.confusion_matrix(*yy))


def test_labels_grow_when_new_class_appears():
    m = Accuracy()
    m.update([0, 1], [0, 1])
    m.update([2, 2], [2, 0])
    assert m.labels_.tolist() == [0, 1, 2]
    assert m.result() == pytest.approx(3 / 4)


def test_string_labels_supported():
    cm = ConfusionMatrix().update(["a", "b", "a"], ["a", "a", "a"])
    assert cm.labels.tolist() == ["a", "b"] and cm.result().tolist() == [[2, 0], [1, 0]]


def test_rolling_window_only_uses_recent_chunks():
    m = Accuracy(window=2)
    m.update([0, 0], [1, 1])  # all wrong
    m.update([0, 0], [0, 0])  # right
    m.update([0, 0], [0, 0])  # right
    assert m.result() == 1.0  # first (bad) chunk slid out
    assert Accuracy().update([0, 0], [1, 1]).update([0, 0], [0, 0]).result() == 0.5


def test_rolling_f1_window(yy):
    m = _stream(F1(window=2), *yy, parts=6)
    yt = np.concatenate(np.array_split(yy[0], 6)[-2:])
    yp = np.concatenate(np.array_split(yy[1], 6)[-2:])
    assert m.result() == pytest.approx(batch.f1(yt, yp, average="macro"))


def test_reset_clears_state(yy):
    m = _stream(Accuracy(), *yy)
    m.reset()
    assert np.isnan(m.result()) and m.n_samples_seen_ == 0


def test_result_nan_before_any_data_and_empty_chunk_ignored():
    m = Accuracy()
    assert np.isnan(m.result())
    m.update([], [])
    assert np.isnan(m.result()) and m.n_updates_ == 0


def test_zero_division_default_and_custom():
    # class 1 never predicted -> precision undefined for it
    p0 = Precision(average=None).update([0, 1], [0, 0]).result()
    assert p0.tolist() == [0.5, 0.0]
    p1 = Precision(average=None, zero_division=1.0).update([0, 1], [0, 0]).result()
    assert p1.tolist() == [0.5, 1.0]


def test_binary_pos_label_missing_returns_zero_division():
    assert Recall(average="binary", pos_label=9, zero_division=0.25).update([0, 1], [0, 1]).result() == 0.25


def test_metric_input_validation():
    with pytest.raises(ValueError, match="different lengths"):
        Accuracy().update([0, 1], [0])
    with pytest.raises(ValueError, match="1-D"):
        Accuracy().update(np.zeros((2, 2)), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="NaN"):
        Accuracy().update([0.0, np.nan], [0, 1])
    with pytest.raises(ValueError):
        Precision(average="bogus")
    with pytest.raises(ValueError):
        Accuracy(window=0)
    with pytest.raises(TypeError):
        Accuracy(window=1.5)


def test_auc_accumulates_equals_batch():
    rng = np.random.default_rng(3)
    yt = rng.integers(0, 2, 400)
    sc = np.clip(yt * 0.4 + rng.random(400) * 0.8, 0, 1)
    m = AUC()
    for a, b in zip(np.array_split(yt, 8), np.array_split(sc, 8)):
        m.update(a, b)
    assert m.result() == pytest.approx(batch.roc_auc_score(yt, sc))


def test_auc_nan_with_single_class_and_window():
    m = AUC().update([1, 1], [0.2, 0.9])
    assert np.isnan(m.result())
    w = AUC(window=1)
    w.update([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    w.update([0, 1], [0.9, 0.1])  # only this (inverted) chunk is kept
    assert w.result() == 0.0
    with pytest.raises(ValueError):
        AUC().update([0, 1], [0.5])
    with pytest.raises(ValueError, match="NaN"):
        AUC().update([0, 1], [0.5, np.nan])
