"""Streaming classification metrics.

Every metric exposes the same protocol::

    m.update(y_true_chunk, y_pred_chunk)   # absorb a chunk
    m.result()                             # value over everything seen (or the window)
    m.reset()                              # start over

Counts are accumulated in a confusion matrix whose label set grows when new
classes appear, so cumulative streaming results equal the batch metric on the
concatenated data. ``window=k`` switches to a *rolling* metric over the last
``k`` chunks.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np

from numcompute.metrics import roc_auc_score

from .utils import check_labels, check_positive_int

__all__ = ["StreamingMetric", "Accuracy", "Precision", "Recall", "F1", "ConfusionMatrix", "AUC", "confusion_counts"]


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Confusion matrix (rows = true, cols = predicted) over a fixed label set.

    Parameters
    ----------
    y_true, y_pred : numpy.ndarray, shape (n,)
        Labels; every value must be in ``labels``.
    labels : numpy.ndarray, shape (k,)
        Sorted unique labels.

    Returns
    -------
    numpy.ndarray of int64, shape (k, k)

    Notes
    -----
    Vectorised with ``np.bincount``; O(n + k^2).
    """
    k = labels.size
    ti = np.searchsorted(labels, y_true)
    pi = np.searchsorted(labels, y_pred)
    return np.bincount(ti * k + pi, minlength=k * k).reshape(k, k).astype(np.int64)


def _check_pair(y_true, y_pred) -> Tuple[np.ndarray, np.ndarray]:
    yt, yp = check_labels(y_true, "y_true"), check_labels(y_pred, "y_pred")
    if yt.shape != yp.shape:
        raise ValueError(f"y_true and y_pred have different lengths: {yt.shape[0]} vs {yp.shape[0]}")
    return yt, yp


class StreamingMetric:
    """Base class: confusion-matrix based streaming classification metric.

    Parameters
    ----------
    window : int or None, optional
        ``None`` (default) -> cumulative over the whole stream. An int ``k``
        -> rolling metric over the last ``k`` non-empty chunks.

    Attributes
    ----------
    labels_ : numpy.ndarray
        Sorted labels seen so far.
    n_samples_seen_ : int
    n_updates_ : int
        Non-empty chunks absorbed.

    Notes
    -----
    Subclasses implement :meth:`_value` mapping a confusion matrix to the
    metric. Memory is O(k^2) cumulative, plus O(window * chunk) when rolling.
    """

    name = "metric"

    def __init__(self, window: Optional[int] = None):
        self.window = None if window is None else check_positive_int(window, "window")
        self.reset()

    def reset(self) -> "StreamingMetric":
        """Discard all accumulated state."""
        self.labels_ = np.array([])
        self._cm = np.zeros((0, 0), dtype=np.int64)
        self._recent: Deque[Tuple[np.ndarray, np.ndarray]] = deque(maxlen=self.window)
        self.n_samples_seen_ = 0
        self.n_updates_ = 0
        return self

    def update(self, y_true_chunk, y_pred_chunk) -> "StreamingMetric":
        """Absorb a chunk of true and predicted labels.

        Parameters
        ----------
        y_true_chunk, y_pred_chunk : array_like, shape (n,)
            Labels (numbers or strings); an empty chunk is ignored.

        Returns
        -------
        StreamingMetric
            ``self``.

        Raises
        ------
        ValueError
            If the arrays are not 1-D, differ in length or contain NaN.
        """
        yt, yp = _check_pair(y_true_chunk, y_pred_chunk)
        if yt.size == 0:
            return self
        new_labels = np.union1d(self.labels_, np.union1d(yt, yp)) if self.labels_.size else np.union1d(yt, yp)
        if new_labels.size != self.labels_.size:  # grow the matrix and remap old rows/cols
            grown = np.zeros((new_labels.size,) * 2, dtype=np.int64)
            if self.labels_.size:
                pos = np.searchsorted(new_labels, self.labels_)
                grown[np.ix_(pos, pos)] = self._cm
            self._cm, self.labels_ = grown, new_labels
        self._cm += confusion_counts(yt, yp, self.labels_)
        if self.window is not None:
            self._recent.append((yt, yp))
        self.n_samples_seen_ += yt.size
        self.n_updates_ += 1
        return self

    def _current_confusion(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.window is None or not self._recent:
            return self._cm, self.labels_
        yt = np.concatenate([a for a, _ in self._recent])
        yp = np.concatenate([b for _, b in self._recent])
        labels = np.union1d(yt, yp)
        return confusion_counts(yt, yp, labels), labels

    def result(self):
        """Current metric value (``nan`` before any data)."""
        cm, labels = self._current_confusion()
        if cm.size == 0 or cm.sum() == 0:
            return float("nan")
        return self._value(cm, labels)

    __call__ = result

    def _value(self, cm: np.ndarray, labels: np.ndarray):  # pragma: no cover - abstract
        raise NotImplementedError


class Accuracy(StreamingMetric):
    """Fraction of correct predictions (streaming)."""

    name = "accuracy"

    def _value(self, cm, labels):
        return float(np.trace(cm) / cm.sum())


class _PRF(StreamingMetric):
    """Shared machinery for precision / recall / F1."""

    def __init__(self, average: Optional[str] = "macro", pos_label=1, zero_division: float = 0.0, window=None):
        if average not in (None, "binary", "macro", "micro", "weighted"):
            raise ValueError(f"average must be None, 'binary', 'macro', 'micro' or 'weighted', got {average!r}")
        self.average = average
        self.pos_label = pos_label
        self.zero_division = float(zero_division)
        super().__init__(window)

    @staticmethod
    def _div(num, den, zd):
        num, den = np.asarray(num, float), np.asarray(den, float)
        return np.divide(num, den, out=np.full(np.broadcast(num, den).shape, zd), where=den > 0)

    def _per_class(self, cm):
        tp = np.diag(cm).astype(float)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        return tp, fp, fn, cm.sum(axis=1).astype(float)

    def _score(self, tp, fp, fn):  # pragma: no cover - abstract
        raise NotImplementedError

    def _value(self, cm, labels):
        tp, fp, fn, support = self._per_class(cm)
        if self.average == "binary":
            hit = np.flatnonzero(labels == self.pos_label)
            if hit.size == 0:
                return self.zero_division
            i = hit[0]
            return float(self._score(tp[i], fp[i], fn[i]))
        if self.average == "micro":
            return float(self._score(tp.sum(), fp.sum(), fn.sum()))
        per = self._score(tp, fp, fn)
        if self.average is None:
            return per
        if self.average == "macro":
            return float(per.mean())
        return float(np.average(per, weights=support)) if support.sum() > 0 else self.zero_division


class Precision(_PRF):
    """Streaming precision ``tp / (tp + fp)``.

    Parameters
    ----------
    average : {'macro', 'micro', 'weighted', 'binary', None}, optional
        Multi-class averaging. ``'binary'`` scores only ``pos_label``;
        ``None`` returns a per-class array. Default ``'macro'``.
    pos_label : object, optional
        Positive class for ``'binary'``. Default 1.
    zero_division : float, optional
        Value used when a denominator is zero. Default 0.0.
    window : int or None, optional
        Rolling window in chunks.
    """

    name = "precision"

    def _score(self, tp, fp, fn):
        return self._div(tp, tp + fp, self.zero_division)


class Recall(_PRF):
    """Streaming recall ``tp / (tp + fn)`` (parameters as :class:`Precision`)."""

    name = "recall"

    def _score(self, tp, fp, fn):
        return self._div(tp, tp + fn, self.zero_division)


class F1(_PRF):
    """Streaming F1 score ``2tp / (2tp + fp + fn)`` (parameters as :class:`Precision`)."""

    name = "f1"

    def _score(self, tp, fp, fn):
        return self._div(2 * tp, 2 * tp + fp + fn, self.zero_division)


class ConfusionMatrix(StreamingMetric):
    """Streaming confusion matrix; ``result()`` returns the count matrix.

    Rows are true labels, columns predicted labels, both ordered like
    ``labels_`` (sorted). The label set grows as new classes appear.
    """

    name = "confusion_matrix"

    def result(self) -> np.ndarray:
        """Return the (k, k) integer matrix (empty (0, 0) before any data)."""
        cm, _ = self._current_confusion()
        return cm.copy()

    __call__ = result

    @property
    def labels(self) -> np.ndarray:
        """Labels indexing rows/columns of the current matrix."""
        return self._current_confusion()[1].copy()


class AUC:
    """Streaming binary ROC-AUC (accumulates scores over time).

    Parameters
    ----------
    pos_label : object, optional
        Label treated as positive. Default 1.
    window : int or None, optional
        Rolling window in chunks (``None`` = whole stream).

    Notes
    -----
    ``update(y_true_chunk, y_score_chunk)`` takes *scores* (e.g. the positive
    column of ``predict_proba``), not hard labels. The exact AUC needs the
    ranking of all scores, so memory is O(samples kept). ``result()`` is
    ``nan`` until both classes have been observed. Reuses
    :func:`numcompute.metrics.roc_auc_score`.
    """

    name = "auc"

    def __init__(self, pos_label=1, window: Optional[int] = None):
        self.pos_label = pos_label
        self.window = None if window is None else check_positive_int(window, "window")
        self.reset()

    def reset(self) -> "AUC":
        """Discard all accumulated scores."""
        self._chunks: Deque[Tuple[np.ndarray, np.ndarray]] = deque(maxlen=self.window)
        self.n_samples_seen_ = 0
        return self

    def update(self, y_true_chunk, y_score_chunk) -> "AUC":
        """Absorb labels and scores (1-D, equal length; empty chunks ignored)."""
        yt = check_labels(y_true_chunk, "y_true")
        ys = np.asarray(y_score_chunk, dtype=float)
        if ys.ndim != 1 or ys.shape != yt.shape:
            raise ValueError(f"y_score must be 1-D with the length of y_true ({yt.shape[0]}), got shape {ys.shape}")
        if np.isnan(ys).any():
            raise ValueError("y_score contains NaN")
        if yt.size:
            self._chunks.append(((yt == self.pos_label).astype(int), ys))
            self.n_samples_seen_ += yt.size
        return self

    def result(self) -> float:
        """AUC over the stored chunks; ``nan`` if one class is missing."""
        if not self._chunks:
            return float("nan")
        yt = np.concatenate([a for a, _ in self._chunks])
        ys = np.concatenate([b for _, b in self._chunks])
        if yt.min() == yt.max():
            return float("nan")
        return float(roc_auc_score(yt, ys))

    __call__ = result
