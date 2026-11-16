"""``StreamTrainer``: drives a streaming model/pipeline and logs per-chunk metrics."""

from __future__ import annotations

import logging
import tracemalloc
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from .metrics import Accuracy, StreamingMetric
from .utils import NotFittedError, Timer, check_X_y, estimate_nbytes, is_streaming

__all__ = ["StreamTrainer"]

logger = logging.getLogger("numcompute_stream")


def _final_estimator(model: Any) -> Any:
    steps = getattr(model, "steps", None)
    return steps[-1][1] if steps else model


class StreamTrainer:
    """Manage a streaming model (or pipeline), its metrics and its log.

    Parameters
    ----------
    model : object
        Anything with ``partial_fit(X, y)`` and ``predict(X)`` - a
        :class:`~numcompute_stream.tree.DecisionTreeClassifier` or a
        :class:`~numcompute_stream.pipeline.Pipeline`.
    metrics : mapping of str -> StreamingMetric, optional
        Extra metrics updated by :meth:`score_chunk`. A cumulative
        ``accuracy`` metric is always present (``Accuracy()`` is added if you
        do not supply one), e.g. ``{"rolling_acc": Accuracy(window=5)}``.
    name : str, optional
        Label used in log lines. Defaults to the model class name.
    track_peak_memory : bool, optional
        Also record the ``tracemalloc`` peak (bytes) of every ``fit_chunk``.
        Adds overhead, so it is off by default.

    Attributes
    ----------
    history : list of dict
        One record per chunk with keys ``chunk, n_samples, scored,
        chunk_accuracy, cumulative_accuracy, <metric names>, fit_time,
        score_time, memory_bytes, buffer_fill, n_retrains, retrained``.

    Raises
    ------
    TypeError
        If ``model`` lacks ``partial_fit`` / ``predict`` or ``metrics`` hold
        non-metric objects.

    Notes
    -----
    ``process_chunk`` implements *prequential* (test-then-train) evaluation:
    each chunk is scored by the model as it stood **before** seeing the chunk,
    then used for training. Before the first retrain the model cannot predict;
    those chunks are logged with ``scored=False`` and NaN accuracy.
    """

    def __init__(
        self,
        model: Any,
        metrics: Optional[Mapping[str, StreamingMetric]] = None,
        name: Optional[str] = None,
        track_peak_memory: bool = False,
    ):
        if not is_streaming(model):
            raise TypeError(f"model {type(model).__name__} has no partial_fit method; it cannot be streamed")
        if not callable(getattr(model, "predict", None)):
            raise TypeError(f"model {type(model).__name__} has no predict method")
        metrics = dict(metrics or {})
        for k, m in metrics.items():
            if not all(callable(getattr(m, a, None)) for a in ("update", "result", "reset")):
                raise TypeError(f"metric {k!r} must provide update/result/reset, got {type(m).__name__}")
        metrics.setdefault("accuracy", Accuracy())
        self.model = model
        self.metrics: Dict[str, StreamingMetric] = metrics
        self.name = name or type(model).__name__
        self.track_peak_memory = track_peak_memory
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self) -> "StreamTrainer":
        """Clear the log and all metrics (the model itself is left untouched)."""
        self._records: Dict[int, Dict[str, Any]] = {}
        self._score_idx = 0
        self._fit_idx = 0
        for m in self.metrics.values():
            m.reset()
        return self

    def _record(self, idx: int) -> Dict[str, Any]:
        return self._records.setdefault(idx, {"chunk": idx})

    @property
    def history(self) -> List[Dict[str, Any]]:
        """Per-chunk log records in chunk order."""
        return [self._records[k] for k in sorted(self._records)]

    # ------------------------------------------------------------------ #
    def fit_chunk(self, X, y) -> Dict[str, Any]:
        """Train on one chunk (``model.partial_fit``) and log time, memory and buffer state.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
        y : array_like, shape (n_samples,)

        Returns
        -------
        dict
            The chunk's log record (also stored in :attr:`history`).

        Raises
        ------
        ValueError
            For inconsistent shapes (raised by validation / the model).
        """
        Xa, ya = check_X_y(X, y)
        rec = self._record(self._fit_idx)
        self._fit_idx += 1
        est = _final_estimator(self.model)
        before = getattr(est, "n_retrains_", 0)
        if self.track_peak_memory:
            tracemalloc.start()
        try:
            with Timer() as t:
                self.model.partial_fit(Xa, ya)
            if self.track_peak_memory:
                rec["peak_bytes"] = tracemalloc.get_traced_memory()[1]
        finally:
            if self.track_peak_memory:
                tracemalloc.stop()
        rec.update(
            n_samples=int(Xa.shape[0]),
            fit_time=t.elapsed,
            memory_bytes=estimate_nbytes(self.model),
            buffer_fill=int(getattr(est, "buffer_fill_", 0)),
            n_retrains=int(getattr(est, "n_retrains_", 0)),
            retrained=bool(getattr(est, "n_retrains_", 0) > before),
        )
        logger.info(
            "[%s] chunk %d fit: n=%d time=%.4fs mem=%.1fKiB buffer=%s retrains=%s",
            self.name,
            rec["chunk"],
            rec["n_samples"],
            t.elapsed,
            rec["memory_bytes"] / 1024,
            rec["buffer_fill"],
            rec["n_retrains"],
        )
        return rec

    def score_chunk(self, X, y) -> Dict[str, Any]:
        """Predict a chunk, update every metric and log the result.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
        y : array_like, shape (n_samples,)
            True labels used only for scoring.

        Returns
        -------
        dict
            The chunk's log record. ``scored`` is False (and the accuracies
            NaN / unchanged) when the model cannot predict yet.
        """
        Xa, ya = check_X_y(X, y)
        rec = self._record(self._score_idx)
        self._score_idx += 1
        rec.setdefault("n_samples", int(Xa.shape[0]))
        try:
            with Timer() as t:
                y_pred = self.model.predict(Xa)
        except NotFittedError:
            rec.update(scored=False, chunk_accuracy=float("nan"), score_time=0.0)
            for k, m in self.metrics.items():
                rec[k] = m.result()
            rec["cumulative_accuracy"] = rec["accuracy"]
            logger.info("[%s] chunk %d: model not trained yet, chunk not scored", self.name, rec["chunk"])
            return rec
        y_pred = np.asarray(y_pred)
        for m in self.metrics.values():
            m.update(ya, y_pred)
        rec.update(
            scored=True,
            chunk_accuracy=float(np.mean(y_pred == ya)) if ya.size else float("nan"),
            score_time=t.elapsed,
        )
        for k, m in self.metrics.items():
            rec[k] = m.result()
        rec["cumulative_accuracy"] = rec["accuracy"]
        logger.info(
            "[%s] chunk %d score: chunk_acc=%.4f cumulative_acc=%.4f",
            self.name,
            rec["chunk"],
            rec["chunk_accuracy"],
            rec["cumulative_accuracy"],
        )
        return rec

    def process_chunk(self, X, y) -> Dict[str, Any]:
        """Prequential step: score the chunk, then train on it. Returns the merged record."""
        self.score_chunk(X, y)
        idx = self._fit_idx
        self.fit_chunk(X, y)
        return self._records[idx]

    def run(
        self,
        stream: Iterable[Tuple[Any, Any]],
        mode: str = "prequential",
        max_chunks: Optional[int] = None,
    ) -> "StreamTrainer":
        """Consume a whole stream of ``(X_chunk, y_chunk)``.

        Parameters
        ----------
        stream : iterable of (X, y)
            E.g. from :func:`numcompute_stream.io.iter_csv_xy_chunks`.
        mode : {'prequential', 'train_only'}, optional
            ``'prequential'`` scores then trains each chunk (default).
        max_chunks : int, optional
            Stop after this many chunks.

        Returns
        -------
        StreamTrainer
            ``self``.

        Raises
        ------
        ValueError
            For an unknown ``mode``.
        """
        if mode not in ("prequential", "train_only"):
            raise ValueError(f"mode must be 'prequential' or 'train_only', got {mode!r}")
        for i, (X, y) in enumerate(stream):
            if max_chunks is not None and i >= max_chunks:
                break
            (self.process_chunk if mode == "prequential" else self.fit_chunk)(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        """Predict with the current model."""
        return self.model.predict(X)

    # ------------------------------------------------------------------ #
    def metric_history(self, key: str = "cumulative_accuracy") -> np.ndarray:
        """Values of a logged key across chunks as a float array (NaN where missing)."""
        return np.array([r.get(key, np.nan) for r in self.history], dtype=float)

    def summary(self) -> Dict[str, Any]:
        """Aggregate view: chunks, samples, retrains, final accuracy, total times, memory."""
        h = self.history
        scored = [r for r in h if r.get("scored")]
        est = _final_estimator(self.model)
        return {
            "n_chunks": len(h),
            "n_samples": int(sum(r.get("n_samples", 0) for r in h if "fit_time" in r)),
            "n_scored_chunks": len(scored),
            "n_retrains": int(getattr(est, "n_retrains_", 0)),
            "final_cumulative_accuracy": self.metrics["accuracy"].result(),
            "mean_chunk_accuracy": float(np.mean([r["chunk_accuracy"] for r in scored])) if scored else float("nan"),
            "total_fit_time": float(sum(r.get("fit_time", 0.0) for r in h)),
            "total_score_time": float(sum(r.get("score_time", 0.0) for r in h)),
            "max_memory_bytes": int(max((r.get("memory_bytes", 0) for r in h), default=0)),
        }
