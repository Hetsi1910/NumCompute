"""Streaming pipeline: chains transformers and a final model with ``partial_fit``."""

from __future__ import annotations

from typing import Any

import numpy as np

from numcompute import pipeline as _nc

from .utils import check_X_y, is_streaming

__all__ = ["Pipeline", "BaseEstimator", "Transformer", "Estimator"]

BaseEstimator = _nc.BaseEstimator
Transformer = _nc.Transformer
Estimator = _nc.Estimator


class Pipeline(_nc.Pipeline):
    """Pipeline whose steps can all be updated incrementally.

    Extends :class:`numcompute.pipeline.Pipeline` (so ``fit``, ``transform``,
    ``predict`` still work) with:

    * :meth:`partial_fit` - each chunk flows through the chain: every
      transformer is updated with ``partial_fit`` and then transforms the
      chunk for the next step; the last step is updated with
      ``partial_fit(Xt, y)``.
    * :meth:`predict_proba`, :meth:`score`, :meth:`flush`.

    Parameters
    ----------
    steps : list of (str, object)
        ``(name, step)`` pairs. All steps must implement ``partial_fit`` to be
        used for streaming.

    Examples
    --------
    >>> from numcompute_stream.preprocessing import StandardScaler
    >>> from numcompute_stream.tree import DecisionTreeClassifier
    >>> pipe = Pipeline([("scale", StandardScaler()),
    ...                  ("model", DecisionTreeClassifier(max_depth=5, buffer_size=500))])
    >>> # pipe.partial_fit(X_chunk, y_chunk)
    """

    @property
    def supports_streaming(self) -> bool:
        """True if every step implements ``partial_fit``."""
        return all(is_streaming(step) for _, step in self.steps)

    def partial_fit(self, X, y=None) -> "Pipeline":
        """Update every step with one chunk (incremental fit + transform + fit).

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
        y : array_like, shape (n_samples,), optional
            Labels for the final model (required by supervised models).

        Returns
        -------
        Pipeline
            ``self``.

        Raises
        ------
        TypeError
            If a step has no ``partial_fit`` (names the offending step).
        ValueError
            If ``X``/``y`` shapes are inconsistent.
        """
        for name, step in self.steps:
            if not is_streaming(step):
                raise TypeError(
                    f"step {name!r} ({type(step).__name__}) has no partial_fit; "
                    "it cannot be used in a streaming pipeline"
                )
        Xt, yt = check_X_y(X, y) if y is not None else (check_X_y(X)[0], None)
        for name, step in self.steps[:-1]:
            step.partial_fit(Xt, yt)
            Xt = step.transform(Xt)
        last = self.steps[-1][1]
        if yt is None:
            last.partial_fit(Xt)
        else:
            last.partial_fit(Xt, yt)
        return self

    def predict_proba(self, X):
        """Transform with all but the last step, then ``predict_proba``.

        Raises
        ------
        TypeError
            If the final step has no ``predict_proba``.
        """
        Xt = X
        for _, step in self.steps[:-1]:
            Xt = step.transform(Xt)
        last = self.steps[-1][1]
        if not hasattr(last, "predict_proba"):
            raise TypeError("final step has no predict_proba method")
        return last.predict_proba(Xt)

    def score(self, X, y) -> float:
        """Accuracy of ``predict(X)`` against ``y``."""
        Xa, ya = check_X_y(X, y)
        return float(np.mean(self.predict(Xa) == ya)) if ya.size else float("nan")

    def flush(self) -> "Pipeline":
        """Call ``flush`` on the final step if it has one (e.g. train on a partial buffer)."""
        last: Any = self.steps[-1][1]
        if hasattr(last, "flush"):
            last.flush()
        return self

    @property
    def final_estimator(self):
        """The last step's object."""
        return self.steps[-1][1]
