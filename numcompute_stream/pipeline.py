"""Streaming pipeline: chains transformers and a final model with ``partial_fit``."""

from __future__ import annotations

import inspect
from typing import Any, Callable

import numpy as np

from numcompute import pipeline as _nc

from .utils import check_X_y, is_streaming

__all__ = ["Pipeline", "BaseEstimator", "Transformer", "Estimator"]

BaseEstimator = _nc.BaseEstimator
Transformer = _nc.Transformer
Estimator = _nc.Estimator


def _accepts_y(method: Callable) -> bool:
    """True if ``method(X, y)`` is a valid call (second positional parameter or ``*args``)."""
    try:
        params = list(inspect.signature(method).parameters.values())
    except (TypeError, ValueError):  # builtins / C extensions without a signature
        return True
    if any(p.kind is p.VAR_POSITIONAL for p in params):
        return True
    positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    return len(positional) >= 2


def _call(method: Callable, X, y):
    """Call ``method(X, y)`` or ``method(X)`` depending on what the step's signature supports."""
    return method(X, y) if _accepts_y(method) else method(X)


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

    # -- batch fitting: dispatch on signatures so minimal ``fit(X)`` / ``fit_transform(X)`` steps work ---------
    def _fit_intermediate(self, X, y):
        Xt = X
        for _, step in self.steps[:-1]:
            if hasattr(step, "fit_transform"):
                Xt = _call(step.fit_transform, Xt, y)
            else:
                _call(step.fit, Xt, y)
                Xt = step.transform(Xt)
        return Xt

    def fit(self, X, y=None) -> "Pipeline":
        """Fit every step in order (steps whose ``fit`` takes only ``X`` are called without ``y``)."""
        Xt = self._fit_intermediate(X, y)
        _call(self.steps[-1][1].fit, Xt, y)
        return self

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
            _call(step.partial_fit, Xt, yt)
            Xt = step.transform(Xt)
        last = self.steps[-1][1]
        if yt is None:
            last.partial_fit(Xt)
        else:
            _call(last.partial_fit, Xt, yt)
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
