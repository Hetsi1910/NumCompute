"""Shared helpers for :mod:`numcompute_stream`.

Re-exports the validation helpers of the group ``numcompute`` package and adds
the streaming-specific checks (chunk validation, memory estimation, timing).
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Optional, Set, Tuple

import numpy as np

from numcompute.utils import (  # noqa: F401  (re-exported)
    NotFittedError,
    as_float_array,
    check_2d,
    check_is_fitted,
)

__all__ = [
    "NotFittedError",
    "as_float_array",
    "check_2d",
    "check_is_fitted",
    "check_positive_int",
    "check_X_y",
    "check_labels",
    "is_streaming",
    "estimate_nbytes",
    "Timer",
]


def check_positive_int(value, name: str, minimum: int = 1) -> int:
    """Validate that ``value`` is an integer ``>= minimum``.

    Parameters
    ----------
    value : int
        Value to check (``bool`` is rejected).
    name : str
        Parameter name used in the error message.
    minimum : int, optional
        Smallest allowed value. Default 1.

    Returns
    -------
    int
        ``value`` as a built-in ``int``.

    Raises
    ------
    TypeError
        If ``value`` is not an integer.
    ValueError
        If ``value < minimum``.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an int, got {type(value).__name__} ({value!r})")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return int(value)


def check_labels(y, name: str = "y") -> np.ndarray:
    """Return ``y`` as a 1-D label array (any dtype) without NaN.

    Parameters
    ----------
    y : array_like, shape (n_samples,)
        Class labels (numbers or strings). A column vector ``(n, 1)`` is
        flattened.

    Returns
    -------
    numpy.ndarray, shape (n_samples,)

    Raises
    ------
    ValueError
        If ``y`` is not 1-D (after flattening a column vector) or holds NaN.
    """
    arr = np.asarray(y)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.ravel()
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if arr.dtype.kind == "f" and np.isnan(arr).any():
        raise ValueError(f"{name} contains NaN; labels must be fully observed")
    return arr


def check_X_y(X, y=None, name_X: str = "X", name_y: str = "y") -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Validate a ``(X, y)`` chunk.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        Feature chunk. Zero rows are allowed (an empty chunk).
    y : array_like, shape (n_samples,), optional
        Labels. ``None`` skips label validation.

    Returns
    -------
    X : numpy.ndarray, float64, shape (n_samples, n_features)
    y : numpy.ndarray or None, shape (n_samples,)

    Raises
    ------
    TypeError
        If ``X`` is not numeric.
    ValueError
        If ``X`` is not 2-D, contains infinity, ``y`` is not 1-D / has NaN, or
        the lengths of ``X`` and ``y`` differ.
    """
    Xa = check_2d(X, name_X)
    if np.isinf(Xa).any():
        raise ValueError(f"{name_X} contains infinity; clip or remove infinite values first")
    if y is None:
        return Xa, None
    ya = check_labels(y, name_y)
    if ya.shape[0] != Xa.shape[0]:
        raise ValueError(f"{name_X} and {name_y} have inconsistent lengths: {Xa.shape[0]} rows vs {ya.shape[0]} labels")
    return Xa, ya


def is_streaming(obj: Any) -> bool:
    """Return True if ``obj`` exposes a callable ``partial_fit``."""
    return callable(getattr(obj, "partial_fit", None))


def estimate_nbytes(obj: Any, _seen: Optional[Set[int]] = None) -> int:
    """Estimate the memory held by ``obj`` (arrays, containers, attributes).

    Parameters
    ----------
    obj : object
        Model, transformer, array or container.

    Returns
    -------
    int
        Approximate number of bytes (NumPy buffers + ``sys.getsizeof`` of
        scalars/containers). It is an *estimate* used for logging the
        footprint of streaming components, not an exact allocator figure.

    Notes
    -----
    Cycle-safe (each object is counted once). Time O(#objects).
    """
    import sys

    seen = set() if _seen is None else _seen
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)
    size = sys.getsizeof(obj, 0)
    if isinstance(obj, dict):
        size += sum(estimate_nbytes(k, seen) + estimate_nbytes(v, seen) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set, frozenset, deque)):
        size += sum(estimate_nbytes(v, seen) for v in obj)
    elif hasattr(obj, "__dict__") and not isinstance(obj, type):
        size += estimate_nbytes(vars(obj), seen)
    return int(size)


class Timer:
    """Context manager measuring wall-clock seconds with ``perf_counter``.

    Examples
    --------
    >>> with Timer() as t:
    ...     _ = sum(range(10))
    >>> t.elapsed >= 0
    True
    """

    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = time.perf_counter() - self._start
