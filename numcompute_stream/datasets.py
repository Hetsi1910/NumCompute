"""Synthetic stream generators (with optional concept drift and missing data)."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .utils import check_positive_int

__all__ = ["make_drift_stream"]


def _bin_codes(col: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile-bin a column into ``n_bins`` equally populated integer codes."""
    if n_bins == 1:
        return np.zeros(col.shape[0], dtype=np.int64)
    edges = np.quantile(col, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.searchsorted(edges, col, side="right").astype(np.int64)


def make_drift_stream(
    n_samples: int = 6000,
    n_features: int = 6,
    n_classes: int = 3,
    label_noise: float = 0.03,
    drift_at: Optional[float] = 0.5,
    missing_rate: float = 0.0,
    seed: Optional[int] = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a classification stream whose concept can change mid-way.

    The label is the quantile bin (``n_classes`` equally populated classes) of
    the score ``x_a + 0.6 * x_b``: an oblique boundary that a depth-limited
    tree approximates with axis-aligned splits and that greedy splitting can
    find from a few hundred samples. Before ``drift_at`` the informative pair is ``(0, 1)``; afterwards
    it switches to ``(2, 3)``, so a model trained on old data becomes wrong
    until it is retrained on new data.

    Parameters
    ----------
    n_samples : int, optional
        Stream length. Default 6000.
    n_features : int, optional
        Number of features, ``>= 4``. Default 6.
    n_classes : int, optional
        Number of classes, ``>= 2``. Default 3.
    label_noise : float, optional
        Probability a label is replaced by a random class, in [0, 1].
    drift_at : float or None, optional
        Fraction of the stream (0-1) at which the concept switches; ``None``
        disables drift. Default 0.5.
    missing_rate : float, optional
        Fraction of feature entries set to NaN, in [0, 1). Default 0.
    seed : int, optional
        Random seed (reproducible output).

    Returns
    -------
    X : numpy.ndarray, shape (n_samples, n_features)
    y : numpy.ndarray of int64, shape (n_samples,)

    Raises
    ------
    ValueError
        For out-of-range arguments.
    """
    check_positive_int(n_samples, "n_samples")
    check_positive_int(n_features, "n_features", minimum=4)
    check_positive_int(n_classes, "n_classes", minimum=2)
    if not 0.0 <= label_noise <= 1.0:
        raise ValueError(f"label_noise must be in [0, 1], got {label_noise}")
    if not 0.0 <= missing_rate < 1.0:
        raise ValueError(f"missing_rate must be in [0, 1), got {missing_rate}")
    if drift_at is not None and not 0.0 < drift_at < 1.0:
        raise ValueError(f"drift_at must be in (0, 1) or None, got {drift_at}")
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    cut = n_samples if drift_at is None else int(drift_at * n_samples)
    y = np.empty(n_samples, dtype=np.int64)
    for lo, hi, (a, b) in ((0, cut, (0, 1)), (cut, n_samples, (2, 3))):
        if hi > lo:
            y[lo:hi] = _bin_codes(X[lo:hi, a] + 0.6 * X[lo:hi, b], n_classes)
    flip = rng.random(n_samples) < label_noise
    y[flip] = rng.integers(0, n_classes, size=int(flip.sum()))
    if missing_rate > 0:
        X[rng.random(X.shape) < missing_rate] = np.nan
    return X, y
