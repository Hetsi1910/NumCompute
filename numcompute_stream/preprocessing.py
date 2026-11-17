"""Streaming preprocessing: scalers, imputer and one-hot encoder with ``partial_fit``.

Each class extends its batch counterpart in :mod:`numcompute.preprocessing`
(so ``fit``/``transform`` keep working) and adds ``partial_fit(X_chunk)``,
which updates the learned statistics incrementally in O(chunk) time and
O(n_features) memory.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence

import numpy as np

from numcompute import preprocessing as _nc

from .stats import RunningStats, StreamStats
from .utils import check_2d

__all__ = ["StandardScaler", "MinMaxScaler", "Imputer", "OneHotEncoder"]

_EPS = np.finfo(float).eps


def _prepare_chunk(X) -> np.ndarray:
    X = check_2d(X, "X_chunk")
    if np.isinf(X).any():
        raise ValueError("X_chunk contains infinity; clip or remove infinite values first")
    return X


def _check_width(est, X: np.ndarray) -> None:
    n = getattr(est, "n_features_in_", None)
    if n is not None and X.shape[1] != n:
        raise ValueError(f"X_chunk has {X.shape[1]} features, but {type(est).__name__} saw {n} earlier")


class StandardScaler(_nc.StandardScaler):
    """Z-score scaler whose statistics can be updated chunk by chunk.

    Parameters
    ----------
    with_mean, with_std : bool, optional
        As in :class:`numcompute.preprocessing.StandardScaler`.
    momentum : float or None, optional
        ``None`` (default) -> exact running mean/variance over *all* chunks
        (Welford/Chan). A float in (0, 1] -> exponential moving average where
        each new chunk gets weight ``momentum`` (adapts to drift, forgets old data).

    Attributes
    ----------
    mean_, var_, scale_ : numpy.ndarray, shape (n_features,)
        Current statistics (constant / all-NaN columns get ``scale_ = 1``).
    n_samples_seen_ : numpy.ndarray, shape (n_features,)
        Non-NaN count per feature (exact mode).

    Notes
    -----
    NaN entries are ignored and stay NaN in the output. In EMA mode the new
    variance is the mixture ``(1-a) var + a var_b + a(1-a)(mean_b - mean)^2``.
    ``fit`` discards previous state and recomputes from the given data.
    """

    def __init__(self, with_mean: bool = True, with_std: bool = True, momentum: Optional[float] = None):
        super().__init__(with_mean=with_mean, with_std=with_std)
        if momentum is not None and not (isinstance(momentum, (int, float, np.floating)) and 0.0 < momentum <= 1.0):
            raise ValueError(f"momentum must be None or a float in (0, 1], got {momentum!r}")
        self.momentum = momentum
        self._rs: Optional[RunningStats] = None

    def fit(self, X, y=None) -> "StandardScaler":
        """Reset and fit on ``X`` (equivalent to one ``partial_fit`` on a fresh scaler)."""
        self._rs = None
        for attr in ("mean_", "var_", "scale_", "n_features_in_"):
            self.__dict__.pop(attr, None)
        return self.partial_fit(X)

    def partial_fit(self, X, y=None) -> "StandardScaler":
        """Update the running mean/variance with a chunk.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            New chunk (NaN allowed; an empty chunk is a no-op once fitted).
        y : ignored

        Returns
        -------
        StandardScaler
            ``self``.

        Raises
        ------
        ValueError
            If the feature count changes, ``X`` is not 2-D or has infinity.
        """
        X = _prepare_chunk(X)
        _check_width(self, X)
        if X.shape[0] == 0 and not hasattr(self, "n_features_in_"):
            raise ValueError("X_chunk must not be empty for the first partial_fit")
        if self._rs is None:
            self._rs = RunningStats()
            self._ema_ready = False
        self.n_features_in_ = X.shape[1]
        if X.shape[0] == 0:
            return self
        if self.momentum is None:
            self._rs.update(X)
            mean, var = np.nan_to_num(self._rs.mean, nan=0.0), np.nan_to_num(self._rs.var(), nan=0.0)
            scale = np.nan_to_num(self._rs.std(), nan=0.0)  # from the rescaled variance: no underflow
            self.n_samples_seen_ = self._rs.count
        else:
            batch = RunningStats().update(X)
            mb, vb = batch.mean, batch.var()
            has = batch.count > 0
            if not self._ema_ready:
                mean, var = np.where(has, mb, 0.0), np.where(has, vb, 0.0)
                self._ema_ready = True
            else:
                a = self.momentum
                m0, v0 = self.mean_, self.var_
                mean = np.where(has, m0 + a * (np.nan_to_num(mb) - m0), m0)
                var = np.where(
                    has, (1 - a) * v0 + a * np.nan_to_num(vb) + a * (1 - a) * (np.nan_to_num(mb) - m0) ** 2, v0
                )
            self.n_samples_seen_ = batch.count
            scale = np.sqrt(var)
        # A column is "constant" only if its spread is round-off relative to its own magnitude. An absolute
        # threshold would wrongly flatten genuine data measured in tiny units (e.g. values around 1e-20).
        scale = np.where(scale <= 10 * _EPS * np.abs(mean), 1.0, scale)
        self.mean_, self.var_, self.scale_ = mean, var, scale
        return self


class MinMaxScaler(_nc.MinMaxScaler):
    """Min-max scaler with ``partial_fit`` (running per-feature min and max).

    Parameters
    ----------
    feature_range, clip
        As in :class:`numcompute.preprocessing.MinMaxScaler`.

    Notes
    -----
    Transformed values of earlier chunks used the ranges known at that time;
    only the statistics, not past output, are updated. NaN is ignored.
    """

    def _check_range(self) -> None:
        try:
            low, high = (float(v) for v in self.feature_range)
        except (TypeError, ValueError) as exc:
            raise ValueError("feature_range must be a (low, high) pair of numbers") from exc
        if not (np.isfinite(low) and np.isfinite(high)):
            raise ValueError(f"feature_range must be finite, got {self.feature_range!r}")

    def fit(self, X, y=None) -> "MinMaxScaler":
        """Reset and fit on ``X``."""
        self.__dict__.pop("_running", None)
        return self.partial_fit(X)

    def partial_fit(self, X, y=None) -> "MinMaxScaler":
        """Extend the running min/max with a chunk and refresh the batch attributes.

        Raises
        ------
        ValueError
            If the feature count changes, the input is invalid or ``feature_range``
            is not a finite ``low < high`` pair.
        """
        self._check_range()
        X = _prepare_chunk(X)
        _check_width(self, X)
        if X.shape[0] == 0 and "_running" not in self.__dict__:
            raise ValueError("X_chunk must not be empty for the first partial_fit")
        if X.shape[0] > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                lo, hi = np.nanmin(X, axis=0), np.nanmax(X, axis=0)
            if "_running" in self.__dict__:
                lo, hi = np.fmin(self._running[0], lo), np.fmax(self._running[1], hi)
            self._running = (lo, hi)
        lo, hi = self._running
        # delegate attribute derivation (scale_, min_, constant/all-NaN handling) to the batch fit
        super().fit(np.vstack([np.where(np.isnan(lo), np.nan, lo), np.where(np.isnan(hi), np.nan, hi)]))
        self.n_features_in_ = X.shape[1]
        return self


class Imputer(_nc.Imputer):
    """Missing-value imputer whose estimates are updated on the fly.

    Parameters
    ----------
    strategy : {'mean', 'median', 'most_frequent', 'constant'}, optional
        ``mean`` is exact (running mean). ``median`` is estimated from a
        reservoir sample (exact while the stream fits in ``reservoir_size``).
        ``most_frequent`` keeps exact per-column value counts.
    fill_value, missing_values
        As in :class:`numcompute.preprocessing.Imputer`.
    reservoir_size : int, optional
        Rows kept for the streaming median. Default 5000.

    Attributes
    ----------
    statistics_ : numpy.ndarray, shape (n_features,)
        Current replacement values.

    Notes
    -----
    Columns never observed yet fall back to ``fill_value``. Time per chunk is
    O(chunk * d); the ``most_frequent`` update loops over features only.
    """

    def __init__(
        self,
        strategy: str = "mean",
        fill_value: float = 0.0,
        missing_values: float = np.nan,
        reservoir_size: int = 5000,
    ):
        super().__init__(strategy=strategy, fill_value=fill_value, missing_values=missing_values)
        self.reservoir_size = reservoir_size
        self._reset_state()

    def _reset_state(self) -> None:
        self._rs: Optional[RunningStats] = None
        self._sketch: Optional[StreamStats] = None
        self._counts: List[Dict[float, int]] = []

    def fit(self, X, y=None) -> "Imputer":
        """Reset and fit on ``X``."""
        self._reset_state()
        for attr in ("statistics_", "n_features_in_"):
            self.__dict__.pop(attr, None)
        return self.partial_fit(X)

    def partial_fit(self, X, y=None) -> "Imputer":
        """Update the replacement values with a chunk.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Chunk containing missing entries (marked by ``missing_values``).

        Returns
        -------
        Imputer
            ``self``.

        Raises
        ------
        ValueError
            For an unknown strategy, a changed feature count or invalid input.
        """
        if self.strategy not in self._STRATEGIES:
            raise ValueError(f"strategy must be one of {self._STRATEGIES}, got {self.strategy!r}")
        if not np.isfinite(self.fill_value):
            raise ValueError(f"fill_value must be finite, got {self.fill_value!r}")
        X = check_2d(X, "X_chunk")
        _check_width(self, X)
        first = not hasattr(self, "n_features_in_")
        if first and X.shape[0] == 0:
            raise ValueError("X_chunk must not be empty for the first partial_fit")
        d = X.shape[1]
        fill = float(self.fill_value)
        if first:
            self.n_features_in_ = d
            self._rs = RunningStats()
            self._sketch = StreamStats(reservoir_size=self.reservoir_size, n_bins=1)
            self._counts = [dict() for _ in range(d)]
            self.statistics_ = np.full(d, fill)
        if X.shape[0] == 0 or self.strategy == "constant":
            return self
        mask = self._mask(X)
        if np.isinf(np.where(mask, 0.0, X)).any():
            raise ValueError("X_chunk contains infinity; clip or remove infinite values first")
        Xm = np.where(mask, np.nan, X)
        if self.strategy == "mean":
            self._rs.update(Xm)
            self.statistics_ = np.where(self._rs.count > 0, np.nan_to_num(self._rs.mean), fill)
        elif self.strategy == "median":
            self._sketch.update_stats(Xm)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(self._sketch._res[: self._sketch._res_n], axis=0)
            self.statistics_ = np.where(np.isnan(med), fill, med)
        else:  # most_frequent: exact counts, smallest value wins ties
            stats = np.full(d, fill)
            for j in range(d):  # loop over features only
                col = Xm[:, j]
                col = col[~np.isnan(col)]
                if col.size:
                    vals, cnt = np.unique(col, return_counts=True)
                    table = self._counts[j]
                    for v, c in zip(vals.tolist(), cnt.tolist()):
                        table[v] = table.get(v, 0) + c
                if self._counts[j]:
                    best = max(self._counts[j].items(), key=lambda kv: (kv[1], -kv[0]))
                    stats[j] = best[0]
            self.statistics_ = stats
        return self


class OneHotEncoder(_nc.OneHotEncoder):
    """One-hot encoder whose category sets expand as new chunks arrive.

    Parameters
    ----------
    handle_unknown : {'error', 'ignore'}, optional
        As in the batch encoder. Default ``'ignore'`` here, because a stream
        may show a category before ``partial_fit`` has seen it.
    dtype : type, optional
        Output dtype.
    categories : list of array_like or None, optional
        Pre-declared categories per column. Fixing them keeps the output width
        constant, which is required when a downstream model (e.g. the buffered
        tree) is trained on the encoded stream.

    Attributes
    ----------
    categories_ : list of numpy.ndarray
    n_output_features_ : int

    Notes
    -----
    Growing the categories changes ``n_output_features_`` and shifts the
    column layout (blocks are ordered by column, then by sorted category), so
    declare ``categories`` up front if the encoder feeds a model.
    """

    def __init__(self, handle_unknown: str = "ignore", dtype=float, categories: Optional[Sequence] = None):
        super().__init__(handle_unknown=handle_unknown, dtype=dtype)
        self.categories = categories

    def fit(self, X, y=None) -> "OneHotEncoder":
        """Reset and fit on ``X``."""
        for attr in ("categories_", "n_features_in_", "n_output_features_"):
            self.__dict__.pop(attr, None)
        return self.partial_fit(X)

    def partial_fit(self, X, y=None) -> "OneHotEncoder":
        """Add the categories found in a chunk.

        Parameters
        ----------
        X : array_like, shape (n_samples,) or (n_samples, n_features)
            Categorical chunk (strings, ints or NaN-free floats).

        Returns
        -------
        OneHotEncoder
            ``self``.

        Raises
        ------
        ValueError
            For NaN, an invalid ``handle_unknown`` or a changed column count.
        """
        if self.handle_unknown not in ("error", "ignore"):
            raise ValueError(f"handle_unknown must be 'error' or 'ignore', got {self.handle_unknown!r}")
        X = self._as_2d(X)
        d = X.shape[1]
        if hasattr(self, "n_features_in_") and d != self.n_features_in_:
            raise ValueError(f"X_chunk has {d} columns, but OneHotEncoder saw {self.n_features_in_} earlier")
        if not hasattr(self, "categories_"):
            if self.categories is not None:
                if len(self.categories) != d:
                    raise ValueError(f"categories has {len(self.categories)} entries for {d} columns")
                self.categories_ = [np.unique(np.asarray(c)) for c in self.categories]
            else:
                if X.shape[0] == 0:
                    raise ValueError("X_chunk must not be empty for the first partial_fit")
                self.categories_ = [np.unique(X[:, j]) for j in range(d)]
            self.n_features_in_ = d
        elif self.categories is None and X.shape[0]:
            self.categories_ = [np.union1d(c, X[:, j]) for j, c in enumerate(self.categories_)]
        self.n_output_features_ = int(sum(len(c) for c in self.categories_))
        return self
