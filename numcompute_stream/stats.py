"""Chunk-based (streaming) statistics.

:class:`StreamStats` consumes data in chunks through :meth:`~StreamStats.update_stats`
and keeps O(n_features + reservoir + bins) state regardless of stream length:

* mean / variance / std  - Welford + Chan merge (``numcompute.stats.RunningStats``),
* min / max / count      - running reductions,
* quantiles              - uniform row *reservoir sample* (exact while the
  stream is no longer than the reservoir capacity, an unbiased approximation after),
* histogram              - fixed-edge counts, optionally over a sliding window of chunks.
"""

from __future__ import annotations

import warnings
from collections import deque
from typing import Deque, Dict, Optional, Tuple

import numpy as np

from numcompute.stats import RunningStats as _BatchRunningStats, quantile as _batch_quantile

from .utils import as_float_array, check_positive_int

__all__ = ["StreamStats", "RunningStats", "update_stats"]


class RunningStats(_BatchRunningStats):
    """Welford/Chan running mean and variance, stabilised against offsets and tiny scales.

    Extends :class:`numcompute.stats.RunningStats`. Every feature is internally centred on its first observed
    value (``shift``) and divided by the size of its first non-zero spread (``unit``) before the usual Welford
    update, i.e. the "shifted data" algorithm. This matters in two situations where the plain update loses
    accuracy:

    * **large offsets** - for data like ``1e9 + noise`` the running mean needs ~1e-7 absolute resolution
      that a float64 near 1e9 cannot provide; centring first keeps all intermediate values small.
    * **tiny scales** - squaring deviations of order ``1e-200`` underflows to 0, so the spread would vanish.

    The public API (``update``, ``merge``, ``count``, ``mean``, ``var``, ``std``) is unchanged.

    Notes
    -----
    ``var`` is returned in the original units and can itself underflow for spreads below ~1e-154; ``std`` is
    computed from the rescaled variance and stays accurate. Time/space are those of the parent class.

    Examples
    --------
    >>> rs = RunningStats().update([[1e9 + 1.0], [1e9 + 2.0], [1e9 + 3.0]])
    >>> float(rs.mean[0]), round(float(rs.var()[0]), 6)
    (1000000002.0, 0.666667)
    """

    def _init(self, n_features: int) -> None:
        super()._init(n_features)
        self._shift = np.zeros(n_features)
        self._shift_set = np.zeros(n_features, dtype=bool)
        self._unit = np.ones(n_features)
        self._unit_set = np.zeros(n_features, dtype=bool)

    def update(self, x) -> "RunningStats":
        """Add a batch (NaN ignored). See :meth:`numcompute.stats.RunningStats.update`."""
        arr = as_float_array(x, name="x", ndim=(1, 2))
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if self.n_features_ is None:
            self._init(arr.shape[1])
        elif arr.shape[1] != self.n_features_:
            raise ValueError(f"x has {arr.shape[1]} features, but earlier batches had {self.n_features_}")
        if arr.shape[0] == 0:
            return self
        valid = ~np.isnan(arr)
        fresh = ~self._shift_set & valid.any(axis=0)
        if fresh.any():
            first = arr[np.argmax(valid, axis=0), np.arange(arr.shape[1])]
            self._shift = np.where(fresh, first, self._shift)
            self._shift_set |= fresh
        centred = arr - self._shift
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            spread = np.nanmax(np.abs(centred), axis=0)
        # a feature that has only been constant so far has zero state, so its unit may still be chosen
        adopt = ~self._unit_set & np.isfinite(spread) & (spread > 0)
        self._unit = np.where(adopt, spread, self._unit)
        self._unit_set |= adopt
        super().update(centred / self._unit)
        return self

    def merge(self, other: "RunningStats") -> "RunningStats":
        """Combine another stabilised ``RunningStats`` into this one (frames are aligned first)."""
        if not isinstance(other, RunningStats):
            raise TypeError("other must be a numcompute_stream.stats.RunningStats")
        if other.n_features_ is None:
            return self
        if self.n_features_ is None:
            self._init(other.n_features_)
        elif self.n_features_ != other.n_features_:
            raise ValueError(f"feature counts differ: {self.n_features_} and {other.n_features_}")
        take = ~self._shift_set & other._shift_set
        self._shift = np.where(take, other._shift, self._shift)
        self._shift_set |= take
        adopt = ~self._unit_set & other._unit_set
        self._unit = np.where(adopt, other._unit, self._unit)
        self._unit_set |= adopt
        mean_b = (other._shift - self._shift + other._unit * other._mean) / self._unit
        m2_b = other._m2 * (other._unit / self._unit) ** 2
        self._merge(other._n, mean_b, m2_b)
        return self

    @property
    def mean(self) -> np.ndarray:
        """Running mean per feature in the original units (NaN where nothing was observed)."""
        return self._shift + self._unit * super().mean

    def var(self, ddof: int = 0) -> np.ndarray:
        """Running variance per feature in the original units."""
        return super().var(ddof) * self._unit**2

    def std(self, ddof: int = 0) -> np.ndarray:
        """Running standard deviation, computed from the rescaled variance so it cannot underflow."""
        return np.sqrt(super().var(ddof)) * self._unit


class StreamStats:
    """Incremental per-feature statistics over a stream of chunks.

    Parameters
    ----------
    reservoir_size : int, optional
        Rows kept for quantile estimation. Default 5000.
    n_bins : int, optional
        Histogram bins per feature. Default 20.
    hist_range : (float, float) or None, optional
        Shared histogram range. ``None`` -> per-feature range of the first
        chunk. Values outside the range are clipped into the edge bins.
    window : int or None, optional
        If set, the histogram only covers the last ``window`` chunks
        (sliding window). ``None`` -> cumulative. Default ``None``.
    seed : int or None, optional
        Seed of the reservoir sampler (reproducibility).

    Attributes
    ----------
    n_features_ : int or None
    n_chunks_ : int
        Number of non-empty chunks processed.

    Raises
    ------
    ValueError
        For invalid sizes or an invalid ``hist_range``.

    Notes
    -----
    NaN entries are ignored by every statistic. State is O(d * (bins +
    1) + reservoir_size * d); each update costs O(chunk_rows * d).

    Examples
    --------
    >>> s = StreamStats(reservoir_size=100).update_stats([[1.0], [2.0]]).update_stats([[3.0]])
    >>> float(s.mean[0]), float(s.quantile(0.5)[0])
    (2.0, 2.0)
    """

    def __init__(
        self,
        reservoir_size: int = 5000,
        n_bins: int = 20,
        hist_range: Optional[Tuple[float, float]] = None,
        window: Optional[int] = None,
        seed: Optional[int] = 0,
    ):
        self.reservoir_size = check_positive_int(reservoir_size, "reservoir_size")
        self.n_bins = check_positive_int(n_bins, "n_bins")
        if hist_range is not None:
            lo, hi = float(hist_range[0]), float(hist_range[1])
            if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
                raise ValueError(f"hist_range must be finite with low < high, got {hist_range!r}")
            hist_range = (lo, hi)
        self.hist_range = hist_range
        self.window = None if window is None else check_positive_int(window, "window")
        self.seed = seed
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self) -> "StreamStats":
        """Forget everything seen so far."""
        self._rs = RunningStats()
        self.n_features_: Optional[int] = None
        self.n_chunks_ = 0
        self._min = np.zeros(0)
        self._max = np.zeros(0)
        self._res = np.zeros((0, 0))
        self._res_n = 0
        self._seen_rows = 0
        self._rng = np.random.default_rng(self.seed)
        self._edges = np.zeros((0, 0))
        self._hist_total = np.zeros((0, 0))
        self._hist_win: Deque[np.ndarray] = deque(maxlen=self.window)
        return self

    def _init(self, X: np.ndarray) -> None:
        d = X.shape[1]
        self.n_features_ = d
        self._min = np.full(d, np.inf)
        self._max = np.full(d, -np.inf)
        self._res = np.full((self.reservoir_size, d), np.nan)
        if self.hist_range is not None:
            lo = np.full(d, self.hist_range[0])
            hi = np.full(d, self.hist_range[1])
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                lo, hi = np.nanmin(X, axis=0), np.nanmax(X, axis=0)
            lo = np.where(np.isnan(lo), 0.0, lo)
            hi = np.where(np.isnan(hi), 1.0, hi)
            flat = hi <= lo
            lo, hi = np.where(flat, lo - 0.5, lo), np.where(flat, hi + 0.5, hi)
        self._edges = np.linspace(lo, hi, self.n_bins + 1, axis=1)  # (d, bins+1)
        self._hist_total = np.zeros((d, self.n_bins))

    # ------------------------------------------------------------------ #
    def update_stats(self, X_chunk) -> "StreamStats":
        """Absorb a chunk of data.

        Parameters
        ----------
        X_chunk : array_like, shape (n_rows, n_features) or (n_rows,)
            New observations; a 1-D input is one feature. NaN is ignored.

        Returns
        -------
        StreamStats
            ``self`` (calls can be chained).

        Raises
        ------
        TypeError
            If the data is not numeric.
        ValueError
            If it is not 1-D/2-D, contains infinity, or its feature count
            differs from earlier chunks.
        """
        X = as_float_array(X_chunk, "X_chunk", ndim=(1, 2))
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if np.isinf(X).any():
            raise ValueError("X_chunk contains infinity; clip or remove infinite values first")
        if self.n_features_ is not None and X.shape[1] != self.n_features_:
            raise ValueError(f"X_chunk has {X.shape[1]} features, but earlier chunks had {self.n_features_}")
        if X.shape[0] == 0:
            return self
        if self.n_features_ is None:
            self._init(X)
        self.n_chunks_ += 1
        self._rs.update(X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self._min = np.fmin(self._min, np.nanmin(X, axis=0))
            self._max = np.fmax(self._max, np.nanmax(X, axis=0))
        self._reservoir_add(X)
        self._hist_add(X)
        return self

    update = update_stats  # short alias used by the other streaming modules

    def _reservoir_add(self, rows: np.ndarray) -> None:
        cap = self.reservoir_size
        space = cap - self._res_n
        if space > 0:
            take = rows[:space]
            start, stop = self._res_n, self._res_n + take.shape[0]
            self._res[start:stop] = take
            self._res_n += take.shape[0]
            self._seen_rows += take.shape[0]
            rows = rows[space:]
        if rows.shape[0]:
            # Algorithm R, vectorised: row t (0-based) replaces slot j ~ U[0, t] if j < cap.
            t = self._seen_rows + np.arange(rows.shape[0])
            j = self._rng.integers(0, t + 1)
            keep = j < cap
            self._res[j[keep]] = rows[keep]  # later rows win duplicate slots, as in the sequential algorithm
            self._seen_rows += rows.shape[0]

    def _hist_add(self, X: np.ndarray) -> None:
        d, b = self.n_features_, self.n_bins
        lo, hi = self._edges[:, 0], self._edges[:, -1]
        valid = ~np.isnan(X)
        pos = np.floor((np.where(valid, X, lo) - lo) / (hi - lo) * b)
        bins = np.clip(pos, 0, b - 1).astype(np.int64)
        flat = (np.arange(d) * b + bins)[valid]
        counts = np.bincount(flat, minlength=d * b).reshape(d, b).astype(float)
        self._hist_total += counts
        if self.window is not None:
            self._hist_win.append(counts)

    # ------------------------------------------------------------------ #
    @property
    def count(self) -> np.ndarray:
        """Non-NaN observations per feature."""
        return self._rs.count

    @property
    def mean(self) -> np.ndarray:
        """Running mean per feature (NaN where nothing was observed)."""
        return self._rs.mean

    def var(self, ddof: int = 0) -> np.ndarray:
        """Running variance per feature (see ``RunningStats.var``)."""
        return self._rs.var(ddof)

    def std(self, ddof: int = 0) -> np.ndarray:
        """Running standard deviation per feature."""
        return self._rs.std(ddof)

    @property
    def min(self) -> np.ndarray:
        """Running minimum per feature (NaN if nothing observed)."""
        return np.where(np.isinf(self._min), np.nan, self._min)

    @property
    def max(self) -> np.ndarray:
        """Running maximum per feature (NaN if nothing observed)."""
        return np.where(np.isinf(self._max), np.nan, self._max)

    @property
    def is_exact_quantile(self) -> bool:
        """True while the reservoir still holds every observed row."""
        return self._seen_rows <= self.reservoir_size

    def quantile(self, q) -> np.ndarray:
        """Per-feature quantile estimate from the reservoir.

        Parameters
        ----------
        q : float or array_like in [0, 1]

        Returns
        -------
        numpy.ndarray, shape (n_features,) for scalar ``q`` else (len(q), n_features)
            Exact while ``is_exact_quantile``; otherwise estimated from a
            uniform sample of the stream.

        Raises
        ------
        ValueError
            If no data was seen or ``q`` is outside [0, 1].
        """
        if self.n_features_ is None or self._res_n == 0:
            raise ValueError("no data seen yet; call update_stats first")
        return np.asarray(_batch_quantile(self._res[: self._res_n], q, axis=0, nan_policy="omit"))

    def histogram(self) -> Tuple[np.ndarray, np.ndarray]:
        """Histogram counts and edges.

        Returns
        -------
        counts : numpy.ndarray, shape (n_features, n_bins)
            Cumulative, or over the last ``window`` chunks if ``window`` is set.
        edges : numpy.ndarray, shape (n_features, n_bins + 1)

        Raises
        ------
        ValueError
            If no data was seen yet.
        """
        if self.n_features_ is None:
            raise ValueError("no data seen yet; call update_stats first")
        if self.window is not None:
            counts = (
                np.sum(np.stack(list(self._hist_win)), axis=0) if self._hist_win else np.zeros_like(self._hist_total)
            )
        else:
            counts = self._hist_total.copy()
        return counts, self._edges.copy()

    def summary(self) -> Dict[str, np.ndarray]:
        """Dict with ``count, mean, std, min, q25, median, q75, max`` per feature."""
        q = self.quantile([0.25, 0.5, 0.75])
        return {
            "count": self.count,
            "mean": self.mean,
            "std": self.std(),
            "min": self.min,
            "q25": q[0],
            "median": q[1],
            "q75": q[2],
            "max": self.max,
        }


def update_stats(stats: StreamStats, X_chunk) -> StreamStats:
    """Functional form of :meth:`StreamStats.update_stats`."""
    return stats.update_stats(X_chunk)
