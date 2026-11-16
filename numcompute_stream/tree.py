"""Decision tree classifier with buffered retraining for streaming data.

The tree is grown from scratch with NumPy only. Split search is vectorised:
for every candidate feature the samples are sorted once and *all* thresholds
are scored at the same time from cumulative class counts. A deliberately
naive loop-based search (:func:`best_split_loop`) is kept for benchmarking
and for cross-checking the vectorised implementation.

Streaming
---------
``partial_fit`` appends each chunk to an internal buffer. Once the buffer
holds at least ``buffer_size`` samples the tree is retrained on the buffered
data and the buffer is cleared; between retrains ``predict`` uses the most
recently trained tree.
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from numcompute.pipeline import Estimator

from .utils import NotFittedError, Timer, check_positive_int, check_X_y

__all__ = ["DecisionTreeClassifier", "impurity", "best_split_vectorised", "best_split_loop"]

_TIE_TOL = 1e-12  # impurity differences below this are treated as ties
_CRITERIA = ("gini", "entropy")


# --------------------------------------------------------------------------- #
# impurity + split search
# --------------------------------------------------------------------------- #
def impurity(counts, criterion: str = "gini") -> np.ndarray:
    """Impurity of class-count vectors.

    Parameters
    ----------
    counts : array_like, shape (..., n_classes)
        Non-negative class counts along the last axis.
    criterion : {'gini', 'entropy'}, optional
        ``gini = 1 - sum(p^2)``; ``entropy = -sum(p log2 p)`` (``0 log 0 = 0``).

    Returns
    -------
    numpy.ndarray, shape (...)
        Impurity per vector; 0 for an all-zero count vector.

    Raises
    ------
    ValueError
        If ``criterion`` is unknown.

    Notes
    -----
    Fully vectorised, O(size) time and space.
    """
    if criterion not in _CRITERIA:
        raise ValueError(f"criterion must be one of {_CRITERIA}, got {criterion!r}")
    c = np.asarray(counts, dtype=float)
    total = c.sum(axis=-1, keepdims=True)
    p = np.divide(c, total, out=np.zeros_like(c), where=total > 0)
    if criterion == "gini":
        # an empty node (total == 0) has no impurity; 1 - 0 would wrongly give 1
        return np.where(total[..., 0] > 0, 1.0 - np.sum(p * p, axis=-1), 0.0)
    safe = np.where(p > 0, p, 1.0)  # log2(1) = 0 -> 0*log 0 contributes 0
    return -np.sum(p * np.log2(safe), axis=-1)


def _threshold_between(lo: float, hi: float) -> float:
    """Midpoint of two consecutive distinct values, never equal to ``hi``."""
    thr = lo + (hi - lo) / 2.0
    return lo if thr >= hi else thr


def best_split_vectorised(
    X: np.ndarray,
    y_onehot: np.ndarray,
    idx: np.ndarray,
    features: Sequence[int],
    criterion: str,
    min_samples_leaf: int,
) -> Optional[Tuple[int, float, float]]:
    """Find the best axis-aligned split of the samples ``idx`` (vectorised).

    Parameters
    ----------
    X : numpy.ndarray, shape (n_samples, n_features)
        Training matrix (no NaN).
    y_onehot : numpy.ndarray, shape (n_samples, n_classes)
        One-hot encoded labels (float).
    idx : numpy.ndarray of int, shape (m,)
        Rows belonging to the node.
    features : sequence of int
        Candidate features, in ascending order.
    criterion : {'gini', 'entropy'}
    min_samples_leaf : int
        Minimum samples allowed in each child.

    Returns
    -------
    (feature, threshold, weighted_child_impurity) or None
        ``None`` if no valid split exists (constant features, leaf limits).

    Notes
    -----
    Tie resolution is deterministic: among equal-impurity splits the one with
    the lowest feature index, then the lowest threshold, wins. For each
    feature the cost is one sort plus a cumulative sum, O(m log m + m k) with
    ``k`` classes; there is **no** Python loop over thresholds.
    """
    m = idx.size
    if m < 2:
        return None
    Yn = y_onehot[idx]
    total = Yn.sum(axis=0)
    n_left = np.arange(1, m, dtype=float)
    n_right = m - n_left
    size_ok = (n_left >= min_samples_leaf) & (n_right >= min_samples_leaf)
    best: Optional[Tuple[int, float, float]] = None
    for j in features:
        col = X[idx, j]
        order = np.argsort(col, kind="stable")
        xs = col[order]
        left = np.cumsum(Yn[order], axis=0)[:-1]  # class counts left of each cut
        right = total - left
        valid = (xs[1:] > xs[:-1]) & size_ok
        if not valid.any():
            continue
        cost = (n_left * impurity(left, criterion) + n_right * impurity(right, criterion)) / m
        cost = np.where(valid, cost, np.inf)
        cmin = cost.min()
        i = int(np.flatnonzero(cost <= cmin + _TIE_TOL)[0])  # lowest threshold among ties
        if best is None or cmin < best[2] - _TIE_TOL:
            best = (int(j), _threshold_between(xs[i], xs[i + 1]), float(cmin))
    return best


def best_split_loop(
    X: np.ndarray,
    y_codes: np.ndarray,
    n_classes: int,
    idx: np.ndarray,
    features: Sequence[int],
    criterion: str,
    min_samples_leaf: int,
) -> Optional[Tuple[int, float, float]]:
    """Reference split search with an explicit Python loop over thresholds.

    Same contract and tie rules as :func:`best_split_vectorised`, but every
    candidate threshold recomputes its class counts from scratch
    (``O(m)`` each), i.e. ``O(m^2)`` per feature. It exists for benchmarking
    (loop vs vectorised) and as an oracle in the unit tests.
    """
    m = idx.size
    if m < 2:
        return None
    yn = y_codes[idx]
    best: Optional[Tuple[int, float, float]] = None
    for j in features:
        col = X[idx, j]
        uniq = np.unique(col)
        if uniq.size < 2:
            continue
        costs: List[float] = []
        thrs: List[float] = []
        for a, b in zip(uniq[:-1], uniq[1:]):  # python loop over thresholds
            thr = _threshold_between(a, b)
            mask = col <= thr
            nl = int(mask.sum())
            nr = m - nl
            if nl < min_samples_leaf or nr < min_samples_leaf:
                continue
            cl = np.bincount(yn[mask], minlength=n_classes)
            cr = np.bincount(yn[~mask], minlength=n_classes)
            costs.append((nl * float(impurity(cl, criterion)) + nr * float(impurity(cr, criterion))) / m)
            thrs.append(thr)
        if not costs:
            continue
        carr = np.asarray(costs)
        cmin = carr.min()
        i = int(np.flatnonzero(carr <= cmin + _TIE_TOL)[0])
        if best is None or cmin < best[2] - _TIE_TOL:
            best = (int(j), thrs[i], float(cmin))
    return best


# --------------------------------------------------------------------------- #
# classifier
# --------------------------------------------------------------------------- #
class DecisionTreeClassifier(Estimator):
    """Depth-limited CART classifier with buffered streaming retraining.

    Parameters
    ----------
    criterion : {'gini', 'entropy'}, optional
        Split quality measure. Default ``'gini'``.
    max_depth : int or None, optional
        Maximum tree depth (root has depth 0). ``None`` grows until leaves are
        pure or too small. Default ``None``.
    min_samples_split : int, optional
        Minimum samples a node needs to be split, ``>= 2``. Default 2.
    min_samples_leaf : int, optional
        Minimum samples in each child, ``>= 1``. Default 1.
    max_features : int, float, {'sqrt', 'log2'} or None, optional
        Features examined per split: an int count, a float fraction in (0, 1],
        ``'sqrt'``/``'log2'`` of the feature count, or ``None`` for all.
        Subsets are drawn with ``random_state``. Default ``None``.
    buffer_size : int, optional
        Streaming: retrain once the buffer holds at least this many samples.
        Default 500.
    min_impurity_decrease : float, optional
        A split is made only if its impurity decrease is at least this value
        (``>= 0``). Default 0.0.
    random_state : int or None, optional
        Seed for the feature subsampling (``max_features``).
    vectorised : bool, optional
        Use the vectorised split search (default) or the slow loop-based
        reference (for benchmarks/tests). Both give identical trees.

    Attributes
    ----------
    classes_ : numpy.ndarray, shape (n_classes,)
        Sorted class labels of the last trained tree.
    n_features_in_ : int
    feature_importances_ : numpy.ndarray, shape (n_features,)
        Normalised total impurity decrease per feature (zeros for a stump).
    n_retrains_ : int
        Number of (re)trainings performed.
    n_samples_seen_ : int
        Samples received through ``partial_fit`` since the last ``fit``.
    retrain_times_ : list of float
        Wall-clock seconds of every retraining.
    tree_ : dict
        Parallel arrays ``feature``, ``threshold``, ``left``, ``right``,
        ``value`` (class counts), ``n_samples``, ``impurity``; ``feature`` is
        -1 on leaves.

    Raises
    ------
    TypeError, ValueError
        For invalid hyper-parameters (checked at construction and at fit).

    Notes
    -----
    * NaN in ``X`` is replaced by the per-feature training median (stored in
      ``nan_fill_``) both at training and prediction time.
    * Ties between equally good splits resolve to the lowest feature index and
      lowest threshold, so results are deterministic.
    * Predictions between retrains use the last trained tree; before the first
      retrain ``predict`` raises :class:`NotFittedError`.

    Examples
    --------
    >>> tree = DecisionTreeClassifier(max_depth=3, buffer_size=4)
    >>> tree = tree.partial_fit([[0.0], [1.0]], [0, 1])      # buffered only
    >>> tree.is_ready
    False
    >>> tree = tree.partial_fit([[0.1], [0.9]], [0, 1])      # retrains now
    >>> tree.predict([[0.0], [1.0]]).tolist()
    [0, 1]
    """

    def __init__(
        self,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Union[int, float, str, None] = None,
        buffer_size: int = 500,
        min_impurity_decrease: float = 0.0,
        random_state: Optional[int] = None,
        vectorised: bool = True,
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.buffer_size = buffer_size
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.vectorised = vectorised
        self._check_params()
        self._reset_stream_state()

    # ------------------------------------------------------------------ #
    # validation / state
    # ------------------------------------------------------------------ #
    def _check_params(self) -> None:
        if self.criterion not in _CRITERIA:
            raise ValueError(f"criterion must be one of {_CRITERIA}, got {self.criterion!r}")
        if self.max_depth is not None:
            check_positive_int(self.max_depth, "max_depth", minimum=0)
        check_positive_int(self.min_samples_split, "min_samples_split", minimum=2)
        check_positive_int(self.min_samples_leaf, "min_samples_leaf", minimum=1)
        check_positive_int(self.buffer_size, "buffer_size", minimum=1)
        if not isinstance(self.min_impurity_decrease, (int, float, np.floating)) or self.min_impurity_decrease < 0:
            raise ValueError(f"min_impurity_decrease must be a number >= 0, got {self.min_impurity_decrease!r}")
        mf = self.max_features
        if mf is not None and not (
            (isinstance(mf, str) and mf in ("sqrt", "log2"))
            or (isinstance(mf, (int, np.integer)) and not isinstance(mf, bool) and mf >= 1)
            or (isinstance(mf, (float, np.floating)) and 0.0 < mf <= 1.0)
        ):
            raise ValueError(f"max_features must be None, 'sqrt', 'log2', an int >= 1 or a float in (0, 1], got {mf!r}")

    def _reset_stream_state(self) -> None:
        self._buf_X: List[np.ndarray] = []
        self._buf_y: List[np.ndarray] = []
        self._buf_n = 0
        self.n_retrains_ = 0
        self.n_samples_seen_ = 0
        self.retrain_times_: List[float] = []

    @property
    def is_ready(self) -> bool:
        """True once a tree has been trained (``predict`` is usable)."""
        return hasattr(self, "tree_")

    @property
    def buffer_fill_(self) -> int:
        """Number of samples waiting in the buffer."""
        return self._buf_n

    def _require_ready(self) -> None:
        if not self.is_ready:
            raise NotFittedError(
                f"This DecisionTreeClassifier has no trained tree yet: the buffer holds "
                f"{self._buf_n}/{self.buffer_size} samples. Call partial_fit with more data, "
                f"flush() to train on the buffer now, or fit(X, y)."
            )

    def _n_candidate_features(self, d: int) -> int:
        mf = self.max_features
        if mf is None:
            return d
        if mf == "sqrt":
            return max(1, int(np.sqrt(d)))
        if mf == "log2":
            return max(1, int(np.log2(d)))
        if isinstance(mf, (float, np.floating)):
            return max(1, int(mf * d))
        return min(int(mf), d)

    # ------------------------------------------------------------------ #
    # training
    # ------------------------------------------------------------------ #
    def fit(self, X, y) -> "DecisionTreeClassifier":
        """Train the tree from scratch on ``(X, y)`` (resets the stream state).

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Training data; NaN allowed (median-imputed).
        y : array_like, shape (n_samples,)
            Class labels (numbers or strings).

        Returns
        -------
        DecisionTreeClassifier
            ``self``.

        Raises
        ------
        ValueError
            For empty data, bad shapes, infinity, NaN labels or invalid parameters.
        TypeError
            If ``X`` is not numeric.

        Notes
        -----
        Time O(d n log n) per tree level; space O(n k) for the one-hot labels.
        """
        self._check_params()
        Xa, ya = check_X_y(X, y)
        if Xa.shape[0] == 0:
            raise ValueError("cannot fit on an empty dataset (0 samples)")
        self._reset_stream_state()
        self.n_samples_seen_ = Xa.shape[0]
        with Timer() as t:
            self._train(Xa, ya)
        self.n_retrains_ = 1
        self.retrain_times_ = [t.elapsed]
        return self

    def partial_fit(self, X, y) -> "DecisionTreeClassifier":
        """Add a chunk to the buffer; retrain when the buffer is full.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Chunk of features. A chunk with zero rows is ignored.
        y : array_like, shape (n_samples,)
            Chunk labels.

        Returns
        -------
        DecisionTreeClassifier
            ``self``.

        Raises
        ------
        ValueError
            If the feature count differs from earlier chunks / the trained
            tree, or the shapes are inconsistent.

        Notes
        -----
        Retraining uses **only** the buffered samples (the buffer is cleared
        afterwards), so the model adapts to recent data. Amortised cost per
        sample is the training cost divided by ``buffer_size``.
        """
        self._check_params()
        Xa, ya = check_X_y(X, y)
        if Xa.shape[0] == 0:
            return self
        known = self._buf_X[0].shape[1] if self._buf_X else (self.n_features_in_ if self.is_ready else None)
        if known is not None and Xa.shape[1] != known:
            raise ValueError(f"chunk has {Xa.shape[1]} features, but earlier data had {known}")
        self._buf_X.append(Xa)
        self._buf_y.append(ya)
        self._buf_n += Xa.shape[0]
        self.n_samples_seen_ += Xa.shape[0]
        if self._buf_n >= self.buffer_size:
            self._retrain_from_buffer()
        return self

    def flush(self) -> "DecisionTreeClassifier":
        """Train immediately on whatever is buffered (no-op if the buffer is empty)."""
        if self._buf_n > 0:
            self._retrain_from_buffer()
        return self

    def _retrain_from_buffer(self) -> None:
        Xb = np.concatenate(self._buf_X, axis=0)
        yb = np.concatenate(self._buf_y, axis=0)
        with Timer() as t:
            self._train(Xb, yb)
        self.n_retrains_ += 1
        self.retrain_times_.append(t.elapsed)
        self._buf_X, self._buf_y, self._buf_n = [], [], 0

    def _train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Grow a tree on validated data and store the fitted attributes."""
        n, d = X.shape
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            fill = np.nanmedian(X, axis=0)
        fill = np.where(np.isnan(fill), 0.0, fill)  # all-NaN column -> 0
        Xf = np.where(np.isnan(X), fill, X)
        classes, codes = np.unique(y, return_inverse=True)
        codes = codes.astype(np.int64).ravel()
        k = classes.size
        y1h = np.zeros((n, k))
        y1h[np.arange(n), codes] = 1.0
        rng = np.random.default_rng(self.random_state)
        m_feat = self._n_candidate_features(d)
        max_depth = np.inf if self.max_depth is None else self.max_depth

        feat: List[int] = []
        thr: List[float] = []
        left: List[int] = []
        right: List[int] = []
        value: List[np.ndarray] = []
        nsamp: List[int] = []
        imp_list: List[float] = []
        importances = np.zeros(d)

        # explicit stack => no recursion limit for max_depth=None
        stack = [(np.arange(n), 0, -1, True)]
        while stack:
            idx, depth, parent, is_left = stack.pop()
            node = len(feat)
            if parent >= 0:
                (left if is_left else right)[parent] = node
            counts = y1h[idx].sum(axis=0)
            node_imp = float(impurity(counts, self.criterion))
            feat.append(-1)
            thr.append(np.nan)
            left.append(-1)
            right.append(-1)
            value.append(counts)
            nsamp.append(idx.size)
            imp_list.append(node_imp)
            if (
                depth >= max_depth
                or idx.size < self.min_samples_split
                or idx.size < 2 * self.min_samples_leaf
                or node_imp <= 1e-15
            ):
                continue
            features = np.arange(d) if m_feat >= d else np.sort(rng.choice(d, size=m_feat, replace=False))
            if self.vectorised:
                split = best_split_vectorised(Xf, y1h, idx, features, self.criterion, self.min_samples_leaf)
            else:
                split = best_split_loop(Xf, codes, k, idx, features, self.criterion, self.min_samples_leaf)
            if split is None:
                continue
            j, t_val, child_cost = split
            decrease = node_imp - child_cost
            if decrease < self.min_impurity_decrease - _TIE_TOL:
                continue
            go_left = Xf[idx, j] <= t_val
            li, ri = idx[go_left], idx[~go_left]
            if li.size == 0 or ri.size == 0:  # pragma: no cover - guarded by valid mask
                continue
            feat[node], thr[node] = j, t_val
            importances[j] += idx.size * decrease
            stack.append((ri, depth + 1, node, False))  # right pushed first -> left built first
            stack.append((li, depth + 1, node, True))

        self.classes_ = classes
        self.n_features_in_ = d
        self.nan_fill_ = fill
        self.tree_ = {
            "feature": np.asarray(feat, dtype=np.int64),
            "threshold": np.asarray(thr, dtype=float),
            "left": np.asarray(left, dtype=np.int64),
            "right": np.asarray(right, dtype=np.int64),
            "value": np.vstack(value),
            "n_samples": np.asarray(nsamp, dtype=np.int64),
            "impurity": np.asarray(imp_list, dtype=float),
        }
        total = importances.sum()
        self.feature_importances_ = importances / total if total > 0 else importances

    # ------------------------------------------------------------------ #
    # inference
    # ------------------------------------------------------------------ #
    def _prepare(self, X) -> np.ndarray:
        self._require_ready()
        Xa, _ = check_X_y(X)
        if Xa.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {Xa.shape[1]} features, but the tree was trained with {self.n_features_in_}")
        return np.where(np.isnan(Xa), self.nan_fill_, Xa)

    def apply(self, X) -> np.ndarray:
        """Return the leaf index reached by every sample.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)

        Returns
        -------
        numpy.ndarray of int, shape (n_samples,)

        Notes
        -----
        Vectorised level-by-level traversal: the Python loop runs once per
        tree level (at most ``max_depth``), not once per sample.
        """
        Xa = self._prepare(X)
        t = self.tree_
        node = np.zeros(Xa.shape[0], dtype=np.int64)
        while True:
            f = t["feature"][node]
            rows = np.flatnonzero(f >= 0)
            if rows.size == 0:
                return node
            nr = node[rows]
            go_left = Xa[rows, f[rows]] <= t["threshold"][nr]
            node[rows] = np.where(go_left, t["left"][nr], t["right"][nr])

    def predict_proba(self, X) -> np.ndarray:
        """Class probabilities (training class frequencies in the reached leaf).

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)

        Returns
        -------
        numpy.ndarray, shape (n_samples, n_classes)
            Columns follow ``classes_``; rows sum to 1.

        Raises
        ------
        NotFittedError
            If no tree has been trained yet.
        ValueError
            If the feature count differs from training.
        """
        leaf = self.apply(X)
        v = self.tree_["value"][leaf]
        return v / v.sum(axis=1, keepdims=True)

    def predict(self, X) -> np.ndarray:
        """Predict class labels (ties between classes go to the lowest label).

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)

        Returns
        -------
        numpy.ndarray, shape (n_samples,)
            Labels with the dtype of the training labels.
        """
        proba = self.predict_proba(X)  # raises NotFittedError first if nothing is trained
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X, y) -> float:
        """Mean accuracy of ``predict(X)`` against ``y``."""
        Xa, ya = check_X_y(X, y)
        if ya.size == 0:
            return float("nan")
        return float(np.mean(self.predict(Xa) == ya))

    # ------------------------------------------------------------------ #
    # introspection
    # ------------------------------------------------------------------ #
    def get_depth(self) -> int:
        """Depth of the trained tree (a single leaf has depth 0)."""
        self._require_ready()
        t = self.tree_
        depth = np.zeros(t["feature"].size, dtype=np.int64)
        for node in range(t["feature"].size):  # preorder: parents precede children
            if t["feature"][node] >= 0:
                depth[t["left"][node]] = depth[t["right"][node]] = depth[node] + 1
        return int(depth.max())

    def get_n_leaves(self) -> int:
        """Number of leaves of the trained tree."""
        self._require_ready()
        return int(np.sum(self.tree_["feature"] < 0))

    def export_text(self, feature_names: Optional[Sequence[str]] = None, max_depth: int = 10) -> str:
        """Human-readable rules of the trained tree.

        Parameters
        ----------
        feature_names : sequence of str, optional
            Names for the features (default ``x[j]``).
        max_depth : int, optional
            Truncate printing below this depth. Default 10.
        """
        self._require_ready()
        t = self.tree_
        if feature_names is not None and len(feature_names) != self.n_features_in_:
            raise ValueError(f"feature_names needs {self.n_features_in_} entries, got {len(feature_names)}")
        names = feature_names or [f"x[{j}]" for j in range(self.n_features_in_)]
        lines: List[str] = []

        def walk(node: int, depth: int) -> None:
            pad = "|   " * depth
            if t["feature"][node] < 0 or depth >= max_depth:
                lab = self.classes_[int(np.argmax(t["value"][node]))]
                lines.append(f"{pad}|--- class: {lab}")
                return
            f, th = names[t["feature"][node]], t["threshold"][node]
            lines.append(f"{pad}|--- {f} <= {th:.4g}")
            walk(int(t["left"][node]), depth + 1)
            lines.append(f"{pad}|--- {f} >  {th:.4g}")
            walk(int(t["right"][node]), depth + 1)

        walk(0, 0)
        return "\n".join(lines)
