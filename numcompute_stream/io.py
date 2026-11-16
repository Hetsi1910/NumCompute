"""Chunked data loading for streaming experiments.

Builds on :mod:`numcompute.io` (CSV parsing, missing-value handling) and adds
the ``X / y`` split plus helpers that turn arrays or CSV files into a stream
of ``(X_chunk, y_chunk)`` pairs.
"""

from __future__ import annotations

from typing import Iterator, List, Optional, Tuple

import numpy as np

from numcompute import io as _nc_io
from numcompute.utils import iter_batches

from .utils import check_labels, check_positive_int

__all__ = [
    "split_xy",
    "load_csv_xy",
    "iter_csv_xy_chunks",
    "iter_array_chunks",
    "read_feature_names",
    "write_csv_xy",
]


def _maybe_int_labels(y: np.ndarray, integer_labels: bool) -> np.ndarray:
    """Cast integral-valued float labels to int64 (leave anything else)."""
    if integer_labels and y.dtype.kind == "f" and y.size and np.all(np.isfinite(y)) and np.all(y == np.round(y)):
        return y.astype(np.int64)
    return y


def split_xy(data, target_col: int = -1, integer_labels: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Split a numeric table into features and a label column.

    Parameters
    ----------
    data : array_like, shape (n_rows, n_cols)
        Table whose column ``target_col`` holds the class label.
    target_col : int, optional
        Index of the label column (negative values count from the end).
        Default -1 (last column).
    integer_labels : bool, optional
        Cast integral-valued labels to ``int64``. Default True.

    Returns
    -------
    X : numpy.ndarray, shape (n_rows, n_cols - 1)
    y : numpy.ndarray, shape (n_rows,)

    Raises
    ------
    ValueError
        If ``data`` is not 2-D with at least 2 columns or ``target_col`` is
        out of range.
    """
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"data must be 2-D with >= 2 columns (features + label), got shape {arr.shape}")
    n_cols = arr.shape[1]
    if not -n_cols <= target_col < n_cols:
        raise ValueError(f"target_col {target_col} is out of range for {n_cols} columns")
    t = target_col % n_cols
    y = arr[:, t]
    X = np.delete(arr, t, axis=1)
    return X, _maybe_int_labels(y, integer_labels)


def load_csv_xy(path, target_col: int = -1, skip_header: int = 1, integer_labels: bool = True, **kwargs):
    """Load a whole numeric CSV and split it into ``X`` and ``y``.

    Parameters
    ----------
    path : str or os.PathLike
        CSV file.
    target_col : int, optional
        Label column. Default -1.
    skip_header : int, optional
        Header lines to skip. Default 1.
    integer_labels : bool, optional
        See :func:`split_xy`.
    **kwargs
        Forwarded to :func:`numcompute.io.load_csv` (``delimiter``, ``missing``...).

    Returns
    -------
    X : numpy.ndarray, shape (n_rows, n_features)
    y : numpy.ndarray, shape (n_rows,)

    Notes
    -----
    Rows whose label is missing (NaN) are dropped. Features keep NaN.
    """
    data = _nc_io.load_csv(path, skip_header=skip_header, **kwargs)
    X, y = split_xy(data, target_col, integer_labels=False)
    keep = ~np.isnan(y)
    return X[keep], _maybe_int_labels(y[keep], integer_labels)


def iter_csv_xy_chunks(
    path,
    chunk_size: int,
    target_col: int = -1,
    skip_header: int = 1,
    integer_labels: bool = True,
    **kwargs,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Stream a numeric CSV as ``(X_chunk, y_chunk)`` pairs.

    Parameters
    ----------
    path : str or os.PathLike
        CSV file.
    chunk_size : int
        Rows per chunk (the last chunk may be smaller), ``>= 1``.
    target_col : int, optional
        Label column. Default -1.
    skip_header : int, optional
        Header lines to skip. Default 1.
    integer_labels : bool, optional
        Cast integral labels to int64. Default True.
    **kwargs
        Forwarded to :func:`numcompute.io.iter_csv_chunks`.

    Yields
    ------
    X_chunk : numpy.ndarray, shape (<= chunk_size, n_features)
    y_chunk : numpy.ndarray, shape (<= chunk_size,)

    Notes
    -----
    Only one chunk is in memory at a time: space O(chunk_size * n_cols).
    Rows with a missing label are dropped; chunks that become empty are skipped.
    """
    check_positive_int(chunk_size, "chunk_size")
    for block in _nc_io.iter_csv_chunks(path, chunksize=chunk_size, skip_header=skip_header, **kwargs):
        X, y = split_xy(block, target_col, integer_labels=False)
        keep = ~np.isnan(y)
        if keep.any():
            yield X[keep], _maybe_int_labels(y[keep], integer_labels)


def iter_array_chunks(
    X, y, chunk_size: int, shuffle: bool = False, seed: Optional[int] = None
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Split in-memory arrays into a stream of ``(X_chunk, y_chunk)``.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
    y : array_like, shape (n_samples,)
    chunk_size : int
        Rows per chunk, ``>= 1``.
    shuffle : bool, optional
        Shuffle rows before chunking. Default False (keeps temporal order,
        which matters for concept-drift experiments).
    seed : int, optional
        Seed for the shuffle.

    Yields
    ------
    X_chunk, y_chunk : numpy.ndarray

    Raises
    ------
    ValueError
        If ``X`` and ``y`` differ in length or ``chunk_size < 1``.
    """
    check_positive_int(chunk_size, "chunk_size")
    y = check_labels(y)
    yield from iter_batches(np.asarray(X), chunk_size, y=y, shuffle=shuffle, seed=seed)


def read_feature_names(path, target_col: int = -1, delimiter: str = ",") -> List[str]:
    """Return the header names of the feature columns (label column removed)."""
    names = list(_nc_io.read_header(path, delimiter=delimiter))
    t = target_col % len(names)
    return [n for i, n in enumerate(names) if i != t]


def write_csv_xy(path, X, y, header: Optional[List[str]] = None, fmt: str = "%.6g") -> None:
    """Write ``X`` and ``y`` to a CSV (label last); NaN is written as ``nan``.

    Parameters
    ----------
    path : str or os.PathLike
        Output file.
    X : array_like, shape (n, d)
    y : array_like, shape (n,)
    header : list of str, optional
        Column names (``d + 1`` entries). Default ``f0..f{d-1},label``.
    fmt : str, optional
        ``numpy.savetxt`` format for numbers.
    """
    Xa = np.asarray(X, dtype=float)
    ya = check_labels(y).astype(float)
    if Xa.shape[0] != ya.shape[0]:
        raise ValueError(f"X and y differ in length: {Xa.shape[0]} vs {ya.shape[0]}")
    if header is None:
        header = [f"f{i}" for i in range(Xa.shape[1])] + ["label"]
    if len(header) != Xa.shape[1] + 1:
        raise ValueError(f"header needs {Xa.shape[1] + 1} names, got {len(header)}")
    np.savetxt(path, np.column_stack([Xa, ya]), delimiter=",", header=",".join(header), comments="", fmt=fmt)
