"""NumCompute-Stream: a decision-tree based streaming ML framework (NumPy + matplotlib).

Extends the group ``numcompute`` package with incremental (``partial_fit``)
preprocessing, statistics, metrics and pipelines, a buffered-retraining
decision tree, a stream trainer and plotting helpers.

Examples
--------
>>> from numcompute_stream.tree import DecisionTreeClassifier
>>> tree = DecisionTreeClassifier(max_depth=5, buffer_size=500)
>>> tree.is_ready
False
"""

from . import (  # noqa: F401
    datasets,
    io,
    metrics,
    pipeline,
    preprocessing,
    stats,
    stream,
    tree,
    utils,
    visualise,
)

__version__ = "0.1.0"

__all__ = [
    "datasets",
    "io",
    "metrics",
    "pipeline",
    "preprocessing",
    "stats",
    "stream",
    "tree",
    "utils",
    "visualise",
]
