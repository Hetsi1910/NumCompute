
# NumCompute-Stream (Assignment 2.2)

> **A decision-tree based streaming ML framework** built on top of NumCompute, using **only NumPy and matplotlib**.
> Individual work, on branch `individual-hetsi` (`main` holds the original group code).

## Layout

```text
numcompute_stream/
├── tree.py           # DecisionTreeClassifier: Gini/entropy, vectorised split search, buffered partial_fit
├── stream.py         # StreamTrainer: prequential evaluation + per-chunk logging (metrics, memory, time)
├── preprocessing.py  # StandardScaler / MinMaxScaler / Imputer / OneHotEncoder with partial_fit
├── stats.py          # StreamStats.update_stats: mean/var (Welford-Chan), min/max, quantiles, histograms (window)
├── metrics.py        # Accuracy, Precision, Recall, F1, ConfusionMatrix, AUC: update/result/reset, rolling window
├── pipeline.py       # Pipeline.partial_fit chaining transformers and the model
├── visualise.py      # plot_metric_over_time, compare_models, plot_predictions_vs_ground_truth (+ extras)
├── io.py             # CSV -> (X, y) chunk streams (reuses numcompute.io)
├── datasets.py       # synthetic stream with missing values and concept drift
└── utils.py          # validation, memory estimation, timer
tests/test_stream_*.py      # 218 unit tests (98% coverage), incl. numerical-robustness and protocol tests
tools/mutation_check.py     # 20 deliberate defects; all must be caught by the tests (results: tools/mutation_results.md)
benchmark/run_stream_benchmarks.py  -> benchmark/stream_results.md (+ figures/)
demo/stream_demo.ipynb      # executed end-to-end demo (loads demo/stream_data.csv)
```

## Install and run

```bash
git clone <your fork> && cd NumCompute && git checkout individual-hetsi
python -m pip install -e ".[dev]"            # numpy + matplotlib (+ pytest, jupyter)
python -m pytest tests -q                    # whole suite (group + streaming): 499 tests
python -m pytest tests/test_stream_*.py -q   # streaming tests only: 218 tests
python tools/mutation_check.py               # mutation check (about 1 minute)
python benchmark/run_stream_benchmarks.py    # loop vs vectorised + buffer_size sweep (about 1 min; --quick for seconds)
jupyter notebook demo/stream_demo.ipynb      # the demo
```

## Usage

```python
from numcompute_stream.io import iter_csv_xy_chunks
from numcompute_stream.preprocessing import Imputer, StandardScaler
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.metrics import Accuracy
from numcompute_stream.stream import StreamTrainer
from numcompute_stream import visualise as vis

pipe = Pipeline([
    ("impute", Imputer()),
    ("scale", StandardScaler()),
    ("model", DecisionTreeClassifier(max_depth=5, buffer_size=500)),
])
trainer = StreamTrainer(pipe, metrics={"rolling_acc": Accuracy(window=5)})
trainer.run(iter_csv_xy_chunks("demo/stream_data.csv", chunk_size=200))   # test-then-train on every chunk

print(trainer.summary())
vis.plot_metric_over_time(trainer.metric_history("cumulative_accuracy"), "Cumulative accuracy", "accuracy",
                          save_path="acc.png")
vis.plot_stream_history(trainer.history, save_path="history.png")
```

Direct use of the tree:

```python
tree = DecisionTreeClassifier(max_depth=5, buffer_size=500)
for X_chunk, y_chunk in stream:
    tree.partial_fit(X_chunk, y_chunk)   # buffered; retrains once >= buffer_size samples, then clears the buffer
tree.predict(X_new)                      # always uses the most recently trained tree
```

## Design decisions in one place

* **Buffered retraining** (spec): `partial_fit` appends the chunk; when `buffer_fill_ >= buffer_size` the tree is retrained on the
  buffered data only, then the buffer is cleared. Before the first retrain `predict` raises `NotFittedError` that states how full the
  buffer is; `flush()` trains on a partial buffer.
* **Vectorised split search**: sort each feature once, cumulative one-hot class counts give the impurity of *every* threshold at once
  (no Python loop over thresholds). A slow loop version (`vectorised=False`) is kept for benchmarks and as a test oracle.
* **Numerical stability**: NaN -> training-median imputation inside the tree; zero-variance and all-NaN columns handled; deterministic
  tie resolution (lowest feature index, then lowest threshold, tolerance 1e-12); `0 log 0 = 0`; Welford/Chan merging for variance.
* **Reuse, not rewrite**: the package extends `numcompute` (`BaseEstimator`, `Pipeline`, scalers, `RunningStats`, `roc_auc_score`,
  CSV reader), so streaming versions are drop-in replacements for the batch classes.
* **Benchmark results**: see `benchmark/stream_results.md`.
