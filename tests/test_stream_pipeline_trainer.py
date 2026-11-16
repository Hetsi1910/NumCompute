"""Unit tests for numcompute_stream.pipeline and numcompute_stream.stream."""

import logging

import numpy as np
import pytest

from numcompute.pipeline import NearestCentroidClassifier
from numcompute_stream.datasets import make_drift_stream
from numcompute_stream.io import iter_array_chunks
from numcompute_stream.metrics import Accuracy
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import Imputer, StandardScaler
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.utils import NotFittedError


def _pipe(buffer_size=300):
    return Pipeline(
        [
            ("imp", Imputer()),
            ("scale", StandardScaler()),
            ("model", DecisionTreeClassifier(max_depth=6, buffer_size=buffer_size)),
        ]
    )


@pytest.fixture
def stream():
    X, y = make_drift_stream(2400, drift_at=None, label_noise=0.0, missing_rate=0.02, seed=5)
    return X, y


# ------------------------------------------------------------------ pipeline
def test_pipeline_partial_fit_updates_all_steps(stream):
    X, y = stream
    p = _pipe()
    for Xc, yc in iter_array_chunks(X[:900], y[:900], 100):
        p.partial_fit(Xc, yc)
    assert p["scale"].n_features_in_ == X.shape[1]
    assert p["model"].n_retrains_ == 3
    assert p.score(X[900:], y[900:]) > 0.6


def test_pipeline_supports_streaming_flag():
    assert _pipe().supports_streaming
    assert not Pipeline([("s", StandardScaler()), ("m", NearestCentroidClassifier())]).supports_streaming


def test_pipeline_partial_fit_names_step_without_partial_fit():
    p = Pipeline([("s", StandardScaler()), ("m", NearestCentroidClassifier())])
    with pytest.raises(TypeError, match="'m'"):
        p.partial_fit(np.ones((4, 2)), [0, 1, 0, 1])


def test_pipeline_predict_before_ready_raises_not_fitted(stream):
    X, y = stream
    p = _pipe(buffer_size=10_000)
    p.partial_fit(X[:100], y[:100])
    with pytest.raises(NotFittedError):
        p.predict(X[:5])


def test_pipeline_flush_and_predict_proba(stream):
    X, y = stream
    p = _pipe(buffer_size=10_000)
    p.partial_fit(X[:200], y[:200]).flush()
    assert p.predict_proba(X[:5]).shape == (5, 3)


def test_pipeline_batch_fit_still_works(stream):
    X, y = stream
    p = _pipe().fit(X[:500], y[:500])
    assert p.predict(X[:10]).shape == (10,)


def test_pipeline_predict_proba_requires_support():
    p = Pipeline([("s", StandardScaler()), ("m", NearestCentroidClassifier())]).fit(np.random.rand(6, 2), [0, 1] * 3)
    with pytest.raises(TypeError, match="predict_proba"):
        p.predict_proba(np.random.rand(2, 2))


def test_pipeline_partial_fit_label_length_mismatch():
    with pytest.raises(ValueError, match="inconsistent"):
        _pipe().partial_fit(np.ones((4, 2)), [0, 1])


# ------------------------------------------------------------------ trainer
def test_trainer_requires_streaming_model():
    with pytest.raises(TypeError, match="partial_fit"):
        StreamTrainer(NearestCentroidClassifier())
    with pytest.raises(TypeError, match="metric"):
        StreamTrainer(DecisionTreeClassifier(), metrics={"bad": object()})


def test_trainer_prequential_logs_one_record_per_chunk(stream):
    X, y = stream
    tr = StreamTrainer(_pipe(), metrics={"rolling": Accuracy(window=3)})
    tr.run(iter_array_chunks(X, y, 200))
    h = tr.history
    assert len(h) == 12 and [r["chunk"] for r in h] == list(range(12))
    for key in (
        "chunk_accuracy",
        "cumulative_accuracy",
        "rolling",
        "fit_time",
        "memory_bytes",
        "buffer_fill",
        "n_retrains",
    ):
        assert key in h[-1]


def test_trainer_unscored_until_first_retrain(stream):
    X, y = stream
    tr = StreamTrainer(_pipe(buffer_size=400))
    tr.run(iter_array_chunks(X[:800], y[:800], 100))
    scored = [r["scored"] for r in tr.history]
    assert scored[:4] == [False] * 4 and all(scored[4:])
    assert np.isnan(tr.history[0]["chunk_accuracy"])


def test_trainer_cumulative_accuracy_matches_pooled(stream):
    X, y = stream
    tr = StreamTrainer(DecisionTreeClassifier(max_depth=6, buffer_size=300))
    preds, truth = [], []
    ref = DecisionTreeClassifier(max_depth=6, buffer_size=300)
    for Xc, yc in iter_array_chunks(X, y, 150):
        tr.process_chunk(Xc, yc)
        if ref.is_ready:
            preds.append(ref.predict(Xc))
            truth.append(yc)
        ref.partial_fit(Xc, yc)
    assert tr.summary()["final_cumulative_accuracy"] == pytest.approx(
        np.mean(np.concatenate(preds) == np.concatenate(truth))
    )


def test_trainer_marks_retrain_events(stream):
    X, y = stream
    tr = StreamTrainer(DecisionTreeClassifier(buffer_size=300))
    tr.run(iter_array_chunks(X[:900], y[:900], 100))
    assert [r["retrained"] for r in tr.history].count(True) == 3
    assert tr.summary()["n_retrains"] == 3


def test_trainer_standalone_fit_and_score_chunk(stream):
    X, y = stream
    tr = StreamTrainer(DecisionTreeClassifier(buffer_size=100))
    rec = tr.fit_chunk(X[:100], y[:100])
    assert rec["retrained"] and rec["n_samples"] == 100
    out = tr.score_chunk(X[100:150], y[100:150])
    assert out["scored"] and 0.0 <= out["chunk_accuracy"] <= 1.0


def test_trainer_train_only_mode_and_max_chunks(stream):
    X, y = stream
    tr = StreamTrainer(DecisionTreeClassifier(buffer_size=200))
    tr.run(iter_array_chunks(X, y, 100), mode="train_only", max_chunks=4)
    assert len(tr.history) == 4 and "chunk_accuracy" not in tr.history[0]
    with pytest.raises(ValueError, match="mode"):
        tr.run([], mode="bogus")


def test_trainer_reset_clears_log_and_metrics(stream):
    X, y = stream
    tr = StreamTrainer(DecisionTreeClassifier(buffer_size=100))
    tr.run(iter_array_chunks(X[:400], y[:400], 100))
    tr.reset()
    assert tr.history == [] and np.isnan(tr.metrics["accuracy"].result())


def test_trainer_logs_via_logging(stream, caplog):
    X, y = stream
    tr = StreamTrainer(DecisionTreeClassifier(buffer_size=100), name="demo")
    with caplog.at_level(logging.INFO, logger="numcompute_stream"):
        tr.process_chunk(X[:100], y[:100])
    assert any("[demo]" in m for m in caplog.messages)


def test_trainer_peak_memory_option(stream):
    X, y = stream
    tr = StreamTrainer(DecisionTreeClassifier(buffer_size=100), track_peak_memory=True)
    assert tr.fit_chunk(X[:100], y[:100])["peak_bytes"] > 0


def test_trainer_concept_drift_dip_then_recovery():
    X, y = make_drift_stream(4000, drift_at=0.5, label_noise=0.0, seed=9)
    tr = StreamTrainer(DecisionTreeClassifier(max_depth=8, buffer_size=400))
    tr.run(iter_array_chunks(X, y, 200))
    acc = tr.metric_history("chunk_accuracy")
    pre, post, end = acc[8:10].mean(), acc[10:12].min(), acc[-2:].mean()
    assert post < pre and end > post


def test_trainer_empty_summary_and_metric_history_nan():
    tr = StreamTrainer(DecisionTreeClassifier())
    s = tr.summary()
    assert s["n_chunks"] == 0 and np.isnan(s["mean_chunk_accuracy"])
