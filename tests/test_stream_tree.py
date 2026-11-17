"""Unit tests for numcompute_stream.tree (training, buffering, edge cases)."""

import numpy as np
import pytest

from numcompute_stream.datasets import make_drift_stream
from numcompute_stream.tree import DecisionTreeClassifier, best_split_loop, best_split_vectorised, impurity
from numcompute_stream.utils import NotFittedError


@pytest.fixture
def data():
    return make_drift_stream(1500, drift_at=None, label_noise=0.0, seed=3)


def _same_tree(a, b):
    return all(np.allclose(a.tree_[k], b.tree_[k], equal_nan=True) for k in a.tree_)


# ------------------------------------------------------------------ impurity
def test_gini_pure_and_uniform():
    assert float(impurity([5, 0])) == 0.0
    assert float(impurity([5, 5])) == pytest.approx(0.5)


def test_entropy_values_and_zero_log_zero():
    assert float(impurity([4, 0, 0], "entropy")) == 0.0
    assert float(impurity([2, 2], "entropy")) == pytest.approx(1.0)


def test_impurity_all_zero_counts_is_zero():
    assert float(impurity([0, 0, 0])) == 0.0


def test_impurity_bad_criterion():
    with pytest.raises(ValueError, match="criterion"):
        impurity([1, 1], "bogus")


# ------------------------------------------------------------------ fitting
def test_fit_separable_data_perfect(data):
    X, y = data
    t = DecisionTreeClassifier().fit(X, y)
    assert t.score(X, y) == 1.0


def test_fit_returns_self_and_counts_retrain():
    t = DecisionTreeClassifier()
    assert t.fit([[0.0], [1.0]], [0, 1]) is t
    assert t.n_retrains_ == 1 and len(t.retrain_times_) == 1


def test_max_depth_limits_tree(data):
    X, y = data
    for d in (0, 1, 3):
        assert DecisionTreeClassifier(max_depth=d).fit(X, y).get_depth() <= d


def test_max_depth_zero_is_majority_stump():
    t = DecisionTreeClassifier(max_depth=0).fit([[0], [1], [2]], [1, 1, 0])
    assert t.get_n_leaves() == 1
    assert t.predict([[10]]).tolist() == [1]


def test_min_samples_split_and_leaf_respected(data):
    X, y = data
    t = DecisionTreeClassifier(min_samples_leaf=40).fit(X, y)
    leaves = t.tree_["feature"] < 0
    assert t.tree_["n_samples"][leaves].min() >= 40
    t2 = DecisionTreeClassifier(min_samples_split=2000).fit(X, y)
    assert t2.get_n_leaves() == 1


def test_entropy_criterion_learns(data):
    X, y = data
    assert DecisionTreeClassifier(criterion="entropy").fit(X, y).score(X, y) == 1.0


def test_string_labels_roundtrip():
    X = np.array([[0.0], [0.1], [1.0], [1.1]])
    y = np.array(["cat", "cat", "dog", "dog"])
    t = DecisionTreeClassifier().fit(X, y)
    assert t.predict([[0.05], [1.05]]).tolist() == ["cat", "dog"]


def test_single_class_training():
    t = DecisionTreeClassifier().fit([[1.0], [2.0], [3.0]], [7, 7, 7])
    assert t.predict([[0.0]]).tolist() == [7]
    assert t.get_n_leaves() == 1


def test_constant_features_make_leaf():
    t = DecisionTreeClassifier().fit(np.ones((6, 2)), [0, 1, 0, 1, 0, 0])
    assert t.get_n_leaves() == 1
    assert t.predict([[1.0, 1.0]]).tolist() == [0]


def test_zero_variance_feature_ignored_in_importance():
    rng = np.random.default_rng(0)
    X = np.column_stack([np.zeros(100), rng.normal(size=100)])
    y = (X[:, 1] > 0).astype(int)
    t = DecisionTreeClassifier().fit(X, y)
    assert t.feature_importances_[0] == 0.0 and t.feature_importances_[1] == pytest.approx(1.0)


def test_xor_needs_zero_gain_first_split():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 5, dtype=float)
    y = np.array([0, 1, 1, 0] * 5)
    assert DecisionTreeClassifier().fit(X, y).score(X, y) == 1.0


def test_tie_resolution_lowest_feature_then_threshold():
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    y = np.array([0, 0, 1, 1])
    t = DecisionTreeClassifier(max_depth=1).fit(X, y)
    assert t.tree_["feature"][0] == 0  # identical features: lowest index wins
    assert t.tree_["threshold"][0] == pytest.approx(1.5)


def test_duplicate_values_threshold_between_distinct():
    X = np.array([[1.0], [1.0], [1.0], [2.0], [2.0]])
    y = np.array([0, 0, 0, 1, 1])
    t = DecisionTreeClassifier().fit(X, y)
    assert t.tree_["threshold"][0] == pytest.approx(1.5)
    assert t.score(X, y) == 1.0


def test_nan_in_training_and_prediction_median_imputed():
    X = np.array([[0.0], [np.nan], [1.0], [10.0], [11.0], [np.nan]])
    y = np.array([0, 0, 0, 1, 1, 1])
    t = DecisionTreeClassifier().fit(X, y)
    assert np.isfinite(t.nan_fill_).all()
    assert t.predict([[np.nan]]).shape == (1,)


def test_all_nan_column_does_not_crash():
    X = np.column_stack([np.full(8, np.nan), np.arange(8.0)])
    t = DecisionTreeClassifier().fit(X, (np.arange(8) > 3).astype(int))
    assert t.score(np.column_stack([np.full(8, np.nan), np.arange(8.0)]), (np.arange(8) > 3).astype(int)) == 1.0


def test_max_features_options_reproducible(data):
    X, y = data
    a = DecisionTreeClassifier(max_features="sqrt", random_state=1).fit(X, y)
    b = DecisionTreeClassifier(max_features="sqrt", random_state=1).fit(X, y)
    assert _same_tree(a, b)
    for mf in (2, 0.5, "log2"):
        assert DecisionTreeClassifier(max_features=mf, random_state=0).fit(X, y).is_ready


def test_vectorised_matches_loop_reference(data):
    X, y = data
    a = DecisionTreeClassifier(max_depth=5).fit(X[:300], y[:300])
    b = DecisionTreeClassifier(max_depth=5, vectorised=False).fit(X[:300], y[:300])
    assert _same_tree(a, b)
    assert np.array_equal(a.predict(X), b.predict(X))


def test_split_functions_agree_on_random_node():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(60, 3))
    codes = rng.integers(0, 3, 60)
    y1h = np.eye(3)[codes]
    idx = np.arange(60)
    v = best_split_vectorised(X, y1h, idx, [0, 1, 2], "gini", 1)
    ref = best_split_loop(X, codes, 3, idx, [0, 1, 2], "gini", 1)
    assert v[0] == ref[0] and v[1] == pytest.approx(ref[1]) and v[2] == pytest.approx(ref[2])


def test_split_none_for_single_row_or_constant():
    X = np.ones((5, 1))
    y1h = np.eye(2)[[0, 1, 0, 1, 0]]
    assert best_split_vectorised(X, y1h, np.arange(5), [0], "gini", 1) is None
    assert best_split_vectorised(X, y1h, np.arange(1), [0], "gini", 1) is None


def test_predict_proba_rows_sum_to_one(data):
    X, y = data
    p = DecisionTreeClassifier(max_depth=3).fit(X, y).predict_proba(X)
    assert p.shape == (len(X), 3) and np.allclose(p.sum(axis=1), 1.0)


def test_apply_and_export_text(data):
    X, y = data
    t = DecisionTreeClassifier(max_depth=2).fit(X, y)
    leaves = t.apply(X)
    assert np.all(t.tree_["feature"][leaves] == -1)
    txt = t.export_text(feature_names=[f"f{i}" for i in range(X.shape[1])])
    assert "class:" in txt and "f" in txt


def test_does_not_modify_input():
    X = np.array([[0.0], [np.nan], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])
    Xc = X.copy()
    DecisionTreeClassifier().fit(X, y).predict(X)
    assert np.array_equal(X, Xc, equal_nan=True)


# ------------------------------------------------------------------ validation
@pytest.mark.parametrize(
    "kw",
    [
        {"criterion": "mse"},
        {"max_depth": -1},
        {"min_samples_split": 1},
        {"min_samples_leaf": 0},
        {"buffer_size": 0},
        {"max_features": "all"},
        {"max_features": 0},
        {"min_impurity_decrease": -1.0},
    ],
)
def test_invalid_hyperparameters_raise(kw):
    with pytest.raises((ValueError, TypeError)):
        DecisionTreeClassifier(**kw)


def test_non_integer_buffer_size_type_error():
    with pytest.raises(TypeError, match="buffer_size"):
        DecisionTreeClassifier(buffer_size=2.5)


def test_fit_shape_and_value_errors():
    t = DecisionTreeClassifier()
    with pytest.raises(ValueError, match="inconsistent"):
        t.fit([[1.0], [2.0]], [0])
    with pytest.raises(ValueError, match="ndim"):
        t.fit([1.0, 2.0], [0, 1])
    with pytest.raises(ValueError, match="empty"):
        t.fit(np.empty((0, 2)), [])
    with pytest.raises(ValueError, match="infinity"):
        t.fit([[np.inf], [1.0]], [0, 1])
    with pytest.raises(ValueError, match="NaN"):
        t.fit([[1.0], [2.0]], [0.0, np.nan])
    with pytest.raises(TypeError):
        t.fit([["a"], ["b"]], [0, 1])


def test_predict_feature_mismatch(data):
    X, y = data
    t = DecisionTreeClassifier().fit(X, y)
    with pytest.raises(ValueError, match="features"):
        t.predict(X[:, :2])


# ------------------------------------------------------------------ streaming
def test_predict_before_first_retrain_raises_informative():
    t = DecisionTreeClassifier(buffer_size=100)
    t.partial_fit([[0.0], [1.0]], [0, 1])
    with pytest.raises(NotFittedError, match="2/100"):
        t.predict([[0.0]])


def test_buffer_retrains_only_when_full_and_clears():
    X, y = make_drift_stream(400, drift_at=None, seed=1)
    t = DecisionTreeClassifier(max_depth=4, buffer_size=200)
    t.partial_fit(X[:100], y[:100])
    assert not t.is_ready and t.buffer_fill_ == 100 and t.n_retrains_ == 0
    t.partial_fit(X[100:200], y[100:200])
    assert t.is_ready and t.buffer_fill_ == 0 and t.n_retrains_ == 1
    t.partial_fit(X[200:250], y[200:250])
    assert t.buffer_fill_ == 50 and t.n_retrains_ == 1


def test_retrain_triggers_on_overshoot_chunk():
    X, y = make_drift_stream(500, drift_at=None, seed=2)
    t = DecisionTreeClassifier(buffer_size=100).partial_fit(X[:350], y[:350])
    assert t.n_retrains_ == 1 and t.buffer_fill_ == 0  # one chunk > buffer_size -> trains on all 350


def test_between_retrains_predict_uses_last_tree():
    X, y = make_drift_stream(600, drift_at=None, label_noise=0, seed=4)
    t = DecisionTreeClassifier(max_depth=6, buffer_size=300)
    t.partial_fit(X[:300], y[:300])
    before = t.predict(X[300:400]).copy()
    t.partial_fit(X[300:400], y[300:400])  # buffered only
    assert np.array_equal(t.predict(X[300:400]), before)


def test_retrain_uses_only_buffered_samples():
    Xa = np.array([[0.0], [1.0]] * 10)
    t = DecisionTreeClassifier(buffer_size=20)
    t.partial_fit(Xa, [0, 1] * 10)
    assert t.predict([[0.0]]).tolist() == [0]
    t.partial_fit(Xa, [1, 0] * 10)  # concept flipped; old data no longer used
    assert t.predict([[0.0]]).tolist() == [1]


def test_partial_fit_matches_fit_on_same_buffer(data):
    X, y = data
    a = DecisionTreeClassifier(max_depth=4, buffer_size=600)
    for i in range(0, 600, 100):
        a.partial_fit(X[i : i + 100], y[i : i + 100])
    b = DecisionTreeClassifier(max_depth=4).fit(X[:600], y[:600])
    assert _same_tree(a, b)


def test_empty_chunk_is_noop():
    t = DecisionTreeClassifier(buffer_size=2)
    t.partial_fit(np.empty((0, 3)), [])
    assert t.buffer_fill_ == 0 and t.n_samples_seen_ == 0


def test_chunk_feature_count_mismatch_raises():
    t = DecisionTreeClassifier(buffer_size=10)
    t.partial_fit(np.ones((2, 3)), [0, 1])
    with pytest.raises(ValueError, match="features"):
        t.partial_fit(np.ones((2, 4)), [0, 1])


def test_feature_mismatch_after_training_raises():
    t = DecisionTreeClassifier(buffer_size=2).partial_fit(np.ones((2, 3)), [0, 1])
    with pytest.raises(ValueError, match="features"):
        t.partial_fit(np.ones((2, 5)), [0, 1])


def test_flush_trains_on_partial_buffer_and_noop_when_empty():
    t = DecisionTreeClassifier(buffer_size=1000)
    t.flush()
    assert not t.is_ready
    t.partial_fit([[0.0], [1.0]], [0, 1]).flush()
    assert t.is_ready and t.buffer_fill_ == 0
    n = t.n_retrains_
    t.flush()
    assert t.n_retrains_ == n


def test_new_class_appears_in_later_chunk():
    t = DecisionTreeClassifier(buffer_size=4)
    t.partial_fit([[0.0], [1.0]], [0, 0]).partial_fit([[0.0], [1.0]], [0, 0])
    assert t.classes_.tolist() == [0]
    t.partial_fit([[0.0], [5.0]], [0, 2]).partial_fit([[0.1], [5.1]], [0, 2])
    assert t.classes_.tolist() == [0, 2]


def test_fit_resets_stream_state():
    t = DecisionTreeClassifier(buffer_size=100)
    t.partial_fit([[0.0], [1.0]], [0, 1])
    t.fit([[0.0], [1.0]], [0, 1])
    assert t.buffer_fill_ == 0 and t.n_samples_seen_ == 2


def test_get_set_params_roundtrip():
    t = DecisionTreeClassifier(max_depth=5, buffer_size=500)
    assert t.get_params()["buffer_size"] == 500
    t.set_params(buffer_size=50)
    assert t.buffer_size == 50


@pytest.mark.parametrize("vectorised", [True, False])
def test_tie_between_thresholds_of_one_feature_takes_lowest(vectorised):
    # cutting after the first or after the last sample gives exactly the same impurity -> lowest threshold wins
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 1, 1, 0])
    t = DecisionTreeClassifier(max_depth=1, vectorised=vectorised).fit(X, y)
    assert t.tree_["threshold"][0] == pytest.approx(0.5)


@pytest.mark.parametrize("vectorised", [True, False])
def test_tie_between_features_takes_lowest_index_even_when_later_one_is_listed_last(vectorised):
    X = np.array([[0.0, 5.0, 9.0], [1.0, 6.0, 8.0], [2.0, 7.0, 7.0], [3.0, 8.0, 6.0]])
    y = np.array([0, 0, 1, 1])
    t = DecisionTreeClassifier(max_depth=1, vectorised=vectorised).fit(X, y)
    assert t.tree_["feature"][0] == 0


def test_prediction_nan_uses_training_median_not_zero():
    X = np.array([[1.0], [2.0], [3.0], [10.0], [11.0], [12.0], [13.0]])
    y = np.array([0, 0, 0, 1, 1, 1, 1])
    t = DecisionTreeClassifier().fit(X, y)
    assert t.nan_fill_[0] == 10.0
    assert t.predict([[np.nan]]).tolist() == [1]  # median 10 lies right of the threshold; 0 would go left


def test_training_nan_rows_follow_the_median_branch():
    X = np.array([[1.0], [2.0], [np.nan], [10.0], [11.0], [np.nan]])
    y = np.array([0, 0, 1, 1, 1, 1])
    t = DecisionTreeClassifier().fit(X, y)
    assert t.nan_fill_[0] == 6.0 and t.score(X, y) == 1.0
