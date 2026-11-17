"""Targeted numerical-robustness and protocol tests for numcompute_stream.

Covers: tiny but non-constant scales, mixed-scale features, huge magnitudes, large offsets (cancellation),
infinite / NaN bounds and parameters, and the minimal transformer/estimator signatures a Pipeline must accept.
"""

import numpy as np
import pytest

from numcompute_stream.metrics import F1, AUC, Accuracy, ConfusionMatrix, Precision, Recall
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import Imputer, MinMaxScaler, OneHotEncoder, StandardScaler
from numcompute_stream.stats import StreamStats
from numcompute_stream.tree import DecisionTreeClassifier, _threshold_between


@pytest.fixture
def base():
    return np.random.default_rng(0).normal(size=(300, 2))


# ------------------------------------------------------------- tiny non-constant scales
@pytest.mark.parametrize("unit", [1e-5, 1e-12, 1e-20, 1e-100, 1e-200])
def test_standard_scaler_preserves_tiny_nonconstant_spread(base, unit):
    z = StandardScaler().partial_fit(base * unit).transform(base * unit)
    assert np.allclose(z.std(axis=0), 1.0)


@pytest.mark.parametrize("unit", [1e-12, 1e-100])
def test_minmax_scaler_preserves_tiny_nonconstant_spread(base, unit):
    w = MinMaxScaler().partial_fit(base * unit).transform(base * unit)
    assert np.allclose(w.max(axis=0) - w.min(axis=0), 1.0)


def test_standard_scaler_tiny_values_streamed_in_chunks_equal_one_shot(base):
    X = base * 1e-30
    s = StandardScaler()
    for c in np.array_split(X, 7):
        s.partial_fit(c)
    assert np.allclose(s.scale_, StandardScaler().partial_fit(X).scale_, rtol=1e-12)


@pytest.mark.parametrize("value", [0.0, 3.0, 0.1, 1e-20, -7.5, 1e12])
def test_constant_column_still_gets_scale_one(value):
    s = StandardScaler().partial_fit(np.full((6, 1), value)).partial_fit(np.full((4, 1), value))
    assert s.scale_[0] == 1.0 and np.allclose(s.transform(np.full((2, 1), value)), 0.0)


# ------------------------------------------------------------- mixed scales / magnitudes
def test_tree_is_invariant_to_per_feature_rescaling(base):
    y = (base[:, 0] + base[:, 1] > 0).astype(int)
    scaled = base * np.array([1e-12, 1e12])
    a = DecisionTreeClassifier(max_depth=5).fit(base, y).predict(base)
    b = DecisionTreeClassifier(max_depth=5).fit(scaled, y).predict(scaled)
    assert np.array_equal(a, b)


def test_pipeline_with_mixed_scale_features_learns(base):
    y = (base[:, 0] > 0).astype(int)
    X = base * np.array([1e-12, 1e12])
    p = Pipeline([("s", StandardScaler()), ("m", DecisionTreeClassifier(max_depth=3, buffer_size=300))])
    p.partial_fit(X, y)
    assert p.score(X, y) > 0.95


def test_threshold_midpoint_does_not_overflow_near_float_max():
    with np.errstate(all="raise"):
        assert _threshold_between(-1e308, 1e308) == 0.0
        assert _threshold_between(1e308, 1.7e308) < 1.7e308


def test_tree_splits_at_extreme_magnitudes_without_warnings():
    X = np.array([[-1e308], [1e308], [-1e308], [1e308]])
    with np.errstate(all="raise"):
        t = DecisionTreeClassifier().fit(X, [0, 1, 0, 1])
        assert t.predict(X).tolist() == [0, 1, 0, 1]


def test_tree_threshold_between_adjacent_floats_separates_them():
    lo = 1.0
    hi = np.nextafter(lo, 2.0)
    t = DecisionTreeClassifier().fit([[lo], [hi]], [0, 1])
    assert t.predict([[lo], [hi]]).tolist() == [0, 1]


def test_tree_with_subnormal_values():
    X = np.array([[5e-324], [1e-323], [5e-324], [1e-323]])
    assert DecisionTreeClassifier().fit(X, [0, 1, 0, 1]).predict(X).tolist() == [0, 1, 0, 1]


# ------------------------------------------------------------- cancellation / large offsets
def test_stream_variance_stable_with_large_offset():
    rng = np.random.default_rng(1)
    X = 1e9 + rng.normal(size=(2000, 1))
    s = StreamStats()
    for c in np.array_split(X, 20):
        s.update_stats(c)
    assert s.var()[0] == pytest.approx(np.var(X), rel=1e-12)
    naive = np.mean(X**2) - np.mean(X) ** 2  # textbook formula: catastrophic cancellation
    assert abs(naive - np.var(X)) > 1.0  # shows why the shifted update is needed


def test_scaler_variance_stable_with_large_offset():
    rng = np.random.default_rng(2)
    X = 1e9 + rng.normal(size=(2000, 1))
    s = StandardScaler()
    for c in np.array_split(X, 20):
        s.partial_fit(c)
    assert s.var_[0] == pytest.approx(np.var(X), rel=1e-12)


def test_stream_stats_tiny_range_histogram_is_valid():
    s = StreamStats(n_bins=4)
    s.update_stats(np.array([[0.0], [1e-300], [2e-300], [3e-300]]))
    counts, _ = s.histogram()
    assert counts.sum() == 4


# ------------------------------------------------------------- infinite / NaN bounds and parameters
@pytest.mark.parametrize("rng_", [(0, np.inf), (-np.inf, 1), (np.nan, 1), (0, np.nan)])
def test_minmax_rejects_non_finite_feature_range(rng_):
    with pytest.raises(ValueError, match="finite|low < high"):
        MinMaxScaler(feature_range=rng_).partial_fit([[1.0], [2.0]])


@pytest.mark.parametrize("rng_", [(0, np.inf), (np.nan, 1)])
def test_histogram_range_rejects_non_finite(rng_):
    with pytest.raises(ValueError):
        StreamStats(hist_range=rng_)


@pytest.mark.parametrize("fill", [np.inf, -np.inf, np.nan])
def test_imputer_rejects_non_finite_fill_value(fill):
    with pytest.raises(ValueError, match="finite"):
        Imputer(fill_value=fill).partial_fit([[np.nan], [1.0]])


@pytest.mark.parametrize(
    "make",
    [StandardScaler, MinMaxScaler, Imputer, StreamStats, lambda: DecisionTreeClassifier(buffer_size=2)],
)
def test_every_component_rejects_infinite_data(make):
    est = make()
    X = np.array([[1.0], [np.inf]])
    with pytest.raises(ValueError, match="infinity"):
        (
            est.update_stats(X)
            if isinstance(est, StreamStats)
            else (est.partial_fit(X, [0, 1]) if isinstance(est, DecisionTreeClassifier) else est.partial_fit(X))
        )


def test_tree_rejects_non_finite_hyperparameters():
    for kw in ({"min_impurity_decrease": np.nan}, {"max_features": np.nan}, {"max_features": np.inf}):
        with pytest.raises(ValueError):
            DecisionTreeClassifier(**kw)


# ------------------------------------------------------------- minimal transformer / estimator signatures
class FitOnly:
    """Smallest legal transformer: fit(X) and transform(X); no y, no fit_transform, no partial_fit."""

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        return self

    def transform(self, X):
        return X - self.mean_


class PartialOnly:
    """Streaming transformer whose partial_fit takes only X."""

    def partial_fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        return self

    def fit(self, X):
        return self.partial_fit(X)

    def transform(self, X):
        return X - self.mean_


class VarArgs:
    """Transformer with *args signature (must receive y)."""

    def __init__(self):
        self.seen = None

    def fit(self, *args):
        self.seen = len(args)
        return self

    def transform(self, X):
        return X


class FitTransformOnly:
    def fit_transform(self, X):
        return X * 2.0

    def fit(self, X):
        return self

    def transform(self, X):
        return X * 2.0


@pytest.fixture
def xy(base):
    return base, (base[:, 0] > 0).astype(int)


def test_pipeline_fit_accepts_fit_X_only_transformer(xy):
    X, y = xy
    p = Pipeline([("m", FitOnly()), ("t", DecisionTreeClassifier(max_depth=3))]).fit(X, y)
    assert p.predict(X).shape == (300,)


def test_pipeline_fit_accepts_fit_transform_X_only(xy):
    X, y = xy
    p = Pipeline([("m", FitTransformOnly()), ("t", DecisionTreeClassifier(max_depth=3))]).fit(X, y)
    assert p.score(X, y) > 0.9


def test_pipeline_partial_fit_accepts_partial_fit_X_only(xy):
    X, y = xy
    p = Pipeline([("m", PartialOnly()), ("t", DecisionTreeClassifier(max_depth=3, buffer_size=100))])
    for i in range(0, 300, 100):
        p.partial_fit(X[i : i + 100], y[i : i + 100])
    assert p.final_estimator.n_retrains_ == 3


def test_pipeline_passes_y_to_varargs_step(xy):
    X, y = xy
    step = VarArgs()
    Pipeline([("v", step), ("t", DecisionTreeClassifier())]).fit(X, y)
    assert step.seen == 2


def test_pipeline_still_passes_y_to_two_argument_steps(xy):
    X, y = xy
    seen = {}

    class NeedsY:
        def fit(self, X, y):
            seen["y"] = y
            return self

        def transform(self, X):
            return X

    Pipeline([("n", NeedsY()), ("t", DecisionTreeClassifier())]).fit(X, y)
    assert np.array_equal(seen["y"], y)


def test_pipeline_unlabelled_streaming_final_step_gets_no_y(base):
    class Collector:
        def __init__(self):
            self.n = 0

        def partial_fit(self, X):
            self.n += len(X)
            return self

        def fit(self, X):
            return self.partial_fit(X)

        def predict(self, X):
            return np.zeros(len(X))

    c = Collector()
    Pipeline([("s", StandardScaler()), ("c", c)]).partial_fit(base)
    assert c.n == 300


def test_pipeline_invalid_steps_raise_clear_errors():
    with pytest.raises(TypeError, match="fit and transform"):
        Pipeline([("bad", object()), ("t", DecisionTreeClassifier())])
    with pytest.raises(TypeError, match="must implement fit"):
        Pipeline([("s", StandardScaler()), ("bad", object())])


# ------------------------------------------------------------- protocol: every streaming object
@pytest.mark.parametrize("cls", [Accuracy, Precision, Recall, F1, ConfusionMatrix, AUC])
def test_every_metric_has_update_result_reset(cls):
    m = cls()
    for name in ("update", "result", "reset"):
        assert callable(getattr(m, name))
    m.reset()
    out = m.result()
    assert np.isnan(out) if np.isscalar(out) else out.size == 0


@pytest.mark.parametrize("make", [StandardScaler, MinMaxScaler, Imputer, OneHotEncoder, DecisionTreeClassifier])
def test_every_model_and_transformer_has_partial_fit(make):
    assert callable(make().partial_fit)


def test_reset_is_idempotent_and_reusable():
    m = Accuracy().update([0, 1], [0, 1])
    m.reset().reset()
    assert m.update([0, 1], [1, 1]).result() == 0.5


# ------------------------------------------------------------- dtype / memory layout
@pytest.mark.parametrize("dtype", [np.float32, np.int64, np.int32])
def test_tree_and_scaler_accept_other_numeric_dtypes(dtype):
    X = np.arange(40).reshape(20, 2).astype(dtype)
    y = (X[:, 0] > 20).astype(int)
    assert DecisionTreeClassifier().fit(X, y).score(X, y) == 1.0
    assert StandardScaler().partial_fit(X).transform(X).dtype == np.float64


def test_non_contiguous_and_fortran_arrays_give_same_result(base):
    y = (base[:, 0] > 0).astype(int)
    ref = DecisionTreeClassifier(max_depth=3).fit(base, y).predict(base)
    strided = np.asfortranarray(base)[::1, :]
    assert np.array_equal(DecisionTreeClassifier(max_depth=3).fit(strided, y).predict(strided), ref)
    wide = np.column_stack([base, base])[:, ::2]
    assert not wide.flags["C_CONTIGUOUS"] or True
    assert np.array_equal(DecisionTreeClassifier(max_depth=3).fit(wide, y).predict(wide), ref)


# ------------------------------------------------------------- stabilised RunningStats (shift + unit)
from numcompute_stream.stats import RunningStats  # noqa: E402


def test_running_stats_chunked_equals_numpy_with_offset_and_scale():
    rng = np.random.default_rng(3)
    X = np.column_stack([1e9 + rng.normal(size=500), rng.normal(size=500) * 1e-30, rng.normal(size=500) * 1e6])
    rs = RunningStats()
    for c in np.array_split(X, 9):
        rs.update(c)
    assert np.allclose(rs.mean, X.mean(axis=0), rtol=1e-12)
    assert np.allclose(rs.std(), X.std(axis=0), rtol=1e-10)
    assert np.allclose(rs.var(ddof=1), X.var(axis=0, ddof=1), rtol=1e-10, atol=0)


def test_running_stats_merge_of_two_streams_with_different_frames():
    rng = np.random.default_rng(4)
    A, B = 1e9 + rng.normal(size=(300, 1)), 5.0 + rng.normal(size=(200, 1)) * 1e-3
    a, b = RunningStats().update(A), RunningStats().update(B)
    a.merge(b)
    both = np.vstack([A, B])
    assert np.allclose(a.mean, both.mean(axis=0), rtol=1e-12)
    assert np.allclose(a.var(), both.var(axis=0), rtol=1e-8)


def test_running_stats_merge_into_empty_and_with_empty():
    rs = RunningStats().update(np.arange(10.0).reshape(-1, 1))
    empty = RunningStats()
    assert np.allclose(empty.merge(rs).mean, rs.mean)
    assert np.allclose(rs.merge(RunningStats()).mean, 4.5)


def test_running_stats_constant_then_varying_chooses_unit_late():
    rs = RunningStats().update(np.zeros((5, 1)))  # constant so far: state is exactly zero
    assert rs.std()[0] == 0.0
    rs.update(np.array([[1e-200], [-1e-200]]))  # genuine but tiny spread arrives later
    expected = np.std(np.array([0.0] * 5 + [1e-200, -1e-200]) * 1e200) * 1e-200
    assert rs.std()[0] == pytest.approx(expected, rel=1e-12)


def test_running_stats_nan_and_all_nan_columns():
    X = np.array([[1.0, np.nan], [3.0, np.nan], [np.nan, np.nan]])
    rs = RunningStats().update(X)
    assert rs.count.tolist() == [2.0, 0.0]
    assert rs.mean[0] == 2.0 and np.isnan(rs.mean[1])
    assert np.isnan(rs.var()[1])


def test_running_stats_errors_and_empty_chunk():
    rs = RunningStats().update(np.ones((3, 2)))
    n = rs.count.copy()
    rs.update(np.empty((0, 2)))
    assert np.array_equal(rs.count, n)
    with pytest.raises(ValueError, match="features"):
        rs.update(np.ones((2, 3)))
    with pytest.raises(TypeError):
        rs.merge(object())
    with pytest.raises(ValueError, match="differ"):
        rs.merge(RunningStats().update(np.ones((2, 5))))
