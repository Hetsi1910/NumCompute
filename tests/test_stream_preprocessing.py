"""Unit tests for numcompute_stream.preprocessing (partial_fit transformers)."""

import numpy as np
import pytest

import numcompute.preprocessing as batch
from numcompute_stream.preprocessing import Imputer, MinMaxScaler, OneHotEncoder, StandardScaler


@pytest.fixture
def X():
    rng = np.random.default_rng(2)
    out = rng.normal(4, 2, size=(600, 3))
    out[3, 0] = np.nan
    out[50:60, 2] = np.nan
    return out


def _feed(est, X, parts=8):
    for c in np.array_split(X, parts):
        est.partial_fit(c)
    return est


def test_standard_scaler_streaming_equals_batch(X):
    s, b = _feed(StandardScaler(), X), batch.StandardScaler().fit(X)
    assert np.allclose(s.mean_, b.mean_) and np.allclose(s.scale_, b.scale_)
    assert np.allclose(s.transform(X), b.transform(X), equal_nan=True)


def test_standard_scaler_fit_resets_previous_state(X):
    s = _feed(StandardScaler(), X)
    s.fit(X[:50] + 100)
    assert np.allclose(s.mean_, np.nanmean(X[:50] + 100, axis=0))


def test_scaler_constant_stream_scale_one():
    s = StandardScaler().partial_fit(np.full((5, 2), 3.0)).partial_fit(np.full((5, 2), 3.0))
    assert np.all(s.scale_ == 1.0)
    assert np.allclose(s.transform(np.full((2, 2), 3.0)), 0.0)


def test_scaler_zero_variance_first_chunk_then_variation():
    s = StandardScaler().partial_fit(np.ones((4, 1))).partial_fit(np.array([[3.0], [5.0]]))
    both = np.array([1, 1, 1, 1, 3, 5.0])
    assert s.mean_[0] == pytest.approx(both.mean()) and s.var_[0] == pytest.approx(both.var())


def test_scaler_ema_tracks_drift():
    s = StandardScaler(momentum=0.5)
    s.partial_fit(np.zeros((10, 1)) + np.arange(10).reshape(-1, 1) * 0.0)
    for _ in range(6):
        s.partial_fit(np.full((10, 1), 10.0))
    assert s.mean_[0] > 9.0  # exact mode would give ~8.6


def test_scaler_ema_mixture_variance_nonnegative(X):
    s = _feed(StandardScaler(momentum=0.3), X)
    assert np.all(s.var_ >= 0) and np.all(np.isfinite(s.scale_))


def test_scaler_invalid_momentum_and_inputs():
    for bad in (0.0, 1.5, "x"):
        with pytest.raises(ValueError):
            StandardScaler(momentum=bad)
    s = StandardScaler().partial_fit(np.ones((2, 3)))
    with pytest.raises(ValueError, match="features"):
        s.partial_fit(np.ones((2, 4)))
    with pytest.raises(ValueError, match="infinity"):
        StandardScaler().partial_fit([[np.inf]])
    with pytest.raises(ValueError, match="empty"):
        StandardScaler().partial_fit(np.empty((0, 2)))


def test_scaler_empty_chunk_after_fit_is_noop(X):
    s = _feed(StandardScaler(), X)
    mean = s.mean_.copy()
    s.partial_fit(np.empty((0, 3)))
    assert np.array_equal(mean, s.mean_)


def test_minmax_streaming_equals_batch(X):
    m, b = _feed(MinMaxScaler(), X), batch.MinMaxScaler().fit(X)
    assert np.allclose(m.transform(X), b.transform(X), equal_nan=True)


def test_minmax_range_expands_with_new_extremes():
    m = MinMaxScaler().partial_fit([[0.0], [1.0]])
    assert m.transform([[2.0]])[0, 0] == pytest.approx(2.0)
    m.partial_fit([[4.0]])
    assert m.transform([[2.0]])[0, 0] == pytest.approx(0.5)


def test_imputer_mean_median_mode_streaming_equal_batch(X):
    Xr = np.round(X)
    for strat in ("mean", "median", "most_frequent"):
        s = _feed(Imputer(strat), Xr)
        assert np.allclose(s.statistics_, batch.Imputer(strat).fit(Xr).statistics_), strat


def test_imputer_transform_fills_nan(X):
    out = _feed(Imputer("mean"), X).transform(X)
    assert not np.isnan(out).any()


def test_imputer_all_missing_column_uses_fill_value():
    s = Imputer("mean", fill_value=-1.0).partial_fit(np.array([[np.nan, 1.0], [np.nan, 3.0]]))
    assert s.statistics_.tolist() == [-1.0, 2.0]


def test_imputer_sentinel_missing_and_constant():
    s = Imputer("constant", fill_value=9.0, missing_values=-999.0).partial_fit([[1.0, -999.0]])
    assert s.transform([[-999.0, 2.0]]).tolist() == [[9.0, 2.0]]


def test_imputer_errors():
    with pytest.raises(ValueError, match="strategy"):
        Imputer("bogus").partial_fit([[1.0]])
    s = Imputer().partial_fit(np.ones((2, 2)))
    with pytest.raises(ValueError, match="features"):
        s.partial_fit(np.ones((2, 3)))


def test_onehot_categories_expand_incrementally():
    oh = OneHotEncoder()
    oh.partial_fit([["a"], ["b"]])
    assert oh.n_output_features_ == 2
    oh.partial_fit([["c"], ["a"]])
    assert oh.n_output_features_ == 3
    assert oh.transform([["c"]]).tolist() == [[0.0, 0.0, 1.0]]


def test_onehot_unknown_ignored_by_default_and_error_mode():
    oh = OneHotEncoder().partial_fit([["a"], ["b"]])
    assert oh.transform([["zzz"]]).sum() == 0
    strict = OneHotEncoder(handle_unknown="error").partial_fit([["a"]])
    with pytest.raises(ValueError, match="unknown"):
        strict.transform([["b"]])


def test_onehot_fixed_categories_keep_width_constant():
    oh = OneHotEncoder(categories=[["a", "b", "c"]])
    oh.partial_fit([["a"]])
    assert oh.n_output_features_ == 3
    oh.partial_fit([["b"], ["z"]])
    assert oh.n_output_features_ == 3


def test_onehot_errors():
    with pytest.raises(ValueError):
        OneHotEncoder(handle_unknown="bad").partial_fit([["a"]])
    oh = OneHotEncoder().partial_fit([["a", "x"]])
    with pytest.raises(ValueError, match="columns"):
        oh.partial_fit([["a"]])
    with pytest.raises(ValueError, match="NaN"):
        OneHotEncoder().partial_fit([[1.0], [np.nan]])
