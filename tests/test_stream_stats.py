"""Unit tests for numcompute_stream.stats (StreamStats)."""

import numpy as np
import pytest

from numcompute_stream.stats import StreamStats, update_stats


@pytest.fixture
def X():
    rng = np.random.default_rng(0)
    out = rng.normal(5, 3, size=(1200, 3))
    out[7, 1] = np.nan
    out[100:110, 2] = np.nan
    return out


def _feed(s, X, parts=9):
    for c in np.array_split(X, parts):
        s.update_stats(c)
    return s


def test_streaming_mean_var_equal_batch(X):
    s = _feed(StreamStats(reservoir_size=5000), X)
    assert np.allclose(s.mean, np.nanmean(X, axis=0))
    assert np.allclose(s.var(), np.nanvar(X, axis=0))
    assert np.allclose(s.std(ddof=1), np.nanstd(X, axis=0, ddof=1))


def test_count_ignores_nan(X):
    s = _feed(StreamStats(), X)
    assert s.count.tolist() == (~np.isnan(X)).sum(axis=0).tolist()


def test_min_max_running(X):
    s = _feed(StreamStats(), X)
    assert np.allclose(s.min, np.nanmin(X, axis=0)) and np.allclose(s.max, np.nanmax(X, axis=0))


def test_quantile_exact_when_reservoir_large(X):
    s = _feed(StreamStats(reservoir_size=5000), X)
    assert s.is_exact_quantile
    assert np.allclose(s.quantile([0.1, 0.5, 0.9]), np.nanquantile(X, [0.1, 0.5, 0.9], axis=0))


def test_quantile_approximate_when_stream_exceeds_reservoir(X):
    s = _feed(StreamStats(reservoir_size=300, seed=1), X)
    assert not s.is_exact_quantile
    assert np.all(np.abs(s.quantile(0.5) - np.nanmedian(X, axis=0)) < 1.0)


def test_reservoir_reproducible_with_seed(X):
    a = _feed(StreamStats(reservoir_size=100, seed=7), X)
    b = _feed(StreamStats(reservoir_size=100, seed=7), X)
    assert np.array_equal(a.quantile(0.5), b.quantile(0.5))


def test_histogram_counts_sum_to_observations(X):
    s = _feed(StreamStats(n_bins=10), X)
    counts, edges = s.histogram()
    assert counts.shape == (3, 10) and edges.shape == (3, 11)
    assert counts.sum(axis=1).tolist() == s.count.tolist()


def test_histogram_window_forgets_old_chunks():
    s = StreamStats(n_bins=4, hist_range=(0, 4), window=2)
    s.update_stats(np.full((10, 1), 0.5))
    s.update_stats(np.full((10, 1), 1.5))
    s.update_stats(np.full((10, 1), 3.5))
    counts, _ = s.histogram()
    assert counts[0].tolist() == [0, 10, 0, 10]  # first chunk slid out of the window


def test_histogram_clips_out_of_range_values():
    s = StreamStats(n_bins=2, hist_range=(0, 1))
    s.update_stats([[-5.0], [9.0], [0.2]])
    assert s.histogram()[0][0].tolist() == [2, 1]


def test_constant_first_chunk_gives_valid_edges():
    s = StreamStats(n_bins=5).update_stats(np.ones((4, 2)))
    _, edges = s.histogram()
    assert np.all(np.diff(edges, axis=1) > 0)


def test_empty_chunk_ignored_and_one_dim_accepted():
    s = StreamStats()
    s.update_stats(np.empty((0, 2)))
    assert s.n_features_ is None
    s.update_stats([1.0, 2.0, 3.0])
    assert s.n_features_ == 1 and float(s.mean[0]) == 2.0


def test_feature_count_change_raises():
    s = StreamStats().update_stats(np.ones((3, 2)))
    with pytest.raises(ValueError, match="features"):
        s.update_stats(np.ones((3, 3)))


def test_invalid_inputs_raise():
    with pytest.raises(ValueError, match="infinity"):
        StreamStats().update_stats([[np.inf]])
    with pytest.raises(TypeError):
        StreamStats().update_stats([["a"]])
    with pytest.raises(ValueError):
        StreamStats(hist_range=(1, 1))
    with pytest.raises(ValueError):
        StreamStats(n_bins=0)
    with pytest.raises(ValueError, match="no data"):
        StreamStats().quantile(0.5)
    with pytest.raises(ValueError, match="no data"):
        StreamStats().histogram()


def test_reset_and_summary_and_functional_api(X):
    s = update_stats(StreamStats(), X)
    summ = s.summary()
    assert set(summ) == {"count", "mean", "std", "min", "q25", "median", "q75", "max"}
    s.reset()
    assert s.n_features_ is None and s.n_chunks_ == 0


def test_reservoir_is_uniform_over_the_whole_stream():
    # Algorithm R keeps each of N items with equal probability cap/N, so about half of the sample must come
    # from the second half of a sorted stream (a biased sampler keeps mostly early or mostly late values).
    n, cap = 10_000, 400
    s = StreamStats(reservoir_size=cap, seed=5)
    for chunk in np.array_split(np.arange(n, dtype=float).reshape(-1, 1), 50):
        s.update_stats(chunk)
    sample = s._res[: s._res_n, 0]
    assert 0.4 < np.mean(sample >= n / 2) < 0.6
    assert np.isclose(sample.mean(), n / 2, rtol=0.1)
    assert abs(float(s.quantile(0.5)[0]) - n / 2) < 0.1 * n
