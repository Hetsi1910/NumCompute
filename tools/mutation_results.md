# Mutation check results

20 of 20 deliberate defects were detected by the streaming tests.

| # | File | Injected defect | Result | First failing test |
|---:|---|---|---|---|
| 1 | `tree.py` | Gini of an empty node is 1 instead of 0 | killed | `test_impurity_all_zero_counts_is_zero` |
| 2 | `tree.py` | ties resolved to the highest threshold | killed | `test_tie_between_thresholds_of_one_feature_takes_lowest[True]` |
| 3 | `tree.py` | ties between features resolved to the last feature | killed | `test_tie_resolution_lowest_feature_then_threshold` |
| 4 | `tree.py` | off-by-one: retrain only when buffer strictly exceeds buffer_size | killed | `test_buffer_retrains_only_when_full_and_clears` |
| 5 | `tree.py` | buffer content is not cleared after retraining | killed | `test_retrain_uses_only_buffered_samples` |
| 6 | `tree.py` | NaN at prediction time filled with 0 instead of the training median | killed | `test_prediction_nan_uses_training_median_not_zero` |
| 7 | `tree.py` | threshold midpoint overflows near +-1e308 | killed | `test_threshold_midpoint_does_not_overflow_near_float_max` |
| 8 | `tree.py` | predict touches classes_ before the readiness check (S-BUG-02) | killed | `test_predict_before_first_retrain_raises_informative` |
| 9 | `metrics.py` | precision uses the recall denominator | killed | `test_prf_equal_batch[macro-Precision-precision]` |
| 10 | `metrics.py` | F1 numerator loses its factor 2 | killed | `test_prf_equal_batch[macro-F1-f1]` |
| 11 | `metrics.py` | rolling window never forgets old chunks | killed | `test_rolling_window_only_uses_recent_chunks` |
| 12 | `metrics.py` | confusion matrix forgets old counts when a new label appears | killed | `test_labels_grow_when_new_class_appears` |
| 13 | `stats.py` | shifted-data update disabled (cancellation with large offsets) | killed | `test_streaming_mean_var_equal_batch` |
| 14 | `stats.py` | reservoir sampler keeps too few replacements | killed | `test_reservoir_is_uniform_over_the_whole_stream` |
| 15 | `stats.py` | histogram edge bin is never filled | killed | `test_histogram_window_forgets_old_chunks` |
| 16 | `preprocessing.py` | absolute threshold flattens tiny but real spread | killed | `test_standard_scaler_preserves_tiny_nonconstant_spread[1e-12]` |
| 17 | `preprocessing.py` | infinite feature_range accepted | killed | `test_minmax_rejects_non_finite_feature_range[rng_0]` |
| 18 | `pipeline.py` | pipeline always passes y to partial_fit (breaks minimal transformers) | killed | `test_pipeline_partial_fit_accepts_partial_fit_X_only` |
| 19 | `stream.py` | unfitted model crashes the trainer instead of being logged | killed | `test_trainer_prequential_logs_one_record_per_chunk` |
| 20 | `utils.py` | infinite values accepted by chunk validation | killed | `test_fit_shape_and_value_errors` |
