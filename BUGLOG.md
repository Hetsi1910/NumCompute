# NumCompute Bug Log

# Assignment 2.2 (NumCompute-Stream) - individual bug log

Defects found while building `numcompute_stream/`. Same loop as above: a check fails -> reproduce ->
root cause -> fix the code -> keep a regression test.

| ID | Module | Type | Short description | Found by | Regression test | Status |
|---|---|---|---|---|---|---|
| S-BUG-01 | `tree.impurity` | Code | Gini of an all-zero count vector returned 1.0 instead of 0 (`1 - sum(p^2)` with p = 0) | Unit test of edge case "empty node" | `test_impurity_all_zero_counts_is_zero` | Fixed |
| S-BUG-02 | `tree.predict` | Code | `predict` before the first retrain raised a bare `AttributeError: no attribute 'classes_'` instead of the informative `NotFittedError` (it evaluated `self.classes_` before `predict_proba` checked readiness) | Smoke run of `StreamTrainer` on a pipeline | `test_predict_before_first_retrain_raises_informative`, `test_trainer_unscored_until_first_retrain` | Fixed |
| S-BUG-03 | `datasets.make_drift_stream` | Design | First concept (sum of two quantile bins, a checkerboard) has zero impurity gain on the first split, so a greedy tree trained on 500 samples chose noise features at random: chunk accuracy jumped 0.3 <-> 0.9 for no reason related to streaming | Demo notebook + benchmark looked implausible | `test_trainer_concept_drift_dip_then_recovery` | Fixed (concept = quantile bins of an oblique score) |
| S-BUG-04 | `demo/stream_demo.ipynb` | Demo | Live-plot loop used `clear_output`, which also erased the per-chunk log lines the demo is supposed to show | Executing the notebook and reading its outputs | n/a (visual) | Fixed (logged chunks run first, live plot after) |
| S-BUG-05 | `preprocessing.StandardScaler` | Code | Absolute threshold `scale <= 1e-12` flattened genuine data measured in tiny units (values around 1e-20 came out with std 0) | Probing against the 2.1 review comment "tiny nonconstant features are treated as constant" | `test_standard_scaler_preserves_tiny_nonconstant_spread[*]`, `test_constant_column_still_gets_scale_one[*]` | Fixed (constant only if spread is round-off *relative to the column's own magnitude*) |
| S-BUG-06 | `tree._threshold_between` | Code | `lo + (hi - lo) / 2` overflows for values near +-1e308 (RuntimeWarning, inf threshold) | Probing with extreme magnitudes under `-W error` | `test_threshold_midpoint_does_not_overflow_near_float_max`, `test_tree_splits_at_extreme_magnitudes_without_warnings` | Fixed (`0.5*lo + 0.5*hi` with a guard) |
| S-BUG-07 | `pipeline.Pipeline` | Code | `fit` / `partial_fit` always called `step.fit(X, y)`; transformers whose `fit` or `partial_fit` only accept `X` raised `TypeError` (the 2.1 comment "Pipeline rejects minimal fit(X) transformers") | Probing with minimal transformer classes | `test_pipeline_fit_accepts_fit_X_only_transformer`, `test_pipeline_partial_fit_accepts_partial_fit_X_only`, `test_pipeline_passes_y_to_varargs_step` | Fixed (dispatch on `inspect.signature`) |
| S-BUG-08 | `preprocessing`, `tree` | Code | Silent acceptance of invalid values: `MinMaxScaler(feature_range=(0, inf))` (only caught by accident through a warning), `Imputer` with infinite data or `fill_value=inf/nan`, `min_impurity_decrease=nan` | Probing parameters and data with non-finite values | `test_minmax_rejects_non_finite_feature_range`, `test_imputer_rejects_non_finite_fill_value`, `test_every_component_rejects_infinite_data`, `test_tree_rejects_non_finite_hyperparameters` | Fixed (explicit validation with informative messages) |
| S-BUG-09 | `stats.RunningStats` (reused by scaler, imputer, `StreamStats`) | Numerical | Running variance of `1e9 + noise` had relative error 1.2e-8 (the textbook `E[x^2]-E[x]^2` is wrong by a factor 125) because float64 cannot resolve a running mean near 1e9 to 1e-7; variance of data with spread below ~1e-154 underflowed to 0 | New numerical tests (large offset, 1e-200 scale) | `test_stream_variance_stable_with_large_offset`, `test_scaler_variance_stable_with_large_offset`, `test_standard_scaler_preserves_tiny_nonconstant_spread[1e-200]`, `test_running_stats_*` | Fixed (shifted-data update: centre on the first value, rescale by the first spread; error now ~1e-15) |
| S-BUG-10 | tests | Test gap | Mutation check (below) showed three defects that no test noticed: ties between *thresholds* of one feature, NaN at prediction time filled with 0 instead of the training median, a biased reservoir sampler | `tools/mutation_check.py` | `test_tie_between_thresholds_of_one_feature_takes_lowest`, `test_prediction_nan_uses_training_median_not_zero`, `test_reservoir_is_uniform_over_the_whole_stream` | Fixed (tests added; 20/20 mutants now killed) |

## Mutation checks

`python tools/mutation_check.py` changes one line of `numcompute_stream` at a time in a temporary copy of the repository (the real files
are never touched) and runs the streaming tests; a mutant is *killed* when at least one test fails. The 20 mutants simulate real
mistakes (wrong denominator, off-by-one buffer test, ties resolved the other way, rolling window that never forgets, shifted-data update
removed, absolute threshold for "constant", infinite values accepted, ...). First run: 18 of 20 killed; the two survivors exposed the
test gaps in S-BUG-10. After adding tests: **20 of 20 killed** (`tools/mutation_results.md` lists every mutant and the first test that
caught it).

Notes on diagnosis
- S-BUG-01 and S-BUG-02 were *edge-case* bugs (empty node, not-yet-trained model) - exactly the streaming situations the assignment stresses.
- S-BUG-05 to S-BUG-09 were found by reading the review comments of the previous assignment (tiny scales, mixed scales, infinite bounds,
  transformer signatures, cancellation) and writing a failing probe for each *before* changing any code.
- Vectorised and loop split searches are cross-checked in `test_vectorised_matches_loop_reference` and `test_split_functions_agree_on_random_node`;
  they produced identical trees on every dataset used.
- Final state for the streaming package: 218 unit tests (98% line coverage), whole repository 499 tests, `pytest -W error` clean,
  doctests pass, `flake8 --max-line-length=130` clean for `numcompute_stream/`, 20/20 mutants killed.
