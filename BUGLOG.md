# NumCompute Bug Log

This log records the defects found while building and testing NumCompute, how each was diagnosed,
how it was fixed, and which test now protects against it. It also records the process problems
we ran into while working as a team, because they affected the code on `main`.

**Final state:** 281 unit tests, all passing (`python -m pytest tests -q`), all doctests passing,
`pyflakes` and `pycodestyle` clean.

## 1. How bugs were found

| Method | What it caught |
|---|---|
| Unit tests with edge cases (empty arrays, ties, NaN, extreme `k`, non-contiguous strides) | BUG-01, BUG-03 |
| Comparing results against reference implementations (NumPy, `bisect`, brute-force loops) | confidence in `rank`, `percentile`, `topk`, `binary_search`, histogram, metrics |
| Running tests with warnings treated as errors (`-W error`) | BUG-02 |
| Doctests (examples in docstrings are executed) | BUG-04 |
| Style checks (`pyflakes`, `pycodestyle`) | BUG-06 |
| Teammates running the suite on their own machines | PROC-01 |

Every defect follows the same loop: **a check fails -> reproduce it in isolation -> find the root
cause -> fix the code (not the symptom) -> keep a test that would fail if the bug came back.**

## 2. Summary

| ID | Module | Type | Short description | Status |
|---|---|---|---|---|
| BUG-01 | `sort_search.topk` | Code | Wrong output width for arrays with zero rows | Fixed |
| BUG-02 | `io` | Code | Empty file raised a NumPy warning instead of a clear error | Fixed |
| BUG-03 | `utils.one_hot` | Code | `one_hot([])` crashed with `IndexError` | Fixed |
| BUG-04 | `stats.histogram` | Documentation | Doctest example failed on floating-point output | Fixed |
| BUG-05 | `optim.grad` | Limitation | `relative=True` test failed because of floating-point rounding | Documented |
| BUG-06 | `io`, `optim` | Style | Indentation and over-long line flagged by `pycodestyle` | Fixed |
| TEST-01 | `tests/` | Test error | Several wrong expectations in newly written tests | Fixed |
| PROC-01 | repo / `main` | Process | Batch 4 tests failed: `io.py` and `utils.py` were missing from `main` | Fixed |
| PROC-02 | git | Process | `git pull` refused: untracked local `utils.py` would be overwritten | Resolved |
| PROC-03 | setup | Process | `pip install` blocked on macOS (externally-managed environment) | Resolved |

## 3. Defect details

### BUG-01 - `topk` returned the wrong width for empty leading dimensions
- **Module:** `numcompute/sort_search.py`
- **Symptom:** `topk(np.empty((0, 5)), 2)[0].shape` was `(0, 0)`; it should be `(0, 2)`.
- **Reproduce:** `topk(np.empty((0, 5)), 2)`
- **Root cause:** the shortcut branch for empty input always built arrays of width 0, ignoring the
  clamped `k`. (`k = 0` and `n = 0` really do give width 0, but zero *rows* with `n > 0` does not.)
- **Fix:** build the empty result with width `k` (already clamped to `n`).
- **Regression test:** `tests/test_sort_search.py::test_topk_noncontiguous_and_empty_inputs`
- **Commit / PR:** `11da362` (first commit of `sort_search.py`, merged in PR #1 `b31cc16`; the fix was already in the code when it was first committed)

### BUG-02 - empty CSV file produced a NumPy warning instead of a clear error
- **Module:** `numcompute/io.py`
- **Symptom:** loading an empty file printed `genfromtxt: Empty input file` as a warning and then
  failed later with an unclear message.
- **Root cause:** `numpy.genfromtxt` signals an empty file with a warning, not an exception.
- **Fix:** the reader catches that specific warning and returns an empty array, so every loader
  can raise `ValueError("no data rows found in ...")` (or yield no chunks when streaming).
- **Regression test:** `tests/test_io.py::test_empty_and_ragged_files`
- **Commit / PR:** `228d08b` (first commit of `io.py`; the fix was already in the code when it was first committed)

### BUG-03 - `one_hot([])` crashed
- **Module:** `numcompute/utils.py`
- **Symptom:** `IndexError: arrays used as indices must be of integer (or boolean) type`
- **Reproduce:** `one_hot([])`
- **Root cause:** an empty list converts to a *float* array, and NumPy refuses float arrays as indices
  even when they are empty.
- **Fix:** cast the validated labels to `intp` before indexing (one line).
- **Regression test:** `tests/test_utils.py::test_one_hot_basic_and_explicit_classes`
- **Commit / PR:** `128ef59` (one-line change to `utils.py`, merged in PR #2 `a34a3a4`). The first version of `utils.py`, in `228d08b` and `11da362`, still had the bug

### BUG-04 - doctest failed on floating-point output
- **Module:** `numcompute/stats.py` (`histogram` docstring)
- **Symptom:** the doctest printed `1.6666666666666665` where the docstring said `1.6666666666666667`.
- **Root cause:** the example used bin edges that are not exactly representable, and the expected
  text had been written by hand.
- **Fix:** changed the example to data with exact bin edges (`[0, 1, 1, 2, 4]`, 2 bins -> edges `[0, 2, 4]`).
- **Regression test:** doctest run (`python -m doctest`-style check over all modules).
- **Commit / PR:** `fcbe3ab` (first commit of `stats.py`; the docstring was corrected before it was committed)

### BUG-05 - `grad(..., relative=True)` test failed (floating-point limitation, not a logic error)
- **Module:** `numcompute/optim.py`
- **Symptom:** a test with `f(x) = x0^2 + x1^2` at `x = [1e6, 1]` returned `0` for the second
  component of the gradient.
- **Root cause:** `f` is about `1e12` there, so the spacing between neighbouring floating-point numbers
  is about `1e-4`. A perturbation of `1e-5` in `x1` changes `f` by less than that and is lost to rounding.
  The code was correct; the test asked for something floating point cannot deliver.
- **Resolution:** the limitation is now stated in the `grad` docstring (use `relative=True` or a
  larger `h`), and the test checks the relative-step option on a single-variable function.
- **Regression test:** `tests/test_optim.py::test_grad_per_coordinate_step_and_relative_step`
- **Commit / PR:** `728a478` (first commit of `optim.py` and `tests/test_optim.py`, merged in PR #4 `60ae707`; the docstring note and corrected test were already included)

### BUG-06 - style findings from `pycodestyle`
- **Modules:** `numcompute/io.py` (E122, continuation line indentation), `numcompute/optim.py` (E501, line 130 characters)
- **Root cause:** a patch to `io.py` left the argument list mis-indented; one error message in
  `optim.py` was written on a single line.
- **Fix:** re-indented the call; split the message across lines.
- **Check:** `python -m pycodestyle --max-line-length=120 numcompute benchmark` reports nothing.
- **Commit / PR:** `io.py` re-indented in `128ef59` (PR #2 `a34a3a4`); the `optim.py` line was split before its first commit `728a478`

### TEST-01 - wrong expectations in newly written tests
These were mistakes in the tests, not in the library. Each was diagnosed by reading the failing
assertion against the data. All were corrected before the test files were first committed
(`fcbe3ab` for `tests/test_stats.py`/`tests/test_metrics.py`, `728a478` for `tests/test_io.py` and `tests/test_optim.py`).

| Test | Wrong expectation | Correct behaviour |
|---|---|---|
| `test_io.py::test_missing_fill_and_skip`, `test_iter_csv_chunks_skip_drops_incomplete_rows` | Expected one surviving row when skipping missing values | Reading only columns 0 and 2, the row `40,,7` is complete, so two rows survive |
| `test_optim.py::test_grad_errors` | `np.log` at a non-positive point inside the test emitted a `RuntimeWarning` | Evaluate it inside `np.errstate`, so the test checks the library's `ValueError` instead |
| `test_optim.py` (relative step) | See BUG-05 | |
| `test_stats.py` / early drafts | `.max()` of an array containing NaN returns NaN | Use `np.nanmax` |
| `test_preprocessing.py::test_minmax_default_range_and_custom_range` | Strict `== 1` on float output failed due to 1-ULP drift (`1.0000000000000002`) | Use `np.isclose` instead of exact float equality |

## 4. Evidence that the tests can detect bugs (mutation checks)

To check the tests are strong, the code was deliberately broken in one place at a time; each time
the suite had to fail, and the original code was then restored and re-verified.

| Module | Deliberate change | Tests that failed |
|---|---|---|
| `rank` | shifted the average-rank formula by +0.5 | 11 |
| `sort_search.topk` | resolved boundary ties to the highest index | 15 |
| `sort_search.binary_search` | used `<` instead of `<=` for the right-hand side | 14 |
| `stats.RunningStats` | dropped a factor from the parallel-merge formula | 4 |
| `metrics.f1` | removed the factor 2 in the numerator | 3 |
| `metrics.roc_curve` | stopped merging tied scores | 1 |
| `optim.grad` | forgot the factor 2 in the central difference | 12 |
| `benchmarking.compare` | inverted the speed-up ratio | 1 |
| `utils.one_hot` | removed the BUG-03 fix | 1 |

## 5. Process incidents

### PROC-01 - batch 4 tests failed on `main` (reported by a teammate)
- **Symptom:** `ModuleNotFoundError: No module named 'numcompute.io'`, raised by
  `tests/test_stats.py::test_running_stats_works_with_chunked_csv`. All other tests passed
  (149 passed, 1 failed). Earlier, the batch 4 tests also failed on `utils` imports.
- **Root cause:** the modules were meant to be merged in dependency order (`utils`, `io`,
  `pipeline`, `preprocessing`, then `sort_search`/`rank`, then `stats`/`metrics`), but batches 1-2
  had not been pushed when batches 3-4 were. `stats.py` imports `utils` and `rank`; one test imports `io`.
- **Diagnosis:** `git ls-tree -r origin/main --name-only` listed the files actually on `main`
  and showed `io.py`, `pipeline.py`, `preprocessing.py` and the later modules missing.
- **Fix:** pushed the missing modules in a branch, merged it, then merged the batch 4 branch.
  Final check: `git checkout main && git pull && python -m pytest tests -q`.
- **Prevention:** later batches carry an instruction that they may only be pushed after the previous
  batch shows as merged on GitHub.
- **Commit / PR:** `11da362` (PR #1) added `utils.py` but not `io.py`, `pipeline.py` or `preprocessing.py`; `fcbe3ab` (stats and metrics) was committed on `feat/stats-metrics`; the missing modules reached `main` in `128ef59` (PR #2 `a34a3a4`), and stats/metrics were merged afterwards in PR #3 `b67f454`. The batch 1 and 2 commits `228d08b` and `5f5f390` sit on the branches `feat/utils-io` and `feat/pipeline-preprocessing`, which were never merged into `main`

### PROC-02 - `git pull` refused to overwrite an untracked file
- **Symptom:** `error: The following untracked working tree files would be overwritten by merge: numcompute/utils.py`
- **Root cause:** a local `utils.py` existed that git did not track, while `main` now contained one.
- **Fix:** moved the local copy to a backup folder, pulled, then replaced it with the newest version.
- **Lesson:** add files by name (`git add numcompute/utils.py ...`), not `git add .`, and keep branches
  in sync with `main` before copying files in.

### PROC-03 - `pip install` blocked on macOS
- **Symptom:** `error: externally-managed-environment`
- **Root cause:** Homebrew's Python refuses to install packages system-wide (PEP 668).
- **Fix:** created a virtual environment (`python3 -m venv ~/venv`, `source ~/venv/bin/activate`)
  and installed `numpy` and `pytest` there.

## 6. What we would do differently
- Merge strictly in dependency order and check `origin/main` before starting the next batch.
- Run the whole suite on `main` after every merge, on at least two operating systems (a teammate
  ran it on Windows; the author's checks ran on Linux and macOS).
- Add a continuous-integration job that runs `pytest` on every pull request, so an out-of-order
  merge fails visibly before it lands on `main`.

## 7. How to add a new entry

Copy this block, fill it in, and add a row to the summary table in section 2.

```markdown
### BUG-NN - one-line title
- **Module:** `numcompute/<file>.py`
- **Symptom:** what you saw (paste the error message).
- **Reproduce:** the smallest snippet that fails.
- **Root cause:** why it happened.
- **Fix:** what changed (not just "fixed").
- **Regression test:** `tests/<file>.py::<test_name>`
- **Commit / PR:** hash from `git log --oneline` or the pull request link
```

Find the commit that fixed something with `git log --oneline -- numcompute/<file>.py`.
---

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
