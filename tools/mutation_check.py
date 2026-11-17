"""Mutation check: break the code on purpose and confirm that the unit tests notice.

For every entry in ``MUTATIONS`` one line of ``numcompute_stream`` is changed in a temporary copy of the
repository and the streaming test-suite is run. A mutation is *killed* if at least one test fails.

Run from the repository root::

    python tools/mutation_check.py            # writes tools/mutation_results.md

The real source files are never modified.
"""

from __future__ import annotations

import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]

# (file, original text, mutated text, what the mutation simulates)
MUTATIONS = [
    (
        "tree.py",
        "np.where(total[..., 0] > 0, 1.0 - np.sum(p * p, axis=-1), 0.0)",
        "1.0 - np.sum(p * p, axis=-1)",
        "Gini of an empty node is 1 instead of 0",
    ),
    (
        "tree.py",
        "cost <= cmin + _TIE_TOL)[0])",
        "cost <= cmin + _TIE_TOL)[-1])",
        "ties resolved to the highest threshold",
    ),
    (
        "tree.py",
        "# lowest threshold among ties\n        if best is None or cmin < best[2] - _TIE_TOL:",
        "# lowest threshold among ties\n        if best is None or cmin <= best[2] + _TIE_TOL:",
        "ties between features resolved to the last feature",
    ),
    (
        "tree.py",
        "if self._buf_n >= self.buffer_size:",
        "if self._buf_n > self.buffer_size:",
        "off-by-one: retrain only when buffer strictly exceeds buffer_size",
    ),
    (
        "tree.py",
        "self._buf_X, self._buf_y, self._buf_n = [], [], 0",
        "self._buf_n = 0",
        "buffer content is not cleared after retraining",
    ),
    (
        "tree.py",
        "return np.where(np.isnan(Xa), self.nan_fill_, Xa)",
        "return np.where(np.isnan(Xa), 0.0, Xa)",
        "NaN at prediction time filled with 0 instead of the training median",
    ),
    ("tree.py", "thr = 0.5 * lo + 0.5 * hi", "thr = lo + (hi - lo) / 2.0", "threshold midpoint overflows near +-1e308"),
    (
        "tree.py",
        "proba = self.predict_proba(X)  # raises NotFittedError first if nothing is trained\n        return self.classes_",
        "return self.classes_[np.argmax(self.predict_proba(X), axis=1)]\n        return self.classes_",
        "predict touches classes_ before the readiness check (S-BUG-02)",
    ),
    (
        "metrics.py",
        "return self._div(tp, tp + fp, self.zero_division)",
        "return self._div(tp, tp + fn, self.zero_division)",
        "precision uses the recall denominator",
    ),
    (
        "metrics.py",
        "return self._div(2 * tp, 2 * tp + fp + fn, self.zero_division)",
        "return self._div(tp, 2 * tp + fp + fn, self.zero_division)",
        "F1 numerator loses its factor 2",
    ),
    (
        "metrics.py",
        "self._recent: Deque[Tuple[np.ndarray, np.ndarray]] = deque(maxlen=self.window)",
        "self._recent: Deque[Tuple[np.ndarray, np.ndarray]] = deque()",
        "rolling window never forgets old chunks",
    ),
    (
        "metrics.py",
        "grown[np.ix_(pos, pos)] = self._cm",
        "pass",
        "confusion matrix forgets old counts when a new label appears",
    ),
    (
        "stats.py",
        "centred = arr - self._shift",
        "centred = arr",
        "shifted-data update disabled (cancellation with large offsets)",
    ),
    ("stats.py", "keep = j < cap", "keep = j < cap // 2", "reservoir sampler keeps too few replacements"),
    (
        "stats.py",
        "bins = np.clip(pos, 0, b - 1).astype(np.int64)",
        "bins = np.clip(pos, 0, b - 2).astype(np.int64)",
        "histogram edge bin is never filled",
    ),
    (
        "preprocessing.py",
        "scale = np.where(scale <= 10 * _EPS * np.abs(mean), 1.0, scale)",
        "scale = np.where(scale <= 1e-12, 1.0, scale)",
        "absolute threshold flattens tiny but real spread",
    ),
    (
        "preprocessing.py",
        '        self._check_range()\n        X = _prepare_chunk(X)\n        _check_width(self, X)\n        if X.shape[0] == 0 and "_running"',
        '        X = _prepare_chunk(X)\n        _check_width(self, X)\n        if X.shape[0] == 0 and "_running"',
        "infinite feature_range accepted",
    ),
    (
        "pipeline.py",
        "            _call(step.partial_fit, Xt, yt)\n",
        "            step.partial_fit(Xt, yt)\n",
        "pipeline always passes y to partial_fit (breaks minimal transformers)",
    ),
    (
        "stream.py",
        "except NotFittedError:",
        "except KeyError:",
        "unfitted model crashes the trainer instead of being logged",
    ),
    ("utils.py", "if np.isinf(Xa).any():", "if False:", "infinite values accepted by chunk validation"),
]


def run_tests(cwd: pathlib.Path) -> tuple[bool, str]:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-x",
            "-q",
            "-p",
            "no:cacheprovider",
            "tests/test_stream_tree.py",
            "tests/test_stream_stats.py",
            "tests/test_stream_metrics.py",
            "tests/test_stream_preprocessing.py",
            "tests/test_stream_pipeline_trainer.py",
            "tests/test_stream_io_datasets_utils.py",
            "tests/test_stream_numerics.py",
        ],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    out = proc.stdout + proc.stderr
    m = re.search(r"FAILED (\S+)", out)
    return proc.returncode != 0, (m.group(1).split("::", 1)[-1] if m else "")


def main() -> None:
    rows, survived = [], []
    with tempfile.TemporaryDirectory() as tmp:
        work = pathlib.Path(tmp)
        for name in ("numcompute", "numcompute_stream", "tests"):
            shutil.copytree(ROOT / name, work / name, ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy(ROOT / "pyproject.toml", work)
        baseline_failed, _ = run_tests(work)
        if baseline_failed:
            sys.exit("baseline test-suite already fails; fix it before running mutation checks")
        for i, (fname, old, new, why) in enumerate(MUTATIONS, 1):
            path = work / "numcompute_stream" / fname
            original = path.read_text(encoding="utf-8")
            if original.count(old) != 1:
                sys.exit(f"mutation {i}: pattern not found exactly once in {fname}: {old[:50]!r}")
            path.write_text(original.replace(old, new), encoding="utf-8")
            try:
                killed, test = run_tests(work)
            finally:
                path.write_text(original, encoding="utf-8")
            rows.append((i, fname, why, "killed" if killed else "SURVIVED", test))
            if not killed:
                survived.append(i)
            print(f"{i:>2} {'killed  ' if killed else 'SURVIVED'} {fname:<17} {why}")
    lines = [
        "# Mutation check results",
        "",
        f"{len(rows) - len(survived)} of {len(rows)} deliberate defects were detected by the streaming tests.",
        "",
        "| # | File | Injected defect | Result | First failing test |",
        "|---:|---|---|---|---|",
    ]
    lines += [f"| {i} | `{f}` | {w} | {r} | `{t}` |" for i, f, w, r, t in rows]
    (ROOT / "tools" / "mutation_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if survived:
        sys.exit(f"survivors: {survived}")


if __name__ == "__main__":
    main()
