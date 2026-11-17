"""Benchmarks for NumCompute-Stream.

Part A - loop-based vs vectorised implementations (split search, full tree fit,
         tree prediction, streaming confusion-matrix update).
Part B - effect of ``buffer_size`` on accuracy and retraining time in a
         streaming scenario (stationary stream and a stream with concept drift).

Run from the repository root::

    python benchmark/run_stream_benchmarks.py            # full run (~1-2 minutes)
    python benchmark/run_stream_benchmarks.py --quick    # small sizes, seconds

Writes ``benchmark/stream_results.md``, ``benchmark/stream_buffer_sweep.csv`` and
figures into ``benchmark/figures/``. Data come from fixed seeds, so every run
uses identical inputs (timings still vary with the machine).
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from numcompute.benchmarking import compare, environment_info, format_table  # noqa: E402
from numcompute_stream.datasets import make_drift_stream  # noqa: E402
from numcompute_stream.io import iter_array_chunks  # noqa: E402
from numcompute_stream.metrics import confusion_counts  # noqa: E402
from numcompute_stream.stream import StreamTrainer  # noqa: E402
from numcompute_stream.tree import DecisionTreeClassifier, best_split_loop, best_split_vectorised  # noqa: E402

FIG_DIR = ROOT / "benchmark" / "figures"


# ----------------------------------------------------------------- Part A helpers
def _data(n: int, rng: np.random.Generator):
    X, y = make_drift_stream(n, drift_at=None, label_noise=0.02, seed=int(rng.integers(1_000_000)))
    return X, y


def split_args(n, rng):
    X, y = _data(n, rng)
    return X, y, np.arange(n)


def split_loop(X, y, idx):
    j, t, c = best_split_loop(X, y, 3, idx, range(X.shape[1]), "gini", 1)
    return np.array([j, t, c])


def split_vec(X, y, idx):
    j, t, c = best_split_vectorised(X, np.eye(3)[y], idx, range(X.shape[1]), "gini", 1)
    return np.array([j, t, c])


def fit_args(n, rng):
    return _data(n, rng)


def fit_loop(X, y):
    return DecisionTreeClassifier(max_depth=6, vectorised=False).fit(X, y).predict(X[:50])


def fit_vec(X, y):
    return DecisionTreeClassifier(max_depth=6, vectorised=True).fit(X, y).predict(X[:50])


_TREE = None


def predict_args(n, rng):
    global _TREE
    X, y = _data(4000, np.random.default_rng(1))
    _TREE = DecisionTreeClassifier(max_depth=8).fit(X, y)
    Xq, _ = make_drift_stream(n, drift_at=None, seed=int(rng.integers(1_000_000)))
    return (Xq,)


def predict_loop(Xq):
    """Walk the tree sample by sample in pure Python."""
    t = _TREE.tree_
    out = np.empty(Xq.shape[0], dtype=_TREE.classes_.dtype)
    for i in range(Xq.shape[0]):
        node = 0
        while t["feature"][node] >= 0:
            node = t["left"][node] if Xq[i, t["feature"][node]] <= t["threshold"][node] else t["right"][node]
        out[i] = _TREE.classes_[int(np.argmax(t["value"][node]))]
    return out


def predict_vec(Xq):
    return _TREE.predict(Xq)


def cm_args(n, rng):
    return rng.integers(0, 5, n), rng.integers(0, 5, n)


def cm_loop(yt, yp):
    cm = np.zeros((5, 5), dtype=np.int64)
    for a, b in zip(yt, yp):
        cm[a, b] += 1
    return cm


def cm_vec(yt, yp):
    return confusion_counts(yt, yp, np.arange(5))


# ----------------------------------------------------------------- Part B
def run_stream(buffer_size: int, X, y, chunk: int, depth: int) -> Dict[str, float]:
    tr = StreamTrainer(DecisionTreeClassifier(max_depth=depth, buffer_size=buffer_size))
    tr.run(iter_array_chunks(X, y, chunk))
    s = tr.summary()
    est = tr.model
    return {
        "accuracy": s["final_cumulative_accuracy"],
        "retrains": s["n_retrains"],
        "retrain_total_s": float(sum(est.retrain_times_)),
        "retrain_mean_s": float(np.mean(est.retrain_times_)) if est.retrain_times_ else float("nan"),
        "fit_total_s": s["total_fit_time"],
        "curve": tr.metric_history("chunk_accuracy"),
    }


def sweep(sizes, n_samples, chunk, depth, seeds, drift_at):
    rows, curves = [], {}
    for b in sizes:
        acc, rt, rtm, ft, nr = [], [], [], [], []
        for sd in seeds:
            X, y = make_drift_stream(n_samples, drift_at=drift_at, label_noise=0.03, seed=sd)
            r = run_stream(b, X, y, chunk, depth)
            acc.append(r["accuracy"])
            rt.append(r["retrain_total_s"])
            rtm.append(r["retrain_mean_s"])
            ft.append(r["fit_total_s"])
            nr.append(r["retrains"])
            if sd == seeds[0]:
                curves[b] = r["curve"]
        rows.append(
            {
                "buffer_size": b,
                "accuracy": float(np.mean(acc)),
                "acc_std": float(np.std(acc)),
                "retrains": float(np.mean(nr)),
                "retrain_total_s": float(np.mean(rt)),
                "retrain_mean_s": float(np.mean(rtm)),
                "fit_total_s": float(np.mean(ft)),
            }
        )
    return rows, curves


def sweep_table(rows, title):
    lines = [
        f"**{title}**",
        "",
        "| buffer_size | Accuracy (mean +/- sd) | Retrains | Mean retrain time | Total retrain time | Total fit time |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['buffer_size']:,} | {r['accuracy']:.3f} +/- {r['acc_std']:.3f} | {r['retrains']:.0f} | "
            f"{r['retrain_mean_s'] * 1e3:.2f} ms | {r['retrain_total_s']:.3f} s | {r['fit_total_s']:.3f} s |"
        )
    return "\n".join(lines)


def plot_sweep(rows_stat, rows_drift, curves_drift, out_dir: pathlib.Path) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for rows, lab in ((rows_stat, "stationary"), (rows_drift, "concept drift")):
        b = [r["buffer_size"] for r in rows]
        ax[0].errorbar(
            b, [r["accuracy"] for r in rows], yerr=[r["acc_std"] for r in rows], marker="o", label=lab, capsize=3
        )
    ax[0].set(xscale="log", xlabel="buffer_size", ylabel="Prequential accuracy", title="Accuracy vs buffer_size")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    b = [r["buffer_size"] for r in rows_stat]
    ax[1].plot(b, [r["retrain_mean_s"] * 1e3 for r in rows_stat], marker="o", label="mean time per retrain (ms)")
    ax2 = ax[1].twinx()
    ax2.plot(b, [r["retrain_total_s"] for r in rows_stat], marker="s", color="tab:red", label="total retrain time (s)")
    ax[1].set(xscale="log", xlabel="buffer_size", ylabel="ms per retrain", title="Retraining cost vs buffer_size")
    ax2.set_ylabel("total seconds")
    ax[1].grid(alpha=0.3)
    h1, l1 = ax[1].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax[1].legend(h1 + h2, l1 + l2, fontsize=8)
    fig.tight_layout()
    p = out_dir / "buffer_size_tradeoff.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    files.append(p.name)

    fig, ax = plt.subplots(figsize=(8, 4))
    for bsz, c in curves_drift.items():
        k = 5
        sm = np.convolve(np.nan_to_num(c, nan=np.nanmean(c)), np.ones(k) / k, mode="same")
        ax.plot(sm, label=f"buffer_size={bsz}")
    ax.set(
        xlabel="Chunk",
        ylabel="Chunk accuracy (5-chunk moving average)",
        title="Streaming accuracy with concept drift (drift at 50%)",
    )
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out_dir / "drift_curves.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    files.append(p.name)
    return files


def plot_loop_vs_vec(tables: Dict[str, list], out_dir: pathlib.Path) -> str:
    fig, axes = plt.subplots(1, len(tables), figsize=(4.2 * len(tables), 3.8))
    for ax, (name, rows) in zip(np.atleast_1d(axes), tables.items()):
        s = [r["size"] for r in rows]
        ax.loglog(s, [r["loop_s"] for r in rows], marker="o", label="loop")
        ax.loglog(s, [r["vec_s"] for r in rows], marker="s", label="vectorised")
        ax.set(title=name, xlabel="size", ylabel="seconds")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.tight_layout()
    p = out_dir / "loop_vs_vectorised.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p.name


# ----------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small sizes for a fast smoke run")
    args = ap.parse_args()
    q = args.quick
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    md: List[str] = ["# NumCompute-Stream benchmark results", "", "## Environment", ""]
    env = environment_info()
    md += [f"- {k}: {v}" for k, v in env.items()]
    md += ["- seed: fixed per benchmark", f"- mode: {'quick' if q else 'full'}", ""]

    # ---- Part A
    md += ["## A. Loop-based vs vectorised", ""]
    cfg = [
        (
            "Best-split search (one node, 6 features)",
            split_loop,
            split_vec,
            split_args,
            [100, 200, 400] if q else [200, 500, 1000, 2000],
            2,
        ),
        ("Full tree fit (max_depth=6)", fit_loop, fit_vec, fit_args, [100, 200] if q else [200, 400, 800, 1600], 2),
        (
            "Tree prediction (per-sample walk vs level-wise)",
            predict_loop,
            predict_vec,
            predict_args,
            [500, 2000] if q else [1000, 10000, 50000],
            3,
        ),
        (
            "Streaming confusion-matrix update",
            cm_loop,
            cm_vec,
            cm_args,
            [1000, 10000] if q else [1000, 10000, 100000],
            3,
        ),
    ]
    tables = {}
    for title, lf, vf, mk, sizes, rep in cfg:
        print(f"[A] {title} ...", flush=True)
        rows = compare(lf, vf, mk, sizes, repeats=rep, seed=0)
        tables[title.split(" (")[0]] = rows
        md += [format_table(rows, title), ""]
    img = plot_loop_vs_vec(tables, FIG_DIR)
    md += [f"![loop vs vectorised](figures/{img})", ""]

    # ---- Part B
    print("[B] buffer_size sweep ...", flush=True)
    sizes = [100, 500, 2000] if q else [100, 250, 500, 1000, 2000, 4000]
    n, chunk, depth = (4000, 100, 6) if q else (20000, 100, 8)
    seeds = [0, 1] if q else [0, 1, 2]
    rows_stat, _ = sweep(sizes, n, chunk, depth, seeds, drift_at=None)
    rows_drift, curves = sweep(sizes, n, chunk, depth, seeds, drift_at=0.5)
    md += [
        "## B. Effect of `buffer_size` (streaming, prequential evaluation)",
        "",
        f"Setup: {n:,} samples, chunks of {chunk}, `max_depth={depth}`, mean over seeds {seeds}. "
        "Accuracy is cumulative test-then-train accuracy (chunks before the first retrain are not scored).",
        "",
        sweep_table(rows_stat, "Stationary stream"),
        "",
        sweep_table(rows_drift, "Stream with concept drift at 50%"),
        "",
    ]
    for f in plot_sweep(rows_stat, rows_drift, curves, FIG_DIR):
        md += [f"![{f}](figures/{f})", ""]
    with open(ROOT / "benchmark" / "stream_buffer_sweep.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["scenario"] + list(rows_stat[0]))
        w.writeheader()
        for scen, rows in (("stationary", rows_stat), ("drift", rows_drift)):
            for r in rows:
                w.writerow({"scenario": scen, **r})
    (ROOT / "benchmark" / "stream_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
