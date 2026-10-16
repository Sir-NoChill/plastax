"""Plot imprinting-learner runs: predictions vs return, errors, feature counts.

Reads the prediction CSVs `imprinting_learner.py --csv` writes (and their
`.counts.csv` companions) for one stream, and renders into an output
directory:

- `<name>_events.png`: thesis Figure 9.5's panels for that run -- the
  prediction against the return around the last `+1` and the last `-1`
  event of the stream (and the best learnable prediction, dotted);
- `lifetime_error.png`: lifetime error vs the learnable target of every run,
  grouped by label, with the GVF-shape pass rate;
- `<name>_counts.png`: live units by kind and tenure over the run;
- `summary.csv`: one row per run: lifetime error and the shape summary.

Run (matplotlib is not a plastax dependency):

    uv run --with matplotlib python examples/plot_imprinting.py OUT_DIR \\
        --stream stream.npz run_a.csv run_b.csv ...

A run's label is its file stem up to the first `__` (so `imprint__s0.csv`
and `imprint__s1.csv` are two seeds of the label `imprint`).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import imprinting_stream as ims  # noqa: E402

PRED = "#c0392b"
RET = "#2c7fb8"
TARGET = "#555555"


def read_prediction(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """The `(reward, prediction)` columns of a run's CSV."""
    rows = np.loadtxt(path, delimiter=",", skiprows=1)
    return rows[:, 1].astype(np.float32), rows[:, 2].astype(np.float32)


def read_counts(path: Path) -> dict[str, np.ndarray]:
    """The columns of a run's `.counts.csv`, keyed by header."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return {k: np.asarray([float(r[k]) for r in rows]) for k in reader.fieldnames or ()}


def _label(path: Path) -> str:
    return path.stem.split("__")[0]


def plot_events(
    name: str, prediction: np.ndarray, stream: ims.Stream, out: Path, margin: int = 60
) -> None:
    """Figure 9.5's two panels: the last +1 and the last -1 event."""
    ret = ims.discounted_return(stream.reward, stream.gamma)
    target = ims.learnable_return(stream)
    rewarded = [
        ev for ev in stream.events if stream.reward[ev[ims.EVENT_REWARD]] != 0.0
    ]
    picks = []
    for sign in (1.0, -1.0):
        events = [
            ev
            for ev in rewarded
            if np.sign(stream.reward[ev[ims.EVENT_REWARD]]) == sign
        ]
        if events:
            picks.append((sign, events[-1]))
    if not picks:
        return
    fig, axes = plt.subplots(
        1, len(picks), figsize=(5.2 * len(picks), 3.4), squeeze=False
    )
    for ax, (sign, ev) in zip(axes[0], picks, strict=True):
        _, t_on, t_off, t_r = (int(x) for x in ev)
        lo, hi = max(0, t_on - margin), min(prediction.shape[0], t_r + margin)
        t = np.arange(lo, hi)
        ax.plot(t, ret[lo:hi], color=RET, lw=1.0, label="return")
        ax.plot(t, target[lo:hi], color=TARGET, lw=1.0, ls=":", label="best learnable")
        ax.plot(t, prediction[lo:hi], color=PRED, lw=1.2, label="prediction")
        ax.axvspan(t_on, t_off, color="#999999", alpha=0.25, lw=0, label="sound")
        ax.axvline(t_r, color="#333333", lw=0.8, ls="--")
        ax.set_title(f"{name}: reward {sign:+.0f} at step {t_r}")
        ax.set_xlabel("time step")
        ax.set_ylabel("prediction")
        ax.set_ylim(
            min(-1.15, float(prediction[lo:hi].min()) - 0.1) if sign < 0 else -0.25,
            1.15,
        )
        ax.grid(alpha=0.3)
    axes[0][0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"{name}_events.png", dpi=140)
    plt.close(fig)


def plot_counts(name: str, counts: dict[str, np.ndarray], out: Path) -> None:
    """Live units by kind and by tenure status over the run."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4))
    step = counts["step"]
    a.plot(step, counts["pattern"], label="pattern")
    a.plot(step, counts["memory"], label="memory")
    a.plot(step, counts["units"], label="all live units", color="#333333", lw=0.8)
    a.set_xlabel("time step")
    a.set_ylabel("live units")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)
    b.plot(step, counts["tenured"], label="tenured")
    b.plot(step, counts["tenure_track"], label="tenure-track")
    b.plot(step, counts["edges"], label="live edges", color="#333333", lw=0.8)
    b.set_xlabel("time step")
    b.legend(fontsize=8)
    b.grid(alpha=0.3)
    fig.suptitle(f"{name}: features over the run")
    fig.tight_layout()
    fig.savefig(out / f"{name}_counts.png", dpi=140)
    plt.close(fig)


def plot_errors(summary: list[dict[str, object]], out: Path) -> None:
    """Lifetime error per label (mean, standard error over seeds) and pass rate."""
    labels = sorted({str(r["label"]) for r in summary})
    means, errs, passes = [], [], []
    for lab in labels:
        rows = [r for r in summary if r["label"] == lab]
        e = np.asarray([float(r["lifetime_error"]) for r in rows])
        p = np.asarray([float(r["pass"]) for r in rows])
        means.append(e.mean())
        errs.append(e.std(ddof=1) / np.sqrt(e.size) if e.size > 1 else 0.0)
        passes.append(np.nanmean(p))
    fig, ax = plt.subplots(figsize=(max(4.0, 1.4 * len(labels)), 3.4))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=errs, color="#8da0cb", capsize=3)
    for xi, pr in zip(x, passes, strict=True):
        ax.text(
            xi, means[xi] * 1.02, f"pass {pr:.0%}", ha="center", va="bottom", fontsize=8
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("lifetime error (vs learnable target)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "lifetime_error.png", dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    """Render the figures and the summary for a set of runs on one stream."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("out", type=Path)
    p.add_argument("--stream", type=Path, required=True)
    p.add_argument("runs", type=Path, nargs="+")
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    stream = ims.load_npz(a.stream)
    target = ims.learnable_return(stream)
    summary: list[dict[str, object]] = []
    for run in a.runs:
        reward, prediction = read_prediction(run)
        if not np.array_equal(reward, stream.reward[: reward.shape[0]]):
            raise SystemExit(f"{run}: its rewards do not match {a.stream}")
        name = run.stem
        plot_events(name, prediction, stream, a.out)
        counts_path = run.with_suffix(".counts.csv")
        if counts_path.exists():
            plot_counts(name, read_counts(counts_path), a.out)
        _, shape = ims.shape_metrics(prediction, stream)
        row: dict[str, object] = {
            "run": name,
            "label": _label(run),
            "lifetime_error": ims.lifetime_error(prediction, target),
        }
        row.update(shape)
        summary.append(row)
        err = float(row["lifetime_error"])
        print(f"{name}: lifetime error {err:.4f}, pass {shape['pass']:.2f}")
    plot_errors(summary, a.out)
    with open(a.out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)


if __name__ == "__main__":
    main()
