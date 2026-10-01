"""Figures for docs/scale_plan/RESULTS.md (all numbers measured on cdol01, RTX
5000 Ada, 2026-09-30/10-01; sources noted per series).

    uv run --with matplotlib python examples/benchmarks/plot_scale_results.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "docs" / "scale_plan"
# One categorical order, reused across figures: plastax, C++ in place, C++
# rebuild (tuned CSR / append+resort).
BLUE, ORANGE, GREY = "#4C72B0", "#DD8452", "#8C8C8C"


def _style(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.8)
    ax.set_axisbelow(True)


def churn_step() -> None:
    """Synthetic churn step: churn_probe --grow propose --align 256 (plastax,
    407ccb1) vs plastix-synth-bench REEVAL_RESULTS.md (C++)."""
    sizes = ["5.4M", "50M", "300M"]
    series = [
        ("plastax (proposals, in place)", [0.259, 2.42, 13.7], BLUE),
        ("C++ Plastix, in place", [0.24, 2.19, 12.9], ORANGE),
        ("C++ tuned CSR, rebuild", [0.716, 11.1, 65.5], GREY),
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.26
    for i, (label, vals, color) in enumerate(series):
        xs = [j + (i - 1) * width for j in range(len(sizes))]
        bars = ax.bar(xs, vals, width - 0.03, color=color, label=label)
        for bar, v in zip(bars, vals, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v * 1.08,
                f"{v:g}",
                ha="center",
                fontsize=8,
            )
    ax.set_yscale("log")
    ax.set_xticks(range(len(sizes)), [f"{s} edges" for s in sizes])
    ax.set_ylabel("ms per churn step (log)")
    ax.set_title("One churn step (forward + prune + regrow 64/level)", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    plt.tight_layout()
    plt.savefig(OUT / "churn_step_sizes.png", dpi=150)
    plt.close(fig)


def batched_layouts() -> None:
    """Batched SGD training, 3 layers, 5.4M edges (.bench/batched_perf.py at
    09fd6d4), ms per sample by layout."""
    bs = [1, 2, 4, 8, 16, 32, 64, 128]
    edge = [0.537, 0.440, 0.405, 0.350, 0.318, 0.294, None, 0.252]
    pallas = [None, 0.386, 0.231, 0.148, 0.116, 0.131, 0.156, 0.489]
    csr = [None, 1.204, 0.651, 0.352, 0.206, 0.128, 0.090, 0.077]
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, vals, color, marker in (
        ("edge list (per-sample vmap)", edge, GREY, "o"),
        ("Pallas edge-once kernel", pallas, BLUE, "s"),
        ("CSR + cuSPARSE (per-step rebuild)", csr, ORANGE, "^"),
    ):
        pts = [(b, v) for b, v in zip(bs, vals, strict=True) if v is not None]
        ax.plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            color=color,
            marker=marker,
            markersize=6,
            linewidth=2,
            label=label,
        )
    ax.axvspan(2, 32, color=BLUE, alpha=0.06)
    ax.text(5.5, 1.1, 'layout="auto": Pallas', fontsize=8, color="#333333")
    ax.text(40, 1.1, "CSR", fontsize=8, color="#333333")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(bs, [str(b) for b in bs])
    ax.set_xlabel("batch size B")
    ax.set_ylabel("ms per sample (log)")
    ax.set_title("Batched training step, 5.4M edges", fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    _style(ax)
    plt.tight_layout()
    plt.savefig(OUT / "batched_layouts.png", dpi=150)
    plt.close(fig)


def deepr() -> None:
    """DEEP R, real MultiMNIST, 300 steps: deepr_scale.py --hash-noise
    (plastax) vs main_deepr.cpp (C++ in place / append+resort)."""
    sizes = ["1M hidden\n65M edges", "4.4M hidden\n286M edges"]
    series = [
        ("plastax (proposals, hash noise)", [11.3, 53.6], BLUE),
        ("C++ Plastix, in place", [7.4, 30.9], ORANGE),
        ("C++ Plastix, append + resort", [35.3, 156.6], GREY),
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.26
    for i, (label, vals, color) in enumerate(series):
        xs = [j + (i - 1) * width for j in range(len(sizes))]
        bars = ax.bar(xs, vals, width - 0.03, color=color, label=label)
        for bar, v in zip(bars, vals, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 2,
                f"{v:g}",
                ha="center",
                fontsize=8,
            )
    ax.set_xticks(range(len(sizes)), sizes)
    ax.set_ylabel("ms per training step")
    ax.set_title(
        "DEEP R on the MultiMNIST stream (state: plastax 7.3 GB vs C++ "
        "29.4 GB at 286M edges)",
        fontsize=10,
    )
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    plt.tight_layout()
    plt.savefig(OUT / "deepr_scale.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    churn_step()
    batched_layouts()
    deepr()
