"""Plot the px vs cx GPU gap analysis (benchmarks/gpu_gap_analysis.md).

Reads the CSVs in DIR (default ``benchmarks/results/gpu_gap``) and writes,
into DIR:

- gap_ratio.png: px/cx time ratio per growth point, with the current XLA
  4-key sort and with the 3-pass stable radix sort experiment.
- gap_components.png: the px - cx gap per growth point, split into its
  causes (sort on the device, sort-driven host launch cost, other device
  work, other host exposure).
- kernel_classes.png: device time per kernel class, px vs cx, at two points.
- sort_micro.png: XLA's 4-key sort vs CUB radix formulations vs n.
- warmup.png: per-call time over the first calls of a fresh process.
- synth_step.png: the plastix-synth-bench E5M step, px vs cx, by component.

Run it with matplotlib and pandas on hand, e.g.:

    uv run --with matplotlib --with pandas \\
        python examples/benchmarks/plot_gpu_gap.py benchmarks/results/gpu_gap
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Categorical slots in fixed order, plus a neutral for "matched" time.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
)
GRAY = "#9a9a94"
INK = "#333333"


def style(ax: plt.Axes) -> None:
    """Recessive axes: no top/right spines, light grid behind the marks.

    Args:
        ax: the axes.
    """
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e4e4e0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK)


def label(row: pd.Series) -> str:
    """A short point label.

    Args:
        row: one growth_points.csv row.

    Returns:
        e.g. "per_unit\\n262K cand".
    """
    c = int(row["candidates"])
    size = (
        f"{c / 2**20:g}M" if c >= 2**20 else (f"{c // 1024}K" if c >= 1024 else str(c))
    )
    return f"{row['strategy']}\n{size} cand"


def plot_ratio(pts: pd.DataFrame, out: Path) -> None:
    """px/cx ratio per point, before and after the radix-sort experiment.

    Args:
        pts: growth_points.csv.
        out: output directory.
    """
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = range(len(pts))
    before = pts["px_total4_synced_ms"] / pts["cx_event_ms"]
    after = pts["px_lsd3_synced_ms"] / pts["cx_event_ms"]
    w = 0.38
    ax.bar(
        [i - w / 2 for i in x],
        before,
        w * 0.95,
        color=BLUE,
        label="px today (XLA 4-key sort)",
    )
    ax.bar(
        [i + w / 2 for i in x],
        after,
        w * 0.95,
        color=ORANGE,
        label="px with 3-pass CUB radix sort",
    )
    ax.axhline(1.0, color=INK, linewidth=1)
    ax.set_yscale("log")
    ax.set_xticks(list(x), [label(r) for _, r in pts.iterrows()], fontsize=8)
    ax.set_ylabel("px time / cx time (log)")
    ax.set_title("Growth call, px vs cx on the GPU (above 1: px slower)", loc="left")
    for i, (b, a) in enumerate(zip(before, after, strict=True)):
        ax.text(i - w / 2, b * 1.06, f"{b:.2f}", ha="center", fontsize=7, color=INK)
        ax.text(i + w / 2, a * 1.06, f"{a:.2f}", ha="center", fontsize=7, color=INK)
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "gap_ratio.png", dpi=150)
    plt.close(fig)


def components(row: pd.Series) -> dict[str, float]:
    """Split one point's px - cx gap (ms) into its causes.

    Device time is px's pipelined per-call time and cx's nsys busy time;
    host exposure is the synced (px) or event (cx) time minus that.

    Args:
        row: one growth_points.csv row with nsys columns.

    Returns:
        The four components, summing to the gap.
    """
    t, l_ = row["px_total4_synced_ms"], row["px_lsd3_synced_ms"]
    pt, pl = row["px_total4_pipelined_ms"], row["px_lsd3_pipelined_ms"]
    c, b = row["cx_event_ms"], row["cx_nsys_busy_ms"]
    return {
        "sort: device (XLA sort vs radix)": pt - pl,
        "sort: host (graph launch of the sort nodes)": (t - pt) - (l_ - pl),
        "other device work": pl - b,
        "other host exposure (dispatch, sync)": (l_ - pl) - (c - b),
    }


def plot_components(pts: pd.DataFrame, out: Path) -> None:
    """Gap components as a share of each point's gap.

    Args:
        pts: growth_points.csv.
        out: output directory.
    """
    sel = pts[
        pts["cx_nsys_busy_ms"].notna()
        & pts["px_lsd3_pipelined_ms"].notna()
        & (pts["px_total4_synced_ms"] > pts["cx_event_ms"])
    ].reset_index(drop=True)
    colors = [BLUE, AQUA, ORANGE, YELLOW]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for i, (_, row) in enumerate(sel.iterrows()):
        comp = components(row)
        gap = row["px_total4_synced_ms"] - row["cx_event_ms"]
        pos = neg = 0.0
        for (name, v), col in zip(comp.items(), colors, strict=True):
            share = 100 * v / gap
            base = pos if share >= 0 else neg
            ax.bar(
                i,
                share,
                0.6,
                bottom=base,
                color=col,
                edgecolor="white",
                linewidth=1.5,
                label=name if i == 0 else None,
            )
            if share >= 0:
                pos += share
            else:
                neg += share
        ax.text(
            i,
            max(pos, 100) + 3,
            f"gap {gap:.3g} ms",
            ha="center",
            fontsize=7,
            color=INK,
        )
    ax.axhline(0, color=INK, linewidth=1)
    ax.set_xticks(range(len(sel)), [label(r) for _, r in sel.iterrows()], fontsize=8)
    ax.set_ylabel("share of the px - cx gap (%)")
    ax.set_title("Where the growth gap goes (negative: px ahead of cx)", loc="left")
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower left", ncol=2)
    fig.tight_layout()
    fig.savefig(out / "gap_components.png", dpi=150)
    plt.close(fig)


def plot_kernel_classes(kc: pd.DataFrame, out: Path) -> None:
    """Device time per kernel class at two per_unit points.

    Args:
        kc: kernel_classes.csv.
        out: output directory.
    """
    impls = [
        ("px_total4", "px today", BLUE),
        ("px_lsd3", "px + radix sort", ORANGE),
        ("cx", "cx", AQUA),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, cand in zip(axes, (262144, 4194304), strict=True):
        sub = kc[(kc["strategy"] == "per_unit") & (kc["candidates"] == cand)]
        classes = (
            sub.groupby("kernel_class")["us"]
            .max()
            .sort_values(ascending=False)
            .index.tolist()
        )
        h = 0.27
        for j, (impl, name, col) in enumerate(impls):
            d = sub[sub["impl"] == impl].set_index("kernel_class")["us"]
            ax.barh(
                [i + (j - 1) * h for i in range(len(classes))],
                [d.get(c, 0.0) for c in classes],
                h * 0.92,
                color=col,
                label=name,
            )
        ax.set_yticks(range(len(classes)), classes, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("device time per call (us)")
        ax.set_title(f"per_unit, {cand // 1024}K candidates", loc="left", fontsize=10)
        style(ax)
        ax.grid(axis="x", color="#e4e4e0", linewidth=0.8)
        ax.grid(axis="y", visible=False)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "kernel_classes.png", dpi=150)
    plt.close(fig)


def plot_sort(sm: pd.DataFrame, out: Path) -> None:
    """Sort time vs n for the three formulations.

    Args:
        sm: sort_micro.csv.
        out: output directory.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for col, name, color in (
        ("xla_4key_sort_ms", "XLA sort, 4 keys (px today)", BLUE),
        ("cub_3pass_lsd_ms", "3 stable CUB radix passes (same order)", ORANGE),
        ("cub_1key_ms", "1 CUB radix pass (score only, lower bound)", AQUA),
    ):
        ax.plot(
            sm["n"],
            sm[col],
            marker="o",
            markersize=5,
            linewidth=2,
            color=color,
            label=name,
        )
    # 4 keys x 4 B x n = 64 MB of L2 at n = 4M.
    ax.axvline(4 * 2**20, color=GRAY, linewidth=1, linestyle="--")
    ax.text(
        4 * 2**20 * 1.08,
        sm["cub_1key_ms"].min(),
        "4 keys x 4 B x n = 64 MB (L2)",
        fontsize=7,
        color=INK,
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("candidates n")
    ax.set_ylabel("ms per sort (pipelined)")
    ax.set_title("Sorting the candidate list on the GPU", loc="left")
    style(ax)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "sort_micro.png", dpi=150)
    plt.close(fig)


def plot_warmup(wu: pd.DataFrame, out: Path) -> None:
    """Per-call time over a fresh process's first calls.

    Args:
        wu: warmup_calls.csv.
        out: output directory.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.plot(
        wu["call"],
        wu["px_per_unit_65536_ms"],
        color=BLUE,
        linewidth=2,
        marker="o",
        markersize=4,
        label="px per_unit, 262K candidates",
    )
    ax.plot(
        wu["call"],
        wu["px_global_ms"],
        color=ORANGE,
        linewidth=2,
        marker="o",
        markersize=4,
        label="px global, P=4",
    )
    ax.plot(
        wu["call"],
        wu["cx_per_unit_65536_ms"],
        color=AQUA,
        linewidth=2,
        marker="o",
        markersize=4,
        label="cx per_unit, 262K candidates",
    )
    ax.axvspan(2.5, 9.5, color="#f0f0ec", zorder=0)
    ax.text(
        6,
        1.25,
        "calls the growth bench times\n(2 warm-up, 7 timed)",
        ha="center",
        fontsize=7,
        color=INK,
    )
    ax.set_ylim(0, 1.5)
    ax.set_xlabel("call number in a fresh process")
    ax.set_ylabel("ms per call (synced)")
    ax.set_title("px needs ~15 calls to reach steady state; cx needs 1", loc="left")
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "warmup.png", dpi=150)
    plt.close(fig)


def plot_synth(ss: pd.DataFrame, out: Path) -> None:
    """The synth-bench E5M step by component, px vs cx.

    Args:
        ss: synth_step.csv.
        out: output directory.
    """
    fig, ax = plt.subplots(figsize=(9, 3.2))
    palette = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]
    for y, impl in enumerate(("cx", "px")):
        left = 0.0
        sub = ss[ss["impl"] == impl]
        for k, (_, r) in enumerate(sub.iterrows()):
            col = GRAY if r["component"].startswith("host") else palette[k]
            ax.barh(
                y,
                r["us_per_step"],
                0.55,
                left=left,
                color=col,
                edgecolor="white",
                linewidth=1.5,
            )
            if r["us_per_step"] > 30:
                short = r["component"].split(" (")[0].split(": ")[-1]
                ax.text(
                    left + r["us_per_step"] / 2,
                    y,
                    f"{short}\n{r['us_per_step']:.0f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if col != YELLOW else INK,
                )
            left += r["us_per_step"]
        ax.text(left + 8, y, f"{left:.0f} us", va="center", fontsize=8, color=INK)
    ax.set_yticks([0, 1], ["cx", "px"])
    ax.set_xlabel("us per step (forward + prune + generate)")
    ax.set_title("synth-bench E5M, s = 0.99, k = 64: one churn step", loc="left")
    style(ax)
    ax.grid(axis="x", color="#e4e4e0", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / "synth_step.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "dir", type=Path, nargs="?", default=Path("benchmarks/results/gpu_gap")
    )
    args = ap.parse_args()
    d: Path = args.dir
    pts = pd.read_csv(d / "growth_points.csv")
    plot_ratio(pts, d)
    plot_components(pts, d)
    plot_kernel_classes(pd.read_csv(d / "kernel_classes.csv"), d)
    plot_sort(pd.read_csv(d / "sort_micro.csv"), d)
    plot_warmup(pd.read_csv(d / "warmup_calls.csv"), d)
    plot_synth(pd.read_csv(d / "synth_step.csv"), d)


if __name__ == "__main__":
    main()
