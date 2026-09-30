"""Plot docs/scale_plan/step_300M.png: one churn step at 300M edges, by phase.

The C++ rows are from plastix-synth-bench (REEVAL_RESULTS.md, in-place
section); the plastax rows are churn_probe.py at --width 387298 --edges
300000000 on an RTX 5000 Ada (2026-09-30), at a145691 (grid growth) and at
9e6fbea (--grow propose, source-major buckets). The "projected" row is the
plan's estimate for P2 (exact-headroom capacities), not a measurement.

    uv run --with matplotlib python examples/benchmarks/plot_step_300M.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# (label, forward, prune, structure update) in ms per step.
ROWS = [
    ("C++ tuned CSR\n(rebuild)", 4.98, 5.13, 55.4),
    ("C++ Plastix\nin place", 7.69, 5.13, 0.01),
    ("plastax a145691\n(grid growth)", 29.7, 9.8, 104.9),
    ("plastax 9e6fbea\n(proposals, source-major)", 9.66, 10.23, 8.01),
    ("plastax + P2\n(projected)", 5.5, 5.8, 1.5),
]
COLORS = ("#4C72B0", "#DD8452", "#8C8C8C")
PHASES = ("forward", "prune", "structure update")
OUT = Path(__file__).resolve().parents[2] / "docs" / "scale_plan" / "step_300M.png"


def main() -> None:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for i, (_, *costs) in enumerate(ROWS):
        left = 0.0
        for cost, color, phase in zip(costs, COLORS, PHASES, strict=True):
            ax.barh(
                i,
                cost,
                left=left,
                color=color,
                edgecolor="white",
                linewidth=1.5,
                label=phase if i == 0 else None,
            )
            left += cost
        ax.text(left + 1.5, i, f"{left:.1f} ms", va="center", fontsize=9)
    ax.set_yticks(range(len(ROWS)), [r[0] for r in ROWS])
    ax.invert_yaxis()
    ax.set_xlabel("ms per step, E = 300M live edges, k = 64 churned per level")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(
        "One churn step at 300M edges (projected rows are estimates)", fontsize=10
    )
    plt.tight_layout()
    plt.savefig(OUT, dpi=150)


if __name__ == "__main__":
    main()
