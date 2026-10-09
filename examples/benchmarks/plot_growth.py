"""Plot the growth bench (growth_bench.py), fit per-unit scaling, compare to cx.

Reads growth_cpu.csv and/or growth_gpu.csv from DIR and writes, into DIR:

- growth_np.png: per-unit time vs live units N, one line per P.
- growth_np_fit.png: per-unit time vs N x P, with the fitted power law.
- growth_n.png, growth_c.png: every strategy vs live units N / connections C.
- growth_p.png: proposers vs P, shortlists vs their size M.
- growth_compile.png: compile time of each point vs its candidates.
- growth_fit.csv: the per-unit fits (slope in N x P, and in N and P apart).

With ``--cx CX_DIR`` (plastax-cpp's ``benchmarks/results/growth``, holding
growth_host.csv and growth_device.csv) it also writes:

- growth_vs_cx.png: per-unit time vs N x P for both libraries.
- growth_vs_cx.csv: every point both libraries ran, with the px/cx ratio
  (px CPU against cx host, px GPU against cx device).

Run it with matplotlib and pandas on hand, e.g.:

    uv run --with matplotlib --with pandas \\
        python examples/benchmarks/plot_growth.py benchmarks/results/growth \\
        --cx ../plastix/benchmarks/results/growth
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Fixed categorical order: a strategy keeps its color and marker in every plot
# (the same encoding as plastax-cpp's plot_growth.py).
STRATEGIES = {
    "per_unit": ("#2a78d6", "o", "per-unit (default)"),
    "per_connection": ("#eb6834", "s", "per-connection"),
    "global": ("#1baf7a", "^", "global"),
    "exhaustive": ("#eda100", "D", "exhaustive"),
    "shortlist": ("#e87ba4", "v", "shortlist"),
    "shortlist_level": ("#4a3aa7", "P", "shortlist per level"),
}
# Sequential blue ramp for P (light = small).
P_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#256abf", "#184f95", "#0d366b"]
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
# Library x backend series for the cx comparison.
SERIES = {
    ("px", "cpu"): ("#2a78d6", "o", "-", "px CPU (XLA:CPU)"),
    ("px", "gpu"): ("#0d366b", "o", "--", "px GPU"),
    ("cx", "host"): ("#eb6834", "s", "-", "cx host (1 thread)"),
    ("cx", "device"): ("#a8431c", "s", "--", "cx device"),
}
PAIRS = {"cpu": "host", "gpu": "device"}

# Below these candidate counts a call is dominated by fixed costs (the slot
# claim over every bucket, dispatch), so each power-law fit starts there.
FIT_RANGES = [("cpu", 1 << 16), ("gpu", 1 << 20), ("gpu", 1 << 23)]
FIT_MIN_CANDIDATES = {b: lo for b, lo in reversed(FIT_RANGES)}


def style(ax: Any, xlabel: str, ylabel: str, title: str) -> None:
    """Log-log axes in the shared style.

    Args:
        ax: the axes.
        xlabel: x label.
        ylabel: y label.
        title: title.
    """
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(which="both", colors=INK2, labelsize=8)


def load(d: Path) -> pd.DataFrame:
    """Every growth_cpu/gpu CSV in `d`.

    Args:
        d: the results directory.

    Returns:
        The concatenated rows.

    Raises:
        SystemExit: when there are none.
    """
    paths = [d / f"growth_{b}.csv" for b in ("cpu", "gpu")]
    frames = [pd.read_csv(p) for p in paths if p.exists()]
    if not frames:
        raise SystemExit(f"no growth_cpu.csv / growth_gpu.csv in {d}")
    return pd.concat(frames, ignore_index=True)


def backends(df: pd.DataFrame) -> list[str]:
    """The backends present, CPU first.

    Args:
        df: the rows.

    Returns:
        The backend names.
    """
    return [b for b in ("cpu", "gpu") if b in set(df.backend)]


def fit(g: pd.DataFrame) -> dict[str, float]:
    """Least-squares log-log fits of time against N x P, and N and P apart.

    Args:
        g: per_unit rows with N, P, candidates and median_ms.

    Returns:
        The slope, intercept and R^2 in N x P, the exponents of N and P, and
        the median time per candidate.
    """
    y = np.log(g.median_ms.to_numpy(float))
    lnp = np.log(g.candidates.to_numpy(float))
    slope, icpt = np.polyfit(lnp, y, 1)
    r2 = 1 - np.sum((y - (slope * lnp + icpt)) ** 2) / np.sum((y - y.mean()) ** 2)
    a = np.column_stack(
        [np.ones(len(g)), np.log(g.N.to_numpy(float)), np.log(g.P.to_numpy(float))]
    )
    (_, a_n, a_p), *_ = np.linalg.lstsq(a, y, rcond=None)
    return {
        "slope_np": slope,
        "intercept": icpt,
        "r2": r2,
        "exp_n": a_n,
        "exp_p": a_p,
        "ns_per_candidate": float(np.median(g.median_ms * 1e6 / g.candidates)),
    }


def fit_per_unit(df: pd.DataFrame) -> pd.DataFrame:
    """The per-unit fits over the np grid, one per FIT_RANGES entry.

    Args:
        df: the rows.

    Returns:
        One row per fit.
    """
    rows = []
    for b, lo in FIT_RANGES:
        g = df[(df.backend == b) & (df.sweep == "np") & (df.candidates >= lo)]
        if len(g) < 3:
            continue
        rows.append(
            {
                "backend": b,
                "points": len(g),
                "min_candidates": int(g.candidates.min()),
                **fit(g),
            }
        )
    cols = ["backend", "points", "min_candidates", "slope_np", "intercept", "r2"]
    return pd.DataFrame(rows, columns=[*cols, "exp_n", "exp_p", "ns_per_candidate"])


def subplots(n: int, rows: int = 1) -> tuple[Any, Any]:
    """A row of `n` panels.

    Args:
        n: panels per row.
        rows: rows.

    Returns:
        The figure and its axes grid.
    """
    return plt.subplots(rows, n, figsize=(5.2 * n, 4.2 * rows), squeeze=False)


def plot_np(df: pd.DataFrame, out: Path) -> None:
    """Per-unit time vs N, one line per P.

    Args:
        df: the rows.
        out: the output directory.
    """
    bs = backends(df)
    fig, axes = subplots(len(bs))
    for ax, b in zip(axes[0], bs, strict=True):
        g = df[(df.backend == b) & (df.sweep == "np")]
        for i, p in enumerate(sorted(g.P.unique())):
            s = g[g.P == p].sort_values("N")
            ax.plot(
                s.N,
                s.median_ms,
                marker="o",
                markersize=4,
                linewidth=1.5,
                color=P_RAMP[i % len(P_RAMP)],
                label=f"P={p}",
            )
        style(ax, "live units N", "growth call (ms, median)", f"Per-unit growth, {b}")
        ax.legend(
            fontsize=8, frameon=False, title="proposals per unit", title_fontsize=8
        )
    fig.tight_layout()
    fig.savefig(out / "growth_np.png", dpi=140)
    plt.close(fig)


def plot_np_fit(df: pd.DataFrame, fits: pd.DataFrame, out: Path) -> None:
    """Per-unit time vs N x P, with the fitted power law.

    Args:
        df: the rows.
        fits: `fit_per_unit`'s rows.
        out: the output directory.
    """
    bs = backends(df)
    fig, axes = subplots(len(bs))
    color, marker, _ = STRATEGIES["per_unit"]
    for ax, b in zip(axes[0], bs, strict=True):
        g = df[(df.backend == b) & (df.sweep == "np")]
        lo = FIT_MIN_CANDIDATES[b]
        used = g[g.candidates >= lo]
        rest = g[g.candidates < lo]
        ax.scatter(
            rest.candidates,
            rest.median_ms,
            s=22,
            facecolors="none",
            edgecolors=color,
            label="below fit range",
        )
        ax.scatter(
            used.candidates,
            used.median_ms,
            s=22,
            color=color,
            marker=marker,
            label="fitted points",
        )
        f = fits[(fits.backend == b) & (fits.min_candidates >= lo)]
        if len(f):
            row = f.iloc[0]
            xs = np.array([used.candidates.min(), used.candidates.max()], dtype=float)
            ax.plot(
                xs,
                np.exp(row.intercept) * xs**row.slope_np,
                "--",
                color=INK2,
                linewidth=1.2,
                label=f"fit: t ~ (N x P)^{row.slope_np:.2f}, R^2={row.r2:.3f}",
            )
        style(
            ax,
            "candidates N x P",
            "growth call (ms, median)",
            f"Per-unit growth vs N x P, {b}",
        )
        ax.legend(fontsize=8, framealpha=0.9, edgecolor="none")
    fig.tight_layout()
    fig.savefig(out / "growth_np_fit.png", dpi=140)
    plt.close(fig)


def plot_sweep(
    df: pd.DataFrame, out: Path, sweep: str, xcol: str, xlabel: str, title: str
) -> None:
    """Every strategy of one sweep against its varied parameter.

    Args:
        df: the rows.
        out: the output directory.
        sweep: "n" or "c".
        xcol: the varied column.
        xlabel: x label.
        title: title stem.
    """
    bs = backends(df)
    fig, axes = subplots(len(bs))
    for ax, b in zip(axes[0], bs, strict=True):
        g = df[(df.backend == b) & (df.sweep == sweep)]
        for name, (color, marker, label) in STRATEGIES.items():
            s = g[g.strategy == name].sort_values(xcol)
            if s.empty:
                continue
            ax.plot(
                s[xcol],
                s.median_ms,
                marker=marker,
                markersize=4,
                linewidth=1.5,
                color=color,
                label=label,
            )
        fixed = "C" if xcol == "N" else "N"
        val = int(g[fixed].iloc[0]) if len(g) else 0
        style(ax, xlabel, "growth call (ms, median)", f"{title}, {b} ({fixed}={val})")
        ax.legend(fontsize=8, framealpha=0.9, edgecolor="none")
    fig.tight_layout()
    fig.savefig(out / f"growth_{sweep}.png", dpi=140)
    plt.close(fig)


def plot_p(df: pd.DataFrame, out: Path) -> None:
    """Proposers vs P and shortlists vs M.

    Args:
        df: the rows.
        out: the output directory.
    """
    bs = backends(df)
    fig, axes = subplots(2, rows=len(bs))
    for row, b in zip(axes, bs, strict=True):
        g = df[(df.backend == b) & (df.sweep == "p")]
        panels = (
            (row[0], ("per_unit", "per_connection", "global"), "P", "Proposers vs P"),
            (row[1], ("shortlist", "shortlist_level"), "M", "Shortlists vs M"),
        )
        for ax, names, knob, title in panels:
            for name in names:
                color, marker, label = STRATEGIES[name]
                s = g[g.strategy == name].sort_values("P")
                if s.empty:
                    continue
                ax.plot(
                    s.P,
                    s.median_ms,
                    marker=marker,
                    markersize=4,
                    linewidth=1.5,
                    color=color,
                    label=label,
                )
            n = int(g.N.iloc[0]) if len(g) else 0
            c = int(g.C.iloc[0]) if len(g) else 0
            xlabel = "proposals per proposer P" if knob == "P" else "shortlist size M"
            style(
                ax, xlabel, "growth call (ms, median)", f"{title}, {b} (N={n}, C={c})"
            )
            ax.legend(fontsize=8, framealpha=0.9, edgecolor="none")
    fig.tight_layout()
    fig.savefig(out / "growth_p.png", dpi=140)
    plt.close(fig)


def plot_compile(df: pd.DataFrame, out: Path) -> None:
    """Compile time of every point against its candidates per call.

    Args:
        df: the rows.
        out: the output directory.
    """
    bs = backends(df)
    fig, axes = subplots(len(bs))
    for ax, b in zip(axes[0], bs, strict=True):
        g = df[df.backend == b]
        for name, (color, marker, label) in STRATEGIES.items():
            s = g[g.strategy == name]
            if s.empty:
                continue
            ax.scatter(
                s.candidates.clip(lower=1),
                s.compile_s,
                s=18,
                color=color,
                marker=marker,
                label=label,
            )
        style(ax, "candidates per call", "compile (s)", f"Compile time, {b}")
        ax.legend(fontsize=8, framealpha=0.9, edgecolor="none")
    fig.tight_layout()
    fig.savefig(out / "growth_compile.png", dpi=140)
    plt.close(fig)


def load_cx(d: Path) -> pd.DataFrame:
    """plastax-cpp's growth rows (host and device).

    Args:
        d: plastax-cpp's results/growth directory.

    Returns:
        The concatenated rows.
    """
    frames = [
        pd.read_csv(p)
        for p in (d / "growth_host.csv", d / "growth_device.csv")
        if p.exists()
    ]
    return pd.concat(frames, ignore_index=True)


def compare_cx(df: pd.DataFrame, cx: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Every point both libraries ran, with the px/cx time ratio.

    Args:
        df: the px rows.
        cx: the cx rows.
        out: the output directory.

    Returns:
        The matched rows.
    """
    key = ["sweep", "strategy", "N", "C", "P"]
    px_rows = df.assign(cx_backend=df.backend.map(PAIRS))
    m = px_rows.merge(
        cx[[*key, "backend", "median_ms"]].rename(
            columns={"backend": "cx_backend", "median_ms": "cx_ms"}
        ),
        on=[*key, "cx_backend"],
    )
    m = m.rename(columns={"median_ms": "px_ms"})
    m["ratio"] = m.px_ms / m.cx_ms
    cols = [*key, "backend", "cx_backend", "candidates", "px_ms", "cx_ms", "ratio"]
    m = m[cols].sort_values(["backend", "sweep", "strategy", "N", "C", "P"])
    m.to_csv(out / "growth_vs_cx.csv", index=False, float_format="%.4f")
    return m


def plot_vs_cx(df: pd.DataFrame, cx: pd.DataFrame, out: Path) -> None:
    """Per-unit time vs N x P for both libraries on both backends.

    Args:
        df: the px rows.
        cx: the cx rows.
        out: the output directory.
    """
    fig, axes = subplots(1)
    ax = axes[0][0]
    for (lib, b), (color, marker, ls, label) in SERIES.items():
        rows = df if lib == "px" else cx
        g = rows[(rows.backend == b) & (rows.sweep == "np")]
        if g.empty:
            continue
        s = g.groupby("candidates").median_ms.median().sort_index()
        ax.plot(
            s.index,
            s.to_numpy(),
            marker=marker,
            markersize=4,
            linewidth=1.5,
            linestyle=ls,
            color=color,
            label=label,
        )
    style(
        ax,
        "candidates N x P",
        "growth call (ms, median over N, P)",
        "Per-unit growth, px vs cx",
    )
    ax.legend(fontsize=8, framealpha=0.9, edgecolor="none")
    fig.tight_layout()
    fig.savefig(out / "growth_vs_cx.png", dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dir", type=Path)
    ap.add_argument("--cx", type=Path, default=None, help="cx results/growth dir")
    args = ap.parse_args()
    df = load(args.dir)
    fits = fit_per_unit(df)
    fits.to_csv(args.dir / "growth_fit.csv", index=False, float_format="%.4f")
    plot_np(df, args.dir)
    plot_np_fit(df, fits, args.dir)
    plot_sweep(df, args.dir, "n", "N", "live units N", "Growth vs live units")
    plot_sweep(df, args.dir, "c", "C", "live connections C", "Growth vs live conns")
    plot_p(df, args.dir)
    plot_compile(df, args.dir)
    with pd.option_context("display.width", 160):
        print(fits.to_string(index=False))
    if args.cx is not None and args.cx.is_dir():
        cx = load_cx(args.cx)
        m = compare_cx(df, cx, args.dir)
        plot_vs_cx(df, cx, args.dir)
        print(f"{len(m)} points matched against cx")


if __name__ == "__main__":
    main()
