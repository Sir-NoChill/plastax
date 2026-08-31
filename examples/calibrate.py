"""Find the density at which each sparse arm's plateau matches the dense one.

Recovery time is NOT comparable across arms of different accuracy -- across the
Stage-0 baselines `pearson(median accuracy, median recovery)` is +0.995, so an
arm that "recovers faster" may only be a weaker network with less to climb back.
See `todo/rl-recovery-metric-confound.md`.

Engineering around it with a scale-free statistic was tried and closed. The one
candidate that decorrelated -- late-over-early recovery, pearson +0.048 -- has a
relative IPR-90 of 0.40-1.87, worse than the recovery-time slope the protocol
already retired at 0.71-0.90, and its decorrelation flips sign with the window
size. So the answer is the one the source papers use: **match the capacity**, as
Liu et al. do when they set "the final network size of NE to be identical to
that of the other methods", and then compare plain mean recovery, which is the
only well-conditioned statistic available (relative IPR 0.10-0.36).

This file finds the matching density. **It must be run at the protocol horizon**
-- the plateau is still climbing at 200 cycles (dense reaches 0.50 there and
0.69 by 300), so a density calibrated on a short run does not transfer.

A coarse 200-cycle, 4-seed probe put the crossing near density 0.5 for all three
arms, and showed accuracy SATURATING there rather than climbing to 1.0 -- which
is what makes the matched comparison meaningful rather than vacuous. Confirm at
the protocol horizon before quoting a number.

Run:  uv run python examples/calibrate.py                 # protocol horizon
      uv run python examples/calibrate.py --cycles 200    # quick look
"""

from __future__ import annotations

import argparse

import numpy as np
import protocol
from nonstationary import run

# The sparse arms. `dense` is the target, not a row.
METHODS = ("static", "set", "rigl")
DENSITIES = (0.1, 0.25, 0.4, 0.5, 0.65, 0.8, 1.0)


def plateau(
    method: str, *, density: float | None, cycles: int, seeds: tuple[int, ...]
) -> float:
    """Median final-window accuracy over the seed block.

    Args:
        method: the arm to run.
        density: live-edge fraction, or None to leave the method's default
            (used for the dense target, which ignores density).
        cycles: cycles per run.
        seeds: the seed block.

    Returns:
        The median across seeds of each run's final-window mean accuracy.
    """
    kwargs: dict[str, object] = {
        "theta": protocol.THETA,
        "switch_period": protocol.SWITCH_PERIOD,
        "num_cycles": cycles,
        "steps_per_cycle": protocol.STEPS_PER_CYCLE,
    }
    if density is not None:
        kwargs["density"] = density
    values = [
        float(
            np.mean(
                [
                    record.accuracy
                    for record in run(method, seed=seed, **kwargs)[  # type: ignore[arg-type]
                        -protocol.FINAL_WINDOW :
                    ]
                ]
            )
        )
        for seed in seeds
    ]
    return float(np.median(values))


def main() -> None:
    """Sweep density per method and report the density that matches dense."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=protocol.NUM_CYCLES)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="accuracy gap that counts as matched",
    )
    args = parser.parse_args()
    seeds = protocol.SEEDS[: args.seeds]

    target = plateau("dense", density=None, cycles=args.cycles, seeds=seeds)
    print(f"capacity calibration: {len(seeds)} seeds, {args.cycles} cycles")
    if args.cycles != protocol.NUM_CYCLES:
        print(
            f"WARNING: not the protocol horizon ({protocol.NUM_CYCLES}). The "
            "plateau is still climbing on short runs, so this will not transfer."
        )
    print(f"dense plateau (the target): {target:.3f}\n")

    print(f"{'density':>8} " + "".join(f"{m:>12}" for m in METHODS))
    table: dict[str, list[float]] = {m: [] for m in METHODS}
    for density in DENSITIES:
        row = [
            plateau(m, density=density, cycles=args.cycles, seeds=seeds)
            for m in METHODS
        ]
        for method, value in zip(METHODS, row, strict=True):
            table[method].append(value)
        print(f"{density:8.2f} " + "".join(f"{v:12.3f}" for v in row), flush=True)

    print()
    print("matched density (lowest that reaches the dense plateau):")
    for method in METHODS:
        reached = [
            d
            for d, v in zip(DENSITIES, table[method], strict=True)
            if v >= target - args.tolerance
        ]
        if reached:
            print(f"  {method:8} {min(reached):.2f}")
        else:
            best = max(table[method])
            print(
                f"  {method:8} NEVER -- best {best:.3f} against {target:.3f}. "
                "This arm cannot be capacity-matched; report the pair instead."
            )


if __name__ == "__main__":
    main()
