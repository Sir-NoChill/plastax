"""The Stage-0 comparison table, as a committed command rather than a script.

Every Stage-0 number quoted so far -- the baseline table, the CBP G2 verdict --
came from an ad-hoc driver that was never committed, so none of it could be
regenerated and two figures have already turned out to disagree with a re-run.
This file is the one entry point that produces the table, so a quoted number is
always `uv run python examples/stage0_table.py --arms ... --seeds N` away.

It only wires arms to `protocol.evaluate`: the seed set, the statistic, the
pairing and the decision rule all live in `protocol` and are not re-decided
here. Each arm is a `(seed) -> (mean_recovery_time, final_accuracy)` callable
over one of the method modules, run at the protocol's own theta, switch period
and horizon so no arm can quietly use a friendlier task.

A stationary arm has no switches and reports `nan` recovery, which `protocol`
prints as unavailable rather than as perfect.

Run:  uv run python examples/stage0_table.py                  # smoke, 4 seeds
      uv run python examples/stage0_table.py --seeds 30       # the real table
      uv run python examples/stage0_table.py --arms dense,cbp-v0,cbp-v1
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence

import numpy as np
import protocol
from nonstationary import CycleRecord, recovery_times

Arm = Callable[[int], tuple[float, float]]

# Named here so `--arms` can be validated before any method module is imported;
# each one builds JAX nets at import time.
ARM_NAMES = (
    "dense",
    "static@10%",
    "set",
    "rigl",
    "cbp-v0",
    "cbp-v1",
    "gmp",
    "ne-capped",
    "upgd-v0",
    "upgd-v1",
)

# Every arm runs at the protocol's task settings. Passed explicitly rather than
# left to each module's defaults, which differ.
_TASK = {
    "theta": protocol.THETA,
    "switch_period": protocol.SWITCH_PERIOD,
    "num_cycles": protocol.NUM_CYCLES,
    "steps_per_cycle": protocol.STEPS_PER_CYCLE,
}


def _score(records: Sequence[CycleRecord]) -> tuple[float, float]:
    """Reduce one run to the protocol's two reported statistics.

    Args:
        records: the per-cycle records of a single run.

    Returns:
        Mean recovery time (nan when the arm never switched) and mean accuracy
        over the final window.
    """
    times = recovery_times(list(records))
    recovery = float(np.mean(times)) if times else float("nan")
    accuracy = float(np.mean([r.accuracy for r in records[-protocol.FINAL_WINDOW :]]))
    return recovery, accuracy


def build_arms(num_cycles: int) -> dict[str, Arm]:
    """Wire every Stage-0 method to a seed-to-statistics callable.

    Args:
        num_cycles: cycles per run, overriding the protocol horizon for smoke
            runs. The full table uses `protocol.NUM_CYCLES`.

    Returns:
        Arm name -> callable, keyed by `ARM_NAMES`. `dense` is the control every
        other arm is differenced against.
    """
    import cbp
    import gmp
    import ne
    import nonstationary
    import upgd

    task = {**_TASK, "num_cycles": num_cycles}

    def baseline(method: str) -> Arm:
        return lambda seed: _score(nonstationary.run(method, seed=seed, **task))

    def cbp_arm(threshold: str) -> Arm:
        return lambda seed: _score(cbp.run(threshold=threshold, seed=seed, **task))

    def upgd_arm(eta: str) -> Arm:
        return lambda seed: _score(upgd.run(eta=eta, seed=seed, **task))

    def gmp_arm(seed: int) -> tuple[float, float]:
        return _score(gmp.run(seed=seed, **task))

    def ne_arm(seed: int) -> tuple[float, float]:
        # Capped: the uncapped arm ends near dense, so there is no sparsity
        # left for the comparison to be about.
        return _score(ne.run(density_target=0.5, seed=seed, **task)[0])

    return {
        "dense": baseline("dense"),
        "static@10%": baseline("static"),
        "set": baseline("set"),
        "rigl": baseline("rigl"),
        "cbp-v0": cbp_arm("v0"),
        "cbp-v1": cbp_arm("v1"),
        "gmp": gmp_arm,
        "ne-capped": ne_arm,
        "upgd-v0": upgd_arm("v0"),
        "upgd-v1": upgd_arm("v1"),
    }


def main() -> None:
    """Run the requested arms over the requested seed block and print the table.

    Raises:
        SystemExit: on an unknown arm name, listing the ones that exist.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arms",
        default="dense,static@10%,set,rigl,cbp-v0,cbp-v1",
        help="comma-separated arm names; must include the control",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=4,
        help="how many of protocol.SEEDS to use (30 for the reported table)",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=protocol.NUM_CYCLES,
        help="cycles per run; shorten only for a smoke run",
    )
    parser.add_argument(
        "--control", default="dense", help="the arm every other arm is differenced from"
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="use the held-out CONFIRMATION_SEEDS block instead",
    )
    args = parser.parse_args()

    names = [name for name in args.arms.split(",") if name]
    unknown = [name for name in names if name not in ARM_NAMES]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; available: {list(ARM_NAMES)}")
    if args.control not in names:
        raise SystemExit(f"the control {args.control!r} must be among --arms")

    block = protocol.CONFIRMATION_SEEDS if args.confirm else protocol.SEEDS
    seeds = block[: args.seeds]
    arms = build_arms(args.cycles)

    print(f"Stage 0 -- {len(names)} arms x {len(seeds)} seeds, theta=pi/4")
    print(f"seeds {'CONFIRMATION' if args.confirm else 'SEEDS'}[:{args.seeds}] {seeds}")
    if args.cycles != protocol.NUM_CYCLES:
        print(
            f"WARNING: {args.cycles} cycles, not the protocol's "
            f"{protocol.NUM_CYCLES} -- a smoke run, not a reportable table"
        )
    print("=" * 100)
    results = protocol.evaluate({name: arms[name] for name in names}, seeds)
    protocol.report(results, control=args.control)


if __name__ == "__main__":
    main()
