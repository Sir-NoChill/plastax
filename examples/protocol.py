"""The pre-registered evaluation protocol for every Stage-0 and Stage-2 arm.

Fixed HERE, in code, before the comparisons run. Deciding a seed count or a
statistic after seeing results is the failure mode this module exists to
prevent, and it already bit once: a single-seed recovery-time slope showed CBP
cutting plasticity loss by 40%, and re-running the same arm gave the opposite
sign.

Adopted from *Performance Variation in Deep Reinforcement Learning* (Tanaka &
Mahmood, AutoRL @ RLC 2026). Their argument is that standard error and
confidence intervals on the mean **shrink with the number of runs regardless of
the real spread**, so they describe how well the mean is pinned down rather than
how much the method actually varies -- which is the wrong question when the
practical concern is run-to-run robustness. They report percentile statistics
instead: min-max IPR-90, the 5th-to-95th inter-percentile range.

## The four choices the protocol has to fix, per todo/rl-eval-protocol.md

1. **Seed count and the seeds themselves.** `SEEDS` is an explicit frozen tuple,
   not a count and not a range drawn at call time, so no arm can be re-rolled
   until it behaves and no seed can be quietly dropped. Every arm runs every
   seed.
2. **The statistic.** Per-run **mean recovery time**, summarised across seeds by
   the **median** and **IPR-90**. NOT the recovery-time slope: measured on the
   dense control alone (which cannot bias a later CBP comparison), the slope's
   spread is 0.71-0.90 of its own magnitude no matter how many switches it fits,
   and its value collapses from 1.79 to 0.03 as the horizon lengthens because
   recovery growth saturates rather than continuing linearly. Mean recovery
   tightens to 0.13 of its magnitude at 39 switches and is comparable across
   horizons.
3. **The final-performance window.** The last `FINAL_WINDOW` cycles, for
   accuracy only. Recovery times use every switch after the first.
4. **Paired or not.** **Paired.** Arms share the seed set, and the reported
   effect is the per-seed difference against the control, which cancels the
   task-realisation variance that dominates the unpaired spread.

## The decision rule, also fixed in advance

An effect is claimed only when the IPR-90 of the *paired per-seed differences*
excludes zero. Overlap is reported as "no effect at this power", never as a
trend.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Sequence
from typing import Final

import numpy as np

# The frozen seed set. Written out rather than generated so it is auditable and
# identical across every arm and every re-run.
SEEDS: Final[tuple[int, ...]] = (
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
)

# Task configuration, frozen with the protocol: 39 switches is where the mean
# recovery statistic reaches an IPR-90 of 0.13 of its magnitude at ~14 s/run.
THETA: Final[float] = math.pi / 4
SWITCH_PERIOD: Final[int] = 15
NUM_CYCLES: Final[int] = 600
STEPS_PER_CYCLE: Final[int] = 100
FINAL_WINDOW: Final[int] = 5


@dataclasses.dataclass(frozen=True)
class Summary:
    """Percentile summary of one statistic across the seed set.

    Attributes:
        p5: 5th percentile across seeds.
        median: 50th percentile, the reported central value.
        p95: 95th percentile across seeds.
        ipr90: the 5th-to-95th inter-percentile range.
        values: the per-seed values, in seed order.
    """

    p5: float
    median: float
    p95: float
    ipr90: float
    values: tuple[float, ...]

    @property
    def relative_ipr(self) -> float:
        """IPR-90 as a fraction of the median's magnitude."""
        return self.ipr90 / abs(self.median) if abs(self.median) > 1e-12 else math.inf

    @property
    def excludes_zero(self) -> bool:
        """Whether the 5th-to-95th range lies entirely on one side of zero."""
        return (self.p5 > 0.0) or (self.p95 < 0.0)


def summarize(values: Sequence[float]) -> Summary:
    """Percentile-summarise one statistic across seeds.

    Args:
        values: one value per seed.

    Returns:
        The Summary.

    Raises:
        ValueError: if `values` is empty.
    """
    if not values:
        raise ValueError("summarize: no values")
    array = np.asarray(values, dtype=float)
    p5, median, p95 = (float(v) for v in np.percentile(array, [5, 50, 95]))
    return Summary(
        p5=p5,
        median=median,
        p95=p95,
        ipr90=p95 - p5,
        values=tuple(float(v) for v in array),
    )


def paired_difference(treatment: Sequence[float], control: Sequence[float]) -> Summary:
    """Summarise the per-seed difference `treatment - control`.

    Pairing is the point: both arms saw the same teacher, the same drift
    sequence and the same data stream on each seed, so differencing removes the
    task-realisation variance that dominates the unpaired spread.

    Args:
        treatment: per-seed values for the arm under test, in seed order.
        control: per-seed values for the control, in the same seed order.

    Returns:
        The Summary of the paired differences.

    Raises:
        ValueError: if the two sequences differ in length.
    """
    if len(treatment) != len(control):
        raise ValueError(
            f"paired_difference: {len(treatment)} vs {len(control)} values"
        )
    return summarize([t - c for t, c in zip(treatment, control, strict=True)])


def verdict(difference: Summary, *, lower_is_better: bool = True) -> str:
    """Apply the pre-registered decision rule to a paired difference.

    Args:
        difference: the paired-difference summary.
        lower_is_better: whether a negative difference favours the treatment.

    Returns:
        One of ``"better"``, ``"worse"`` or ``"no effect at this power"``.
    """
    if not difference.excludes_zero:
        return "no effect at this power"
    improved = (difference.median < 0.0) == lower_is_better
    return "better" if improved else "worse"


def evaluate(
    arms: dict[str, Callable[[int], tuple[float, float]]],
) -> dict[str, tuple[Summary, Summary]]:
    """Run every arm on every seed and summarise both reported statistics.

    Args:
        arms: name -> callable taking a seed and returning
            ``(mean_recovery_time, final_accuracy)``.

    Returns:
        name -> (recovery Summary, accuracy Summary).
    """
    out: dict[str, tuple[Summary, Summary]] = {}
    for name, arm in arms.items():
        recoveries, accuracies = [], []
        for seed in SEEDS:
            recovery, accuracy = arm(seed)
            recoveries.append(recovery)
            accuracies.append(accuracy)
        out[name] = (summarize(recoveries), summarize(accuracies))
    return out


def report(results: dict[str, tuple[Summary, Summary]], control: str) -> None:
    """Print the protocol's table: medians, IPR-90, and paired differences.

    Args:
        results: the mapping returned by `evaluate`.
        control: the arm name every other arm is differenced against.
    """
    print(
        f"protocol: {len(SEEDS)} fixed seeds, paired, "
        f"theta=pi/4 switch_period={SWITCH_PERIOD} cycles={NUM_CYCLES}"
    )
    print("statistic: per-run mean recovery time; median and IPR-90 across seeds")
    print()
    print(
        f"{'arm':18} {'median rec':>11} {'IPR90':>8} {'rel':>6} "
        f"{'paired diff median':>19} {'diff IPR90':>11} {'verdict':>22}"
    )
    base_recovery = results[control][0]
    for name, (recovery, _accuracy) in results.items():
        if name == control:
            print(
                f"{name:18} {recovery.median:11.2f} {recovery.ipr90:8.2f} "
                f"{recovery.relative_ipr:6.2f} {'(control)':>19} {'':>11} {'':>22}"
            )
            continue
        difference = paired_difference(recovery.values, base_recovery.values)
        print(
            f"{name:18} {recovery.median:11.2f} {recovery.ipr90:8.2f} "
            f"{recovery.relative_ipr:6.2f} {difference.median:19.2f} "
            f"{difference.ipr90:11.2f} {verdict(difference):>22}"
        )
