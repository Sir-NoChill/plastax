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

An effect is claimed when a paired SIGN TEST over the per-seed differences
rejects at `ALPHA`, Bonferroni-corrected across the arms in the comparison, AND
the median difference favours the treatment. Failure is reported as "no effect
at this power", never as a trend.

## AMENDMENT, 2026-08-30

The original rule required the IPR-90 of the paired differences to exclude zero.
That was wrong, and the first G2 run exposed it: CBP beat the dense control on
27 of 30 seeds -- a sign test rejects at p = 4e-06 -- and the rule returned "no
effect".

IPR-90 excluding zero means the 5th-to-95th percentile band lies entirely one
side of zero, which requires at least 29 of 30 runs to improve. It reports no
effect however large and consistent the median shift is. The mistake was
importing IPR-90 from Tanaka & Mahmood, where it is a DESCRIPTIVE statistic for
run-to-run spread, and then reusing it as a DECISION rule, which is not what it
measures.

Median and IPR-90 remain the reported statistics; only the decision moves to the
sign test. The amendment was made after seeing G2 and is therefore applied ONLY
to comparisons re-run under it -- the original G2 verdict stands as recorded.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Sequence
from math import comb
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

# A second frozen block, held out so an AMENDED rule can be tested on data it
# has never seen. Re-running an amended rule on the seeds that motivated the
# amendment reproduces the same numbers and confirms nothing.
CONFIRMATION_SEEDS: Final[tuple[int, ...]] = tuple(range(30, 60))

# Task configuration, frozen with the protocol: 39 switches is where the mean
# recovery statistic reaches an IPR-90 of 0.13 of its magnitude at ~14 s/run.
THETA: Final[float] = math.pi / 4
SWITCH_PERIOD: Final[int] = 15
NUM_CYCLES: Final[int] = 600
STEPS_PER_CYCLE: Final[int] = 100
FINAL_WINDOW: Final[int] = 5
# Per-comparison significance level, Bonferroni-corrected across arms in
# `report`. Fixed here rather than chosen per comparison.
ALPHA: Final[float] = 0.01


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


@dataclasses.dataclass(frozen=True)
class SignTest:
    """Paired sign test over per-seed differences.

    Attributes:
        wins: seeds where the treatment beat the control.
        losses: seeds where it lost.
        ties: exact ties, excluded from the test.
        p_value: TWO-SIDED binomial probability under a fair coin. Two-sided so
            a method that reliably HURTS is detected rather than silently
            reported as no effect.
        median_difference: median of the paired differences.
    """

    wins: int
    losses: int
    ties: int
    p_value: float
    median_difference: float

    @property
    def effective_n(self) -> int:
        """Seeds contributing to the test, i.e. excluding ties."""
        return self.wins + self.losses


def sign_test(
    treatment: Sequence[float],
    control: Sequence[float],
    *,
    lower_is_better: bool = True,
) -> SignTest:
    """Count per-seed wins and give the one-sided binomial probability.

    Ties are dropped rather than split, which is the conservative choice: they
    reduce the effective sample instead of being counted as half-successes. The
    probability is two-sided, so an arm that consistently makes things worse is
    reported as such instead of falling through to "no effect".

    Args:
        treatment: per-seed values for the arm under test, in seed order.
        control: per-seed values for the control, in the same order.
        lower_is_better: whether a smaller value favours the treatment.

    Returns:
        The SignTest.

    Raises:
        ValueError: if the two sequences differ in length.
    """
    if len(treatment) != len(control):
        raise ValueError(f"sign_test: {len(treatment)} vs {len(control)} values")
    wins = losses = ties = 0
    for t, c in zip(treatment, control, strict=True):
        if t == c:
            ties += 1
        elif (t < c) == lower_is_better:
            wins += 1
        else:
            losses += 1
    trials = wins + losses
    if trials == 0:
        p_value = 1.0
    else:
        upper = sum(comb(trials, k) for k in range(wins, trials + 1)) / 2**trials
        lower = sum(comb(trials, k) for k in range(0, wins + 1)) / 2**trials
        p_value = min(1.0, 2.0 * min(upper, lower))
    difference = paired_difference(treatment, control)
    return SignTest(
        wins=wins,
        losses=losses,
        ties=ties,
        p_value=p_value,
        median_difference=difference.median,
    )


def verdict(
    test: SignTest, *, alpha: float = ALPHA, lower_is_better: bool = True
) -> str:
    """Apply the amended decision rule to a paired sign test.

    Requires BOTH significance and the right direction, so a rejection driven by
    a trivially small but consistent shift still has to move the median the
    right way.

    Args:
        test: the sign test to judge.
        alpha: significance level, already corrected for multiple arms.
        lower_is_better: whether a negative median difference favours the arm.

    Returns:
        One of ``"better"``, ``"worse"`` or ``"no effect at this power"``.
    """
    if test.p_value > alpha or test.median_difference == 0.0:
        return "no effect at this power"
    improved = (test.median_difference < 0.0) == lower_is_better
    return "better" if improved else "worse"


def evaluate(
    arms: dict[str, Callable[[int], tuple[float, float]]],
    seeds: Sequence[int] = SEEDS,
) -> dict[str, tuple[Summary, Summary]]:
    """Run every arm on every seed and summarise both reported statistics.

    Args:
        arms: name -> callable taking a seed and returning
            ``(mean_recovery_time, final_accuracy)``.
        seeds: the seed block to run; every arm gets the same one.

    Returns:
        name -> (recovery Summary, accuracy Summary).
    """
    out: dict[str, tuple[Summary, Summary]] = {}
    for name, arm in arms.items():
        recoveries, accuracies = [], []
        for seed in seeds:
            recovery, accuracy = arm(seed)
            recoveries.append(recovery)
            accuracies.append(accuracy)
        out[name] = (summarize(recoveries), summarize(accuracies))
    return out


def report(results: dict[str, tuple[Summary, Summary]], control: str) -> None:
    """Print medians, IPR-90, and the paired sign test against the control.

    Args:
        results: the mapping returned by `evaluate`.
        control: the arm every other arm is differenced against.
    """
    arms = [name for name in results if name != control]
    alpha = ALPHA / max(len(arms), 1)
    print(
        f"protocol: {len(SEEDS)} fixed seeds, paired, "
        f"theta=pi/4 switch_period={SWITCH_PERIOD} cycles={NUM_CYCLES}"
    )
    print("statistic: per-run mean recovery time; median and IPR-90 across seeds")
    print(
        f"decision: paired sign test, alpha={ALPHA} Bonferroni-corrected over "
        f"{len(arms)} arm(s) -> {alpha:.4f}"
    )
    print()
    print(
        f"{'arm':18} {'median rec':>11} {'IPR90':>8} {'diff':>7} "
        f"{'wins':>7} {'p':>10} {'verdict':>22}"
    )
    base = results[control][0]
    print(f"{control:18} {base.median:11.2f} {base.ipr90:8.2f} {'(control)':>7}")
    for name in arms:
        recovery = results[name][0]
        test = sign_test(recovery.values, base.values)
        print(
            f"{name:18} {recovery.median:11.2f} {recovery.ipr90:8.2f} "
            f"{test.median_difference:7.2f} "
            f"{test.wins:3d}/{test.effective_n:<3d} {test.p_value:10.2e} "
            f"{verdict(test, alpha=alpha):>22}"
        )
