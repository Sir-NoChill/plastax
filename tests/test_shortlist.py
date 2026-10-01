"""Shortlist coverage (phases.shortlist_coverage / recommended_shortlist).

A shortlisted `add_conn` draws candidates from the M most important sources
crossed with the M most important eligible destinations, so **M bounds how many
distinct units can receive a new edge**. That is a different and usually tighter
constraint than the volume rule `M >= sqrt(zeta * E)` that `dst_sparse`
documents, and violating it is silent: the arena stays sparse, it just drifts
*below* the target density while every downstream comparison believes it is
running at the density it asked for.

These tests pin the diagnostic against the behaviour it predicts -- an
undersized shortlist must both be flagged AND actually bleed the live-edge
count, and a covered one must hold it exactly.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import plastax as px

_EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def _load_example(name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, _EXAMPLES_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_load_example("mlp_xor")
dst = _load_example("dst_sparse")

_LAYERS = (9, 16, 4)
_BUDGETS = (64, 64)


def _nets(shortlist: int | None) -> tuple[type, type]:
    """A (train, churn) net pair at the given shortlist."""
    optimizer = px.optim.adam(0.01, dst.GradPreAct)
    train = dst.make_net(optimizer, method="set", mode="train")
    grow = max(_BUDGETS) if shortlist is None else shortlist * shortlist
    churn = dst.make_net(
        optimizer,
        method="set",
        mode="churn",
        zeta=0.3,
        max_candidates=grow,
        shortlist=shortlist,
    )
    return train, churn


def test_no_shortlist_reports_no_coverage() -> None:
    """The exhaustive grid reaches everything, so there is nothing to report."""
    train, churn = _nets(None)
    static, state = dst.build_sparse_mlp(train, _LAYERS, _BUDGETS, 0)
    assert px.shortlist_coverage(churn, static, state) == ()
    assert px.recommended_shortlist(churn, static, state) == 0


def test_coverage_flags_the_undersized_bucket() -> None:
    """M=8 against a 16-wide hidden layer is reported uncovered."""
    train, churn = _nets(8)
    static, state = dst.build_sparse_mlp(train, _LAYERS, _BUDGETS, 0)
    coverage = px.shortlist_coverage(churn, static, state)
    assert len(coverage) == len(static.level_capacities)
    by_bucket = {c.bucket: c for c in coverage}
    # bucket 0 grows into the 16-unit hidden layer; bucket 1 into 4 outputs.
    assert by_bucket[0].destination_units == 16
    assert not by_bucket[0].covered
    assert by_bucket[1].destination_units == 4
    assert by_bucket[1].covered
    assert px.recommended_shortlist(churn, static, state) == 16


def test_recommended_shortlist_is_covered() -> None:
    """Building at the recommendation covers every bucket."""
    train, churn = _nets(8)
    static, state = dst.build_sparse_mlp(train, _LAYERS, _BUDGETS, 0)
    recommended = px.recommended_shortlist(churn, static, state)
    _, covered_churn = _nets(recommended)
    assert all(c.covered for c in px.shortlist_coverage(covered_churn, static, state))


def _live_counts(shortlist: int, cycles: int = 6) -> set[int]:
    """Live-edge counts across a few train-then-churn cycles."""
    train, churn = _nets(shortlist)
    static, state = dst.build_sparse_mlp(train, _LAYERS, _BUDGETS, 0)
    train_step = px.make_step(train, static)
    churn_step = px.make_step(churn, static)
    teacher, rng = dst.teacher_task(_LAYERS[0] - 1, _LAYERS[-1], 0)
    counts: set[int] = set()
    inputs = None
    for _ in range(cycles):
        for _ in range(10):
            inputs, label = dst._sample(teacher, rng)
            state = train_step(
                state,
                px.StepInputs(inputs=inputs, targets=dst._one_hot(label, _LAYERS[-1])),
            ).state
        state = churn_step(state, px.StepInputs(inputs=inputs, targets=None)).state
        counts.add(int(px.state.live_conn_count(state)))
    return counts


def test_uncovered_shortlist_actually_bleeds_edges() -> None:
    """The diagnostic predicts real behaviour, in both directions.

    This is the test that makes `shortlist_coverage` worth having: an uncovered
    shortlist must lose edges, and a covered one must not.
    """
    uncovered = _live_counts(8)
    covered = _live_counts(16)
    assert len(uncovered) > 1, (
        f"expected an undersized shortlist to bleed edges, got {uncovered}"
    )
    assert len(covered) == 1, (
        f"a covered shortlist must conserve the live count, got {covered}"
    )
    assert max(uncovered) <= max(covered), "under-fill should never grow the arena"
