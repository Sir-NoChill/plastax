"""Gradual magnitude pruning (examples/gmp.py).

GMP is almost entirely a schedule over shipped traits, so the tests target the
schedule and the one conversion that silently corrupts it -- cumulative sparsity
is not the per-churn prune fraction, and treating them as interchangeable
over-prunes every cycle after the first.

Also pins the structural claim: GMP declares no `add_conn`, so the live-edge
count must be monotonically non-increasing. A regrowing "GMP" would still hit
its target sparsity and look correct in aggregate.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def _load_example(name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, _EXAMPLES_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


for _dep in ("mlp_xor", "dst_sparse", "nonstationary"):
    _load_example(_dep)
gmp = _load_example("gmp")


def test_cubic_schedule_respects_its_window() -> None:
    """Zero before the window, the target after it, monotone within."""
    schedule = [
        gmp.cubic_sparsity(t, start=20, end=80, final_sparsity=0.95) for t in range(101)
    ]
    assert all(s == 0.0 for s in schedule[:21]), "pruned before the window opened"
    assert schedule[80] == pytest.approx(0.95)
    assert all(s == pytest.approx(0.95) for s in schedule[80:])
    assert all(a <= b for a, b in zip(schedule[:-1], schedule[1:], strict=True))
    assert schedule[50] > 0.5 * 0.95, "cubic front-loads the window"


def test_cubic_schedule_degenerate_window() -> None:
    """An empty window jumps to the target rather than dividing by zero."""
    assert gmp.cubic_sparsity(5, start=10, end=10, final_sparsity=0.9) == 0.0
    assert gmp.cubic_sparsity(10, start=10, end=10, final_sparsity=0.9) == 0.9


def test_zeta_converts_cumulative_to_fraction_of_live() -> None:
    """The conversion is where a plausible implementation goes wrong.

    Going from 50% to 75% cumulative sparsity means removing HALF of what is
    still live, not 75% and not 25%.
    """
    assert gmp.zeta_for_step(0.75, 0.50) == pytest.approx(0.5)
    assert gmp.zeta_for_step(0.5, 0.0) == pytest.approx(0.5)
    assert gmp.zeta_for_step(0.5, 0.5) == pytest.approx(0.0)
    assert gmp.zeta_for_step(0.4, 0.6) == 0.0, "must never ask for regrowth"
    assert gmp.zeta_for_step(0.99, 1.0) == 0.0, "nothing left to prune"


def _short_run(**kwargs: object) -> list:
    return gmp.run(
        num_cycles=40,
        steps_per_cycle=20,
        hidden_layers=(24, 24),
        d=12,
        classes=4,
        seed=0,
        **kwargs,
    )


def test_no_pruning_before_the_window_opens() -> None:
    """The net stays fully dense while the schedule is still zero."""
    records = _short_run(final_sparsity=0.9, start_fraction=0.5, end_fraction=0.9)
    dense = records[0].live_edges
    for record in records[: int(0.5 * len(records))]:
        assert record.live_edges == dense, "pruned inside the dense phase"


def test_live_edges_never_increase() -> None:
    """GMP declares no add_conn, so a pruned edge can never come back.

    Checked cycle by cycle: a method that over-pruned and regrew would still
    land on a plausible final count.
    """
    counts = [r.live_edges for r in _short_run(final_sparsity=0.9)]
    assert all(a >= b for a, b in zip(counts[:-1], counts[1:], strict=True)), (
        f"live-edge count increased: {counts}"
    )


def test_sparsity_moves_toward_the_target_but_undershoots() -> None:
    """Realized sparsity rises with the target and stays below it.

    `SetPrune` compares each edge against its destination's half-normal
    quantile, and trained weights are not half-normal, so each churn removes
    less than the requested fraction. The host recomputes zeta from the realized
    sparsity each cycle so the error does not compound, but it does not close
    within the window either -- measured ~0.83 against a 0.90 target. Asserting
    exact tracking would be asserting a bug.
    """
    realized = {
        target: 1.0 - _short_run(final_sparsity=target)[-1].density
        for target in (0.75, 0.9)
    }
    assert realized[0.9] > realized[0.75], "higher target must give more sparsity"
    for target, got in realized.items():
        assert got <= target + 1e-6, f"overshot {target}: {got}"
        assert got > 0.5 * target, f"barely pruned for target {target}: {got}"
