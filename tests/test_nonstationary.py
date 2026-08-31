"""Stage 0 harness (examples/nonstationary.py).

The whole RL plan is gated on this harness measuring what it claims, so the
tests target the three places it could quietly lie:

* `drift` must rotate by exactly the requested angle and preserve ||T||. Without
  the Gram-Schmidt step a raw Gaussian overlaps the row it rotates by
  O(1/sqrt(d)), turning a requested 90 degrees into 97 at d=16 -- "severity"
  would not mean what the sweep says, and a teacher whose norm drifted would
  change the task's difficulty as a side effect of switching.
* the DENSE baseline must actually be dense. `build_sparse_mlp` draws distinct
  random pairs and drops collisions, so asking it for a full budget yields ~86%
  of the edges; a silently-14%-sparse baseline makes every sparsity comparison
  in the study unfalsifiable.
* `recovery_times` must censor rather than silently skip a switch that never
  recovers, otherwise plasticity loss makes the metric look BETTER by dropping
  its own worst cases.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

import jax.numpy as jnp
import numpy as np

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
_load_example("dst_sparse")
ns = _load_example("nonstationary")


def _row_angles(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-row angle between two matrices, in radians."""
    dot = np.sum(a * b, axis=1)
    norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.arccos(np.clip(dot / norms, -1.0, 1.0))


def test_drift_rotates_by_exact_angle_and_preserves_norm() -> None:
    """The realised rotation is the requested one, to machine precision."""
    rng = np.random.default_rng(0)
    for d in (16, 64):
        teacher = rng.standard_normal((8, d))
        before = np.linalg.norm(teacher, axis=1)
        for theta in (math.pi / 8, math.pi / 4, math.pi / 2):
            rotated = ns.drift(teacher, theta, rng)
            np.testing.assert_allclose(_row_angles(teacher, rotated), theta, atol=1e-9)
            np.testing.assert_allclose(
                np.linalg.norm(rotated, axis=1), before, rtol=1e-9
            )


def test_drift_at_zero_is_identity() -> None:
    """theta=0 must leave the teacher untouched, so the stationary arm is a
    genuine control rather than a slowly-perturbed one."""
    rng = np.random.default_rng(1)
    teacher = rng.standard_normal((4, 32))
    np.testing.assert_allclose(ns.drift(teacher, 0.0, rng), teacher, atol=1e-12)


def test_drift_at_half_pi_is_orthogonal() -> None:
    """theta=pi/2 must give a fully orthogonal teacher -- a real resample."""
    rng = np.random.default_rng(2)
    teacher = rng.standard_normal((6, 24))
    rotated = ns.drift(teacher, math.pi / 2, rng)
    np.testing.assert_allclose(np.sum(teacher * rotated, axis=1), 0.0, atol=1e-9)


def test_dense_baseline_is_actually_dense() -> None:
    """Every (src, dst) pair of each layer transition is present exactly once."""
    layers = (9, 16, 4)
    optimizer = px.optim.sgd(0.01, ns.GradPreAct)
    net = ns.make_net(optimizer, mode="train")
    static, state = ns.build_dense_mlp(net, layers, 0)
    expected = sum(a * b for a, b in zip(layers[:-1], layers[1:], strict=True))
    assert int(px.state.live_conn_count(state)) == expected
    pairs = set()
    for bucket in state.conns:
        dead = np.asarray(bucket[px.DEAD.name])
        from_ids = np.asarray(bucket[px.FROM_ID.name])[~dead]
        to_ids = np.asarray(bucket[px.TO_ID.name])[~dead]
        pairs.update(zip(from_ids.tolist(), to_ids.tolist(), strict=True))
    assert len(pairs) == expected, "dense baseline has duplicate or missing edges"
    del static


def test_recovery_times_censors_unrecovered_switches() -> None:
    """A switch that never recovers is reported at the horizon, not dropped.

    Dropping it would make a run that got WORSE look better, since the switches
    that never recover are exactly the ones plasticity loss produces.
    """
    records = []
    for cycle in range(20):
        # plateau 1.0 before the switch at 10; never recovers afterwards.
        accuracy = 1.0 if cycle < 10 else 0.1
        records.append(
            ns.CycleRecord(
                cycle=cycle,
                loss=0.0,
                accuracy=accuracy,
                live_edges=1,
                density=1.0,
                dormant=0.0,
                mean_abs_w=0.0,
                switched=(cycle == 10),
                seconds=0.0,
            )
        )
    times = ns.recovery_times(records, window=5, tolerance=0.98)
    assert times == [10], f"expected one censored switch of 10 cycles, got {times}"


def test_recovery_times_finds_a_real_recovery() -> None:
    """A switch that does recover reports the cycle count it took."""
    records = []
    for cycle in range(20):
        accuracy = 1.0 if cycle < 10 else (0.1 if cycle < 14 else 1.0)
        records.append(
            ns.CycleRecord(
                cycle=cycle,
                loss=0.0,
                accuracy=accuracy,
                live_edges=1,
                density=1.0,
                dormant=0.0,
                mean_abs_w=0.0,
                switched=(cycle == 10),
                seconds=0.0,
            )
        )
    assert ns.recovery_times(records, window=5, tolerance=0.98) == [4]


def test_growth_slope_beats_first_vs_last() -> None:
    """The trend statistic must not be fooled by a noisy non-trend.

    [8, 15, 8, 10] rises from first to last but is not growing; the slope says
    so. This is the check that keeps G0 from passing on noise.
    """
    assert ns.growth_slope([8, 15, 8, 10]) < 0.25
    assert ns.growth_slope([1, 2, 3, 4, 5]) > 0.9
    assert ns.growth_slope([5]) == 0.0


def test_rewiring_holds_sparsity_across_a_switch() -> None:
    """G1: SET and RigL keep their live-edge count through a teacher rotation.

    A rule whose density drifts on the switch is mis-implemented, and every
    downstream comparison is at a different sparsity than it claims.

    `shortlist` must cover the LAYER WIDTH, not merely satisfy dst_sparse's
    `M >= sqrt(zeta * E)`: the per-level grid draws candidates from the top-M
    destinations, so M=8 against a 16-unit hidden layer can only ever refill
    into half of it and the count bleeds down (measured: 128 -> {121..125},
    exact at M >= 16). That is a property of the candidate grid, not of the
    rewiring rule, but it silently changes the sparsity a study runs at.
    """
    for method in ("set", "rigl"):
        records = ns.run(
            method,
            theta=math.pi / 2,
            switch_period=2,
            d=8,
            hidden_layers=(16,),
            classes=4,
            density=0.3,
            num_cycles=6,
            steps_per_cycle=10,
            shortlist=16,
            seed=0,
        )
        counts = {r.live_edges for r in records}
        assert len(counts) == 1, f"{method}: live-edge count drifted: {counts}"
        assert any(r.switched for r in records), "no switch occurred"


def test_dormancy_is_scale_invariant() -> None:
    """Sokar's normalised score, not an absolute activation threshold.

    This is the same confound as the recovery metric, in the diagnostic that was
    supposed to be independent of it: under an ABSOLUTE bar an arm whose
    activations are globally smaller reports more dormant units for that reason
    alone, so dormancy and accuracy level could not be told apart. Normalising
    by the layer mean (Plasticine appendix D.1, from Sokar et al. 2023) makes
    the score invariant to any positive rescaling of a layer.

    Scaling by 0.01 puts EVERY unit under the old 0.01 bar, so the old
    definition would report total dormancy for a network that has not changed.
    """
    optimizer = px.optim.sgd(0.1, ns.GradPreAct)
    net = ns.make_net(optimizer, mode="train")
    static, state = ns.build_dense_mlp(net, (4, 9, 6, 3), 0)
    state = ns.mark_outputs(static, state)

    rng = np.random.default_rng(0)
    base = np.abs(rng.random(static.num_units).astype(np.float32)) + 0.05
    # One genuinely quiet unit, so the measurement is not trivially zero.
    base[static.num_units // 2] = 1e-6

    scores = []
    for factor in (1.0, 0.01, 100.0):
        state.units = {**state.units, ns.ACT_EMA.name: jnp.asarray(base * factor)}
        scores.append(ns.dormant_fraction(static, state, tau=0.025))

    assert scores[0] == scores[1] == scores[2], f"dormancy moved with scale: {scores}"
    assert scores[0] > 0.0, "the planted quiet unit was not detected"
    assert scores[0] < 1.0, "every unit reported dormant -- the bar is not relative"
