"""Unit capacity: free unit slots, the PRUNED column, and the live mask.

A net declaring `unit_capacity` gets that many unit slots; the built units are
live and the slots above them are free (marked `PRUNED`). A slot holding no
live unit is skipped by every apply, by the loss and by growth, so spare
capacity must not change what the live units compute -- checked here against
the same model built without it, in every step flavour. The rules below write
a constant offset in every apply, so a skipped apply is visible.
"""

from __future__ import annotations

import dataclasses
import pathlib
import subprocess
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import plastax as px

GRAD = px.FieldSpec.float32("test/grad")
LOSS_GRAD = px.FieldSpec.float32("test/loss_grad")
APPLIED = px.FieldSpec.float32("test/applied")

# Two inputs (0, 1), two hidden (2, 3), two outputs (4, 5).
_N = 6
_INPUTS = (0, 1)
_OUTPUTS = (4, 5)
_SRC = np.asarray([0, 0, 1, 1, 2, 2, 3, 3], np.int32)
_DST = np.asarray([2, 3, 2, 3, 4, 5, 4, 5], np.int32)
_W = np.asarray([0.5, -0.25, 0.75, 0.125, -0.5, 0.25, 0.375, -0.75], np.float32)


class OffsetForward(px.ForwardPass):
    """activation = sum(w * activation[src]) + 1; applied = 1."""

    combine = px.monoid.sum_
    linear_input = px.ACTIVATION

    def map(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: None,
    ) -> jax.Array:
        del dst, g
        return c[px.WEIGHT, cid] * u[px.ACTIVATION, src]

    def apply(
        self, u: px.UnitView, i: px.UnitIdx, g: None, acc: jax.Array
    ) -> px.UnitWrite:
        del u, i, g
        return px.UnitWrite.of(
            (px.ACTIVATION, acc + jnp.float32(1.0)), (APPLIED, jnp.float32(1.0))
        )


class OffsetBackward(px.BackwardPass):
    """grad = sum(w * grad[dst]) + loss_grad + 1."""

    combine = px.monoid.sum_
    linear_input = GRAD

    def map(
        self,
        u: px.UnitView,
        src: px.UnitIdx,
        dst: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: None,
    ) -> jax.Array:
        del src, g
        return c[px.WEIGHT, cid] * u[GRAD, dst]

    def apply(
        self, u: px.UnitView, i: px.UnitIdx, g: None, acc: jax.Array
    ) -> px.UnitWrite:
        del g
        return px.UnitWrite.of((GRAD, acc + u[LOSS_GRAD, i] + jnp.float32(1.0)))


class HalfSquaredLoss(px.Loss):
    """0.5 * (activation - target)^2, staging the difference in loss_grad."""

    def per_output(
        self, u: px.UnitView, i: px.UnitIdx, target: jax.Array, g: None
    ) -> tuple[jax.Array, px.UnitWrite]:
        del g
        diff = u[px.ACTIVATION, i] - target
        return jnp.float32(0.5) * diff * diff, px.UnitWrite.of((LOSS_GRAD, diff))


class DeltaRule(px.UpdateConn):
    """w -= 1/16 * grad[dst] * activation[src]."""

    def incoming(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: None,
    ) -> px.ConnWrite:
        del g
        step = u[GRAD, dst] * u[px.ACTIVATION, src] * jnp.float32(1.0 / 16.0)
        return px.ConnWrite.of((px.WEIGHT, c[px.WEIGHT, cid] - step))

    def outgoing(
        self,
        u: px.UnitView,
        src: px.UnitIdx,
        dst: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: None,
    ) -> px.ConnWrite:
        del u, src, dst, c, cid, g
        return px.ConnWrite.of()


class SmallWeightPrune(px.PruneConn):
    """Tombstone |w| < 1/8, or a source `applied` above 1000 (never, here).

    `applied` does not depend on the forward's accumulator, so a fused prune
    forwards it (see `plan_prune_fusion`) and that copy is checked too.
    """

    def predicate(
        self, u: px.UnitView, c: px.ConnView, cid: px.ConnIdx, g: None
    ) -> jax.Array:
        del g
        small = jnp.abs(c[px.WEIGHT, cid]) < jnp.float32(0.125)
        return small | (u[APPLIED, c[px.FROM_ID, cid]] > jnp.float32(1000.0))


def _init(u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: None) -> px.ConnWrite:
    del u, g
    w = ((src * 3 + dst * 5) % 7).astype(jnp.float32) / jnp.float32(8.0)
    return px.ConnWrite.of((px.WEIGHT, w - jnp.float32(0.375)))


class HighIdGrid(px.ScoreAddConn):
    """Exhaustive growth preferring high unit ids -- the free slots, if seen."""

    max_new_per_level = 2
    max_level_gap = 3

    def score(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: None
    ) -> jax.Array:
        del u, g
        return (src + dst).astype(jnp.float32)

    def init(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: None
    ) -> px.ConnWrite:
        return _init(u, src, dst, g)


class HighIdShortlist(HighIdGrid):
    """A 3-unit shortlist ranked by unit id: the free slots rank first."""

    candidates = "shortlist"
    shortlist_size = 3

    def importance(self, u: px.UnitView, i: px.UnitIdx, g: None) -> jax.Array:
        del u, g
        return i.astype(jnp.float32)


class EveryUnitProposes(px.ProposeAddConn):
    """Each unit proposes edges to fixed ids, some of them free slots."""

    proposals_per_proposer = 2
    max_new_per_level = 2
    max_level_gap = 3

    def propose(
        self, u: px.UnitView, i: px.UnitIdx, j: jax.Array, g: None, rng: Any
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        del u, g
        # A free slot (id >= 6) would propose an edge between live units.
        return i % 6, (i * 3 + j + 1) % 8, rng.uniform()

    def init(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: None
    ) -> px.ConnWrite:
        return _init(u, src, dst, g)


_GROWTH: dict[str, Any] = {
    "grid": HighIdGrid(),
    "shortlist": HighIdShortlist(),
    "propose": EveryUnitProposes(),
}


def _net(
    *,
    capacity: int | None,
    propagation: px.Propagation = px.Propagation.TOPOLOGICAL,
    growth: str | None = None,
    update: bool = True,
) -> type[px.Network[None]]:
    class _Net(px.Network[None]):
        forward_pass = OffsetForward()
        backward_pass = OffsetBackward()
        loss = HalfSquaredLoss()
        update_conn = DeltaRule() if update else None
        prune_conn = SmallWeightPrune()
        add_conn = _GROWTH[growth] if growth is not None else None
        extra_unit_fields = (GRAD, LOSS_GRAD, APPLIED)
        unit_capacity = capacity

    _Net.propagation = propagation
    return _Net


def _build(
    net: type[px.Network[None]], **kw: Any
) -> tuple[px.NetworkStatic, px.NetworkState[None]]:
    return px.NetworkBuilder.from_edges(
        net,
        _N,
        _SRC,
        _DST,
        weights=_W,
        input_ids=_INPUTS,
        output_ids=_OUTPUTS,
        globals_=None,
        capacity_headroom=2.0,
        **kw,
    )


def _live_edges(state: px.NetworkState[None]) -> list[tuple[int, int, float]]:
    out = []
    for bucket in state.conns:
        dead = np.asarray(bucket[px.DEAD.name])
        for s, d, w in zip(
            np.asarray(bucket[px.FROM_ID.name])[~dead],
            np.asarray(bucket[px.TO_ID.name])[~dead],
            np.asarray(bucket[px.WEIGHT.name])[~dead],
            strict=True,
        ):
            out.append((int(s), int(d), float(w)))
    return sorted(out)


# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------


def test_capacity_adds_free_slots_marked_pruned() -> None:
    static, state = _build(_net(capacity=9))
    assert static.num_units == 9
    assert px.PRUNED in static.unit_fields
    pruned = np.asarray(state.units[px.PRUNED.name])
    assert pruned.tolist() == [False] * _N + [True] * 3
    assert int(px.state.live_unit_count(state)) == _N
    for spec in static.unit_fields:
        if spec.name != px.PRUNED.name:
            col = np.asarray(state.units[spec.name])
            assert (col[_N:] == spec.default).all(), spec.name


def test_no_capacity_keeps_the_column_layout() -> None:
    static, state = _build(_net(capacity=None))
    assert static.num_units == _N
    assert px.PRUNED not in static.unit_fields
    assert px.PRUNED.name not in state.units
    assert int(px.state.live_unit_count(state)) == _N


def test_capacity_equal_to_the_unit_count_has_no_free_slot() -> None:
    _, state = _build(_net(capacity=_N))
    assert not np.asarray(state.units[px.PRUNED.name]).any()


def test_capacity_below_the_unit_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="unit_capacity 5 is below the 6 built"):
        _build(_net(capacity=5))


@pytest.mark.parametrize("bad", [0, -1, True, 2.0, "8"])
def test_bad_unit_capacity_is_rejected(bad: object) -> None:
    with pytest.raises(TypeError, match="unit_capacity must be None or an int"):

        class _Bad(px.Network[None]):
            forward_pass = OffsetForward()
            unit_capacity = bad  # type: ignore[assignment]


@pytest.mark.parametrize("bad", [1, 0, False, 3.0])
def test_bad_max_levels_is_rejected(bad: object) -> None:
    with pytest.raises(TypeError, match="max_levels must be an int >= 2"):

        class _Bad(px.Network[None]):
            forward_pass = OffsetForward()
            max_levels = bad  # type: ignore[assignment]


def test_max_levels_defaults_to_the_cpp_bound() -> None:
    assert px.Network.max_levels == 1024


def test_pruned_is_a_reserved_field_name() -> None:
    with pytest.raises(ValueError, match="reserved builtin name"):

        class _Bad(px.Network[None]):
            forward_pass = OffsetForward()
            extra_unit_fields = (px.FieldSpec.boolean("pruned"),)


# ---------------------------------------------------------------------------
# The live mask: a slot holding no live unit takes no part in a pass
# ---------------------------------------------------------------------------


def _kill_unit(state: px.NetworkState[None], uid: int, *, prune: bool) -> Any:
    """Tombstone every edge touching `uid`; mark it pruned when `prune`."""
    conns = []
    for bucket in state.conns:
        touches = (bucket[px.FROM_ID.name] == uid) | (bucket[px.TO_ID.name] == uid)
        conns.append({**bucket, px.DEAD.name: bucket[px.DEAD.name] | touches})
    units = dict(state.units)
    if prune:
        units[px.PRUNED.name] = units[px.PRUNED.name].at[uid].set(True)
    return dataclasses.replace(state, units=units, conns=tuple(conns))


def _inputs() -> px.StepInputs:
    return px.StepInputs(
        inputs=jnp.asarray([0.5, -1.0], jnp.float32),
        targets=jnp.asarray([0.25, 2.0], jnp.float32),
    )


@pytest.mark.parametrize(
    ("flavour", "propagation"),
    [
        ("streaming", px.Propagation.TOPOLOGICAL),
        ("streaming", px.Propagation.PIPELINE),
        ("batched_edge_list", px.Propagation.TOPOLOGICAL),
        ("batched_edge_once", px.Propagation.TOPOLOGICAL),
    ],
)
def test_masked_units_take_no_part_in_forward_and_backward(
    flavour: str, propagation: px.Propagation
) -> None:
    """A pruned unit and the free slots keep their defaults in every pass.

    The reference is the same net without a capacity and with unit 3's edges
    tombstoned, so unit 3 is isolated but live there: every live unit must
    agree, while unit 3 itself is written there (offset 1) and not here.
    """
    kw: dict[str, Any] = dict(_FLAVOURS[flavour])
    batch = kw.get("batch_size")
    masked_net = _net(capacity=8, propagation=propagation)
    plain_net = _net(capacity=None, propagation=propagation)
    masked_static, masked = _build(masked_net)
    plain_static, plain = _build(plain_net)
    masked = _kill_unit(masked, 3, prune=True)
    plain = _kill_unit(plain, 3, prune=False)
    masked_step = px.make_step(masked_net, masked_static, **kw)
    plain_step = px.make_step(plain_net, plain_static, **kw)
    for t in range(3):
        masked = masked_step(masked, _batch(_inputs(), batch, t)).state
        plain = plain_step(plain, _batch(_inputs(), batch, t)).state
    live = [0, 1, 2, 4, 5]
    for name in (px.ACTIVATION.name, GRAD.name, LOSS_GRAD.name, APPLIED.name):
        got = np.asarray(masked.units[name])
        want = np.asarray(plain.units[name])
        np.testing.assert_allclose(
            got[live], want[live], rtol=1e-6, atol=1e-6, err_msg=name
        )
        # Unit 3 and the free slots 6, 7 were never applied.
        assert (got[[3, 6, 7]] == 0.0).all(), (name, got)
    # The isolated-but-live unit 3 of the reference is applied.
    assert float(plain.units[APPLIED.name][3]) == 1.0
    assert float(plain.units[GRAD.name][3]) != 0.0


def test_a_masked_output_adds_no_loss() -> None:
    """An output slot holding no live unit contributes no loss term."""
    net = _net(capacity=8)
    static, state = _build(net)
    state = _kill_unit(state, 5, prune=True)
    result = px.make_step(net, static)(state, _inputs())
    act = np.asarray(result.state.units[px.ACTIVATION.name])
    loss_grad = np.asarray(result.state.units[LOSS_GRAD.name])
    want = np.float32(0.5) * np.float32(act[4] - np.float32(0.25)) ** 2
    assert float(result.loss) == float(want)
    assert loss_grad[5] == 0.0
    assert loss_grad[4] == act[4] - np.float32(0.25)


# ---------------------------------------------------------------------------
# Spare capacity changes nothing the live units compute
# ---------------------------------------------------------------------------

_FLAVOURS = {
    "streaming": {},
    "fused_prune": {"fuse_prune": "xla"},
    "batched_edge_list": {"batch_size": 2, "layout": "edge_list"},
    "batched_edge_once": {"batch_size": 2, "layout": "triton"},
}


def _batch(inputs: px.StepInputs, batch: int | None, t: int) -> px.StepInputs:
    if batch is None:
        return inputs
    assert inputs.targets is not None
    shift = jnp.arange(batch, dtype=jnp.float32)[:, None] * jnp.float32(0.25)
    return px.StepInputs(
        inputs=inputs.inputs[None] + shift + jnp.float32(t),
        targets=inputs.targets[None] - shift,
    )


@pytest.mark.parametrize("growth", sorted(_GROWTH))
@pytest.mark.parametrize("flavour", sorted(_FLAVOURS))
def test_spare_capacity_matches_the_model_without_it(flavour: str, growth: str) -> None:
    _run_side_by_side(
        dict(_FLAVOURS[flavour]), growth=growth, propagation=px.Propagation.TOPOLOGICAL
    )


@pytest.mark.parametrize("growth", sorted(_GROWTH))
@pytest.mark.parametrize("flavour", ["streaming", "fused_prune"])
def test_spare_capacity_matches_the_model_without_it_pipeline(
    flavour: str, growth: str
) -> None:
    _run_side_by_side(
        dict(_FLAVOURS[flavour]), growth=growth, propagation=px.Propagation.PIPELINE
    )


def _assert_same_live_state(
    got: px.NetworkState[None], want: px.NetworkState[None], what: object
) -> None:
    """Live unit columns and live edges agree (floats to an ulp or so).

    The two steps compile to different graphs (different unit-axis lengths),
    so XLA may fuse a float expression differently and round its last bit
    differently; integers, flags and the edge set must match exactly.
    """
    for name, col in want.units.items():
        a, b = np.asarray(got.units[name])[: col.shape[0]], np.asarray(col)
        if np.issubdtype(b.dtype, np.floating):
            np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6, err_msg=name)
        else:
            np.testing.assert_array_equal(a, b, err_msg=name)
    got_edges, want_edges = _live_edges(got), _live_edges(want)
    assert [e[:2] for e in got_edges] == [e[:2] for e in want_edges], what
    np.testing.assert_allclose(
        [e[2] for e in got_edges], [e[2] for e in want_edges], rtol=1e-6, atol=1e-6
    )


def _run_side_by_side(
    kw: dict[str, Any],
    *,
    growth: str,
    propagation: px.Propagation,
    steps: int = 4,
) -> None:
    """Step both models side by side and compare every live column."""
    batch = kw.get("batch_size")
    # A connection update between forward and prune rules fusion out.
    fused = "fuse_prune" in kw
    spare_net = _net(
        capacity=_N + 3, propagation=propagation, growth=growth, update=not fused
    )
    plain_net = _net(
        capacity=None, propagation=propagation, growth=growth, update=not fused
    )
    spare_static, spare = _build(spare_net)
    plain_static, plain = _build(plain_net)
    assert spare_static.level_capacities == plain_static.level_capacities
    spare_step = px.make_step(spare_net, spare_static, **kw)
    plain_step = px.make_step(plain_net, plain_static, **kw)
    grew = 0
    for t in range(steps):
        inputs = _batch(_inputs(), batch, t)
        r_spare = spare_step(spare, inputs)
        r_plain = plain_step(plain, inputs)
        spare, plain = r_spare.state, r_plain.state
        if fused:
            plan = spare_step.prune_fusion.plan  # type: ignore[attr-defined]
            assert plan.fused and plan.forwarded == (APPLIED.name,), plan
        np.testing.assert_allclose(float(r_spare.loss), float(r_plain.loss), rtol=1e-6)
        assert int(spare.grown) == int(plain.grown), t
        assert bool(spare.needs_resort) == bool(plain.needs_resort), t
        grew += int(plain.grown)
        _assert_same_live_state(spare, plain, t)
    assert grew > 0, "growth never fired: the comparison would be vacuous"
    # The free slots stayed free and untouched.
    for spec in spare_static.unit_fields:
        col = np.asarray(spare.units[spec.name])
        assert (col[_N:] == spec.default).all(), spec.name


_SHARDED_SCRIPT = pathlib.Path(__file__).parent / "unit_capacity_sharding_equiv.py"


@pytest.mark.skipif(len(jax.devices()) < 4, reason="needs >= 4 devices")
def test_sharded_spare_capacity_matches_single_device() -> None:
    """Units are replicated under Scheme-A, so the live mask is too.

    Runs in a clean subprocess for the reason `test_sharding.py` gives.
    """
    result = subprocess.run(
        [sys.executable, str(_SHARDED_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "UNIT CAPACITY SHARDING PASS" in result.stdout
