"""The prune_unit phase: permanent unit pruning.

The net below is linear (activation = sum(w * activation[src])), so each
unit's post-forward activation is known in closed form, and the rule prunes a
unit whose activation is below -1/2. With inputs (-1, -2) the forward gives

    unit:        0    1    2    3    4    5
    activation: -1   -2   -1    2   -3    1
    level:       0    0    1    1    3    2

so the rule selects inputs 0 and 1, hidden 2 and output 4; only hidden 2 is
prunable. Unit 5's only input edge comes from 2, so pruning 2 orphans it.
With inputs (1/4, -1) no hidden unit reads below -1/2.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import plastax as px
from plastax import topo
from plastax.phases import build_phases, build_prune_unit_phase, plan_prune_fusion

TRACE = px.FieldSpec.float32("test/trace", default=0.25)
COUNT = px.FieldSpec.int32("test/count", default=7)
FLAG = px.FieldSpec.boolean("test/flag", default=True)

_N = 6
_CAPACITY = 8
_INPUTS = (0, 1)
_OUTPUTS = (4,)
_SRC = np.asarray([0, 1, 2, 5, 3], np.int32)
_DST = np.asarray([2, 3, 5, 4, 4], np.int32)
_W = np.asarray([1.0, -1.0, -1.0, 1.0, -2.0], np.float32)
_PRUNE_INPUTS = jnp.asarray([-1.0, -2.0], jnp.float32)
_KEEP_INPUTS = jnp.asarray([0.25, -1.0], jnp.float32)


class LinearForward(px.ForwardPass):
    """activation = sum(w * activation[src])."""

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
        return px.UnitWrite.of((px.ACTIVATION, acc))


class Negative(px.PruneUnit[None]):
    """Prune a unit whose activation is below -1/2."""

    def predicate(self, u: px.UnitView, i: px.UnitIdx, g: None) -> jax.Array:
        del g
        return u[px.ACTIVATION, i] < jnp.float32(-0.5)


class Everything(px.PruneUnit[None]):
    """Select every unit."""

    def predicate(self, u: px.UnitView, i: px.UnitIdx, g: None) -> jax.Array:
        del u, i, g
        return jnp.bool_(True)


class NoConn(px.PruneConn):
    """Never prunes a connection; only there to enable prune fusion."""

    def predicate(
        self, u: px.UnitView, c: px.ConnView, cid: px.ConnIdx, g: None
    ) -> jax.Array:
        del u, g
        return c[px.WEIGHT, cid] > jnp.float32(100.0)


def _net(
    rule: px.PruneUnit[None] | None = None,
    *,
    prune_conn: px.PruneConn | None = None,
    interval: int = 1,
) -> type[px.Network[None]]:
    class _Net(px.Network[None]):
        forward_pass = LinearForward()
        prune_unit = Negative() if rule is None else rule
        extra_unit_fields = (TRACE, COUNT, FLAG)
        unit_capacity = _CAPACITY
        structural_interval = interval
        propagation = px.Propagation.TOPOLOGICAL
        batch_reduction = px.MeanFloatFirstRest()

    _Net.prune_conn = prune_conn
    return _Net


def _build(net: type[px.Network[None]], **kw: Any) -> tuple[px.NetworkStatic, Any]:
    return px.NetworkBuilder.from_edges(
        net,
        _N,
        _SRC,
        _DST,
        weights=_W,
        input_ids=_INPUTS,
        output_ids=_OUTPUTS,
        globals_=None,
        **kw,
    )


def _x(inputs: jax.Array) -> px.StepInputs:
    return px.StepInputs(inputs=inputs, targets=None)


def _col(state: Any, spec: px.FieldSpec[Any]) -> np.ndarray:
    return np.asarray(state.units[spec.name])


def _live_edges(state: Any) -> set[tuple[int, int]]:
    out = set()
    for bucket in state.conns:
        dead = np.asarray(bucket[px.DEAD.name])
        src = np.asarray(bucket[px.FROM_ID.name])[~dead]
        dst = np.asarray(bucket[px.TO_ID.name])[~dead]
        out |= {(int(s), int(d)) for s, d in zip(src, dst, strict=True)}
    return out


_ALL_EDGES = {(int(s), int(d)) for s, d in zip(_SRC, _DST, strict=True)}
_WITHOUT_2 = _ALL_EDGES - {(0, 2), (2, 5)}


def test_pruning_is_permanent_over_steps() -> None:
    """A pruned unit stays pruned, at its defaults, with its edges dead."""
    net = _net()
    static, state = _build(net)
    step = px.make_step(net, static)
    state = step(state, _x(_PRUNE_INPUTS)).state
    pruned = [False, False, True, False, False, False, True, True]
    assert _col(state, px.PRUNED).tolist() == pruned
    assert _live_edges(state) == _WITHOUT_2
    for _ in range(3):
        # Live, unit 2 would read +1/4 now and its rule would not fire.
        state = step(state, _x(_KEEP_INPUTS)).state
        assert _col(state, px.PRUNED).tolist() == pruned
        assert _live_edges(state) == _WITHOUT_2
        act = _col(state, px.ACTIVATION)
        assert act[2] == 0.0
        # The orphaned unit 5 stays live; its input is gone.
        assert act[5] == 0.0
        assert act[3] == 1.0 and act[4] == -2.0


def test_input_and_output_units_are_exempt() -> None:
    """A rule selecting every unit prunes only the hidden ones."""
    net = _net(Everything())
    static, state = _build(net)
    state = px.make_step(net, static)(state, _x(_KEEP_INPUTS)).state
    pruned = _col(state, px.PRUNED)
    assert pruned.tolist() == [False, False, True, True, False, True, True, True]
    assert _live_edges(state) == set()


def test_io_units_survive_a_selecting_activation() -> None:
    """Inputs 0, 1 and output 4 read below -1/2 but are not pruned."""
    net = _net()
    static, state = _build(net)
    state = px.make_step(net, static)(state, _x(_PRUNE_INPUTS)).state
    act = _col(state, px.ACTIVATION)
    assert act[[0, 1, 4]].tolist() == [-1.0, -2.0, -3.0]
    assert not _col(state, px.PRUNED)[[0, 1, 4]].any()


def _with_fields(state: Any) -> Any:
    """Give every unit slot non-default extra fields."""
    units = dict(state.units)
    units[TRACE.name] = jnp.arange(_CAPACITY, dtype=jnp.float32) + 10.0
    units[COUNT.name] = jnp.arange(_CAPACITY, dtype=jnp.int32) + 100
    units[FLAG.name] = jnp.zeros((_CAPACITY,), jnp.bool_)
    return dataclasses.replace(state, units=units)


def test_a_pruned_unit_resets_to_defaults_and_keeps_its_level() -> None:
    net = _net()
    static, state = _build(net)
    state = _with_fields(state)
    before = {k: np.asarray(v).copy() for k, v in state.units.items()}
    state = px.make_step(net, static)(state, _x(_PRUNE_INPUTS)).state
    assert _col(state, TRACE)[2] == np.float32(0.25)
    assert _col(state, COUNT)[2] == 7
    assert bool(_col(state, FLAG)[2]) is True
    assert _col(state, px.ACTIVATION)[2] == 0.0
    assert _col(state, px.LEVEL)[2] == 1
    # Every other slot keeps its values.
    keep = np.arange(_CAPACITY) != 2
    for spec in (TRACE, COUNT, FLAG, px.LEVEL):
        np.testing.assert_array_equal(
            _col(state, spec)[keep], before[spec.name][keep], err_msg=spec.name
        )


def test_incident_edges_die_in_the_same_phase() -> None:
    """The phase alone tombstones the pruned unit's in- and out-edges."""
    net = _net()
    static, state = _build(net)
    act = jnp.asarray([-1.0, -2.0, -1.0, 2.0, -3.0, 1.0, 0.0, 0.0], jnp.float32)
    state = dataclasses.replace(state, units={**state.units, px.ACTIVATION.name: act})
    assert _live_edges(state) == _ALL_EDGES
    out, _ = build_prune_unit_phase(net, static)(state, _x(_PRUNE_INPUTS))
    assert _col(out, px.PRUNED)[:_N].tolist() == [False] * 2 + [True] + [False] * 3
    assert _live_edges(out) == _WITHOUT_2
    assert not bool(out.needs_resort)


class NeighbourZero(px.PruneUnit[None]):
    """Prune 2 when its activation is below -1/2; prune 3 when 2 reads 0."""

    def predicate(self, u: px.UnitView, i: px.UnitIdx, g: None) -> jax.Array:
        del g
        own = (i == 2) & (u[px.ACTIVATION, i] < jnp.float32(-0.5))
        follows = (i == 3) & (u[px.ACTIVATION, jnp.int32(2)] == jnp.float32(0.0))
        return own | follows


def test_the_rule_reads_the_pre_phase_state() -> None:
    """Unit 3's rule sees unit 2's pre-prune activation, not its reset."""
    net = _net(NeighbourZero())
    static, state = _build(net)
    state = px.make_step(net, static)(state, _x(_PRUNE_INPUTS)).state
    assert _col(state, px.PRUNED)[[2, 3]].tolist() == [True, False]


def test_levels_after_a_prune_follow_the_live_edges() -> None:
    """No resort is requested; a resort puts orphaned units at level 0."""
    net = _net()
    static, state = _build(net)
    assert _col(state, px.LEVEL)[:_N].tolist() == [0, 0, 1, 1, 3, 2]
    state = px.make_step(net, static)(state, _x(_PRUNE_INPUTS)).state
    assert not bool(state.needs_resort)
    levels = np.asarray(topo.recompute_levels(static, state))
    # 2 (pruned) and 5 (orphaned) have no live input edge.
    assert levels[:_N].tolist() == [0, 0, 0, 1, 2, 0]
    _, resorted = topo.resort(static, state)
    assert _col(resorted, px.LEVEL)[:_N].tolist() == [0, 0, 0, 1, 2, 0]
    assert _live_edges(resorted) == _WITHOUT_2


@pytest.mark.parametrize(
    ("inputs", "pruned"),
    [
        # Unit 2 reads -2 and +1.5: the mean -1/4 survives, though sample 0
        # alone would be pruned.
        ([[-2.0, -2.0], [1.5, -2.0]], False),
        # Unit 2 reads -1/4 and -1.5: sample 0 alone survives, the mean -7/8
        # is pruned.
        ([[-0.25, -2.0], [-1.5, -2.0]], True),
    ],
)
def test_a_batch_prunes_on_the_batch_mean(
    inputs: list[list[float]], pruned: bool
) -> None:
    net = _net()
    static, state = _build(net)
    batched = px.make_step(net, static, batch_size=2, layout="edge_list")
    out = batched(
        state, px.StepInputs(inputs=jnp.asarray(inputs, jnp.float32), targets=None)
    ).state
    assert bool(_col(out, px.PRUNED)[2]) is pruned
    assert (_live_edges(out) == _WITHOUT_2) is pruned


def test_a_batch_of_one_is_the_streaming_step() -> None:
    net = _net()
    static, state = _build(net)
    stream = px.make_step(net, static)
    batched = px.make_step(net, static, batch_size=1, layout="edge_list")
    s_stream, s_batch = state, jax.tree.map(jnp.copy, state)
    for x in (_PRUNE_INPUTS, _KEEP_INPUTS, _KEEP_INPUTS):
        s_stream = stream(s_stream, _x(x)).state
        s_batch = batched(s_batch, _x(x[None])).state
    for name, col in s_stream.units.items():
        np.testing.assert_array_equal(
            np.asarray(s_batch.units[name]), np.asarray(col), err_msg=name
        )
    assert _live_edges(s_batch) == _live_edges(s_stream) == _WITHOUT_2


def test_the_structural_interval_does_not_gate_unit_pruning() -> None:
    """Unit pruning runs every step, off-interval steps included."""
    net = _net(interval=3)
    static, state = _build(net)
    step = px.make_step(net, static)
    state = step(state, _x(_KEEP_INPUTS)).state  # step 0: nothing to prune
    state = step(state, _x(_PRUNE_INPUTS)).state  # step 1: 1 % 3 != 0
    assert bool(_col(state, px.PRUNED)[2])
    assert _live_edges(state) == _WITHOUT_2


def test_the_phase_sits_between_update_conn_and_prune_conn() -> None:
    """Declaring prune_unit adds one phase, right before prune_conn."""
    with_net = _net(prune_conn=NoConn())
    without_net = _net(prune_conn=NoConn())
    without_net.prune_unit = None
    static, _ = _build(with_net)
    with_phases = [p.__name__ for p in build_phases(with_net, static)]
    without_phases = [p.__name__ for p in build_phases(without_net, static)]
    assert "prune_unit_phase" not in without_phases
    assert with_phases.index("prune_unit_phase") + 1 == with_phases.index(
        "prune_conn_phase"
    )
    assert [p for p in with_phases if p != "prune_unit_phase"] == without_phases


def test_prune_fusion_is_refused_and_the_fused_engine_matches() -> None:
    """A fused prune_conn would revive the edges prune_unit killed."""
    net = _net(prune_conn=NoConn())
    static, state = _build(net)
    plan = plan_prune_fusion(net, static, state.globals_, engine="xla")
    assert not plan.fused and "prune_unit" in plan.reason
    out = {}
    for mode in ("xla", "off"):
        step = px.make_step(net, static, fuse_prune=mode)  # type: ignore[arg-type]
        out[mode] = step(jax.tree.map(jnp.copy, state), _x(_PRUNE_INPUTS)).state
    assert _live_edges(out["xla"]) == _live_edges(out["off"]) == _WITHOUT_2


def test_prune_unit_requires_a_unit_capacity() -> None:
    with pytest.raises(TypeError, match="prune_unit requires a unit_capacity"):

        class _Bad(px.Network[None]):
            forward_pass = LinearForward()
            prune_unit = Negative()


def test_a_non_conforming_prune_unit_is_rejected() -> None:
    with pytest.raises(TypeError, match="prune_unit must satisfy PruneUnit"):

        class _Bad(px.Network[None]):
            forward_pass = LinearForward()
            prune_unit = object()  # type: ignore[assignment]
            unit_capacity = 4


def test_unit_pruning_under_sharding_is_refused_at_definition() -> None:
    with pytest.raises(NotImplementedError, match="unit pruning under sharding"):

        class _Bad(px.Network[None]):
            forward_pass = LinearForward()
            prune_unit = Negative()
            unit_capacity = 8
            sharding = px.ShardSpec("conns", 2)


def test_unit_pruning_under_sharding_is_refused_at_build() -> None:
    net = _net()
    with pytest.raises(NotImplementedError, match="unit pruning under sharding"):
        _build(net, sharding=px.ShardSpec("conns", 2))
    # Without prune_unit, the same sharded build succeeds.
    net.prune_unit = None
    static, _ = _build(net, sharding=px.ShardSpec("conns", 2))
    assert static.sharding == px.ShardSpec("conns", 2)
