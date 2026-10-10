"""The add_unit phase: lowest-free-id unit addition.

The net below is the linear one of test_prune_unit (activation =
sum(w * activation[src])):

    unit:    0  1  2  3  4  5      edges: 0->2, 1->3, 2->5, 5->4, 3->4
    level:   0  0  1  1  3  2      inputs 0, 1; output 4

Most rules here spawn from the `test/spawn` column, so a test picks the
parents (and their level offsets) by writing that column; the child records
its parent in `test/parent`. Built with 6 units in `_CAPACITY` slots, the free
slots start at 6.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import plastax as px
from plastax.phases import build_add_unit_phase, build_phases

SPAWN = px.FieldSpec.int32("test/spawn", default=0)
OFFSET = px.FieldSpec.int32("test/offset", default=1)
PARENT = px.FieldSpec.int32("test/parent", default=-1)
TRACE = px.FieldSpec.float32("test/trace", default=0.25)

_N = 6
_CAPACITY = 8
_INPUTS = (0, 1)
_OUTPUTS = (4,)
_SRC = np.asarray([0, 1, 2, 5, 3], np.int32)
_DST = np.asarray([2, 3, 5, 4, 4], np.int32)
_W = np.asarray([1.0, -1.0, -1.0, 1.0, -2.0], np.float32)
_LEVELS = [0, 0, 1, 1, 3, 2]
_ALL_EDGES = {(int(s), int(d)) for s, d in zip(_SRC, _DST, strict=True)}
_ZERO_INPUTS = jnp.zeros((2,), jnp.float32)


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


class _Child:
    """Shared init: record the parent, take half its activation."""

    def init(
        self, u: px.UnitView, child: px.UnitIdx, parent: px.UnitIdx, g: None
    ) -> px.UnitWrite:
        del child, g
        return px.UnitWrite.of(
            (PARENT, parent), (px.ACTIVATION, u[px.ACTIVATION, parent] / 2)
        )


class FromColumn(_Child, px.AddUnit[None]):
    """Spawn where `test/spawn` is set, at the `test/offset` level offset."""

    def spawn(
        self, u: px.UnitView, parent: px.UnitIdx, g: None
    ) -> tuple[jax.Array, jax.Array]:
        del g
        return u[SPAWN, parent] != 0, u[OFFSET, parent]


class Everything(_Child, px.AddUnit[None]):
    """Every slot asks to spawn, free slots included."""

    def spawn(
        self, u: px.UnitView, parent: px.UnitIdx, g: None
    ) -> tuple[jax.Array, jax.Array]:
        del u, parent, g
        return jnp.bool_(True), jnp.int32(1)


class HighActivation(_Child, px.AddUnit[None]):
    """Unit 2 spawns when its activation is at least 1/2."""

    def spawn(
        self, u: px.UnitView, parent: px.UnitIdx, g: None
    ) -> tuple[jax.Array, jax.Array]:
        del g
        return (parent == 2) & (u[px.ACTIVATION, parent] >= 0.5), jnp.int32(1)


class Never(_Child, px.AddUnit[None]):
    """Never spawns."""

    def spawn(
        self, u: px.UnitView, parent: px.UnitIdx, g: None
    ) -> tuple[jax.Array, jax.Array]:
        del u, parent, g
        return jnp.bool_(False), jnp.int32(1)


class PruneMarked(px.PruneUnit[None]):
    """Prune a unit whose `test/spawn` is negative."""

    def predicate(self, u: px.UnitView, i: px.UnitIdx, g: None) -> jax.Array:
        del g
        return u[SPAWN, i] < 0


def _net(
    rule: px.AddUnit[None] | None = None,
    *,
    capacity: int = _CAPACITY,
    interval: int = 1,
    max_levels: int = 1024,
    prune_unit: px.PruneUnit[None] | None = None,
    add_conn: px.ProposeAddConn[None] | None = None,
) -> type[px.Network[None]]:
    class _Net(px.Network[None]):
        forward_pass = LinearForward()
        extra_unit_fields = (SPAWN, OFFSET, PARENT, TRACE)
        unit_capacity = capacity
        structural_interval = interval
        propagation = px.Propagation.TOPOLOGICAL
        batch_reduction = px.MeanFloatFirstRest()

    _Net.add_unit = FromColumn() if rule is None else rule
    _Net.max_levels = max_levels
    _Net.prune_unit = prune_unit
    _Net.add_conn = add_conn
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


def _set(state: Any, spec: px.FieldSpec[Any], values: dict[int, Any]) -> Any:
    col = state.units[spec.name]
    for i, v in values.items():
        col = col.at[i].set(v)
    return dataclasses.replace(state, units={**state.units, spec.name: col})


def _live_edges(state: Any) -> set[tuple[int, int]]:
    out = set()
    for bucket in state.conns:
        dead = np.asarray(bucket[px.DEAD.name])
        src = np.asarray(bucket[px.FROM_ID.name])[~dead]
        dst = np.asarray(bucket[px.TO_ID.name])[~dead]
        out |= {(int(s), int(d)) for s, d in zip(src, dst, strict=True)}
    return out


def _children(state: Any) -> dict[int, int]:
    """Child slot -> parent, for every slot holding a child."""
    parent = _col(state, PARENT)
    live = ~_col(state, px.PRUNED)
    return {int(c): int(parent[c]) for c in np.flatnonzero(live & (parent >= 0))}


def test_the_ith_spawner_takes_the_ith_lowest_free_id() -> None:
    """Free slots 2 and 5 (pruned) come before the never-allocated 6, 7."""
    net = _net()
    static, state = _build(net)
    state = _set(state, px.PRUNED, {2: True, 5: True})
    state = _set(state, SPAWN, {0: 1, 3: 1, 4: 1})
    out, _ = build_add_unit_phase(net, static)(state, _x(_ZERO_INPUTS))
    assert _children(out) == {2: 0, 5: 3, 6: 4}
    assert _col(out, px.PRUNED).tolist() == [False] * 7 + [True]
    assert int(out.units_added) == 3
    assert not bool(out.unit_overflow)
    # The phase never touches a connection.
    for old, new in zip(state.conns, out.conns, strict=True):
        for name, col in old.items():
            np.testing.assert_array_equal(np.asarray(new[name]), np.asarray(col))
    assert not bool(out.needs_resort)


def test_a_child_starts_at_defaults_then_init() -> None:
    """The reused slot's stale fields reset; init's writes land on top."""
    net = _net()
    static, state = _build(net)
    state = _set(state, px.PRUNED, {2: True})
    state = _set(state, TRACE, {2: 9.0, 3: 7.0})
    state = _set(state, px.ACTIVATION, {2: 5.0, 3: 3.0})
    state = _set(state, SPAWN, {3: 1})
    out, _ = build_add_unit_phase(net, static)(state, _x(_ZERO_INPUTS))
    assert _col(out, TRACE)[2] == np.float32(0.25)  # reset to default
    assert _col(out, px.ACTIVATION)[2] == 1.5  # init: parent activation / 2
    assert _col(out, SPAWN)[2] == 0 and _col(out, OFFSET)[2] == 1
    assert _col(out, px.LEVEL)[2] == 2  # level(3) + 1
    # The parent and every other slot keep their values.
    assert _col(out, TRACE)[3] == 7.0 and _col(out, px.ACTIVATION)[3] == 3.0
    assert _col(out, SPAWN)[3] == 1


def test_an_id_pruned_in_the_same_step_is_reused() -> None:
    """prune_unit frees 2 this step; spawners 3 and 5 take 2 and then 6."""
    net = _net(prune_unit=PruneMarked())
    static, state = _build(net)
    state = _set(state, SPAWN, {2: -1, 3: 1, 5: 1})
    out = px.make_step(net, static)(state, _x(_ZERO_INPUTS)).state
    assert _children(out) == {2: 3, 6: 5}
    assert _col(out, px.PRUNED).tolist() == [False] * 7 + [True]
    # Slot 2's edges died with the pruned unit; the child has none.
    assert _live_edges(out) == _ALL_EDGES - {(0, 2), (2, 5)}
    assert _col(out, px.LEVEL)[[2, 6]].tolist() == [2, 3]
    assert int(out.units_added) == 2


def test_a_spawn_without_a_free_id_is_dropped_and_flags_overflow() -> None:
    """Two free slots, four spawners: the two lowest ids are served."""
    net = _net()
    static, state = _build(net)
    state = _set(state, SPAWN, {0: 1, 1: 1, 3: 1, 4: 1})
    step = px.make_step(net, static)
    out = step(state, _x(_ZERO_INPUTS)).state
    assert _children(out) == {6: 0, 7: 1}
    assert int(out.units_added) == 2
    assert bool(out.unit_overflow)
    # The flag is per step: with no slot free and no spawner, it clears.
    out = _set(out, SPAWN, {0: 0, 1: 0, 3: 0, 4: 0})
    out = step(out, _x(_ZERO_INPUTS)).state
    assert not bool(out.unit_overflow)
    assert int(out.units_added) == 0


@pytest.mark.parametrize(
    ("parent", "offset", "level"),
    [
        (5, 2, 4),  # level 2 + 2, inside the bounds
        (4, 5000, 15),  # level 3 + 5000, clamped to max_levels - 1
        (3, -7, 1),  # level 1 - 7, clamped to 1
        (0, 0, 1),  # an input's level 0 + 0, clamped to 1
    ],
)
def test_the_child_level_is_clamped(parent: int, offset: int, level: int) -> None:
    net = _net(max_levels=16)
    static, state = _build(net)
    state = _set(state, SPAWN, {parent: 1})
    state = _set(state, OFFSET, {parent: offset})
    out, _ = build_add_unit_phase(net, static)(state, _x(_ZERO_INPUTS))
    assert _children(out) == {6: parent}
    assert _col(out, px.LEVEL)[6] == level


def test_children_and_free_slots_are_not_parents() -> None:
    """Every slot asks to spawn; only the 6 live units at phase start do."""
    net = _net(Everything(), capacity=16)
    static, state = _build(net)
    step = px.make_step(net, static)
    out = step(state, _x(_ZERO_INPUTS)).state
    assert _children(out) == {6 + p: p for p in range(6)}
    assert int(out.units_added) == 6
    assert not bool(out.unit_overflow)
    assert _col(out, px.PRUNED)[12:].all()
    # Next step the children are parents too: 12 spawners, 4 free slots.
    out = step(out, _x(_ZERO_INPUTS)).state
    assert {c: p for c, p in _children(out).items() if c >= 12} == {
        12: 0,
        13: 1,
        14: 2,
        15: 3,
    }
    assert int(out.units_added) == 4
    assert bool(out.unit_overflow)


@pytest.mark.parametrize(
    ("inputs", "spawned"),
    [
        # Unit 2 reads 1.5 and -1: sample 0 alone would spawn, the mean 1/4
        # does not.
        ([[1.5, 0.0], [-1.0, 0.0]], False),
        # Unit 2 reads 1/4 and 1: sample 0 alone would not, the mean 5/8 does.
        ([[0.25, 0.0], [1.0, 0.0]], True),
    ],
)
def test_a_batch_spawns_on_the_batch_mean(
    inputs: list[list[float]], spawned: bool
) -> None:
    net = _net(HighActivation())
    static, state = _build(net)
    batched = px.make_step(net, static, batch_size=2, layout="edge_list")
    out = batched(
        state, px.StepInputs(inputs=jnp.asarray(inputs, jnp.float32), targets=None)
    ).state
    assert _children(out) == ({6: 2} if spawned else {})
    if spawned:
        assert _col(out, px.ACTIVATION)[6] == np.float32(0.625 / 2)


def test_a_batch_of_one_is_the_streaming_step() -> None:
    net = _net(HighActivation())
    static, state = _build(net)
    stream = px.make_step(net, static)
    batched = px.make_step(net, static, batch_size=1, layout="edge_list")
    s_stream, s_batch = state, jax.tree.map(jnp.copy, state)
    for v in (1.0, 0.25, 0.75):
        x = jnp.asarray([v, 0.0], jnp.float32)
        s_stream = stream(s_stream, _x(x)).state
        s_batch = batched(s_batch, _x(x[None])).state
        assert int(s_batch.units_added) == int(s_stream.units_added)
    for name, col in s_stream.units.items():
        np.testing.assert_array_equal(
            np.asarray(s_batch.units[name]), np.asarray(col), err_msg=name
        )
    assert _children(s_stream) == {6: 2, 7: 2}


def test_the_structural_interval_gates_unit_addition() -> None:
    """With interval 3, units are added on steps 0 and 3 only."""
    net = _net(capacity=16, interval=3)
    static, state = _build(net)
    state = _set(state, SPAWN, {0: 1})
    step = px.make_step(net, static)
    added = []
    for _ in range(5):
        state = step(state, _x(_ZERO_INPUTS)).state
        added.append(int(state.units_added))
    assert added == [1, 0, 0, 1, 0]
    assert _children(state) == {6: 0, 7: 0}


class _GrowZeroToThree(px.ProposeAddConn[None]):
    """Propose 0 -> 3 (an edge the net lacks), only when units were added."""

    proposer = "global"
    proposals_per_proposer = 1
    max_new_per_level = 1
    trigger = "on_units_added"

    def propose(
        self, u: px.UnitView, j: jax.Array, g: None, rng: px.rng.Rng
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        del u, j, g, rng
        return jnp.int32(0), jnp.int32(3), jnp.float32(1.0)

    def init(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: None
    ) -> px.ConnWrite:
        del u, src, dst, g
        return px.ConnWrite.of((px.WEIGHT, jnp.float32(0.5)))


def test_growth_on_units_added_fires_on_the_phase_output() -> None:
    """The real addition count drives the growth trigger in the same step."""
    net = _net(add_conn=_GrowZeroToThree())
    static, state = _build(net)
    step = px.make_step(net, static)
    state = step(state, _x(_ZERO_INPUTS)).state
    assert int(state.units_added) == 0
    assert (0, 3) not in _live_edges(state)
    state = _set(state, SPAWN, {5: 1})
    state = step(state, _x(_ZERO_INPUTS)).state
    assert int(state.units_added) == 1
    assert (0, 3) in _live_edges(state)
    assert int(state.grown) == 1


def test_the_phase_sits_between_prune_conn_and_add_conn() -> None:
    class _NoConn(px.PruneConn):
        def predicate(
            self, u: px.UnitView, c: px.ConnView, cid: px.ConnIdx, g: None
        ) -> jax.Array:
            del u, g
            return c[px.WEIGHT, cid] > jnp.float32(100.0)

    with_net = _net(add_conn=_GrowZeroToThree())
    with_net.prune_conn = _NoConn()
    without_net = _net(add_conn=_GrowZeroToThree())
    without_net.prune_conn = _NoConn()
    without_net.add_unit = None
    static, _ = _build(with_net)
    with_phases = [p.__name__ for p in build_phases(with_net, static)]
    without_phases = [p.__name__ for p in build_phases(without_net, static)]
    assert "add_unit_phase" not in without_phases
    # The growth phase is named for its trigger gate.
    assert with_phases == [
        "forward_phase",
        "prune_conn_phase",
        "add_unit_phase",
        "gated",
    ]
    assert [p for p in with_phases if p != "add_unit_phase"] == without_phases


def test_a_rule_that_never_spawns_changes_nothing() -> None:
    """Declaring add_unit that never fires leaves every step bit-identical."""
    with_net = _net(Never(), prune_unit=PruneMarked())
    without_net = _net(Never(), prune_unit=PruneMarked())
    without_net.add_unit = None
    static, state = _build(with_net)
    state = _set(state, SPAWN, {2: -1})
    states = {}
    for name, net in (("with", with_net), ("without", without_net)):
        step = px.make_step(net, static)
        s = jax.tree.map(jnp.copy, state)
        for v in (1.0, -0.5, 0.25):
            s = step(s, _x(jnp.asarray([v, 2.0 * v], jnp.float32))).state
        states[name] = s
    leaves_with = jax.tree.leaves(states["with"])
    leaves_without = jax.tree.leaves(states["without"])
    assert len(leaves_with) == len(leaves_without)
    for a, b in zip(leaves_with, leaves_without, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_add_unit_requires_a_unit_capacity() -> None:
    with pytest.raises(TypeError, match="add_unit requires a unit_capacity"):

        class _Bad(px.Network[None]):
            forward_pass = LinearForward()
            add_unit = FromColumn()


def test_a_non_conforming_add_unit_is_rejected() -> None:
    class _SpawnOnly:
        def spawn(self, u: Any, parent: Any, g: Any) -> Any:
            return jnp.bool_(True), jnp.int32(1)

    with pytest.raises(TypeError, match="add_unit must satisfy AddUnit"):

        class _Bad(px.Network[None]):
            forward_pass = LinearForward()
            add_unit = _SpawnOnly()  # type: ignore[assignment]
            unit_capacity = 8


def test_unit_addition_under_sharding_is_refused_at_definition() -> None:
    with pytest.raises(NotImplementedError, match="unit addition under sharding"):

        class _Bad(px.Network[None]):
            forward_pass = LinearForward()
            add_unit = FromColumn()
            unit_capacity = 8
            sharding = px.ShardSpec("conns", 2)


def test_unit_addition_under_sharding_is_refused_at_build() -> None:
    net = _net()
    with pytest.raises(NotImplementedError, match="unit addition under sharding"):
        _build(net, sharding=px.ShardSpec("conns", 2))
    # Without add_unit, the same sharded build succeeds.
    net.add_unit = None
    static, _ = _build(net, sharding=px.ShardSpec("conns", 2))
    assert static.sharding == px.ShardSpec("conns", 2)
