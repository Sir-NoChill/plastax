"""Growth-only overflow recovery: the Driver finishes an overflowing growth
inside the same step.

A step whose growth claim drops winners for lack of room is completed by the
Driver: the full buckets grow and the dropped winners claim the new room. No
other phase runs again, so the learning update is applied once and the step
counter advances once, and the result equals a run whose buckets were large
enough from the start.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import plastax as px
from plastax.state import live_conn_count

# Units 0, 1: inputs (level 0); 2..5: hidden (level 1); 6: output (level 2).
_EDGES = ((0, 2), (1, 3), (0, 4), (1, 5), (2, 6), (3, 6), (4, 6), (5, 6))
_WEIGHTS = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
_NEW_WEIGHT = 0.5
_STEP = px.StepInputs(inputs=jnp.asarray([1.0, -1.0], jnp.float32), targets=None)
_BATCH = px.StepInputs(
    inputs=jnp.asarray([[1.0, -1.0], [0.5, 2.0]], jnp.float32), targets=None
)


class _SumForward(px.ForwardPass):
    combine = px.monoid.sum_

    def map(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: jax.Array,
    ) -> jax.Array:
        return c[px.WEIGHT, cid] * u[px.ACTIVATION, src]

    def apply(
        self, u: px.UnitView, i: px.UnitIdx, g: jax.Array, acc: jax.Array
    ) -> px.UnitWrite:
        return px.UnitWrite.of((px.ACTIVATION, acc))


class _BumpWeights(px.UpdateConn):
    """The learning update: every live connection's weight += 1 per step."""

    def incoming(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: jax.Array,
    ) -> px.ConnWrite:
        return px.ConnWrite.of((px.WEIGHT, c[px.WEIGHT, cid] + jnp.float32(1.0)))

    def outgoing(
        self,
        u: px.UnitView,
        src: px.UnitIdx,
        dst: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: jax.Array,
    ) -> px.ConnWrite:
        return px.ConnWrite.of()


class _DeeperGrowth(px.ScoreAddConn[jax.Array]):
    """Each source level's three best strictly-deeper pairs, by -(10 src + dst)."""

    max_new_per_level = 3

    def score(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: jax.Array
    ) -> jax.Array:
        deeper = u[px.LEVEL, dst] > u[px.LEVEL, src]
        rank = -(10 * src + dst).astype(jnp.float32)
        return jnp.where(deeper, rank, -jnp.inf)

    def init(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: jax.Array
    ) -> px.ConnWrite:
        return px.ConnWrite.of((px.WEIGHT, jnp.float32(_NEW_WEIGHT)))


class _CountSteps(px.ResetGlobal[jax.Array]):
    """A phase after growth: counts how often the step's tail runs."""

    def reset(self, g: jax.Array) -> jax.Array:
        return g + jnp.int32(1)


class _RetryNet(px.Network[jax.Array]):
    forward_pass = _SumForward()
    update_conn = _BumpWeights()
    add_conn = _DeeperGrowth()
    reset_global = _CountSteps()
    propagation = px.Propagation.TOPOLOGICAL
    batch_reduction = px.MeanFloatFirstRest()


def _build(level0_capacity: int) -> tuple[px.NetworkStatic, px.NetworkState[jax.Array]]:
    """The net with bucket 0 cut or padded to `level0_capacity` slots.

    Bucket 0 holds the 4 input edges first; at 5 slots it has room for one of
    level 0's three winners, so a step overflows.
    """
    static, state = px.NetworkBuilder.from_edges(
        _RetryNet,
        7,
        np.asarray([e[0] for e in _EDGES], np.int32),
        np.asarray([e[1] for e in _EDGES], np.int32),
        weights=np.asarray(_WEIGHTS, np.float32),
        input_ids=[0, 1],
        output_ids=[6],
        globals_=jnp.int32(0),
    )
    assert static.level_capacities[0] >= level0_capacity
    bucket0 = {name: col[:level0_capacity] for name, col in state.conns[0].items()}
    assert int(jnp.sum(~bucket0[px.DEAD.name])) == 4
    return (
        dataclasses.replace(
            static,
            level_capacities=(level0_capacity, *static.level_capacities[1:]),
        ),
        dataclasses.replace(state, conns=(bucket0, *state.conns[1:])),
    )


def _presized(
    capacities: tuple[int, ...],
) -> tuple[px.NetworkStatic, px.NetworkState[jax.Array]]:
    """The cut net with every bucket padded (never-used slots) to `capacities`."""
    static, state = _build(5)
    defaults = {spec.name: spec for spec in static.conn_fields}
    conns = []
    for bucket, cap in zip(state.conns, capacities, strict=True):
        pad = cap - bucket[px.DEAD.name].shape[0]
        assert pad >= 0
        conns.append(
            {
                name: jnp.concatenate(
                    [
                        col,
                        jnp.full(
                            (pad,),
                            np.asarray(defaults[name].default),
                            defaults[name].dtype,
                        ),
                    ]
                )
                for name, col in bucket.items()
            }
        )
    return (
        dataclasses.replace(static, level_capacities=capacities),
        dataclasses.replace(state, conns=tuple(conns)),
    )


def _edges(state: px.NetworkState[jax.Array]) -> dict[tuple[int, int], list[float]]:
    out: dict[tuple[int, int], list[float]] = {}
    for bucket in state.conns:
        live = ~np.asarray(bucket[px.DEAD.name])
        for s, d, w in zip(
            np.asarray(bucket[px.FROM_ID.name])[live],
            np.asarray(bucket[px.TO_ID.name])[live],
            np.asarray(bucket[px.WEIGHT.name])[live],
            strict=True,
        ):
            out.setdefault((int(s), int(d)), []).append(float(w))
    return {k: sorted(v) for k, v in out.items()}


def _assert_states_equal(a: px.NetworkState[jax.Array], b: Any) -> None:
    leaves_a, tree_a = jax.tree_util.tree_flatten(a)
    leaves_b, tree_b = jax.tree_util.tree_flatten(b)
    assert tree_a == tree_b
    for x, y in zip(leaves_a, leaves_b, strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_the_bare_step_overflows() -> None:
    """The scenario under test: one step at 5 slots drops level 0's winners."""
    static, state = _build(5)
    result = px.make_step(_RetryNet, static)(state, _STEP)
    assert bool(result.overflow)
    assert int(result.state.grown) == 1 + 3
    assert result.growth_remainder is not None
    dropped = [int(np.sum(c.growable)) for c in result.growth_remainder.claims]
    assert dropped == [2, 0, 0]


@pytest.mark.parametrize("batch_size", [None, 2])
def test_an_overflowing_step_runs_every_other_phase_once(
    batch_size: int | None,
) -> None:
    static, state = _build(5)
    driver = px.Driver(_RetryNet, static, state, batch_size=batch_size)
    driver.step(_STEP if batch_size is None else _BATCH)
    out = driver.state

    assert driver.static.level_capacities[0] > 5, "the full bucket regrew"
    assert int(out.step) == 1, "the step counter advances once"
    assert int(out.globals_) == 1, "the phases after growth run once"
    assert not bool(out.overflow)
    assert int(out.grown) == 6
    edges = _edges(out)
    # The learning update ran once: each built edge is exactly one bump up.
    for (s, d), w in zip(_EDGES, _WEIGHTS, strict=True):
        assert edges[(s, d)][-1] == w + 1.0, (s, d)
    # Every winner of the step, each committed once and never updated.
    grown = {
        pair: [w for w in ws if w == _NEW_WEIGHT]
        for pair, ws in edges.items()
        if _NEW_WEIGHT in ws
    }
    assert grown == {
        (0, 2): [_NEW_WEIGHT],
        (0, 3): [_NEW_WEIGHT],
        (0, 4): [_NEW_WEIGHT],
        (2, 6): [_NEW_WEIGHT],
        (3, 6): [_NEW_WEIGHT],
        (4, 6): [_NEW_WEIGHT],
    }


@pytest.mark.parametrize("batch_size", [None, 2])
def test_the_recovered_step_equals_a_presized_run(batch_size: int | None) -> None:
    static, state = _build(5)
    driver = px.Driver(_RetryNet, static, state, batch_size=batch_size)
    inputs = _STEP if batch_size is None else _BATCH
    driver.step(inputs)

    big_static, big_state = _presized(driver.static.level_capacities)
    want = px.make_step(_RetryNet, big_static, batch_size=batch_size)(big_state, inputs)
    assert not bool(want.overflow)
    _assert_states_equal(driver.state, want.state)
    assert int(live_conn_count(driver.state)) == len(_EDGES) + 6
