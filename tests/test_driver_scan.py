"""`Driver.scan`: a chunk of steps as one device-side loop, exactly `step`.

The loop runs the step K times in one jitted call and stops itself after a
step that overflowed or (TOPOLOGICAL only) needs a resort; the host finishes
that step the way `Driver.step` does and re-enters the same compiled loop at
the next step. These tests pin that the result, state and records alike,
is step for step what `Driver.step` produces through overflow, regrow and
resort; that the loop compiles once per static configuration; that a
PIPELINE network never resorts; and the argument contract.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import plastax as px
import plastax.driver
from test_growth_retry import _BATCH, _STEP, _RetryNet
from test_growth_retry import _build as _build_retry
from test_inplace_churn import _build_with, _GrowingNet

_ACT = px.ACTIVATION.name


def _activation(result: px.StepResult[Any]) -> jax.Array:
    return result.state.units[_ACT]


def _grown(result: px.StepResult[Any]) -> jax.Array:
    return result.state.grown


def _assert_states_equal(a: Any, b: Any) -> None:
    leaves_a, tree_a = jax.tree_util.tree_flatten(a)
    leaves_b, tree_b = jax.tree_util.tree_flatten(b)
    assert tree_a == tree_b
    for x, y in zip(leaves_a, leaves_b, strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def _repeat(inputs: px.StepInputs, k: int) -> px.StepInputs:
    return jax.tree_util.tree_map(lambda a: jnp.stack([a] * k), inputs)


@pytest.mark.parametrize("batch_size", [None, 2])
def test_scan_equals_steps_through_an_overflow(batch_size: int | None) -> None:
    """The first step overflows (test_growth_retry); the loop halts on it,
    the host regrows and claims the remainder, and the loop resumes."""
    one = _STEP if batch_size is None else _BATCH
    k = 5
    static, state = _build_retry(5)
    by_step = px.Driver(_RetryNet, static, state, batch_size=batch_size)
    want_records = []
    for _ in range(k):
        by_step.step(one)
        want_records.append(np.asarray(by_step.state.grown))

    static, state = _build_retry(5)
    scanned = px.Driver(_RetryNet, static, state, batch_size=batch_size)
    got = scanned.scan(_repeat(one, k), record=_grown)

    assert scanned.static == by_step.static
    assert scanned.static.level_capacities[0] > 5, "the full bucket regrew"
    _assert_states_equal(scanned.state, by_step.state)
    # The overflowing step is re-recorded once its growth is complete.
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want_records))
    assert len(scanned._scan_fns) == 2, "one loop per static configuration"


def test_scan_equals_steps_through_growth_and_resort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"grow": 0, "resort": 0}
    grow, resort = plastax.driver.grow_bucket, plastax.driver.topo.resort

    def counted_grow(*a: Any) -> Any:
        calls["grow"] += 1
        return grow(*a)

    def counted_resort(*a: Any) -> Any:
        calls["resort"] += 1
        return resort(*a)

    monkeypatch.setattr(plastax.driver, "grow_bucket", counted_grow)
    monkeypatch.setattr(plastax.driver.topo, "resort", counted_resort)

    x = np.linspace(-1.0, 1.0, 24).astype(np.float32)
    one = px.StepInputs(inputs=jnp.asarray(x), targets=None)
    chunk, chunks = 8, 6

    static, state = _build_with(_GrowingNet, headroom=0.0)
    by_step = px.Driver(_GrowingNet, static, state)
    want = []
    for _ in range(chunk * chunks):
        by_step.step(one)
        want.append(np.asarray(by_step.state.units[_ACT]))
    step_calls = dict(calls)
    assert step_calls["grow"] and step_calls["resort"], step_calls

    calls.update(grow=0, resort=0)
    static, state = _build_with(_GrowingNet, headroom=0.0)
    scanned = px.Driver(_GrowingNet, static, state)
    got = []
    for _ in range(chunks):
        got.append(np.asarray(scanned.scan(_repeat(one, chunk), record=_activation)))

    assert calls == step_calls, "the same structural events, in the same steps"
    assert scanned.static == by_step.static
    _assert_states_equal(scanned.state, by_step.state)
    np.testing.assert_array_equal(np.concatenate(got), np.asarray(want))


def test_scan_runs_only_the_first_steps_of_a_padded_chunk() -> None:
    static, state = _build_retry(5)
    driver = px.Driver(_RetryNet, static, state)
    got = driver.scan(_repeat(_STEP, 6), record=_grown, steps=4)
    assert int(driver.state.step) == 4
    assert int(driver.state.globals_) == 4
    np.testing.assert_array_equal(np.asarray(got[4:]), 0)
    assert int(got[0]) == 6, "the overflowing first step's full growth"

    static, state = _build_retry(5)
    nothing = px.Driver(_RetryNet, static, state)
    assert np.all(np.asarray(nothing.scan(_repeat(_STEP, 3), steps=0)) == 0)
    assert int(nothing.state.step) == 0


def test_scan_default_record_is_the_loss_and_the_state_is_donated() -> None:
    static, state = _build_retry(5)
    driver = px.Driver(_RetryNet, static, state)
    before = driver.state
    out = driver.scan(_repeat(_STEP, 3))
    assert out.shape == (3,) and out.dtype == jnp.float32
    assert all(leaf.is_deleted() for leaf in jax.tree_util.tree_leaves(before))


def test_scan_rejects_a_chunkless_or_ragged_input() -> None:
    static, state = _build_retry(5)
    driver = px.Driver(_RetryNet, static, state)
    with pytest.raises(ValueError, match="chunk axis"):
        driver.scan(_STEP)
    with pytest.raises(ValueError, match="steps must be"):
        driver.scan(_repeat(_STEP, 3), steps=4)


# --- a PIPELINE network never resorts -----------------------------------------


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
        del dst, g
        return c[px.WEIGHT, cid] * u[px.ACTIVATION, src]

    def apply(
        self, u: px.UnitView, i: px.UnitIdx, g: jax.Array, acc: jax.Array
    ) -> px.UnitWrite:
        del u, i, g
        return px.UnitWrite.of((px.ACTIVATION, acc))


class _SameLevelGrowth(px.ScoreAddConn[jax.Array]):
    """Grows one edge per step between the two hidden units, which sit at the
    same level: a same-level edge, which raises `needs_resort`."""

    max_new_per_level = 1
    max_level_gap = 8

    def score(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: jax.Array
    ) -> jax.Array:
        del g
        same = (u[px.LEVEL, dst] == u[px.LEVEL, src]) & (u[px.LEVEL, dst] > 0)
        return jnp.where(same & (dst > src), 0.0, -jnp.inf).astype(jnp.float32)

    def init(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: jax.Array
    ) -> px.ConnWrite:
        del u, src, dst, g
        return px.ConnWrite.of((px.WEIGHT, jnp.float32(0.5)))


class _Count(px.ResetGlobal[jax.Array]):
    def reset(self, g: jax.Array) -> jax.Array:
        return g + jnp.int32(1)


def _chain_net(propagation: px.Propagation) -> type[px.Network[jax.Array]]:
    class Chain(px.Network[jax.Array]):
        forward_pass = _SumForward()
        add_conn = _SameLevelGrowth()
        reset_global = _Count()

    Chain.propagation = propagation
    return Chain


def _build_chain(
    propagation: px.Propagation,
) -> tuple[type[px.Network[jax.Array]], px.NetworkStatic, px.NetworkState[jax.Array]]:
    net = _chain_net(propagation)
    # 0 -> {1, 2} -> 3: the hidden units 1 and 2 share level 1.
    static, state = px.NetworkBuilder.from_edges(
        net,
        4,
        np.asarray([0, 0, 1, 2], np.int32),
        np.asarray([1, 2, 3, 3], np.int32),
        weights=np.ones(4, np.float32),
        input_ids=[0],
        output_ids=[3],
        globals_=jnp.int32(0),
        capacity_headroom=1.0,
    )
    return net, static, state


_ONE = px.StepInputs(inputs=jnp.asarray([1.0], jnp.float32), targets=None)


@pytest.mark.parametrize("how", ["step", "scan"])
def test_a_pipeline_network_never_resorts(
    how: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"resort": 0}
    resort = plastax.driver.topo.resort

    def counted(*a: Any) -> Any:
        calls["resort"] += 1
        return resort(*a)

    monkeypatch.setattr(plastax.driver.topo, "resort", counted)

    net, static, state = _build_chain(px.Propagation.TOPOLOGICAL)
    topological = px.Driver(net, static, state)
    topological.step(_ONE)
    assert calls["resort"] == 1, "the same-level edge resorts a topological net"
    assert not bool(topological.state.needs_resort)

    calls["resort"] = 0
    net, static, state = _build_chain(px.Propagation.PIPELINE)
    pipeline = px.Driver(net, static, state)
    if how == "step":
        for _ in range(3):
            pipeline.step(_ONE)
    else:
        pipeline.scan(_repeat(_ONE, 3))
    assert calls["resort"] == 0
    assert bool(pipeline.state.needs_resort), "the flag is left for the caller"
    assert pipeline.static == static, "no retrace"
    assert int(pipeline.state.globals_) == 3
    assert int(pipeline.state.grown) == 1
    if how == "scan":
        assert len(pipeline._scan_fns) == 1
