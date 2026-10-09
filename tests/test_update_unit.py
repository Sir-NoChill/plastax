"""The update_unit phase: one write per live unit, after backward.

The net below threads a value through the phase order: update_unit reads the
backward's GRAD to write TRACE, and update_conn reads TRACE. So a unit update
in the wrong place (before backward, or after the connection update) changes
the weights, and the checks below compute what the right order gives.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import plastax as px
from plastax.phases import build_phases, plan_prune_fusion

GRAD = px.FieldSpec.float32("test/grad")
LOSS_GRAD = px.FieldSpec.float32("test/loss_grad")
TRACE = px.FieldSpec.float32("test/trace")
_LR = np.float32(0.125)

# Inputs 0, 1 -> hidden 2 -> output 3, plus a skip edge 1 -> 3.
_N = 4
_SRC = np.asarray([0, 1, 2, 1], np.int32)
_DST = np.asarray([2, 2, 3, 3], np.int32)
_W = np.asarray([0.5, -0.25, 0.75, 0.375], np.float32)


class Forward(px.ForwardPass):
    """activation = sum(w * activation[src]) + 1."""

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
        return px.UnitWrite.of((px.ACTIVATION, acc + jnp.float32(1.0)))


class Backward(px.BackwardPass):
    """grad = sum(w * grad[dst]) + loss_grad."""

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
        return px.UnitWrite.of((GRAD, acc + u[LOSS_GRAD, i]))


class Loss(px.Loss):
    """sum(0.5 * (activation - target)^2), seeding the difference."""

    seed_field = LOSS_GRAD

    def calculate_loss(
        self, u: px.UnitView, outputs: jax.Array, targets: jax.Array, g: None
    ) -> tuple[jax.Array, jax.Array]:
        del g
        diff = u.gather(px.ACTIVATION, outputs) - targets
        return jnp.sum(jnp.float32(0.5) * diff * diff), diff


class Trace(px.UpdateUnit[None]):
    """trace = trace / 2 + grad * activation + 1."""

    def update(self, u: px.UnitView, i: px.UnitIdx, g: None) -> px.UnitWrite:
        del g
        value = (
            u[TRACE, i] * jnp.float32(0.5)
            + u[GRAD, i] * u[px.ACTIVATION, i]
            + jnp.float32(1.0)
        )
        return px.UnitWrite.of((TRACE, value))


class TraceRule(px.UpdateConn):
    """w -= lr * trace[dst] * activation[src]."""

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
        step = jnp.float32(_LR) * u[TRACE, dst] * u[px.ACTIVATION, src]
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


def _net(
    *, update_unit: bool = True, capacity: int | None = None
) -> type[px.Network[None]]:
    rule = Trace() if update_unit else None

    class _Net(px.Network[None]):
        forward_pass = Forward()
        backward_pass = Backward()
        loss = Loss()
        update_unit = rule
        update_conn = TraceRule()
        extra_unit_fields = (GRAD, LOSS_GRAD, TRACE)
        unit_capacity = capacity
        propagation = px.Propagation.TOPOLOGICAL

    return _Net


def _build(net: type[px.Network[None]]) -> tuple[px.NetworkStatic, Any]:
    return px.NetworkBuilder.from_edges(
        net,
        _N,
        _SRC,
        _DST,
        weights=_W,
        input_ids=(0, 1),
        output_ids=(3,),
        globals_=None,
    )


def _weights(state: Any) -> dict[tuple[int, int], float]:
    out = {}
    for bucket in state.conns:
        dead = np.asarray(bucket[px.DEAD.name])
        for s, d, w in zip(
            np.asarray(bucket[px.FROM_ID.name])[~dead],
            np.asarray(bucket[px.TO_ID.name])[~dead],
            np.asarray(bucket[px.WEIGHT.name])[~dead],
            strict=True,
        ):
            out[int(s), int(d)] = float(w)
    return out


def _inputs(t: int = 0) -> px.StepInputs:
    return px.StepInputs(
        inputs=jnp.asarray([0.5 + t, -1.0], jnp.float32),
        targets=jnp.asarray([0.25 * t], jnp.float32),
    )


def test_update_unit_runs_after_backward_and_before_update_conn() -> None:
    """TRACE is written from this step's GRAD, and the weights read it."""
    net = _net()
    static, state = _build(net)
    result = px.make_step(net, static)(state, _inputs())
    units = {k: np.asarray(v) for k, v in result.state.units.items()}
    act, grad = units[px.ACTIVATION.name], units[GRAD.name]
    # Every unit (inputs and the output included) took the update, from the
    # post-backward GRAD: the trace started at 0.
    want_trace = grad * act + np.float32(1.0)
    np.testing.assert_allclose(units[TRACE.name], want_trace, rtol=1e-6)
    assert (units[TRACE.name] != 0.0).all()
    assert grad[2] != 0.0 and grad[3] != 0.0  # the backward fed the trace
    # The connection update read the written trace.
    got = _weights(result.state)
    for (s, d), w in zip(zip(_SRC, _DST, strict=True), _W, strict=True):
        want = np.float32(w) - _LR * units[TRACE.name][d] * act[s]
        np.testing.assert_allclose(got[int(s), int(d)], want, rtol=1e-6)


def test_update_unit_changes_unit_state_over_steps() -> None:
    """The trace accumulates: trace_t = trace_{t-1} / 2 + grad * act + 1."""
    net = _net()
    static, state = _build(net)
    step = px.make_step(net, static)
    trace = np.zeros((_N,), np.float32)
    for t in range(3):
        state = step(state, _inputs(t)).state
        grad = np.asarray(state.units[GRAD.name])
        act = np.asarray(state.units[px.ACTIVATION.name])
        trace = trace * np.float32(0.5) + grad * act + np.float32(1.0)
        np.testing.assert_allclose(
            np.asarray(state.units[TRACE.name]), trace, rtol=1e-6
        )


def test_update_unit_skips_slots_holding_no_live_unit() -> None:
    """With a unit capacity, a pruned unit and the free slots are not updated."""
    net = _net(capacity=_N + 2)
    static, state = _build(net)
    # Hidden unit 2 pruned, with its incident edges tombstoned.
    conns = []
    for bucket in state.conns:
        touches = (bucket[px.FROM_ID.name] == 2) | (bucket[px.TO_ID.name] == 2)
        conns.append({**bucket, px.DEAD.name: bucket[px.DEAD.name] | touches})
    units = dict(state.units)
    units[px.PRUNED.name] = units[px.PRUNED.name].at[2].set(True)
    state = dataclasses.replace(state, units=units, conns=tuple(conns))
    state = px.make_step(net, static)(state, _inputs()).state
    trace = np.asarray(state.units[TRACE.name])
    assert trace[[2, 4, 5]].tolist() == [0.0, 0.0, 0.0]
    assert (trace[[0, 1, 3]] != 0.0).all()


def test_the_phase_sits_between_backward_and_update_conn() -> None:
    """Declaring update_unit adds exactly one phase, before update_conn."""
    with_net, without_net = _net(), _net(update_unit=False)
    static, _ = _build(with_net)
    with_phases = build_phases(with_net, static)
    without_phases = build_phases(without_net, static)
    assert len(with_phases) == len(without_phases) + 1
    names = [p.__name__ for p in with_phases]
    assert names.index("update_unit_phase") == names.index("update_conn_phase") - 1
    assert "update_unit_phase" not in [p.__name__ for p in without_phases]


@pytest.mark.parametrize("layout", ["edge_list", "triton"])
def test_a_batch_of_one_is_the_streaming_step(layout: str) -> None:
    """update_unit runs per sample before update_conn, as when streaming."""
    net = _net()
    static, state = _build(net)
    stream = px.make_step(net, static)
    batched = px.make_step(net, static, batch_size=1, layout=layout)  # type: ignore[arg-type]
    s_stream, s_batch = state, jax.tree.map(jnp.copy, state)
    for t in range(4):
        x = _inputs(t)
        assert x.targets is not None
        s_stream = stream(s_stream, x).state
        s_batch = batched(
            s_batch, px.StepInputs(inputs=x.inputs[None], targets=x.targets[None])
        ).state
    for name, col in s_stream.units.items():
        np.testing.assert_allclose(
            np.asarray(s_batch.units[name]), np.asarray(col), rtol=1e-6, err_msg=name
        )
    got, want = _weights(s_batch), _weights(s_stream)
    assert got.keys() == want.keys()
    for key, w in want.items():
        np.testing.assert_allclose(got[key], w, rtol=1e-6, err_msg=str(key))
    assert want != {
        (int(s), int(d)): float(w) for s, d, w in zip(_SRC, _DST, _W, strict=True)
    }


def test_a_batch_averages_the_per_sample_unit_updates() -> None:
    """Each sample updates its own units; the step keeps their batch mean."""
    net = _net()
    static, state = _build(net)
    batched = px.make_step(net, static, batch_size=2, layout="edge_list")
    x = jnp.asarray([[0.5, -1.0], [1.5, 0.25]], jnp.float32)
    y = jnp.asarray([[0.0], [1.0]], jnp.float32)
    out = batched(
        jax.tree.map(jnp.copy, state), px.StepInputs(inputs=x, targets=y)
    ).state
    traces = []
    stream = px.make_step(net, static)
    for b in range(2):
        s = stream(
            jax.tree.map(jnp.copy, state), px.StepInputs(inputs=x[b], targets=y[b])
        ).state
        traces.append(np.asarray(s.units[TRACE.name]))
    np.testing.assert_allclose(
        np.asarray(out.units[TRACE.name]), np.mean(traces, axis=0), rtol=1e-6
    )


def test_a_non_conforming_update_unit_is_rejected() -> None:
    with pytest.raises(TypeError, match="update_unit must satisfy UpdateUnit"):

        class _Bad(px.Network[None]):
            forward_pass = Forward()
            update_unit = object()  # type: ignore[assignment]


class _TracePrune(px.PruneConn):
    """Reads TRACE, which update_unit writes between forward and prune."""

    def predicate(
        self, u: px.UnitView, c: px.ConnView, cid: px.ConnIdx, g: None
    ) -> jax.Array:
        del g
        return u[TRACE, c[px.FROM_ID, cid]] > jnp.float32(1000.0)


def test_prune_fusion_sees_the_unit_update_between_forward_and_prune() -> None:
    """A predicate reading a field update_unit writes is not fused."""
    net = _net()
    net.prune_conn = _TracePrune()
    net.update_conn = None
    static, state = _build(net)
    plan = plan_prune_fusion(net, static, state.globals_, engine="xla")
    assert not plan.fused
    assert "test/trace" in plan.reason
    net.update_unit = None
    plan = plan_prune_fusion(net, static, state.globals_, engine="xla")
    assert plan.fused, plan
