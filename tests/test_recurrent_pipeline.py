"""Recurrent (cyclic) structure under pipeline propagation.

Topological propagation levels the graph with Kahn's algorithm and rejects
cycles; pipeline propagation admits them -- every conn lands in one flat
bucket and each step is a single synchronous sweep reading the previous
step's activations, so a reservoir->reservoir edge realizes the recurrent
feedback x(t) = f(W x(t-1)). These tests lock in that a cyclic graph builds
in pipeline mode, still raises in topological mode, and that the feedback is
numerically the one-step recurrence (examples/echo_state_network.py is the
worked reservoir).
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

import plastax as px
from plastax.views import UnitWrite


class _SumForward(px.ForwardPass):
    """Weighted-sum forward, identity apply onto ACTIVATION."""

    combine = px.monoid.sum_

    def map(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: None,
    ) -> jnp.ndarray:
        return c[px.WEIGHT, cid] * u[px.ACTIVATION, src]

    def apply(
        self, u: px.UnitView, i: px.UnitIdx, g: None, acc: jnp.ndarray
    ) -> UnitWrite:
        return UnitWrite.of((px.ACTIVATION, acc))


class _PipelineRecurrent(px.Network[None]):
    forward_pass = _SumForward()
    propagation = px.Propagation.PIPELINE


class _TopologicalRecurrent(px.Network[None]):
    forward_pass = _SumForward()
    propagation = px.Propagation.TOPOLOGICAL


# unit 0: input; unit 1: reservoir with a self-loop (1->1) and an input
# edge (0->1). A self-loop is a cycle -- Kahn never dequeues unit 1 -- so
# this is the minimal graph that topological mode rejects.
_W_IN, _W_SELF = 0.5, 0.8


def _build(
    net: type[px.Network[None]],
) -> tuple[px.NetworkStatic, px.NetworkState[None]]:
    builder = px.NetworkBuilder(net, None)
    builder.add_unit()  # 0: input
    builder.add_unit()  # 1: reservoir
    builder.mark_input(0)
    builder.add_conn(0, 1, weight=_W_IN)
    builder.add_conn(1, 1, weight=_W_SELF)  # self-loop: recurrent feedback
    return builder.finalize()


def test_pipeline_builds_a_cyclic_reservoir() -> None:
    # The whole point: a graph with a cycle finalizes in pipeline mode.
    static, state = _build(_PipelineRecurrent)
    assert len(static.level_capacities) == 1  # single flat bucket
    assert int(jnp.sum(~state.conns[0][px.DEAD.name])) == 2


def test_topological_still_rejects_a_cycle() -> None:
    with pytest.raises(ValueError, match="DAG"):
        _build(_TopologicalRecurrent)


def test_self_loop_realizes_one_step_recurrence() -> None:
    static, state = _build(_PipelineRecurrent)
    step = px.make_step(_PipelineRecurrent, static)

    drive = np.array([1.0, -0.5, 0.25, 2.0], dtype=np.float32)

    # Reference: x1(t) = w_in * u(t) + w_self * x1(t-1), x1(-1) = 0.
    ref = 0.0
    expected = []
    for u_t in drive:
        ref = _W_IN * float(u_t) + _W_SELF * ref
        expected.append(ref)

    got = []
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error", message=".*Some donated buffers were not usable.*"
        )
        for u_t in drive:
            inputs = px.StepInputs(
                inputs=jnp.asarray([u_t], dtype=jnp.float32), targets=None
            )
            result = step(state, inputs)
            state = result.state
            got.append(float(np.asarray(state.units[px.ACTIVATION.name])[1]))

    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)
    # A stateless (non-recurrent) sweep would give x1(t) = w_in * u(t),
    # dropping the w_self * x1(t-1) term -- assert the feedback is present.
    assert got[1] != pytest.approx(_W_IN * float(drive[1]))
