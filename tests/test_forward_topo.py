"""Topological forward level-walk (M3).

Level walk vs a numpy reference, and the milestone's correctness oracle:
one TOPOLOGICAL step on a layered net must equal exactly `num_levels`
PIPELINE steps on the identical net (dispatch_cpu.hpp:41-67 vs :202-223).
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np

import plastax as px
from plastax.views import UnitWrite


class _SumForward(px.ForwardPass):
    """Weighted-sum forward (test_forward_pipeline.py's _SumForward,
    reused verbatim): map = weight * activation[src], combine = sum,
    apply = identity onto ACTIVATION."""

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


class _TopoNet(px.Network[None]):
    forward_pass = _SumForward()
    propagation = px.Propagation.TOPOLOGICAL


class _PipelineNet(px.Network[None]):
    forward_pass = _SumForward()
    propagation = px.Propagation.PIPELINE


# 5 units: 0, 1 input (level 0); 2 hidden (level 1); 3 hidden2 (level 2);
# 4 output (level 3) -- 3 levels/buckets, satisfying "3+ levels". A direct
# 0->4 SKIP edge (source level 0, destination level 3) runs alongside the
# 0->2->3->4 chain: this is the case that actually discriminates a correct
# level walk from a buggy one that resets the accumulator every bucket
# instead of persisting it across buckets (IMPLEMENTATION_PLAN.md M3's
# correctness crux) -- unit 4's accumulator must carry the skip's
# contribution (written while processing bucket 0) all the way through
# buckets 1 and 2 before being finalized at the end.
_W02, _W12, _W23, _W34, _WSKIP04 = 0.5, -0.25, 2.0, -1.5, 3.0
_NUM_LEVELS = 3  # buckets sourced at level 0, 1, 2; unit 4 sits at level 3
_STALE_SENTINEL = -99.0  # pre-step activation for non-input units


def _build(
    net: type[px.Network[None]],
) -> tuple[px.NetworkStatic, px.NetworkState[None]]:
    """Fresh, independently-allocated (static, state) each call (matches
    test_forward_pipeline.py's _build) so a TOPOLOGICAL build and a
    PIPELINE build never alias the same donated buffers."""
    builder = px.NetworkBuilder(net, None)
    builder.add_unit()  # 0: input
    builder.add_unit()  # 1: input
    builder.add_unit(activation=_STALE_SENTINEL)  # 2: hidden
    builder.add_unit(activation=_STALE_SENTINEL)  # 3: hidden2
    builder.add_unit(activation=_STALE_SENTINEL)  # 4: output
    builder.mark_input(0)
    builder.mark_input(1)
    builder.mark_output(4)
    builder.add_conn(0, 2, weight=_W02)
    builder.add_conn(1, 2, weight=_W12)
    builder.add_conn(2, 3, weight=_W23)
    builder.add_conn(3, 4, weight=_W34)
    builder.add_conn(0, 4, weight=_WSKIP04)
    return builder.finalize()


def _numpy_forward_reference(x0: float, x1: float) -> np.ndarray:
    """Hand-computed level walk (dispatch_cpu.hpp:41-67, `L = 1..NumLevels`
    reindexed to 0-based buckets): activation[4] combines the skip's
    contribution (from bucket 0) with the chain's (from bucket 2) in the
    SAME accumulator, finalized only once, after bucket 2."""
    a = np.zeros(5, dtype=np.float32)
    a[0], a[1] = np.float32(x0), np.float32(x1)

    # bucket 0 (sources at level 0): 0->2, 1->2, 0->4 (skip)
    acc2 = np.float32(_W02) * a[0] + np.float32(_W12) * a[1]
    acc4 = np.float32(_WSKIP04) * a[0]
    a[2] = acc2  # finalize level 1

    # bucket 1 (sources at level 1): 2->3
    acc3 = np.float32(_W23) * a[2]
    a[3] = acc3  # finalize level 2

    # bucket 2 (sources at level 2): 3->4 -- ADDS to the skip's acc4, the
    # accumulator carried in from bucket 0 two iterations earlier.
    acc4 = acc4 + np.float32(_W34) * a[3]
    a[4] = acc4  # finalize level 3

    return a


def test_forward_topo_one_step_matches_numpy_level_walk_reference() -> None:
    static, state = _build(_TopoNet)
    assert len(static.level_capacities) == _NUM_LEVELS
    step = px.make_step(_TopoNet, static)

    x0, x1 = 1.0, 2.0
    inputs = px.StepInputs(
        inputs=jnp.asarray([x0, x1], dtype=jnp.float32), targets=None
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error", message=".*Some donated buffers were not usable.*"
        )
        result = step(state, inputs)

    got = np.asarray(result.state.units[px.ACTIVATION.name])
    expected = _numpy_forward_reference(x0, x1)
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)

    # A full forward propagation reaches the output in ONE topological
    # step (unlike pipeline's one-hop-per-call latency): no unit is left at
    # its pre-step sentinel.
    assert not np.any(np.asarray(got[2:]) == _STALE_SENTINEL)
    # Skip-contribution sanity: if the accumulator were wrongly reset
    # between buckets (losing bucket 0's contribution to unit 4), unit 4
    # would equal exactly _W34 * unit 3, omitting the skip term entirely.
    assert not np.isclose(float(got[4]), _W34 * float(got[3]))
    # Input units are never Applied (dispatch_cpu.hpp:59's NumInput bound,
    # generalized to input_ids): activation is exactly the scattered value.
    assert float(got[0]) == x0
    assert float(got[1]) == x1


def test_forward_topo_one_step_equals_pipeline_after_num_levels_steps() -> None:
    """CRITICAL self-validation (IMPLEMENTATION_PLAN.md M3): the identical
    layered net, one TOPOLOGICAL step must equal exactly `num_levels`
    PIPELINE steps fed the SAME inputs each time. This validates the level
    walk against M2's already-trusted pipeline sweep independently of the
    hand-computed reference above."""
    topo_static, topo_state = _build(_TopoNet)
    pipe_static, pipe_state = _build(_PipelineNet)
    assert len(topo_static.level_capacities) == _NUM_LEVELS
    assert len(pipe_static.level_capacities) == 1  # pipeline: 1-bucket flat arena

    topo_step = px.make_step(_TopoNet, topo_static)
    pipe_step = px.make_step(_PipelineNet, pipe_static)

    x0, x1 = 1.0, 2.0

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error", message=".*Some donated buffers were not usable.*"
        )
        topo_inputs = px.StepInputs(
            inputs=jnp.asarray([x0, x1], dtype=jnp.float32), targets=None
        )
        topo_result = topo_step(topo_state, topo_inputs)
        del topo_state  # may be donated; use topo_result.state from here on

        for _ in range(_NUM_LEVELS):
            pipe_inputs = px.StepInputs(
                inputs=jnp.asarray([x0, x1], dtype=jnp.float32), targets=None
            )
            pipe_result = pipe_step(pipe_state, pipe_inputs)
            pipe_state = pipe_result.state  # donated; always use the fresh return

    got_topo = np.asarray(topo_result.state.units[px.ACTIVATION.name])
    got_pipe = np.asarray(pipe_state.units[px.ACTIVATION.name])
    np.testing.assert_allclose(got_topo, got_pipe, rtol=1e-5, atol=1e-5)

    # Fewer than num_levels pipeline steps must NOT yet match (otherwise the
    # comparison above would be vacuous, e.g. if the net had no real depth):
    # after only num_levels - 1 steps, the slowest (chain) path has not
    # finished propagating into unit 4.
    pipe_static2, pipe_state2 = _build(_PipelineNet)
    for _ in range(_NUM_LEVELS - 1):
        pipe_inputs2 = px.StepInputs(
            inputs=jnp.asarray([x0, x1], dtype=jnp.float32), targets=None
        )
        pipe_state2 = pipe_step(pipe_state2, pipe_inputs2).state
    got_pipe_early = np.asarray(pipe_state2.units[px.ACTIVATION.name])
    assert not np.allclose(got_topo, got_pipe_early, rtol=1e-5, atol=1e-5)
