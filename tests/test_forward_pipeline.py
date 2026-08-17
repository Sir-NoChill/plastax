"""Pipeline forward sweep (M2).

Flat sweep vs numpy reference; one-hop latency semantics
(dispatch_cpu.hpp:202-223); dead-slot null-scatter.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np

import plastax as px
from plastax import state as state_mod
from plastax.views import UnitWrite


class _SumForward(px.ForwardPass):
    """Weighted-sum forward, the DefaultForwardPass analogue
    (traits.hpp:111-121): map = weight * activation[src], combine = sum,
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


class _PipelineNet(px.Network[None]):
    forward_pass = _SumForward()
    propagation = px.Propagation.PIPELINE


# 4 units: 0, 1 input; 2 hidden (one hop from inputs); 3 output (two hops).
# Edge weights and the hidden unit's pre-step activation are chosen so the
# one-hop-latency assertion is unambiguous (unit 3 must reflect unit 2's OLD
# activation, not the value this same step just computed for unit 2).
_EDGES = ((0, 2, 0.5), (1, 2, -0.25), (2, 3, 2.0))
_HIDDEN_INITIAL_ACTIVATION = 5.0


def _build(
    net: type[px.Network[None]] = _PipelineNet,
) -> tuple[px.NetworkStatic, px.NetworkState[None]]:
    """Fresh, independently-allocated (static, state) each call -- no shared
    array objects across calls, so two builds can each be donated by their
    own make_step call without aliasing into each other."""
    builder = px.NetworkBuilder(net, None)
    builder.add_unit()  # 0: input
    builder.add_unit()  # 1: input
    builder.add_unit(activation=_HIDDEN_INITIAL_ACTIVATION)  # 2: hidden
    builder.add_unit()  # 3: output
    builder.mark_input(0)
    builder.mark_input(1)
    builder.mark_output(3)
    for src, dst, weight in _EDGES:
        builder.add_conn(src, dst, weight=weight)
    return builder.finalize()


def _numpy_reference(
    num_units: int,
    input_ids: tuple[int, ...],
    activation_before: np.ndarray,
    inputs: np.ndarray,
) -> np.ndarray:
    """Hand-computed one-hop reference matching DoForwardPipeline
    (dispatch_cpu.hpp:202-223): scatter inputs, accumulate every live edge
    into its destination, then overwrite every NON-input unit's activation
    with its accumulator -- input units keep exactly the scattered value."""
    activation = activation_before.astype(np.float32).copy()
    for k, unit_id in enumerate(input_ids):
        activation[unit_id] = inputs[k]

    acc = np.zeros(num_units, dtype=np.float32)
    for src, dst, weight in _EDGES:
        acc[dst] += np.float32(weight) * activation[src]

    result = activation.copy()
    for unit_id in range(num_units):
        if unit_id in input_ids:
            continue
        result[unit_id] = acc[unit_id]
    return result


def test_forward_pipeline_matches_numpy_reference_and_one_hop_latency() -> None:
    static, built_state = _build()
    step = px.make_step(_PipelineNet, static)

    x0, x1 = 1.0, 2.0
    inputs = px.StepInputs(
        inputs=jnp.asarray([x0, x1], dtype=jnp.float32), targets=None
    )

    activation_before = np.asarray(built_state.units[px.ACTIVATION.name])
    with warnings.catch_warnings():
        # Donation-failure warning is a global pytest error (pyproject); also
        # assert locally (targeted, not a blanket simplefilter, so an
        # unrelated warning elsewhere can't make this assertion spuriously
        # strict) so this test's intent doesn't depend on that global config
        # alone.
        warnings.filterwarnings(
            "error", message=".*Some donated buffers were not usable.*"
        )
        result = step(built_state, inputs)
    # Use only the returned state from here on; `built_state` may be donated.
    del built_state

    got = np.asarray(result.state.units[px.ACTIVATION.name])
    expected = _numpy_reference(
        static.num_units, static.input_ids, activation_before, np.array([x0, x1])
    )
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)

    # Explicit one-hop-latency check: unit 3 (two hops from the inputs) must
    # reflect unit 2's OLD activation (5.0), not the value this same step
    # just computed for unit 2 (0.5*1.0 + -0.25*2.0 == 0.0). If the sweep
    # read its own freshly-written activations, unit 3 would come out 0.0.
    assert float(got[2]) == 0.0
    assert float(got[3]) == 2.0 * _HIDDEN_INITIAL_ACTIVATION
    assert float(got[3]) != 2.0 * float(got[2])

    # Input units are never Apply'd (dispatch_cpu.hpp:217-222): activation
    # is exactly the scattered value, unaffected by their own (identity,
    # since inputs have no incoming edges) accumulator.
    assert float(got[0]) == x0
    assert float(got[1]) == x1


def test_forward_pipeline_dead_slot_null_scatter_matches_exactly_live() -> None:
    """A conn bucket with extra dead/headroom slots must give the identical
    result to one with only the exactly-live slots (rung0 design section 3
    null-slot trick): grow one build's bucket well past what
    capacity_policy gave it by construction, run both, compare."""
    static_a, state_a = _build()
    static_b, state_b = _build()
    assert static_a.level_capacities == static_b.level_capacities  # same start

    grown_static, grown_state = state_mod.grow_bucket(static_b, state_b, level=0)
    assert grown_static.level_capacities[0] > static_a.level_capacities[0]
    del state_b  # grow_bucket's new_state aliases state_b.units; don't reuse

    inputs_a = px.StepInputs(
        inputs=jnp.asarray([1.0, 2.0], dtype=jnp.float32), targets=None
    )
    inputs_b = px.StepInputs(
        inputs=jnp.asarray([1.0, 2.0], dtype=jnp.float32), targets=None
    )

    step_a = px.make_step(_PipelineNet, static_a)
    step_b = px.make_step(_PipelineNet, grown_static)
    result_a = step_a(state_a, inputs_a)
    result_b = step_b(grown_state, inputs_b)

    got_a = np.asarray(result_a.state.units[px.ACTIVATION.name])
    got_b = np.asarray(result_b.state.units[px.ACTIVATION.name])
    np.testing.assert_array_equal(got_a, got_b)
