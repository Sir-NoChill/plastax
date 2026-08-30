"""Loss phase vectorization.

`_build_loss_phase` vmaps the user's scalar `per_output` policy over the whole
output set and scatters its writes in one pass, rather than unrolling a Python
loop over `static.output_ids`. The unrolled form emitted a gather plus a
scatter per output unit, so trace and XLA compile cost grew superlinearly in
the output count -- unnoticeable for the 1-10 outputs of an MLP, but the wall
for extreme multi-label classification, whose output layer is 10^5-10^6 units.

Two properties are pinned here: the vectorized phase computes the same total
and the same per-output writes a scalar reference does (including for a policy
whose write depends on the unit id, which a mis-ordered scatter would break),
and its trace size is INDEPENDENT of the output count -- the property the
unrolled loop violated.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import plastax as px
from plastax.phases import StepInputs, _build_loss_phase
from plastax.views import UnitWrite

LossGrad = px.FieldSpec.float32("loss_grad")

# The loss phase ignores StepInputs.inputs (step.py scatters it before any
# phase runs); these tests build the phase directly, so any (1,) value does.
_NO_INPUT = jnp.zeros((1,), dtype=jnp.float32)


class _IdSensitiveLoss(px.Loss):
    """0.5*(pred-target)^2, staging a write that depends on the unit id.

    Scaling the staged gradient by the unit's own id makes the per-output
    writes mutually distinguishable, so a scatter that pairs the vmapped
    results with the wrong output ids cannot pass.
    """

    def per_output(
        self, u: px.UnitView, i: px.UnitIdx, target: jax.Array, g: None
    ) -> tuple[jax.Array, UnitWrite]:
        del g
        diff = u[px.ACTIVATION, i] - target
        loss = jnp.float32(0.5) * diff * diff
        return loss, UnitWrite.of((LossGrad, diff * i.astype(jnp.float32)))


class _SumForward(px.ForwardPass):
    """Trivial weighted sum; present only because a Network needs a forward
    pass -- these tests build the loss phase directly and never run it."""

    combine = px.monoid.sum_

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
    ) -> UnitWrite:
        del u, i, g
        return UnitWrite.of((px.ACTIVATION, acc))


class _Net(px.Network[None]):
    forward_pass = _SumForward()
    loss = _IdSensitiveLoss()
    extra_unit_fields = (LossGrad,)
    propagation = px.Propagation.TOPOLOGICAL


def _fan_out_net(
    num_outputs: int,
) -> tuple[px.NetworkStatic, px.NetworkState[None]]:
    """One input unit fanning out to `num_outputs` output units."""
    outs = np.arange(1, num_outputs + 1, dtype=np.int32)
    return px.NetworkBuilder.from_edges(
        _Net,
        num_outputs + 1,
        np.zeros((num_outputs,), dtype=np.int32),
        outs,
        weights=np.ones((num_outputs,), dtype=np.float32),
        input_ids=(0,),
        output_ids=tuple(range(1, num_outputs + 1)),
        globals_=None,
    )


def _loss_eqn_count(num_outputs: int) -> int:
    """Equations in the traced loss phase for a net with `num_outputs` outputs."""
    static, state = _fan_out_net(num_outputs)
    phase = _build_loss_phase(_Net, static)
    targets = jnp.zeros((num_outputs,), dtype=jnp.float32)
    jaxpr = jax.make_jaxpr(phase)(state, StepInputs(inputs=_NO_INPUT, targets=targets))
    return len(jaxpr.jaxpr.eqns)


def test_loss_matches_scalar_reference() -> None:
    """Total loss and every staged write match a per-output numpy reference."""
    num_outputs = 96
    static, state = _fan_out_net(num_outputs)
    rng = np.random.default_rng(0)
    activations = rng.standard_normal(num_outputs + 1).astype(np.float32)
    targets = rng.standard_normal(num_outputs).astype(np.float32)
    state.units[px.ACTIVATION.name] = jnp.asarray(activations)

    phase = _build_loss_phase(_Net, static)
    new_state, total = phase(
        state, StepInputs(inputs=_NO_INPUT, targets=jnp.asarray(targets))
    )

    output_ids = np.asarray(static.output_ids)
    diff = activations[output_ids] - targets
    np.testing.assert_allclose(
        float(total), 0.5 * float(np.sum(diff * diff)), rtol=1e-5
    )

    written = np.asarray(new_state.units[LossGrad.name])
    np.testing.assert_allclose(written[output_ids], diff * output_ids, rtol=1e-5)
    # Nothing outside the output set is touched (the input unit stays default).
    assert written[0] == 0.0


def test_trace_size_independent_of_output_count() -> None:
    """The loss phase traces to the same equation count at 8 and 512 outputs.

    This is the regression guard: the unrolled loop emitted work per output, so
    its equation count grew with the output layer and 10^6 labels never traced.
    """
    assert _loss_eqn_count(8) == _loss_eqn_count(512)
