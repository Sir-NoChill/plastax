"""The whole-output loss phase.

`_build_loss_phase` calls the policy's `calculate_loss` once over every output
and scatters the returned seed into the policy's declared `seed_field` in one
pass. Pinned here: the phase reports the policy's scalar and writes each seed
to the right output id (a policy whose seed depends on the unit id catches a
mis-ordered scatter), nothing outside the outputs is touched, the trace size
is independent of the output count (10^5-10^6 outputs must trace), the
validation rejects the retired per-output signature and undeclared seed
fields, and the shipped `SoftmaxCrossEntropyLoss` is stable for logits whose
unshifted exponentials overflow float32.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import plastax as px
from plastax.phases import StepInputs, _build_loss_phase
from plastax.views import UnitWrite

LossGrad = px.FieldSpec.float32("loss_grad")

# The loss phase ignores StepInputs.inputs (step.py scatters it before any
# phase runs); these tests build the phase directly, so any (1,) value does.
_NO_INPUT = jnp.zeros((1,), dtype=jnp.float32)


class _IdSensitiveLoss(px.Loss):
    """0.5*sum((pred-target)^2), with a seed that depends on the unit id.

    Scaling each seed by its unit's own id makes the per-output writes
    mutually distinguishable, so a scatter that pairs seeds with the wrong
    output ids cannot pass.
    """

    seed_field = LossGrad

    def calculate_loss(
        self, u: px.UnitView, outputs: jax.Array, targets: jax.Array, g: None
    ) -> tuple[jax.Array, jax.Array]:
        del g
        diff = u.gather(px.ACTIVATION, outputs) - targets
        loss = jnp.sum(jnp.float32(0.5) * diff * diff)
        return loss, diff * outputs.astype(jnp.float32)


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
    """The reported loss and every seed write match a numpy reference."""
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


def _net_with_loss(loss: object, fields: tuple[px.FieldSpec[np.float32], ...]) -> None:
    type(
        "_Checked",
        (px.Network,),
        {"forward_pass": _SumForward(), "loss": loss, "extra_unit_fields": fields},
    )


def test_retired_per_output_signature_is_rejected_with_a_migration_hint() -> None:
    """A per-output loss fails at class definition, naming the replacement."""

    class _PerOutput:
        def per_output(self, u: object, i: object, t: object, g: object) -> None:
            del u, i, t, g

    with pytest.raises(TypeError, match="calculate_loss"):
        _net_with_loss(_PerOutput(), (LossGrad,))


def test_seed_field_must_be_a_declared_float_unit_column() -> None:
    """The seed column must exist on the network and hold floats."""
    with pytest.raises(TypeError, match="seed_field"):
        _net_with_loss(px.SoftmaxCrossEntropyLoss(LossGrad), ())
    counts = px.FieldSpec.int32("counts")
    with pytest.raises(TypeError, match="float"):
        _net_with_loss(px.SoftmaxCrossEntropyLoss(counts), (counts,))  # type: ignore[arg-type]
    _net_with_loss(px.SoftmaxCrossEntropyLoss(LossGrad), (LossGrad,))


class _SoftmaxNet(px.Network[None]):
    forward_pass = _SumForward()
    loss = px.SoftmaxCrossEntropyLoss(LossGrad)
    extra_unit_fields = (LossGrad,)
    propagation = px.Propagation.TOPOLOGICAL


def test_softmax_cross_entropy_is_stable_for_overflowing_logits() -> None:
    """Logits near 1000 stay finite and match a float64 log-sum-exp reference."""
    num_outputs = 5
    static, state = px.NetworkBuilder.from_edges(
        _SoftmaxNet,
        num_outputs + 1,
        np.zeros((num_outputs,), dtype=np.int32),
        np.arange(1, num_outputs + 1, dtype=np.int32),
        weights=np.ones((num_outputs,), dtype=np.float32),
        input_ids=(0,),
        output_ids=tuple(range(1, num_outputs + 1)),
        globals_=None,
    )
    logits = np.asarray([1000.0, 999.0, 997.5, 1000.25, 990.0], np.float32)
    targets = np.asarray([0.0, 0.25, 0.0, 0.75, 0.0], np.float32)
    state.units[px.ACTIVATION.name] = jnp.asarray(np.concatenate([[0.0], logits]))
    phase = _build_loss_phase(_SoftmaxNet, static)
    new_state, loss = phase(
        state, StepInputs(inputs=_NO_INPUT, targets=jnp.asarray(targets))
    )
    z = logits.astype(np.float64) - logits.max()
    soft = np.exp(z) / np.exp(z).sum()
    seed = np.asarray(new_state.units[LossGrad.name])[1:]
    assert np.isfinite(float(loss)) and np.all(np.isfinite(seed))
    np.testing.assert_allclose(seed, soft - targets, atol=1e-6)
    want = float(np.sum(targets * (np.log(np.exp(z).sum()) - z)))
    np.testing.assert_allclose(float(loss), want, rtol=1e-6)
