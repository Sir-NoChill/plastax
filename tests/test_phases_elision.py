"""Phase elision (M2).

Absent phases produce identical jaxprs to a hand-assembled subset (compare
jax.make_jaxpr output structure). build_phases is a Python-level `if
... : phases.append(...)` (phases.py), so an absent trait slot must
contribute exactly zero equations to the trace -- never a lax.cond branch
that would show up as a `cond` primitive regardless of which side is live.

jax.make_jaxpr on an already-`jax.jit`-wrapped callable collapses to a
single opaque `jit`/`pjit` equation wrapping a nested closed jaxpr
(confirmed empirically), which would make a top-level equation-count
comparison vacuous. So these tests trace `build_phases`' composed output
directly (the same fold make_step/step.py performs, minus the jit
wrapper), which is the right boundary for a phases.py-level property
anyway.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import jax
import jax.numpy as jnp

import plastax as px
from plastax.phases import Phase, StepInputs, build_phases
from plastax.views import UnitWrite


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
    ) -> UnitWrite:
        return UnitWrite.of((px.ACTIVATION, acc))


class _ResetToZero(px.ResetGlobal):
    def reset(self, g: jax.Array) -> jax.Array:
        return g * jnp.float32(0.0)


class _SquaredErrorLoss(px.Loss):
    def per_output(
        self, u: px.UnitView, i: px.UnitIdx, target: jax.Array, g: jax.Array
    ) -> tuple[jax.Array, UnitWrite]:
        pred = u[px.ACTIVATION, i]
        diff = pred - target
        return jnp.float32(0.5) * diff * diff, UnitWrite.of((px.ACTIVATION, pred))


_shared_forward = _SumForward()


class _ForwardOnlyNet(px.Network[jax.Array]):
    forward_pass = _shared_forward
    propagation = px.Propagation.PIPELINE


class _ForwardResetNet(px.Network[jax.Array]):
    forward_pass = _shared_forward
    reset_global = _ResetToZero()
    propagation = px.Propagation.PIPELINE


class _ForwardLossNet(px.Network[jax.Array]):
    forward_pass = _shared_forward
    loss = _SquaredErrorLoss()
    propagation = px.Propagation.PIPELINE


def _build(
    net: type[px.Network[jax.Array]],
) -> tuple[px.NetworkStatic, px.NetworkState[jax.Array]]:
    builder = px.NetworkBuilder(net, jnp.float32(3.0))
    builder.add_unit()  # 0: input
    builder.add_unit()  # 1: input
    builder.add_unit()  # 2: output
    builder.mark_input(0)
    builder.mark_input(1)
    builder.mark_output(2)
    builder.add_conn(0, 2, weight=0.5)
    builder.add_conn(1, 2, weight=-0.25)
    return builder.finalize()


def _trace_phases[GS](
    net: type[px.Network[GS]],
    static: px.NetworkStatic,
    state: px.NetworkState[GS],
    inputs: StepInputs,
) -> jax.core.ClosedJaxpr:
    """The same present-phases fold make_step/step.py performs, minus the
    jax.jit wrapper (so the jaxpr isn't hidden behind one opaque `jit`
    equation) and minus the input scatter (irrelevant to phase presence)."""
    phases: tuple[Phase[GS], ...] = build_phases(net, static)

    def run(
        state: px.NetworkState[GS], inputs: StepInputs
    ) -> tuple[px.NetworkState[GS], jax.Array]:
        total = jnp.float32(0.0)
        for phase in phases:
            state, contribution = phase(state, inputs)
            total = total + contribution
        return state, total

    return jax.make_jaxpr(run)(state, inputs)


def _primitive_names(jaxpr: jax.core.ClosedJaxpr) -> list[str]:
    return [eqn.primitive.name for eqn in jaxpr.eqns]


def _assert_prefix_and_strictly_more(
    smaller: jax.core.ClosedJaxpr, larger: jax.core.ClosedJaxpr
) -> None:
    small_names = _primitive_names(smaller)
    large_names = _primitive_names(larger)
    assert len(large_names) > len(small_names), (
        "the present extra phase must contribute at least one equation"
    )
    assert large_names[: len(small_names)] == small_names, (
        "the shared (forward) portion of the trace must be byte-for-byte "
        "identical in equation structure regardless of the other net's "
        "extra phase -- an absent phase must never perturb it"
    )


def test_absent_reset_global_adds_no_equations_to_the_shared_forward_trace() -> None:
    static, state = _build(_ForwardOnlyNet)
    inputs = StepInputs(inputs=jnp.asarray([1.0, 2.0], dtype=jnp.float32), targets=None)

    phases_without = build_phases(_ForwardOnlyNet, static)
    phases_with = build_phases(_ForwardResetNet, static)
    assert len(phases_without) == 1
    assert len(phases_with) == 2

    jaxpr_without = _trace_phases(_ForwardOnlyNet, static, state, inputs)
    jaxpr_with = _trace_phases(_ForwardResetNet, static, state, inputs)
    _assert_prefix_and_strictly_more(jaxpr_without, jaxpr_with)


def test_absent_loss_adds_no_equations_to_the_shared_forward_trace() -> None:
    static, state = _build(_ForwardOnlyNet)
    inputs_without = StepInputs(
        inputs=jnp.asarray([1.0, 2.0], dtype=jnp.float32), targets=None
    )
    inputs_with = StepInputs(
        inputs=jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        targets=jnp.asarray([0.5], dtype=jnp.float32),
    )

    phases_without = build_phases(_ForwardOnlyNet, static)
    phases_with = build_phases(_ForwardLossNet, static)
    assert len(phases_without) == 1
    assert len(phases_with) == 2

    jaxpr_without = _trace_phases(_ForwardOnlyNet, static, state, inputs_without)
    jaxpr_with = _trace_phases(_ForwardLossNet, static, state, inputs_with)
    _assert_prefix_and_strictly_more(jaxpr_without, jaxpr_with)


def test_no_phase_ever_lowers_to_a_cond() -> None:
    """Elision is a Python-level list append, never lax.cond (design
    invariant #1) -- so `cond` must never appear, present phase or not."""
    static, state = _build(_ForwardResetNet)
    inputs = StepInputs(inputs=jnp.asarray([1.0, 2.0], dtype=jnp.float32), targets=None)
    jaxpr = _trace_phases(_ForwardResetNet, static, state, inputs)
    assert "cond" not in _primitive_names(jaxpr)


def test_build_phases_returns_a_plain_python_tuple() -> None:
    """Presence is decided once, at Python trace-assembly time, not per-call
    inside the traced program (no data-dependent phase selection is even
    expressible this way)."""
    static, _ = _build(_ForwardOnlyNet)
    phases = build_phases(_ForwardOnlyNet, static)
    assert isinstance(phases, tuple)
    assert all(isinstance(phase, Callable) for phase in phases)


def test_loss_phase_runs_end_to_end_through_make_step_with_donation() -> None:
    """The two structural tests above never exercise the real jit +
    donation path (they trace build_phases' output directly); the forward
    pipeline tests exercise that path but never with a loss phase present.
    Close the gap: run a loss net through make_step for real and check the
    reduced scalar this milestone's loss-target Deviation lands in
    StepResult.loss."""
    static, state = _build(_ForwardLossNet)
    step = px.make_step(_ForwardLossNet, static)
    # Phases run forward-then-loss (build_phases order), so loss reads
    # unit 2's freshly-computed (not pre-step default) activation.
    inputs = StepInputs(
        inputs=jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        targets=jnp.asarray([0.5], dtype=jnp.float32),
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error", message=".*Some donated buffers were not usable.*"
        )
        result = step(state, inputs)
    # pred (unit 2, post-forward) == 0.5*1.0 + -0.25*2.0 == 0.0; target 0.5.
    assert float(result.loss) == 0.5 * (0.0 - 0.5) ** 2
    assert bool(result.overflow) is False
