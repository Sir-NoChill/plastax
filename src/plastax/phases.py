"""Phase builders: each returns a pure state->state function for one Do*
phase, or None when the trait slot is absent (trace-time elision, rung0
design section 2). Phase order matches plastix.hpp's 11-phase step for the
v1 subset: forward, loss, backward, update_conn, prune_conn, add_conn,
reset_global.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from plastax._types import Propagation, UnitIdx
from plastax.state import NetworkState, NetworkStatic
from plastax.sweep import build_forward_sweep
from plastax.traits import Network
from plastax.views import UnitView

# PEP 695 generic alias: lazily evaluated, so the NetworkState/StepInputs
# forward references need no quoting. Every phase also returns a scalar loss
# contribution -- Deviation (IMPLEMENTATION_PLAN.md M2): globals_ is a fully
# opaque user pytree (GS may be None, plain dict, ...), so the loss phase's
# reduced scalar has no generic slot to land in inside NetworkState; thread
# it out as a second return so make_step can fold it into StepResult.loss
# (sibling to `overflow`, itself a framework-computed, state-external signal)
# instead of assuming globals_ has a loss field. Non-loss phases return 0.0.
type Phase[GS] = Callable[
    [NetworkState[GS], StepInputs], tuple[NetworkState[GS], Float[Array, ""]]  # noqa: F722
]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class StepInputs:
    """Clamped inputs + targets for one step; fixed pytree structure.

    inputs: (num_inputs,) values scattered to input unit ids (static tuple)
    targets: (num_outputs,) for the loss phase, or None when loss is absent

    Registered as a dataclass pytree (Deviation, IMPLEMENTATION_PLAN.md M2):
    the stub was a bare annotated class, not constructible/traceable. `None`
    is a valid data field (jax/_src/tree_util.py:1034) -- an empty subtree,
    not a leaf -- so `targets=None` flattens to zero leaves for a loss-less
    net, matching "fixed structure for a given net" (structure is static per
    net.loss presence, never per-call).
    """

    inputs: Float[Array, " num_inputs"]  # noqa: F722  jaxtyping named-axis string
    targets: Float[Array, " num_outputs"] | None  # noqa: F722  jaxtyping named-axis


def build_phases[GS](
    net: type[Network[GS]], static: NetworkStatic
) -> tuple[Phase[GS], ...]:
    """Assemble only the present phases; absent slots contribute nothing to
    the trace. Topological forward walks buckets 1..L (Python loop, static
    slices); backward walks L..1; pipeline is the 1-bucket flat sweep.

    M2 scope: PIPELINE only, among {forward, loss, reset_global}; every
    other slot is M3/M4 and raises. Presence is a Python-level `if`, so an
    absent phase is never traced -- zero equations, never lax.cond [D:2],
    which is exactly what test_phases_elision checks.
    """
    if net.propagation is not Propagation.PIPELINE:
        raise NotImplementedError("topological mode: M3")
    if net.backward_pass is not None:
        raise NotImplementedError("backward_pass: M3/M4")
    if net.update_conn is not None:
        raise NotImplementedError("update_conn: M3/M4")
    if net.prune_conn is not None:
        raise NotImplementedError("prune_conn: M3/M4")
    if net.add_conn is not None:
        raise NotImplementedError("add_conn: M3/M4")

    phases: list[Phase[GS]] = [_build_forward_phase(net, static)]
    if net.loss is not None:
        phases.append(_build_loss_phase(net, static))
    if net.reset_global is not None:
        phases.append(_build_reset_global_phase(net))
    return tuple(phases)


def _build_forward_phase[GS](
    net: type[Network[GS]], static: NetworkStatic
) -> Phase[GS]:
    # Pipeline: level_capacities is a 1-tuple (rung0 design section 3) --
    # the single flat bucket is state.conns[0]. indices_are_sorted=True
    # holds from NetworkBuilder.finalize's per-bucket (dead, to_id) sort.
    sweep = build_forward_sweep(
        net.forward_pass,
        num_units=static.num_units,
        indices_are_sorted=True,
        input_ids=static.input_ids,
    )

    def forward_phase(
        state: NetworkState[GS], inputs: StepInputs
    ) -> tuple[NetworkState[GS], Float[Array, ""]]:  # noqa: F722
        del inputs
        new_units = sweep(state.units, state.conns[0], state.globals_)
        return dataclasses.replace(state, units=new_units), jnp.float32(0.0)

    return forward_phase


def _build_loss_phase[GS](net: type[Network[GS]], static: NetworkStatic) -> Phase[GS]:
    loss = net.loss
    assert loss is not None  # build_phases only calls this when set
    output_ids = static.output_ids

    def loss_phase(
        state: NetworkState[GS], inputs: StepInputs
    ) -> tuple[NetworkState[GS], Float[Array, ""]]:  # noqa: F722
        # StepInputs.targets is None only when net.loss is None (phases.py
        # docstring); build_phases only reaches here when net.loss is set.
        assert inputs.targets is not None
        u_view = UnitView(state.units)
        units = dict(state.units)
        total = jnp.float32(0.0)
        # Static Python loop over the static output_ids tuple (unrolled at
        # trace time), matching the forward topological level loop's style:
        # small and static, so no vmap machinery is needed.
        for k, unit_id in enumerate(output_ids):
            value, write = loss.per_output(
                u_view, UnitIdx(unit_id), inputs.targets[k], state.globals_
            )
            total = total + value
            for name, field_value in write.fields.items():
                units[name] = units[name].at[unit_id].set(field_value)
        return dataclasses.replace(state, units=units), total

    return loss_phase


def _build_reset_global_phase[GS](net: type[Network[GS]]) -> Phase[GS]:
    reset_global = net.reset_global
    assert reset_global is not None  # build_phases only calls this when set

    def reset_global_phase(
        state: NetworkState[GS], inputs: StepInputs
    ) -> tuple[NetworkState[GS], Float[Array, ""]]:  # noqa: F722
        del inputs
        new_globals = reset_global.reset(state.globals_)
        return dataclasses.replace(state, globals_=new_globals), jnp.float32(0.0)

    return reset_global_phase


def build_add_conn_phase[GS](
    net: type[Network[GS]], static: NetworkStatic
) -> Phase[GS]:
    """K-bounded candidates from the neighbourhood window; lax.top_k
    (static k, stable); per-bucket prefix-sum slot claim; overflow ->
    dropped scatter + flag; level-preserving adds must NOT set
    needs_resort (rung0 design section 5)."""
    raise NotImplementedError
