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

from plastax._types import LEVEL, Propagation, UnitIdx
from plastax.state import NetworkState, NetworkStatic
from plastax.sweep import (
    build_backward_accumulate,
    build_backward_apply,
    build_backward_sweep,
    build_forward_accumulate,
    build_forward_apply,
    build_forward_sweep,
    identity_accumulator,
    unit_id_mask,
)
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

    M3 scope: PIPELINE and TOPOLOGICAL, among {forward, loss, backward,
    reset_global}; update_conn/prune_conn/add_conn are M3b/M4 and still
    raise. Presence is a Python-level `if`, so an absent phase is never
    traced -- zero equations, never lax.cond [D:2], which is exactly what
    test_phases_elision checks. Phase order (module docstring): forward,
    loss, backward, update_conn, prune_conn, add_conn, reset_global.
    """
    if net.update_conn is not None:
        raise NotImplementedError("update_conn: M3b/M4")
    if net.prune_conn is not None:
        raise NotImplementedError("prune_conn: M3b/M4")
    if net.add_conn is not None:
        raise NotImplementedError("add_conn: M3b/M4")

    phases: list[Phase[GS]] = [_build_forward_phase(net, static)]
    if net.loss is not None:
        phases.append(_build_loss_phase(net, static))
    if net.backward_pass is not None:
        phases.append(_build_backward_phase(net, static))
    if net.reset_global is not None:
        phases.append(_build_reset_global_phase(net))
    return tuple(phases)


def _build_forward_phase[GS](
    net: type[Network[GS]], static: NetworkStatic
) -> Phase[GS]:
    if net.propagation is Propagation.PIPELINE:
        # level_capacities is a 1-tuple (rung0 design section 3) -- the
        # single flat bucket is state.conns[0]. indices_are_sorted=True
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

    return _build_forward_topological_phase(net, static)


def _build_forward_topological_phase[GS](
    net: type[Network[GS]], static: NetworkStatic
) -> Phase[GS]:
    """Level walk, source-level buckets 0..num_levels-1 in order
    (dispatch_cpu.hpp:41-67's `L = 1..NumLevels`, reindexed to plastax's
    0-based `conns` tuple: C++ `Ranges[L-1]` is plastax `conns[L-1]`; unit
    LEVEL values themselves need no reindexing, they already match the
    oracle's 1-based-from-inputs numbering).

    One accumulate call per bucket, combined into a carried accumulator
    (sweep.build_forward_accumulate) so a unit's contributions from EVERY
    earlier bucket survive even if its incoming edges are not all in bucket
    level-1 (a skip connection sources from an earlier level still). A
    unit is only finalized (sweep.build_forward_apply, write + accumulator
    reset) once every bucket that could feed it has been accumulated --
    which for a level-`level_idx+1` unit is exactly buckets `0..level_idx`,
    i.e. right after bucket `level_idx` is processed, since no edge sources
    from a level >= its own destination's level (the leveling invariant).
    """
    num_units = static.num_units
    num_levels = len(static.level_capacities)
    fp = net.forward_pass
    accumulate = build_forward_accumulate(
        fp, num_units=num_units, indices_are_sorted=True
    )
    apply = build_forward_apply(fp, num_units=num_units)
    not_input = ~unit_id_mask(static.input_ids, num_units)

    def forward_phase(
        state: NetworkState[GS], inputs: StepInputs
    ) -> tuple[NetworkState[GS], Float[Array, ""]]:  # noqa: F722
        del inputs
        units = state.units
        unit_level = units[LEVEL.name]
        acc = identity_accumulator(fp.combine, num_units)
        for level_idx in range(num_levels):
            acc = accumulate(units, state.conns[level_idx], acc, state.globals_)
            finalize = (unit_level == level_idx + 1) & not_input
            units, acc = apply(units, acc, state.globals_, finalize)
        return dataclasses.replace(state, units=units), jnp.float32(0.0)

    return forward_phase


def _build_backward_phase[GS](
    net: type[Network[GS]], static: NetworkStatic
) -> Phase[GS]:
    bp = net.backward_pass
    assert bp is not None  # build_phases only calls this when set

    if net.propagation is Propagation.PIPELINE:
        # dispatch_cpu.hpp:390-411: no level structure, one flat bucket,
        # every unit Applied unconditionally (build_backward_sweep takes no
        # input_ids -- see its docstring). indices_are_sorted=False: backward
        # indexes segments by FROM_ID, but finalize sorts each bucket by
        # (dead, TO_ID), so those indices are not sorted -- correct on CPU
        # either way, honest for GPU/TPU, matching the topological backward.
        sweep = build_backward_sweep(
            bp, num_units=static.num_units, indices_are_sorted=False
        )

        def backward_phase(
            state: NetworkState[GS], inputs: StepInputs
        ) -> tuple[NetworkState[GS], Float[Array, ""]]:  # noqa: F722
            del inputs
            new_units = sweep(state.units, state.conns[0], state.globals_)
            return dataclasses.replace(state, units=new_units), jnp.float32(0.0)

        return backward_phase

    return _build_backward_topological_phase(net, static)


def _build_backward_topological_phase[GS](
    net: type[Network[GS]], static: NetworkStatic
) -> Phase[GS]:
    """Reverse level walk (dispatch_cpu.hpp:232-258, `L = NumLevels..1`).

    Bucket `level_idx` holds edges SOURCED at level_idx; backward
    accumulates into the source, so accumulating bucket `level_idx` is
    exactly what completes a level-`level_idx` unit's accumulator (every
    outgoing edge of a level-`level_idx` unit sources at level_idx, by
    definition of the bucketing -- unlike forward, there is no cross-bucket
    spread on the finalizing side). Walking buckets high-to-low is what
    guarantees a bucket's Map (which reads destination-side state, at a
    strictly higher level) only ever runs after that destination has
    already been finalized.

    The top level (== num_levels) has no source-level bucket of its own --
    no edge sources from the deepest level, since that would need a
    destination one level deeper still (dispatch_cpu.hpp:328-333 makes this
    explicit: "no edge has source level MaxLevels anywhere") -- so it is
    primed directly from the identity accumulator before the bucket loop,
    picking up only whatever an earlier phase (e.g. loss) wrote into unit
    columns Map/Apply itself read.

    The loop stops at bucket 1, never touching bucket 0 (input units' own
    outgoing edges): input units are excluded from `finalize` exactly like
    forward (dispatch_cpu.hpp:250 bounds Apply at NumInput same as :59), so
    accumulating bucket 0 would only feed an accumulator that is never
    read -- matching the oracle loop, which structurally never reaches
    L=0 either (`for (L = NumLevels; L >= 1; --L)`).
    """
    num_units = static.num_units
    num_levels = len(static.level_capacities)
    bp = net.backward_pass
    assert bp is not None  # build_phases only calls this when set
    accumulate = build_backward_accumulate(
        bp, num_units=num_units, indices_are_sorted=False
    )
    apply = build_backward_apply(bp, num_units=num_units)
    not_input = ~unit_id_mask(static.input_ids, num_units)

    def backward_phase(
        state: NetworkState[GS], inputs: StepInputs
    ) -> tuple[NetworkState[GS], Float[Array, ""]]:  # noqa: F722
        del inputs
        units = state.units
        unit_level = units[LEVEL.name]
        acc = identity_accumulator(bp.combine, num_units)
        finalize = (unit_level == num_levels) & not_input
        units, acc = apply(units, acc, state.globals_, finalize)
        for level_idx in range(num_levels - 1, 0, -1):
            acc = accumulate(units, state.conns[level_idx], acc, state.globals_)
            finalize = (unit_level == level_idx) & not_input
            units, acc = apply(units, acc, state.globals_, finalize)
        return dataclasses.replace(state, units=units), jnp.float32(0.0)

    return backward_phase


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
