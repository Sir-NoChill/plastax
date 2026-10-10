"""Network state: static config (jit cache key).

NetworkStatic meta fields must stay hashable primitives, so the SoA fields
need to be of known size before the jit. These are only checked once, when
jax traces the function/network.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Int32

from plastax._types import PRUNED, FieldSpec, Propagation, ShardSpec


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class NetworkStatic:
    """Static network configuration.

    Attributes:
        num_units: number of unit slots: the unit count, or the network's
            unit capacity when it declares one (`Network.unit_capacity`).
        propagation: propagation mode used to advance the network.
        unit_fields: field specs defining the unit column layout.
        conn_fields: field specs defining the connection column layout.
        level_capacities: bucket capacities. PIPELINE uses a 1-tuple;
            TOPOLOGICAL uses one bucket per source level.
        kahn_max_depth: Kahn-order depth bound, or None to derive it as
            num_units.
        input_ids: builder-recorded unit ids that StepInputs scatters onto.
        output_ids: builder-recorded unit ids that the loss clamps targets
            to.
        sharding: Scheme-A sharding config, or None for a single device.
        capacity_headroom: the dead-slot fraction every bucket sizing
            (build, grow_bucket, resort) reserves above the live count.
        capacity_align: the capacity rounding for that sizing: None for a
            power of two, an int for a multiple of it (topo.capacity_policy).
        seed: the network seed keying the framework's counter-based RNG
            (`plastax.rng`), copied from `Network.seed` at build time.
        deepest_grows: whether the TOPOLOGICAL bucket list includes a bucket
            for the deepest unit level. Set when the net declares a growth
            rule: a forward DAG never sources an edge at its deepest level,
            but growth may (the committed edge is backward and triggers a
            resort), so the bucket must exist for the claim to land in.
            Resort preserves the convention.
    """

    num_units: int = dataclasses.field(metadata=dict(static=True))
    propagation: Propagation = dataclasses.field(metadata=dict(static=True))
    unit_fields: tuple[FieldSpec[np.generic], ...] = dataclasses.field(
        metadata=dict(static=True)
    )
    conn_fields: tuple[FieldSpec[np.generic], ...] = dataclasses.field(
        metadata=dict(static=True)
    )
    level_capacities: tuple[int, ...] = dataclasses.field(metadata=dict(static=True))
    kahn_max_depth: int | None = dataclasses.field(metadata=dict(static=True))
    input_ids: tuple[int, ...] = dataclasses.field(metadata=dict(static=True))
    output_ids: tuple[int, ...] = dataclasses.field(metadata=dict(static=True))
    sharding: ShardSpec | None = dataclasses.field(
        default=None, metadata=dict(static=True)
    )
    capacity_headroom: float = dataclasses.field(
        default=0.0, metadata=dict(static=True)
    )
    capacity_align: int | None = dataclasses.field(
        default=None, metadata=dict(static=True)
    )
    seed: int = dataclasses.field(default=0, metadata=dict(static=True))
    deepest_grows: bool = dataclasses.field(default=False, metadata=dict(static=True))


Columns = dict[str, Array]  # keyed by FieldSpec.name; one array per SOA tag


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class NetworkState[GS]:
    """Mutable arena state: unit/conn columns plus user-defined globals.

    Type Args:
        GS: the user's global-state pytree, opaque to the framework.

    Attributes:
        units: per-unit columns, each array shaped (num_units,).
        conns: per-level connection columns; conns[i] is sized to
            level_capacities[i].
        globals_: user-defined global state, opaque to the framework.
        needs_resort: scalar flag marking whether a topological resort is
            due; checked host-side by the driver between steps.
        step: scalar count of completed framework steps. It reads 0 during
            the first step's phases and is incremented once at the end of
            every step (a batched step counts as one); rules and the
            counter-based RNG key on the pre-increment value.
        grown: connections committed by this step's growth phase (0 when the
            phase was elided, skipped by its trigger, or grew nothing).
        overflow: whether this step's growth phase dropped selected
            candidates for lack of free capacity.
        units_added: units added by this step's unit-addition phase (0 when
            the phase was elided or skipped by `structural_interval`); the
            growth trigger "on_units_added" reads it.
        unit_overflow: whether this step's unit-addition phase dropped a
            spawn for lack of a free unit slot.
        tail_start: PIPELINE only (0 under TOPOLOGICAL): the single bucket's
            high-water mark. Slots below it have held a connection; slots
            from it up are the never-used tail that growth spills into once
            a source level's own dead slots run out.
    """

    units: Columns
    conns: tuple[Columns, ...]
    globals_: GS
    needs_resort: Bool[Array, ""]
    step: Int32[Array, ""] = dataclasses.field(default_factory=lambda: jnp.int32(0))
    grown: Int32[Array, ""] = dataclasses.field(default_factory=lambda: jnp.int32(0))
    overflow: Bool[Array, ""] = dataclasses.field(
        default_factory=lambda: jnp.bool_(False)
    )
    units_added: Int32[Array, ""] = dataclasses.field(
        default_factory=lambda: jnp.int32(0)
    )
    unit_overflow: Bool[Array, ""] = dataclasses.field(
        default_factory=lambda: jnp.bool_(False)
    )
    tail_start: Int32[Array, ""] = dataclasses.field(
        default_factory=lambda: jnp.int32(0)
    )


def _filled_columns(specs: tuple[FieldSpec[np.generic], ...], capacity: int) -> Columns:
    """One (capacity,) array per spec, filled with that spec's default."""
    # np.asarray: jnp.full's ArrayLike excludes the bare np.generic that
    # FieldSpec[np.generic].default erases to; an ndarray is accepted.
    return {
        spec.name: jnp.full((capacity,), np.asarray(spec.default), dtype=spec.dtype)
        for spec in specs
    }


def make_empty_state[GS](static: NetworkStatic, globals_: GS) -> NetworkState[GS]:
    """Allocate arenas at capacity, with all conn slots dead (tombstoned).

    Type Args:
        GS: the user's global-state pytree, opaque to the framework.

    Args:
        static: static network configuration giving the arena sizes.
        globals_: initial user-defined global state.

    Returns:
        A freshly allocated NetworkState with no live connections.
    """
    units = _filled_columns(static.unit_fields, static.num_units)
    conns = tuple(
        _filled_columns(static.conn_fields, capacity)
        for capacity in static.level_capacities
    )
    return NetworkState(
        units=units,
        conns=conns,
        globals_=globals_,
        needs_resort=jnp.bool_(False),
        step=jnp.int32(0),
        grown=jnp.int32(0),
        overflow=jnp.bool_(False),
        units_added=jnp.int32(0),
        unit_overflow=jnp.bool_(False),
        tail_start=jnp.int32(0),
    )


def live_conn_count[GS](state: NetworkState[GS], level: int | None = None) -> Array:
    """Count live connections.

    Type Args:
        GS: the user's global-state pytree, opaque to the framework.

    Args:
        state: network state to count connections in.
        level: bucket to count, or None to sum across all buckets.

    Returns:
        Scalar array with the live connection count.
    """
    if level is not None:
        return jnp.sum(~state.conns[level]["dead"])
    total = jnp.asarray(0, dtype=jnp.int32)
    for columns in state.conns:
        total = total + jnp.sum(~columns["dead"])
    return total


def live_unit_mask(units: Columns) -> Bool[Array, " num_units"] | None:
    """The unit slots that hold a live unit, or None when every slot does.

    A network that declares a unit capacity carries the `PRUNED` column, and a
    slot is live when it is not marked there. Without that column every slot
    is a live unit, and callers skip the mask entirely, so a network without a
    unit capacity traces no masking equations.

    Args:
        units: the unit columns (batched or not; the mask has their shape).

    Returns:
        The boolean live mask, or None when the network has no `PRUNED`
        column.
    """
    pruned = units.get(PRUNED.name)
    return None if pruned is None else ~pruned


def live_unit_count[GS](state: NetworkState[GS]) -> Array:
    """Count live units: every slot, minus those `PRUNED` marks.

    Type Args:
        GS: the user's global-state pytree, opaque to the framework.

    Args:
        state: network state to count units in.

    Returns:
        Scalar int32 array with the live unit count.
    """
    live = live_unit_mask(state.units)
    if live is None:
        return jnp.asarray(next(iter(state.units.values())).shape[0], jnp.int32)
    return jnp.sum(live, dtype=jnp.int32)


def grow_bucket[GS](
    static: NetworkStatic, state: NetworkState[GS], level: int
) -> tuple[NetworkStatic, NetworkState[GS]]:
    """Pad one bucket's columns, host-side, and produce a new static/state.

    Pure old-state -> new-state; the caller retraces once against the new
    static config and host-side reallocation. The padding is never-used, so
    in PIPELINE mode it extends the bucket's tail (`tail_start` stays put).

    Type Args:
        GS: the user's global-state pytree, opaque to the framework.

    Args:
        static: current static config, whose level_capacities is grown.
        state: current network state, whose bucket columns are padded.
        level: bucket index to grow.

    Returns:
        The new (static, state) pair with the bucket at `level` grown.
    """
    # Local import: topo.py imports NetworkState/NetworkStatic from this
    # module at top level, so a module-level import here would cycle.
    from plastax.topo import capacity_policy

    old_capacity = static.level_capacities[level]
    live = int(live_conn_count(state, level))
    # Seeding the policy with live+1 seeks headroom for one more live slot;
    # the geometric floor (2x for power-of-two rounding, 1.5x for aligned
    # rounding, which exists to keep memory tight) guarantees genuine growth
    # even when grow_bucket is invoked well before the bucket is full, so
    # repeated overflows cost O(log) retraces.
    policy = capacity_policy(
        live + 1, headroom=static.capacity_headroom, align=static.capacity_align
    )
    if static.capacity_align is None:
        floor = old_capacity * 2
    else:
        floor = capacity_policy(
            old_capacity + old_capacity // 2, align=static.capacity_align
        )
    new_capacity = max(policy, floor)
    pad = new_capacity - old_capacity

    old_columns = state.conns[level]
    grown_columns: Columns = {
        spec.name: jnp.concatenate(
            [
                old_columns[spec.name],
                jnp.full((pad,), np.asarray(spec.default), dtype=spec.dtype),
            ]
        )
        for spec in static.conn_fields
    }
    new_conns = tuple(
        grown_columns if i == level else columns
        for i, columns in enumerate(state.conns)
    )
    new_level_capacities = tuple(
        new_capacity if i == level else capacity
        for i, capacity in enumerate(static.level_capacities)
    )

    new_static = dataclasses.replace(static, level_capacities=new_level_capacities)
    new_state = NetworkState(
        units=state.units,
        conns=new_conns,
        globals_=state.globals_,
        needs_resort=state.needs_resort,
        step=state.step,
        grown=state.grown,
        overflow=state.overflow,
        units_added=state.units_added,
        unit_overflow=state.unit_overflow,
        tail_start=state.tail_start,
    )
    return new_static, new_state
