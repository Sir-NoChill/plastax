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
from jaxtyping import Array, Bool

from plastax._types import FieldSpec, Propagation, ShardSpec


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class NetworkStatic:
    """Static network configuration.

    Attributes:
        num_units: total number of units in the network.
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
    """

    units: Columns
    conns: tuple[Columns, ...]
    globals_: GS
    needs_resort: Bool[Array, ""]


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


def grow_bucket[GS](
    static: NetworkStatic, state: NetworkState[GS], level: int
) -> tuple[NetworkStatic, NetworkState[GS]]:
    """Pad one bucket's columns, host-side, and produce a new static/state.

    Pure old-state -> new-state; the caller retraces once against the new
    static config and host-side reallocation.

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
    # the old_capacity*2 floor guarantees genuine growth even when grow_bucket
    # is invoked well before the bucket is actually full.
    new_capacity = max(capacity_policy(live + 1), old_capacity * 2)
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

    new_static = NetworkStatic(
        num_units=static.num_units,
        propagation=static.propagation,
        unit_fields=static.unit_fields,
        conn_fields=static.conn_fields,
        level_capacities=new_level_capacities,
        kahn_max_depth=static.kahn_max_depth,
        input_ids=static.input_ids,
        output_ids=static.output_ids,
        sharding=static.sharding,
    )
    new_state = NetworkState(
        units=state.units,
        conns=new_conns,
        globals_=state.globals_,
        needs_resort=state.needs_resort,
    )
    return new_static, new_state
