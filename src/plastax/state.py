"""Network state: static config (jit cache key) and arena leaves.

See rung0 design sections 1 and 4. NetworkStatic meta fields must stay
hashable primitives (jaxlib/pytree.cc:295-313 enforces lazily).
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool

from plastax._types import FieldSpec, Propagation


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class NetworkStatic:
    num_units: int = dataclasses.field(metadata=dict(static=True))
    propagation: Propagation = dataclasses.field(metadata=dict(static=True))
    unit_fields: tuple[FieldSpec[np.generic], ...] = dataclasses.field(
        metadata=dict(static=True)
    )
    conn_fields: tuple[FieldSpec[np.generic], ...] = dataclasses.field(
        metadata=dict(static=True)
    )
    # PIPELINE: 1-tuple. TOPOLOGICAL: one bucket per source level.
    level_capacities: tuple[int, ...] = dataclasses.field(metadata=dict(static=True))
    # None -> Kahn bound derived as num_units (static in v1).
    kahn_max_depth: int | None = dataclasses.field(metadata=dict(static=True))
    # Builder-recorded unit ids consumed by M2 (loss clamps targets to
    # output_ids; StepInputs scatters onto input_ids). Not in the rung0
    # design's NetworkStatic sketch -- Deviation, IMPLEMENTATION_PLAN.md.
    input_ids: tuple[int, ...] = dataclasses.field(metadata=dict(static=True))
    output_ids: tuple[int, ...] = dataclasses.field(metadata=dict(static=True))


Columns = dict[str, Array]  # keyed by FieldSpec.name; one array per SOA tag


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class NetworkState[GS]:
    units: Columns  # each (num_units,)
    conns: tuple[Columns, ...]  # conns[i] columns sized level_capacities[i]
    globals_: GS
    needs_resort: Bool[Array, ""]  # noqa: F722  jaxtyping scalar-shape string


def _filled_columns(specs: tuple[FieldSpec[np.generic], ...], capacity: int) -> Columns:
    """One (capacity,) array per spec, filled with that spec's default."""
    # np.asarray: jnp.full's ArrayLike excludes the bare np.generic that
    # FieldSpec[np.generic].default erases to; an ndarray is accepted.
    return {
        spec.name: jnp.full((capacity,), np.asarray(spec.default), dtype=spec.dtype)
        for spec in specs
    }


def make_empty_state[GS](static: NetworkStatic, globals_: GS) -> NetworkState[GS]:
    """Allocate arenas at capacity; all conn slots dead (tombstoned)."""
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
    """Derived, never stored: sum(~dead) per bucket or total (traced)."""
    if level is not None:
        return jnp.sum(~state.conns[level]["dead"])
    total = jnp.asarray(0, dtype=jnp.int32)
    for columns in state.conns:
        total = total + jnp.sum(~columns["dead"])
    return total


def grow_bucket[GS](
    static: NetworkStatic, state: NetworkState[GS], level: int
) -> tuple[NetworkStatic, NetworkState[GS]]:
    """Host-side reallocation: pad one bucket's columns, new static, one
    retrace. Pure old-state -> new-state (rung0 design section 5)."""
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
    )
    new_state = NetworkState(
        units=state.units,
        conns=new_conns,
        globals_=state.globals_,
        needs_resort=state.needs_resort,
    )
    return new_static, new_state
