"""Network state: static config (jit cache key) and arena leaves.

See rung0 design sections 1 and 4. NetworkStatic meta fields must stay
hashable primitives (jaxlib/pytree.cc:295-313 enforces lazily).
"""

from __future__ import annotations

import dataclasses

import jax
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


Columns = dict[str, Array]  # keyed by FieldSpec.name; one array per SOA tag


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class NetworkState[GS]:
    units: Columns  # each (num_units,)
    conns: tuple[Columns, ...]  # conns[i] columns sized level_capacities[i]
    globals_: GS
    needs_resort: Bool[Array, ""]  # noqa: F722  jaxtyping scalar-shape string


def make_empty_state[GS](static: NetworkStatic, globals_: GS) -> NetworkState[GS]:
    """Allocate arenas at capacity; all conn slots dead (tombstoned)."""
    raise NotImplementedError


def live_conn_count[GS](state: NetworkState[GS], level: int | None = None) -> Array:
    """Derived, never stored: sum(~dead) per bucket or total (traced)."""
    raise NotImplementedError


def grow_bucket[GS](
    static: NetworkStatic, state: NetworkState[GS], level: int
) -> tuple[NetworkStatic, NetworkState[GS]]:
    """Host-side reallocation: pad one bucket's columns, new static, one
    retrace. Pure old-state -> new-state (rung0 design section 5)."""
    raise NotImplementedError
