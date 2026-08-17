"""Level assignment and resort (rung0 design section 4).

Deletion never resorts. Level-preserving adds never resort. Resort runs
only on level reassignment; bucket growth is handled by state.grow_bucket.
"""

from __future__ import annotations

import numpy as np
from jaxtyping import Array, Int32

from plastax.state import NetworkState, NetworkStatic


def initial_levels(
    num_units: int,
    edges: np.ndarray,  # (E, 2) int32 host array, from builder
) -> np.ndarray:
    """Host-side Kahn/BFS for initial construction (pre-jit, plain numpy)."""
    raise NotImplementedError


def recompute_levels[GS](
    static: NetworkStatic, state: NetworkState[GS]
) -> Int32[Array, " num_units"]:  # noqa: F722  jaxtyping named-axis string
    """On-device Kahn relaxation: lax.fori_loop bounded by
    static.kahn_max_depth or num_units (both static). Carry is
    fixed-capacity (loops.py:545-619 constraints)."""
    raise NotImplementedError


def resort[GS](
    static: NetworkStatic, state: NetworkState[GS]
) -> tuple[NetworkStatic, NetworkState[GS]]:
    """Host-driven: recompute levels, redistribute conns into new buckets
    (gather per level), stable sort each bucket by (dead, to_id) via
    lax.sort_key_val (lax.py:3509) — doubles as compaction — then derive
    new level_capacities via capacity_policy. Per-level live counts are the
    only host transfer. Returns new (static, state); caller retraces."""
    raise NotImplementedError


def capacity_policy(live: int, *, min_bucket: int = 64) -> int:
    """Default: max(next_pow2(live), min_bucket) so buckets carry headroom
    by construction. Constants are an open tuning item (design section 9)."""
    raise NotImplementedError
