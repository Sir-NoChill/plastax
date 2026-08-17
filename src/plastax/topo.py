"""Level assignment and resort (rung0 design section 4).

Deletion never resorts. Level-preserving adds never resort. Resort runs
only on level reassignment; bucket growth is handled by state.grow_bucket.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from jaxtyping import Array, Int32

from plastax.state import NetworkState, NetworkStatic


def initial_levels(
    num_units: int,
    edges: np.ndarray,  # (E, 2) int32 host array, from builder
) -> np.ndarray:
    """Host-side Kahn/BFS for initial construction (pre-jit, plain numpy).

    Longest-path levels: in-degree-0 units are level 0; level(v) = max over
    incoming edges (u -> v) of level(u) + 1. Raises on a cycle (edges must
    form a DAG; M1 topologies -- dense/conv2d/sequential -- always do).
    """
    levels = np.zeros(num_units, dtype=np.int32)
    if edges.shape[0] == 0:
        return levels

    src = edges[:, 0]
    dst = edges[:, 1]
    in_degree = np.zeros(num_units, dtype=np.int64)
    np.add.at(in_degree, dst, 1)

    adjacency: list[list[int]] = [[] for _ in range(num_units)]
    for u, v in zip(src.tolist(), dst.tolist(), strict=True):
        adjacency[u].append(v)

    remaining = in_degree.copy()
    queue: deque[int] = deque(int(u) for u in np.flatnonzero(in_degree == 0))
    processed = 0
    while queue:
        u = queue.popleft()
        processed += 1
        for v in adjacency[u]:
            if levels[u] + 1 > levels[v]:
                levels[v] = levels[u] + 1
            remaining[v] -= 1
            if remaining[v] == 0:
                queue.append(v)

    if processed != num_units:
        raise ValueError("initial_levels: edges do not form a DAG (cycle detected)")
    return levels


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
    if live <= 0:
        return min_bucket
    next_pow2 = 1 << (live - 1).bit_length()
    return max(next_pow2, min_bucket)
