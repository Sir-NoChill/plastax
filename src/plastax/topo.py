"""Level assignment and resort (rung0 design section 4).

Deletion never resorts. Level-preserving adds never resort. Resort runs
only on level reassignment; bucket growth is handled by state.grow_bucket.
"""

from __future__ import annotations

import dataclasses
from collections import deque
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Int32

from plastax import monoid
from plastax._types import DEAD, FROM_ID, LEVEL, TO_ID, Propagation
from plastax.state import Columns, NetworkState, NetworkStatic
from plastax.sweep import unit_id_mask


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
    fixed-capacity (loops.py:545-619 constraints).

    Bellman-Ford-style longest-path relaxation: the vectorized analogue of
    dispatch_cpu.hpp:1407-1474's frontier BFS, which resets every non-input
    unit's Level to 0 and then only ever RAISES a unit's Level by walking
    forward from the NumInput-sized input frontier (an input's Level is
    simply never visited by that walk). Every unit here likewise starts at
    level 0; each of the `max_depth` rounds gathers, for every LIVE conn,
    its source's current level + 1 as a candidate for its destination, and
    folds candidates into each destination via a max-segment-reduce
    (monoid.max_, jax.ops.segment_max -- dead conns are routed to the
    out-of-range null slot `num_units`, dropped by FILL_OR_DROP, matching
    sweep._accumulate_into); declared inputs (static.input_ids) are re-
    pinned to 0 at the end of every round so an input can never become a
    relaxation TARGET regardless of what the live edge set looks like,
    mirroring the oracle's Level array never being written for I <
    NumInput. A DAG's longest path visits fewer than num_units edges, so
    kahn_max_depth-or-num_units rounds always converges to the exact same
    longest-path level `initial_levels` computes host-side from the same
    edge set, when the graph's structurally-in-degree-0 units are exactly
    the declared inputs (true for every M1 topology and this module's own
    test graphs -- the acceptance bar the task sets).
    """
    num_units = static.num_units
    max_depth = (
        static.kahn_max_depth if static.kahn_max_depth is not None else num_units
    )
    from_id = jnp.concatenate([bucket[FROM_ID.name] for bucket in state.conns])
    to_id = jnp.concatenate([bucket[TO_ID.name] for bucket in state.conns])
    dead = jnp.concatenate([bucket[DEAD.name] for bucket in state.conns])
    safe_to = jnp.where(dead, jnp.int32(num_units), to_id)
    is_input = unit_id_mask(static.input_ids, num_units)

    def relax(
        _: Int32[Array, ""],  # noqa: F722  jaxtyping scalar-shape string
        level: Int32[Array, " num_units"],  # noqa: F722
    ) -> Int32[Array, " num_units"]:  # noqa: F722
        candidate = level[from_id] + jnp.int32(1)
        incoming_max = monoid.max_.segment_reduce(
            candidate, safe_to, num_units, indices_are_sorted=False
        )
        relaxed = jnp.maximum(level, incoming_max.astype(jnp.int32))
        return jnp.where(is_input, jnp.int32(0), relaxed)

    init = jnp.zeros((num_units,), dtype=jnp.int32)
    # jax.lax.fori_loop's stub resolves the carry generically enough that
    # mypy loses the concrete Array return type here (confirmed: the SAME
    # cast-not-ignore choice as step.py's jax.jit call, for the same
    # follow_imports reason cited there).
    return cast(
        Int32[Array, " num_units"], jax.lax.fori_loop(0, max_depth, relax, init)
    )  # noqa: F722


def resort[GS](
    static: NetworkStatic, state: NetworkState[GS]
) -> tuple[NetworkStatic, NetworkState[GS]]:
    """Host-driven: recompute levels, redistribute conns into new buckets
    (gather per level), stable sort each bucket by (dead, to_id) via
    lax.sort_key_val (lax.py:3509) -- doubles as compaction -- then derive
    new level_capacities via capacity_policy. Per-level live counts are the
    only host transfer. Returns new (static, state); caller retraces.

    PIPELINE mode keeps exactly one bucket (rung0 design section 3, mirrored
    from NetworkBuilder.finalize's own PIPELINE branch); TOPOLOGICAL's new
    bucket count is `max(new_level.max(), 1)`, exactly finalize's "the
    highest level ever used as a source is max(levels) - 1" derivation --
    unlike construction, a resort's bucket count can move in EITHER
    direction versus the old static: AddConn can deepen the graph (more
    buckets) and PruneConn can orphan a formerly-deep subtree (fewer).

    Redistribution is two device-side passes per new bucket, both reusing
    the prefix-sum null-slot idiom `build_add_conn_phase` already
    establishes: (1) a cumsum-rank COMPACTING scatter of every old conn
    (concatenated across every OLD bucket) whose (live, new source level)
    matches this bucket, into a fresh `capacity_b`-sized column (capacity_b
    from capacity_policy, sized off the live count that same predicate
    yields -- so every match provably fits and the scatter's "no such rank"
    sink, one past capacity_b, only ever catches non-matches); (2) a stable
    lax.sort_key_val over a single combined `dead * num_units + to_id` key
    (to_id < num_units always, so the two key ranges never collide) that
    restores the (dead, to_id) order the segment reductions' `indices_are_
    sorted=True` needs -- step (1) preserves each match's OLD relative
    order, not to_id order, so this second pass is not redundant with it.
    """
    num_units = static.num_units
    new_level = recompute_levels(static, state)

    flat: Columns = {
        spec.name: jnp.concatenate([bucket[spec.name] for bucket in state.conns])
        for spec in static.conn_fields
    }
    dead = flat[DEAD.name]
    bucket_of = new_level[flat[FROM_ID.name]]
    is_pipeline = static.propagation is Propagation.PIPELINE

    live_counts: list[int]
    if is_pipeline:
        new_num_buckets = 1
        live_counts = [int(jnp.sum(~dead))]
    else:
        # Host transfer 1/2: a single scalar, the deepest unit's new level
        # (module docstring: NOT always recoverable from the live-conn
        # histogram alone -- a mid-graph level can be a legitimate unit
        # level with zero live OUTGOING conns of its own this round).
        max_level = int(jnp.max(new_level)) if num_units else 0
        new_num_buckets = max(max_level, 1)
        safe_bucket = jnp.where(dead, jnp.int32(num_units), bucket_of)
        histogram = monoid.sum_.segment_reduce(
            jnp.ones_like(safe_bucket, dtype=jnp.int32),
            safe_bucket,
            num_units,
            indices_are_sorted=False,
        )
        # Host transfer 2/2: the per-level live-conn counts (module
        # docstring / design section 4), sliced to just the buckets that
        # will actually exist.
        live_counts = [int(c) for c in np.asarray(histogram[:new_num_buckets])]

    new_level_capacities = tuple(capacity_policy(live) for live in live_counts)

    new_conns: list[Columns] = []
    for bucket_idx in range(new_num_buckets):
        capacity_b = new_level_capacities[bucket_idx]
        in_bucket = ~dead if is_pipeline else (~dead) & (bucket_of == bucket_idx)
        rank = jnp.cumsum(in_bucket.astype(jnp.int32)) - 1
        scatter_target = jnp.where(in_bucket, rank, jnp.int32(capacity_b))

        bucket_cols: Columns = {
            spec.name: jnp.full(
                (capacity_b,), np.asarray(spec.default), dtype=spec.dtype
            )
            .at[scatter_target]
            .set(flat[spec.name], mode="drop")
            for spec in static.conn_fields
        }

        sort_key = bucket_cols[DEAD.name].astype(jnp.int32) * jnp.int32(
            num_units
        ) + bucket_cols[TO_ID.name].astype(jnp.int32)
        _, perm = jax.lax.sort_key_val(
            sort_key, jnp.arange(capacity_b, dtype=jnp.int32), is_stable=True
        )
        new_conns.append({name: col[perm] for name, col in bucket_cols.items()})

    new_static = dataclasses.replace(static, level_capacities=new_level_capacities)
    new_state = dataclasses.replace(
        state,
        units={**state.units, LEVEL.name: new_level},
        conns=tuple(new_conns),
        needs_resort=jnp.bool_(False),
    )
    return new_static, new_state


def capacity_policy(live: int, *, min_bucket: int = 64) -> int:
    """Default: max(next_pow2(live), min_bucket) so buckets carry headroom
    by construction. Constants are an open tuning item (design section 9)."""
    if live <= 0:
        return min_bucket
    next_pow2 = 1 << (live - 1).bit_length()
    return max(next_pow2, min_bucket)
