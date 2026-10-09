"""Level assignment and resort.

Deletion never resorts. Level-preserving adds never resort. Resort runs
only on level reassignment; bucket growth is handled by state.grow_bucket.
"""

from __future__ import annotations

import dataclasses
import math
from collections import deque
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Int32

from plastax import monoid
from plastax._types import DEAD, FROM_ID, LEVEL, TO_ID, Propagation
from plastax.state import Columns, NetworkState, NetworkStatic
from plastax.sweep import unit_id_mask

# Frontier-Kahn round cap before the Python longest-path takes over. Rounds
# equal graph depth; the shallow-wide nets this library builds (a handful of
# layers over a huge width) settle in a few rounds, so the vectorized path
# covers them. A graph deeper than this -- e.g. a long chain -- would make the
# O(depth*E) vectorized pass lose to the O(N+E) Python one, so past the cap we
# hand off to `_kahn_levels`. 64 comfortably clears any realistic layered net
# while capping the vectorized work spent before falling back on a deep one.
_MAX_VECTORIZED_LEVEL_ROUNDS = 64


def initial_levels(
    num_units: int,
    edges: np.ndarray,
    *,
    allow_cycles: bool = False,
    input_ids: tuple[int, ...] = (),
) -> np.ndarray:
    """Compute initial longest-path levels host-side, before jit.

    Longest-path levels: in-degree-0 units are level 0; level(v) = max over
    incoming edges (u -> v) of level(u) + 1. Raises on a cycle (topological
    propagation needs a DAG; the dense/conv2d/sequential topologies always
    give one) unless `allow_cycles` is set.

    The `input_ids` sit at level 0: an edge into an input does not raise
    the input's level, as in `recompute_levels`. It still counts toward the
    cycle check, so a cycle through an input raises.

    Computed by a **vectorized frontier Kahn** (BFS by level): each round
    relaxes, in one numpy scatter, every out-edge whose source settled last
    round, and settles any unit whose remaining in-degree hits zero. A unit
    settles only after all its predecessors, so its level is final when its
    out-edges relax -- identical longest-path values to a per-node Kahn queue,
    but O(depth) numpy rounds instead of a per-edge Python walk. Depth beyond
    `_MAX_VECTORIZED_LEVEL_ROUNDS` (a rare deep graph) falls back to the
    O(N+E) `_kahn_levels` so this never regresses versus the old scalar path.

    With `allow_cycles` (pipeline propagation, where recurrent reservoirs
    are legal), a cycle is not an error: the frontier drains once the acyclic
    prefix settles, leaving cycle units at the best-effort longest-path level
    reached from their acyclic predecessors (0 when they have none) -- exactly
    what the scalar Kahn left them at. Levels are cosmetic in pipeline mode
    (every conn lands in the single flat bucket), so a partial assignment is
    harmless.

    Args:
        num_units: Total number of units in the network.
        edges: (E, 2) int32 host array of edges, from builder.
        allow_cycles: If True, tolerate cycles and return best-effort
            levels instead of raising.
        input_ids: The input unit ids, pinned to level 0.

    Returns:
        Per-unit levels as an int32 host array.

    Raises:
        ValueError: The edges do not form a DAG and `allow_cycles` is False.
    """
    if input_ids and edges.shape[0]:
        into_input = np.isin(edges[:, 1], np.asarray(input_ids))
        if into_input.any():
            if not allow_cycles:
                # The cycle check sees every edge, the edges into inputs too.
                initial_levels(num_units, edges)
            return initial_levels(
                num_units, edges[~into_input], allow_cycles=allow_cycles
            )
    levels = np.zeros(num_units, dtype=np.int32)
    if edges.shape[0] == 0:
        return levels

    src = edges[:, 0]
    dst = edges[:, 1]
    # bincount, not np.add.at: the unbuffered ufunc.at scatters are slow.
    in_degree = np.bincount(dst, minlength=num_units).astype(np.int64)

    remaining = in_degree.copy()
    settled = in_degree == 0  # (num_units,) bool: level-0 units start settled
    newly = settled  # this round's frontier (units that just settled)
    processed = int(settled.sum())
    rounds = 0
    while bool(newly.any()):
        if rounds >= _MAX_VECTORIZED_LEVEL_ROUNDS:
            # Frontier still advancing past the cap -> a graph deeper than the
            # vectorized path is worth. The scalar longest-path is O(N+E) and
            # correct for any depth; recompute from scratch.
            return _kahn_levels(num_units, edges, allow_cycles=allow_cycles)
        active = newly[src]  # edges leaving the current frontier
        active_dst = dst[active]
        # level[v] = max(level[v], level[u] + 1) over this round's edges. Every
        # frontier unit settled this round sits at level `rounds` (a unit
        # settles the round after its last predecessor, so by induction its
        # longest-path level is its settle round), so the max is a plain
        # assignment of rounds + 1 -- monotone across rounds, duplicates
        # harmless.
        levels[active_dst] = np.int32(rounds + 1)
        remaining -= np.bincount(active_dst, minlength=num_units)
        reached = (remaining == 0) & ~settled
        settled = settled | reached
        newly = reached
        processed += int(reached.sum())
        rounds += 1

    if processed != num_units and not allow_cycles:
        raise ValueError("initial_levels: edges do not form a DAG (cycle detected)")
    return levels


def _kahn_levels(
    num_units: int,
    edges: np.ndarray,
    *,
    allow_cycles: bool,
) -> np.ndarray:
    """Scalar Kahn longest-path leveling: the deep-graph fallback.

    The per-node adjacency-plus-queue longest-path `initial_levels` used before
    it was vectorized, kept as the O(N+E) fallback for graphs deeper than the
    vectorized round cap. Produces byte-identical levels to the frontier pass
    on any input (both are longest-path Kahn); the randomized equivalence is
    pinned in tests/test_topo.py.

    Args:
        num_units: Total number of units in the network.
        edges: (E, 2) int32 host array of edges.
        allow_cycles: If True, tolerate cycles and return best-effort levels.

    Returns:
        Per-unit levels as an int32 host array.

    Raises:
        ValueError: The edges do not form a DAG and `allow_cycles` is False.
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

    if processed != num_units and not allow_cycles:
        raise ValueError("initial_levels: edges do not form a DAG (cycle detected)")
    return levels


def recompute_levels[GS](
    static: NetworkStatic, state: NetworkState[GS]
) -> Int32[Array, " num_units"]:
    """Relax unit levels on-device via bounded Kahn/Bellman-Ford iteration.

    Runs as a jax.lax.fori_loop bounded by static.kahn_max_depth or
    num_units (both static); the carry is fixed-capacity as the loop
    requires. Bellman-Ford-style longest-path relaxation: every unit
    starts at level 0; each of the `max_depth` rounds gathers, for every
    LIVE conn, its source's current level + 1 as a candidate for its
    destination, and folds candidates into each destination via a
    max-segment-reduce (monoid.max_, jax.ops.segment_max -- dead conns are
    routed to the out-of-range null slot `num_units`, dropped by
    FILL_OR_DROP, matching sweep._accumulate_into); declared inputs
    (static.input_ids) are re-pinned to 0 at the end of every round so an
    input can never become a relaxation TARGET regardless of what the
    live edge set looks like. A DAG's longest path visits fewer than
    num_units edges, so kahn_max_depth-or-num_units rounds always
    converges to the exact same longest-path level `initial_levels`
    computes host-side from the same edge set, when the graph's
    structurally-in-degree-0 units are exactly the declared inputs (true
    for every topology in this module's own test graphs). On a graph with a
    cycle among non-input units the levels do not converge; `resort`
    rejects such a graph in TOPOLOGICAL mode (`has_cycle`).

    Type Args:
        GS: Growth-state type parameter carried by NetworkState.

    Args:
        static: Static network configuration.
        state: Current network state.

    Returns:
        Per-unit levels after relaxation.
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
        _: Int32[Array, ""],
        level: Int32[Array, " num_units"],
    ) -> Int32[Array, " num_units"]:
        candidate = level[from_id] + jnp.int32(1)
        incoming_max = monoid.max_.segment_reduce(
            candidate, safe_to, num_units, indices_are_sorted=False
        )
        relaxed = jnp.maximum(level, incoming_max.astype(jnp.int32))
        return jnp.where(is_input, jnp.int32(0), relaxed)

    init = jnp.zeros((num_units,), dtype=jnp.int32)
    # jax.lax.fori_loop's stub resolves the carry generically enough that
    # mypy loses the concrete Array return type, hence the cast.
    return cast(
        Int32[Array, " num_units"], jax.lax.fori_loop(0, max_depth, relax, init)
    )


def has_cycle[GS](static: NetworkStatic, state: NetworkState[GS]) -> Bool[Array, ""]:
    """Whether the live connections contain a directed cycle.

    Every live edge counts, edges into input units included, so a cycle
    through an input is a cycle (a topological schedule cannot order it).
    Longest-path relaxation over the live edges with no unit pinned: on an
    acyclic graph every level settles within num_units - 1 rounds, while on
    a cycle some level rises every round. A `jax.lax.while_loop` relaxes
    until a round changes nothing or num_units rounds have run, so an
    acyclic graph costs its depth plus one round.

    Type Args:
        GS: Growth-state type parameter carried by NetworkState.

    Args:
        static: Static network configuration.
        state: Current network state.

    Returns:
        A scalar bool, True when the live edges contain a cycle.
    """
    num_units = static.num_units
    from_id = jnp.concatenate([bucket[FROM_ID.name] for bucket in state.conns])
    to_id = jnp.concatenate([bucket[TO_ID.name] for bucket in state.conns])
    dead = jnp.concatenate([bucket[DEAD.name] for bucket in state.conns])
    safe_to = jnp.where(dead, jnp.int32(num_units), to_id)

    def cond(carry: tuple[Int32[Array, ""], Int32[Array, " n"], Array]) -> Array:
        rounds, _, changed = carry
        return changed & (rounds < num_units)

    def body(
        carry: tuple[Int32[Array, ""], Int32[Array, " n"], Array],
    ) -> tuple[Int32[Array, ""], Int32[Array, " n"], Array]:
        rounds, level, _ = carry
        incoming_max = monoid.max_.segment_reduce(
            level[from_id] + jnp.int32(1), safe_to, num_units, indices_are_sorted=False
        )
        relaxed = jnp.maximum(level, incoming_max.astype(jnp.int32))
        return rounds + jnp.int32(1), relaxed, jnp.any(relaxed != level)

    init = (
        jnp.int32(0),
        jnp.zeros((num_units,), dtype=jnp.int32),
        jnp.bool_(num_units > 0),
    )
    changed: Bool[Array, ""] = jax.lax.while_loop(cond, body, init)[2]
    return changed


def resort[GS](
    static: NetworkStatic, state: NetworkState[GS]
) -> tuple[NetworkStatic, NetworkState[GS]]:
    """Recompute levels, redistribute conns into new buckets, and resort.

    Host-driven: recompute levels, redistribute conns into new buckets
    (gather per level), stable sort each bucket by (dead, from_id) via
    lax.sort_key_val -- doubles as compaction -- then derive new
    level_capacities via capacity_policy. Per-level live counts are the
    only host transfer besides, in TOPOLOGICAL mode, the `has_cycle` flag:
    a cycle in the live edges (through an input or not) has no level
    schedule and raises. Returns new (static, state); caller retraces.

    PIPELINE mode keeps exactly one bucket, mirrored from
    NetworkBuilder.finalize's own PIPELINE branch; TOPOLOGICAL's new
    bucket count is `max(new_level.max(), 1)`, exactly finalize's "the
    highest level ever used as a source is max(levels) - 1" derivation --
    unlike construction, a resort's bucket count can move in EITHER
    direction versus the old static: growth can deepen the graph (more
    buckets) and PruneConn can orphan a formerly-deep subtree (fewer).

    Redistribution is two device-side passes per new bucket, both reusing
    the prefix-sum null-slot idiom `build_add_conn_phase` already
    establishes: (1) a cumsum-rank COMPACTING scatter of every old conn
    (concatenated across every OLD bucket) whose (live, new source level)
    matches this bucket, into a fresh `capacity_b`-sized column (capacity_b
    from capacity_policy, sized off the live count that same predicate
    yields -- so every match provably fits and the scatter's "no such rank"
    sink, one past capacity_b, only ever catches non-matches); (2) a stable
    lax.sort_key_val over a single combined `dead * num_units + from_id` key
    (from_id < num_units always, so the two key ranges never collide) that
    groups the live edges by source, first -- step (1) preserves each
    match's OLD relative order, not from_id order, so this second pass is
    not redundant with it; within a source the old relative order is kept
    (not the builder's to_id tie-break). The order
    is for performance, not a precondition: grouping by source keeps
    consecutive scatter-adds off a single destination (see
    NetworkBuilder._assemble), and in-place prune and add loosen it on the
    next step, so no sweep passes a sorted-segment hint.

    Type Args:
        GS: Growth-state type parameter carried by NetworkState.

    Args:
        static: Static network configuration.
        state: Current network state.

    Returns:
        The new (static, state) pair with resorted conns.

    Raises:
        ValueError: The network propagates TOPOLOGICALLY and its live
            connections contain a cycle.
    """
    num_units = static.num_units
    if static.propagation is not Propagation.PIPELINE and bool(
        has_cycle(static, state)
    ):
        raise ValueError(
            "resort: topological propagation requires an acyclic graph; the "
            "live connections contain a cycle"
        )
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
        new_num_buckets = max(max_level + (1 if static.deepest_grows else 0), 1)
        safe_bucket = jnp.where(dead, jnp.int32(num_units), bucket_of)
        histogram = monoid.sum_.segment_reduce(
            jnp.ones_like(safe_bucket, dtype=jnp.int32),
            safe_bucket,
            num_units,
            indices_are_sorted=False,
        )
        # Host transfer 2/2: the per-level live-conn counts (module
        # docstring), sliced to just the buckets that will actually exist.
        live_counts = [int(c) for c in np.asarray(histogram[:new_num_buckets])]

    # The build-time sizing policy (headroom and rounding, recorded in the
    # static config) carries through: a resort sized to the bare live count
    # would leave every bucket full and turn the next growth into an
    # overflow -> grow_bucket -> retrace.
    # Never shrink a bucket that carries over: the space it had (maybe grown
    # by grow_bucket just before) is what its growth needs, and re-tightening
    # it to the policy would turn the next growth into another overflow ->
    # grow -> retrace.
    old_capacities = static.level_capacities
    new_level_capacities = tuple(
        max(
            capacity_policy(
                live, headroom=static.capacity_headroom, align=static.capacity_align
            ),
            old_capacities[i] if i < len(old_capacities) else 0,
        )
        for i, live in enumerate(live_counts)
    )

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
        ) + bucket_cols[FROM_ID.name].astype(jnp.int32)
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


def capacity_policy(
    live: int,
    *,
    min_bucket: int = 64,
    headroom: float = 0.0,
    align: int | None = None,
) -> int:
    """Compute a bucket capacity with headroom above the live count.

    Default policy: max(next_pow2(live), min_bucket). `headroom` pre-allocates
    more dead slots for device-resident growth: the live count is inflated by
    (1 + headroom) before the next-power-of-two rounding, reserving slots that
    add_conn can grow into without an overflow -> host `grow_bucket` rebuild
    (each rebuild is a retrace). The result stays a power of two, so it stays
    divisible by a power-of-two Scheme-A shard count. The rounding is coarse:
    any headroom > 0 on a power-of-two live count lands at the next power of two
    (a full doubling). headroom=0.0 is the historical policy; constants are an
    open tuning item.

    With `align` set, the target is instead rounded up to a multiple of
    `align` (and of at least `min_bucket`), so capacity tracks
    `live * (1 + headroom)` to within `align` slots. Power-of-two rounding can
    leave up to half a bucket empty, and every pass that streams the whole
    bucket (forward, backward, prune, the growth free-slot scan) pays for the
    empty half. Under Scheme-A, `align` must be a multiple of the shard count.

    Args:
        live: Number of live conns the bucket must hold.
        min_bucket: Minimum capacity to allocate regardless of live count.
        headroom: Extra dead-slot fraction to pre-allocate above `live`
            (0.0 = none, 1.0 = at least double). Must be non-negative.
        align: Round up to a multiple of this instead of to a power of two,
            or None for the power-of-two policy. Must be positive.

    Returns:
        The capacity to allocate for the bucket.

    Raises:
        ValueError: If `headroom` is negative or `align` is not positive.
    """
    if headroom < 0.0:
        raise ValueError(f"capacity_policy: headroom must be >= 0, got {headroom}")
    if align is not None and align < 1:
        raise ValueError(f"capacity_policy: align must be >= 1, got {align}")
    target = math.ceil(max(live, 0) * (1.0 + headroom))
    if align is not None:
        target = max(target, min_bucket)
        return -(-target // align) * align
    if live <= 0:
        return min_bucket
    next_pow2 = 1 << (target - 1).bit_length()
    return max(next_pow2, min_bucket)
