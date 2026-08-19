"""Phase builders: each builds a pure state->state function for one Do* phase.

Returns None when the trait slot is absent (trace-time elision). Phase
order: forward, loss, backward, update_conn, prune_conn, add_conn,
reset_global.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float

from plastax._types import DEAD, FROM_ID, LEVEL, TO_ID, ConnIdx, Propagation, UnitIdx
from plastax.state import Columns, NetworkState, NetworkStatic
from plastax.sweep import (
    build_backward_accumulate,
    build_backward_apply,
    build_backward_sweep,
    build_forward_accumulate,
    build_forward_apply,
    build_forward_sweep,
    build_incoming_conn_update,
    build_outgoing_conn_update,
    identity_accumulator,
    unit_id_mask,
)
from plastax.traits import Network
from plastax.views import ConnView, UnitView

# PEP 695 generic alias: lazily evaluated, so the NetworkState/StepInputs
# forward references need no quoting. Every phase also returns a scalar loss
# contribution: globals_ is a fully opaque user pytree (GS may be None,
# plain dict, ...), so the loss phase's reduced scalar has no generic slot
# to land in inside NetworkState; thread it out as a second return so
# make_step can fold it into StepResult.loss (sibling to `overflow`, itself
# a framework-computed, state-external signal) instead of assuming
# globals_ has a loss field. Non-loss phases return 0.0.
type Phase[GS] = Callable[
    [NetworkState[GS], StepInputs], tuple[NetworkState[GS], Float[Array, ""]]
]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class StepInputs:
    """Clamped inputs + targets for one step; fixed pytree structure.

    Attributes:
        inputs: the (num_inputs,) values scattered to input unit ids.
        targets: the (num_outputs,) loss targets, or None when the net
            has no loss phase.
    """

    inputs: Float[Array, " num_inputs"]
    targets: Float[Array, " num_outputs"] | None


def build_phases[GS](
    net: type[Network[GS]],
    static: NetworkStatic,
    *,
    overflow_sink: list[Bool[Array, ""]] | None = None,
) -> tuple[Phase[GS], ...]:
    """Assemble the phases present for this net; absent slots trace nothing.

    Topological forward walks buckets 1..L (Python loop, static slices);
    backward walks L..1; pipeline is the 1-bucket flat sweep. Phase order:
    forward, loss, backward, update_conn, prune_conn, add_conn,
    reset_global.

    Type Args:
        GS: the user's global-state pytree, opaque to the framework.

    Args:
        net: the network's trait class, supplying each phase's callbacks.
        static: static network configuration giving the arena shapes.
        overflow_sink: optional length-1 out-parameter that
            build_add_conn_phase overwrites with its computed overflow
            flag on every call.

    Returns:
        The tuple of phase functions to run in order, one per present
        trait slot.
    """
    phases: list[Phase[GS]] = [_build_forward_phase(net, static)]
    if net.loss is not None:
        phases.append(_build_loss_phase(net, static))
    if net.backward_pass is not None:
        phases.append(_build_backward_phase(net, static))
    if net.update_conn is not None:
        phases.append(build_update_conn_phase(net, static))
    if net.prune_conn is not None:
        phases.append(build_prune_conn_phase(net, static))
    if net.add_conn is not None:
        phases.append(build_add_conn_phase(net, static, overflow_sink=overflow_sink))
    if net.reset_global is not None:
        phases.append(_build_reset_global_phase(net))
    return tuple(phases)


def _shard_axis(static: NetworkStatic) -> str | None:
    """Return the Scheme-A mesh axis name, or None when unsharded."""
    return static.sharding.axis_name if static.sharding is not None else None


def _build_forward_phase[GS](
    net: type[Network[GS]], static: NetworkStatic
) -> Phase[GS]:
    if net.propagation is Propagation.PIPELINE:
        # level_capacities is a 1-tuple -- the single flat bucket is
        # state.conns[0]. indices_are_sorted=True holds from
        # NetworkBuilder.finalize's per-bucket (dead, to_id) sort.
        sweep = build_forward_sweep(
            net.forward_pass,
            num_units=static.num_units,
            indices_are_sorted=True,
            input_ids=static.input_ids,
            shard_axis=_shard_axis(static),
        )

        def forward_phase(
            state: NetworkState[GS], inputs: StepInputs
        ) -> tuple[NetworkState[GS], Float[Array, ""]]:
            del inputs
            new_units = sweep(state.units, state.conns[0], state.globals_)
            return dataclasses.replace(state, units=new_units), jnp.float32(0.0)

        return forward_phase

    return _build_forward_topological_phase(net, static)


def _build_forward_topological_phase[GS](
    net: type[Network[GS]], static: NetworkStatic
) -> Phase[GS]:
    """Level walk over source-level buckets, in order from 0 to num_levels-1.

    One accumulate call per bucket, combined into a carried accumulator
    (sweep.build_forward_accumulate) so a unit's contributions from every
    earlier bucket survive even if its incoming edges are not all in the
    immediately preceding bucket (a skip connection sources from an
    earlier level still). A unit is only finalized
    (sweep.build_forward_apply, write + accumulator reset) once every
    bucket that could feed it has been accumulated -- which for a
    level-`level_idx + 1` unit is exactly buckets `0..level_idx`, i.e.
    right after bucket `level_idx` is processed, since no edge sources
    from a level at or above its own destination's level (the leveling
    invariant).
    """
    num_units = static.num_units
    num_levels = len(static.level_capacities)
    fp = net.forward_pass
    accumulate = build_forward_accumulate(
        fp,
        num_units=num_units,
        indices_are_sorted=True,
        shard_axis=_shard_axis(static),
    )
    apply = build_forward_apply(fp, num_units=num_units)
    not_input = ~unit_id_mask(static.input_ids, num_units)

    def forward_phase(
        state: NetworkState[GS], inputs: StepInputs
    ) -> tuple[NetworkState[GS], Float[Array, ""]]:
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
        # No level structure, one flat bucket, every unit Applied
        # unconditionally (build_backward_sweep takes no input_ids -- see
        # its docstring). indices_are_sorted=False: backward indexes
        # segments by FROM_ID, but finalize sorts each bucket by (dead,
        # TO_ID), so those indices are not sorted -- correct on CPU either
        # way, honest for GPU/TPU, matching the topological backward.
        sweep = build_backward_sweep(
            bp,
            num_units=static.num_units,
            indices_are_sorted=False,
            shard_axis=_shard_axis(static),
        )

        def backward_phase(
            state: NetworkState[GS], inputs: StepInputs
        ) -> tuple[NetworkState[GS], Float[Array, ""]]:
            del inputs
            new_units = sweep(state.units, state.conns[0], state.globals_)
            return dataclasses.replace(state, units=new_units), jnp.float32(0.0)

        return backward_phase

    return _build_backward_topological_phase(net, static)


def _build_backward_topological_phase[GS](
    net: type[Network[GS]], static: NetworkStatic
) -> Phase[GS]:
    """Reverse level walk, from bucket num_levels down to bucket 1.

    Bucket `level_idx` holds edges sourced at level_idx; backward
    accumulates into the source, so accumulating bucket `level_idx` is
    exactly what completes a level-`level_idx` unit's accumulator (every
    outgoing edge of a level-`level_idx` unit sources at level_idx, by
    definition of the bucketing -- unlike forward, there is no
    cross-bucket spread on the finalizing side). Walking buckets
    high-to-low is what guarantees a bucket's Map (which reads
    destination-side state, at a strictly higher level) only ever runs
    after that destination has already been finalized.

    The top level (== num_levels) has no source-level bucket of its own,
    since no edge sources from the deepest level -- that would need a
    destination one level deeper still -- so it is primed directly from
    the identity accumulator before the bucket loop, picking up only
    whatever an earlier phase (e.g. loss) wrote into unit columns
    Map/Apply itself read.

    The loop stops at bucket 1, never touching bucket 0 (input units' own
    outgoing edges): input units are excluded from `finalize` exactly like
    forward, so accumulating bucket 0 would only feed an accumulator that
    is never read.
    """
    num_units = static.num_units
    num_levels = len(static.level_capacities)
    bp = net.backward_pass
    assert bp is not None  # build_phases only calls this when set
    accumulate = build_backward_accumulate(
        bp,
        num_units=num_units,
        indices_are_sorted=False,
        shard_axis=_shard_axis(static),
    )
    apply = build_backward_apply(bp, num_units=num_units)
    not_input = ~unit_id_mask(static.input_ids, num_units)

    def backward_phase(
        state: NetworkState[GS], inputs: StepInputs
    ) -> tuple[NetworkState[GS], Float[Array, ""]]:
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
    ) -> tuple[NetworkState[GS], Float[Array, ""]]:
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
                u_view,
                UnitIdx(jnp.asarray(unit_id, dtype=jnp.int32)),
                inputs.targets[k],
                state.globals_,
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
    ) -> tuple[NetworkState[GS], Float[Array, ""]]:
        del inputs
        new_globals = reset_global.reset(state.globals_)
        return dataclasses.replace(state, globals_=new_globals), jnp.float32(0.0)

    return reset_global_phase


def build_update_conn_phase[GS](
    net: type[Network[GS]], static: NetworkStatic
) -> Phase[GS]:
    """Sweep every live conn twice: an incoming pass, then an outgoing pass.

    `state.conns` is swept unconditionally bucket-by-bucket for each pass
    (a 1-tuple in PIPELINE, one per source level in TOPOLOGICAL): unlike
    forward/backward, this phase has no level structure of its own, it
    just walks every live conn twice.

    The two passes are sequenced across all buckets (every bucket's
    incoming write is merged before any bucket's outgoing pass reads conn
    state) so an edge's `outgoing` call observes that same edge's
    `incoming` write already landed. UpdateConn writes only ConnWrite
    (never a UnitWrite, traits.py's Protocol), so no edge's write is ever
    visible to a different edge's callback regardless of bucketing -- the
    cross-bucket sequencing only matters for an edge observing its own
    prior write, which per-bucket sequencing alone would already
    guarantee; kept global (all incoming buckets, then all outgoing
    buckets) as the simpler of the two equally-correct orderings.

    Type Args:
        GS: the user's global-state pytree, opaque to the framework.

    Args:
        net: the network's trait class, supplying the update_conn policy.
        static: static network configuration giving the arena shapes.

    Returns:
        The update_conn phase function.
    """
    uc = net.update_conn
    assert uc is not None  # build_phases only calls this when set
    incoming = build_incoming_conn_update(uc.incoming)
    outgoing = build_outgoing_conn_update(uc.outgoing)

    def update_conn_phase(
        state: NetworkState[GS], inputs: StepInputs
    ) -> tuple[NetworkState[GS], Float[Array, ""]]:
        del inputs
        conns = tuple(
            incoming(state.units, bucket, state.globals_) for bucket in state.conns
        )
        conns = tuple(outgoing(state.units, bucket, state.globals_) for bucket in conns)
        return dataclasses.replace(state, conns=conns), jnp.float32(0.0)

    return update_conn_phase


def build_prune_conn_phase[GS](
    net: type[Network[GS]], static: NetworkStatic
) -> Phase[GS]:
    """Tombstone write: vmap the prune predicate over every conn row.

    vmap's `prune_conn.predicate` runs over every conn row of every
    bucket, OR-ing the result into that bucket's own `dead` column.
    Static shapes, no resort, no counter -- live counts stay derived
    (`state.live_conn_count`, `sum(~dead)`).

    Already-dead rows are not skipped before evaluating the predicate
    (vmap forbids data-dependent control flow), but this is harmless:
    `dead[i] | predicate(...)` is `True` regardless of the (possibly
    meaningless) predicate result whenever `dead[i]` already is, so
    evaluating every row unconditionally and OR-merging is equivalent to
    skipping already-dead rows first.

    Type Args:
        GS: the user's global-state pytree, opaque to the framework.

    Args:
        net: the network's trait class, supplying the prune_conn policy.
        static: static network configuration (unused directly, kept for
            signature symmetry with the other phase builders).

    Returns:
        The prune_conn phase function.
    """
    pc = net.prune_conn
    assert pc is not None  # build_phases only calls this when set

    def prune_conn_phase(
        state: NetworkState[GS], inputs: StepInputs
    ) -> tuple[NetworkState[GS], Float[Array, ""]]:
        del inputs
        u_view = UnitView(state.units)
        g = state.globals_

        def prune_bucket(bucket_conns: Columns) -> Columns:
            c_view = ConnView(bucket_conns)
            dead = bucket_conns[DEAD.name]
            cids = jnp.arange(dead.shape[0], dtype=jnp.int32)

            def per_conn(cid: jax.Array) -> jax.Array:
                return pc.predicate(u_view, c_view, ConnIdx(cid), g)

            should_die = jax.vmap(per_conn)(cids)
            new_bucket: Columns = dict(bucket_conns)
            new_bucket[DEAD.name] = dead | should_die
            return new_bucket

        new_conns = tuple(prune_bucket(bucket) for bucket in state.conns)
        return dataclasses.replace(state, conns=new_conns), jnp.float32(0.0)

    return prune_conn_phase


def build_add_conn_phase[GS](
    net: type[Network[GS]],
    static: NetworkStatic,
    *,
    overflow_sink: list[Bool[Array, ""]] | None = None,
) -> Phase[GS]:
    """Select each bucket's top-k candidates and claim free slots via prefix sum.

    Candidates come from the full (src, dst) unit-id grid, filtered to a
    level-gap window: `abs(level[dst] - level[src]) <= net.neighbourhood`,
    self-loops excluded (see the `window_ok` comment below for the
    per-ordered-pair derivation). In TOPOLOGICAL mode a bucket only
    sources candidates from units at its own level (matching
    NetworkBuilder.finalize's bucket-of-conn convention); PIPELINE's
    single bucket accepts a source at any level, since every live conn
    lives in one flat arena regardless of source level -- the level
    window itself is still consulted in both modes, only the destination
    bucket differs. Candidates already present as a live edge in the bucket
    are masked out (a pair-id occupancy grid built by scatter, gathered per
    candidate) so growth never regrows an existing pair as a duplicate. Each
    bucket runs an independent top_k (static k) over its own scored,
    windowed candidates, with no cross-bucket sequencing.

    Free slots are claimed by a prefix-sum scan over each bucket's own
    `dead` mask: the scan turns dead-row rank into a slot assignment, so a
    committed candidate lands in the position of the rank-th free dead
    slot. A candidate that is invalid or for which the bucket has no free
    slot scatters to one past the bucket's valid range, which the
    scatter's default drop mode discards rather than mis-writing a live
    slot.

    Overflow is a real (valid, top-k-selected) candidate for which its
    own bucket ran out of dead slots; it is dropped and the flag is
    raised via `overflow_sink` rather than committed. A committed
    candidate whose destination is not strictly deeper than its source
    (the window admits same-level and behind-src pairs) sets
    `needs_resort`, since it breaks the leveling invariant that every
    edge sources from a level strictly below its destination.

    Type Args:
        GS: the user's global-state pytree, opaque to the framework.

    Args:
        net: the network's trait class, supplying the add_conn policy.
        static: static network configuration giving the arena shapes.
        overflow_sink: optional length-1 out-parameter overwritten with
            this call's computed overflow flag.

    Returns:
        The add_conn phase function.
    """
    ac = net.add_conn
    assert ac is not None  # build_phases only calls this when set
    num_units = static.num_units
    num_buckets = len(static.level_capacities)
    neighbourhood = net.neighbourhood
    is_pipeline = net.propagation is Propagation.PIPELINE
    # Static (Python-int) candidate-pool bound: top_k requires k <= pool
    # size, and a small test network's unit count squared can undercut a
    # generously-configured max_candidates.
    k = max(0, min(ac.max_candidates, num_units * num_units))

    # Full (src, dst) unit-id grid, built once (num_units is static): the
    # per-bucket validity mask (level window, and TOPOLOGICAL's source-level
    # filter) is a boolean over this SAME fixed pool every step, never a
    # dynamically-sized candidate list.
    unit_ids = jnp.arange(num_units, dtype=jnp.int32)
    flat_src = jnp.broadcast_to(unit_ids[:, None], (num_units, num_units)).reshape(-1)
    flat_dst = jnp.broadcast_to(unit_ids[None, :], (num_units, num_units)).reshape(-1)
    not_self = flat_src != flat_dst

    def add_conn_phase(
        state: NetworkState[GS], inputs: StepInputs
    ) -> tuple[NetworkState[GS], Float[Array, ""]]:
        del inputs
        units = state.units
        g = state.globals_
        u_view = UnitView(units)
        unit_level = units[LEVEL.name]
        src_level = unit_level[flat_src]
        dst_level = unit_level[flat_dst]
        level_gap = dst_level - src_level
        # Per ordered (src, dst) pair, the level window is `abs(gap) <=
        # neighbourhood`: it admits any pair within `neighbourhood` levels
        # of each other, including same-level pairs and edges toward a
        # shallower destination, in both directions. Self-loops are
        # excluded explicitly (`not_self`), since gap == 0 no longer
        # implies src == dst now that same-level pairs are admitted.
        window_ok = (jnp.abs(level_gap) <= neighbourhood) & not_self

        def scored(s: jax.Array, d: jax.Array, ok: jax.Array) -> jax.Array:
            raw = ac.score(u_view, UnitIdx(s), UnitIdx(d), g)
            return jnp.where(ok, raw.astype(jnp.float32), jnp.float32(-jnp.inf))

        def init_one(s: jax.Array, d: jax.Array) -> dict[str, jax.Array]:
            # ConnWrite is not pytree-registered (views.py); unwrap .fields
            # to a plain dict before vmap, matching _build_conn_update /
            # _apply_masked's UnitWrite handling in sweep.py.
            write = ac.init(u_view, UnitIdx(s), UnitIdx(d), g)
            return dict(write.fields)

        new_conns: list[Columns] = []
        overflow = jnp.bool_(False)
        reassigning = jnp.bool_(False)
        for bucket_idx in range(num_buckets):
            bucket_conns = state.conns[bucket_idx]
            capacity_b = static.level_capacities[bucket_idx]
            src_ok = (
                jnp.ones_like(src_level, dtype=jnp.bool_)
                if is_pipeline
                else src_level == bucket_idx
            )
            # Exclude candidates already present as a live edge in this
            # bucket, so growth never regrows an existing pair as a
            # duplicate. Each edge has a static pair id `src * num_units +
            # dst`; scatter every LIVE edge's pair id into an occupancy grid
            # (dead slots routed to the throwaway sink `num_units**2`, so
            # their stale from/to cannot mark a real pair occupied), then
            # gather it at each candidate's pair id. A scatter + a gather,
            # O(num_units**2) -- the same order as the candidate grid
            # itself, no edge-list search.
            dead_bucket = bucket_conns[DEAD.name]
            live_pair = jnp.where(
                dead_bucket,
                jnp.int32(num_units * num_units),
                bucket_conns[FROM_ID.name].astype(jnp.int32) * jnp.int32(num_units)
                + bucket_conns[TO_ID.name].astype(jnp.int32),
            )
            occupied = (
                jnp.zeros((num_units * num_units + 1,), dtype=jnp.bool_)
                .at[live_pair]
                .set(True, mode="drop")
            )
            not_duplicate = ~occupied[flat_src * jnp.int32(num_units) + flat_dst]
            valid = window_ok & src_ok & not_duplicate

            flat_scores = jax.vmap(scored)(flat_src, flat_dst, valid)
            _, top_idx = jax.lax.top_k(flat_scores, k)
            top_src = flat_src[top_idx]
            top_dst = flat_dst[top_idx]
            top_valid = valid[top_idx]

            # Prefix-sum slot claim over this bucket's own dead mask:
            # rank[i] is dead-position i's 0-based rank among this bucket's
            # dead slots; scattering `positions` by `rank` (dropping live
            # positions via the always-OOB `sink_len` index) inverts that
            # into slot_for_rank[j] = the position of the j-th free slot,
            # `capacity_b` (never a real position) where fewer than j+1
            # free slots exist. `sink_len = max(capacity_b, k)` keeps both
            # the scatter (indices < capacity_b <= sink_len, in bounds) and
            # the later static `[:k]` slice in bounds without a runtime
            # clamp, even if k > capacity_b.
            dead_b = bucket_conns[DEAD.name]
            rank = jnp.cumsum(dead_b.astype(jnp.int32)) - 1
            positions = jnp.arange(capacity_b, dtype=jnp.int32)
            sink_len = max(capacity_b, k)
            scatter_target = jnp.where(dead_b, rank, jnp.int32(sink_len))
            slot_for_rank = (
                jnp.full((sink_len,), capacity_b, dtype=jnp.int32)
                .at[scatter_target]
                .set(positions, mode="drop")
            )
            free_slot = slot_for_rank[:k]
            has_room = free_slot < capacity_b
            committed = top_valid & has_room
            # Overflow: a real (valid, top-k-selected) candidate for which
            # this bucket ran out of dead slots. Uncommitted candidates
            # (invalid or no room) scatter to `capacity_b`, one past this
            # bucket's own valid range -- the scatter's default drop mode
            # discards them rather than mis-writing a live slot.
            overflow = overflow | jnp.any(top_valid & ~has_room)
            target_slot = jnp.where(committed, free_slot, jnp.int32(capacity_b))

            # A committed candidate whose destination is not strictly
            # deeper than its source breaks the leveling invariant, so it
            # marks the network as needing a topological resort.
            level_preserving = unit_level[top_dst] > unit_level[top_src]
            reassigning = reassigning | jnp.any(committed & ~level_preserving)

            batched_init = jax.vmap(init_one)(top_src, top_dst)

            new_bucket: Columns = dict(bucket_conns)
            for spec in static.conn_fields:
                value: jax.Array
                if spec.name == FROM_ID.name:
                    value = top_src.astype(spec.dtype)
                elif spec.name == TO_ID.name:
                    value = top_dst.astype(spec.dtype)
                elif spec.name == DEAD.name:
                    value = jnp.zeros((k,), dtype=spec.dtype)
                elif spec.name in batched_init:
                    value = batched_init[spec.name].astype(spec.dtype)
                else:
                    # Not touched by ac.init: reset to the FieldSpec
                    # default rather than inheriting whatever a previous
                    # tenant (a conn tombstoned by this same step's
                    # prune_conn pass, or the builder's initial padding)
                    # left behind.
                    value = jnp.full((k,), np.asarray(spec.default), dtype=spec.dtype)
                new_bucket[spec.name] = (
                    bucket_conns[spec.name].at[target_slot].set(value, mode="drop")
                )
            new_conns.append(new_bucket)

        if overflow_sink is not None:
            overflow_sink[0] = overflow
        new_state = dataclasses.replace(
            state,
            conns=tuple(new_conns),
            needs_resort=state.needs_resort | reassigning,
        )
        return new_state, jnp.float32(0.0)

    return add_conn_phase
