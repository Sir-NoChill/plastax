"""Phase builders: each builds a pure state->state function for one Do* phase.

Returns None when the trait slot is absent (trace-time elision). Phase
order: forward, loss, backward, update_conn, prune_conn, add_conn,
reset_global.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int32, Shaped

from plastax import monoid
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
from plastax.traits import AddConn, Network, ProposeAddConn
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
        # state.conns[0]. Not indices_are_sorted: buckets are laid out
        # source-major (builder, resort), so TO_ID is not sorted, and in-place
        # prune (a tombstone's null target) and add (a new edge in any dead
        # slot) would break any order anyway; a violated hint is undefined
        # in XLA.
        sweep = build_forward_sweep(
            net.forward_pass,
            num_units=static.num_units,
            indices_are_sorted=False,
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
    # Not indices_are_sorted, for the same reason as the pipeline forward.
    accumulate = build_forward_accumulate(
        fp,
        num_units=num_units,
        indices_are_sorted=False,
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
        # segments by FROM_ID, and although buckets start source-major,
        # dead slots' null targets and in-place adds break that order.
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
    # vmapped over the whole output set rather than unrolled per output id:
    # `per_output` is a pure scalar policy over views (views.py docstring), so
    # one trace of it serves every output. The unrolled Python loop this
    # replaced emitted a gather + a scatter *per output unit*, so trace and XLA
    # compile cost grew superlinearly in the output count -- fine for the 1-10
    # outputs of an MLP, but the wall for extreme multi-label classification,
    # whose whole point is 10^5-10^6 output units.
    output_ids = jnp.asarray(static.output_ids, dtype=jnp.int32)

    def loss_phase(
        state: NetworkState[GS], inputs: StepInputs
    ) -> tuple[NetworkState[GS], Float[Array, ""]]:
        # StepInputs.targets is None only when net.loss is None (phases.py
        # docstring); build_phases only reaches here when net.loss is set.
        assert inputs.targets is not None
        u_view = UnitView(state.units)
        globals_ = state.globals_

        def one(
            unit_id: Int32[Array, ""], target: Float[Array, ""]
        ) -> tuple[Float[Array, ""], dict[str, Shaped[Array, ""]]]:
            # UnitWrite is not a registered pytree, so vmap carries the
            # underlying field mapping and the scatter below rebuilds columns.
            value, write = loss.per_output(u_view, UnitIdx(unit_id), target, globals_)
            return value, dict(write.fields)

        values, columns = jax.vmap(one)(output_ids, inputs.targets)
        units = dict(state.units)
        for name, column in columns.items():
            units[name] = units[name].at[output_ids].set(column)
        return dataclasses.replace(state, units=units), jnp.sum(values)

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


# jnp.searchsorted's default method ("scan") is a while loop, one tiny kernel
# launch per halving step on GPU; unrolled, the whole search fuses. At 5.4M
# edges this took the add phase from 0.34 to 0.07 ms per step.
_SEARCH = "scan_unrolled"

# Largest num_units whose pair id `src * num_units + dst` still fits in int32.
_INT32_PAIR_UNITS = 46340


def live_pair_member(
    from_id: Int32[Array, " cap"],
    to_id: Int32[Array, " cap"],
    dead: Bool[Array, " cap"],
    cand_src: Int32[Array, " p"],
    cand_dst: Int32[Array, " p"],
    num_units: int,
) -> Bool[Array, " p"]:
    """Whether each candidate `(src, dst)` is already a live edge of the bucket.

    Exact for every `num_units`: the live pair ids `src * num_units + dst`
    are sorted once and each candidate is binary-searched. While the id fits
    in int32 (num_units <= 46340) it is computed in int32; past that it would
    wrap, so it is computed in uint64 under a scoped `jax.enable_x64` (still a
    single-key radix sort on GPU, about 3x the int32 cost). Dead rows get an
    id no candidate can have. Cost is O((P + cap) log cap).

    Args:
        from_id: The bucket's source column.
        to_id: The bucket's destination column.
        dead: The bucket's tombstone mask.
        cand_src: Candidate source ids.
        cand_dst: Candidate destination ids, parallel to `cand_src`.
        num_units: Total unit count, the id bound.

    Returns:
        A mask over the candidates, True where the pair is live in the bucket.
    """
    cap = from_id.shape[0]
    last = jnp.int32(cap - 1)
    src = from_id.astype(jnp.int32)
    dst = to_id.astype(jnp.int32)
    if num_units <= _INT32_PAIR_UNITS:
        live_pair = jnp.where(dead, jnp.int32(-1), src * jnp.int32(num_units) + dst)
        sorted_live = jnp.sort(live_pair)
        cand_pair = cand_src * jnp.int32(num_units) + cand_dst
        pos = jnp.minimum(
            jnp.searchsorted(sorted_live, cand_pair, method=_SEARCH), last
        )
        hit: Bool[Array, " p"] = sorted_live[pos] == cand_pair
        return hit
    # Past the bound, the same search on a uint64 pair id. x64 is enabled
    # only while tracing these ops; nothing 64-bit escapes (the result is a
    # bool mask), so the rest of the step keeps the default 32-bit types.
    with jax.enable_x64(True):
        n = jnp.uint64(num_units)
        sentinel = jnp.uint64(num_units) * n  # above every real pair id
        live_pair64 = jnp.where(
            dead, sentinel, src.astype(jnp.uint64) * n + dst.astype(jnp.uint64)
        )
        sorted_live64 = jnp.sort(live_pair64)
        cand_pair64 = cand_src.astype(jnp.uint64) * n + cand_dst.astype(jnp.uint64)
        pos = jnp.minimum(
            jnp.searchsorted(sorted_live64, cand_pair64, method=_SEARCH), last
        )
        wide_hit: Bool[Array, " p"] = sorted_live64[pos] == cand_pair64
    return wide_hit


def repeats_earlier(
    src: Int32[Array, " k"], dst: Int32[Array, " k"]
) -> Bool[Array, " k"]:
    """Mark each `(src, dst)` pair that also occurs at an earlier position.

    Two stable single-key sorts (dst, then src) order the pairs
    lexicographically while keeping equal pairs in their original order, so
    in every run of equal pairs all but the first-positioned copy is marked.
    O(k log k), for the (small) per-step top-k.

    Args:
        src: Source ids.
        dst: Destination ids, parallel to `src`.

    Returns:
        True where the same pair occurs at a lower index.
    """
    order = jnp.arange(src.shape[0], dtype=jnp.int32)
    _, order = jax.lax.sort_key_val(dst, order, is_stable=True)
    _, order = jax.lax.sort_key_val(src[order], order, is_stable=True)
    s_src, s_dst = src[order], dst[order]
    same_as_prev = jnp.concatenate(
        [
            jnp.zeros((1,), dtype=jnp.bool_),
            (s_src[1:] == s_src[:-1]) & (s_dst[1:] == s_dst[:-1]),
        ]
    )
    repeated: Bool[Array, " k"] = (
        jnp.zeros_like(same_as_prev).at[order].set(same_as_prev)
    )
    return repeated


# Block length for the two-level free-slot search: the largest power of two up
# to this that divides the bucket, so the per-block counts are a plain reshape.
_FREE_BLOCK = 1024


def count_free_blocks(
    dead: Bool[Array, " cap"],
) -> tuple[Int32[Array, " blocks"], int]:
    """Inclusive running count of free (dead) slots per block of the bucket.

    The first level of the free-slot search: `blocks[b]` counts the dead slots
    in blocks 0..b. Reads the mask once (a reduction), where a slot-level
    cumsum would also write 4 bytes per slot.

    Args:
        dead: The bucket's tombstone mask.

    Returns:
        The per-block inclusive counts and the (static) block length. The last
        count is the bucket's total free slots.
    """
    cap = dead.shape[0]
    block = _FREE_BLOCK
    while block > 1 and cap % block:
        block //= 2
    if block < 64:  # an awkward capacity: pad rather than use tiny blocks
        block = _FREE_BLOCK
        dead = jnp.pad(dead, (0, -cap % block))
    counts = dead.reshape(-1, block).sum(axis=1, dtype=jnp.int32)
    running: Int32[Array, " blocks"] = jnp.cumsum(counts)
    return running, block


def nth_free_slot(
    dead: Bool[Array, " cap"],
    free_blocks: Int32[Array, " blocks"],
    block: int,
    rank: Int32[Array, " k"],
) -> Int32[Array, " k"]:
    """Position of each `rank`-th (0-based) free slot of the bucket.

    The second level: a binary search over the block counts finds each rank's
    block, then a cumsum over just that block finds the slot -- O(k * block)
    work, independent of the capacity. Ranks at or beyond the free count return
    an arbitrary in-bounds position; callers mask them.

    Args:
        dead: The bucket's tombstone mask.
        free_blocks: `count_free_blocks(dead)`'s running counts.
        block: `count_free_blocks(dead)`'s block length.
        rank: The free-slot ranks to locate.

    Returns:
        One slot position per rank.
    """
    cap = dead.shape[0]
    target = rank + jnp.int32(1)
    b = jnp.minimum(
        jnp.searchsorted(free_blocks, target, method=_SEARCH).astype(jnp.int32),
        jnp.int32(free_blocks.shape[0] - 1),
    )
    before = jnp.where(b > 0, free_blocks[jnp.maximum(b - 1, 0)], jnp.int32(0))
    idx = b[:, None] * jnp.int32(block) + jnp.arange(block, dtype=jnp.int32)[None, :]
    in_block = jnp.where(idx < cap, dead[jnp.minimum(idx, cap - 1)], False)
    running = jnp.cumsum(in_block.astype(jnp.int32), axis=1)
    offset = jax.vmap(functools.partial(jnp.searchsorted, method=_SEARCH))(
        running, target - before
    ).astype(jnp.int32)
    slot: Int32[Array, " k"] = jnp.minimum(b * jnp.int32(block) + offset, cap - 1)
    return slot


@dataclasses.dataclass(frozen=True)
class ShortlistCoverage:
    """How much of one bucket's destination population a shortlist can reach.

    Attributes:
        bucket: the source-level bucket index.
        candidate_units: the shortlist size M the AddConn declares.
        source_units: units sitting at this bucket's source level.
        destination_units: units this bucket may grow INTO, i.e. those within
            the neighbourhood window and strictly deeper.
        covered: whether M reaches every eligible destination.
    """

    bucket: int
    candidate_units: int
    source_units: int
    destination_units: int
    covered: bool


def shortlist_coverage[GS](
    net: type[Network[GS]], static: NetworkStatic, state: NetworkState[GS]
) -> tuple[ShortlistCoverage, ...]:
    """Report, per bucket, whether the growth shortlist reaches every destination.

    A shortlisted `add_conn` draws candidates from the M most important sources
    at a bucket's own level crossed with the M most important eligible
    destinations. **M therefore bounds how many distinct units can receive a new
    edge**, which is a different and usually tighter constraint than the
    "M >= sqrt(zeta * E)" volume rule: a bucket whose destination layer is wider
    than M can only ever refill into M of those units, so a
    count-conserving churn quietly under-fills and the realized sparsity drifts
    away from the target. Measured on a 16-unit hidden layer, M=8 bled a
    128-edge arena down to 121-125 while M>=16 held it exactly.

    This runs on the host against a built state, because the level assignment
    that decides which units are eligible is runtime data, not static config --
    so the traced phase cannot check it for you.

    Args:
        net: the network type whose ``add_conn`` declares the shortlist.
        static: static network configuration.
        state: a built state, read for its LEVEL column.

    Returns:
        One ShortlistCoverage per bucket, empty when the net declares no
        shortlist (the exhaustive grid always covers everything).
    """
    add_conn = net.add_conn
    max_candidate_units: int | None = getattr(add_conn, "max_candidate_units", None)
    if add_conn is None or max_candidate_units is None:
        return ()
    levels = np.asarray(state.units[LEVEL.name])
    neighbourhood = net.neighbourhood
    out: list[ShortlistCoverage] = []
    for bucket in range(len(static.level_capacities)):
        sources = int(np.sum(levels == bucket))
        # matches build_add_conn_phase's own window: strictly deeper, within
        # the neighbourhood radius.
        destinations = int(
            np.sum((levels > bucket) & (levels <= bucket + neighbourhood))
        )
        out.append(
            ShortlistCoverage(
                bucket=bucket,
                candidate_units=max_candidate_units,
                source_units=sources,
                destination_units=destinations,
                covered=max_candidate_units >= destinations,
            )
        )
    return tuple(out)


def recommended_shortlist[GS](
    net: type[Network[GS]], static: NetworkStatic, state: NetworkState[GS]
) -> int:
    """Smallest shortlist that reaches every eligible destination in every bucket.

    Use as a floor, not a target: it satisfies the destination-coverage
    constraint only. The candidate *volume* rule still applies on top -- a churn
    frees about ``zeta * E`` slots and can refill only from the shortlisted
    grid, so size M above ``sqrt(zeta * E)`` as well.

    Args:
        net: the network type whose ``add_conn`` declares the shortlist.
        static: static network configuration.
        state: a built state, read for its LEVEL column.

    Returns:
        The maximum eligible-destination count over buckets, or 0 when the net
        declares no shortlist.
    """
    coverage = shortlist_coverage(net, static, state)
    return max((c.destination_units for c in coverage), default=0)


def build_add_conn_phase[GS](
    net: type[Network[GS]],
    static: NetworkStatic,
    *,
    overflow_sink: list[Bool[Array, ""]] | None = None,
) -> Phase[GS]:
    """Select each bucket's top-k candidates and claim free slots via prefix sum.

    Candidates come from the (src, dst) unit-id grid -- the full num_units^2
    grid, or, when the AddConn declares `max_candidate_units` (M) and an
    `importance(u, i, g)` method, the M x M grid of that step's top-M most
    important units (an O(num_units + M^2) shortlist replacing the O(num_units^2)
    sweep) -- filtered to a level-gap window:
    `abs(level[dst] - level[src]) <= net.neighbourhood`,
    self-loops excluded (see the `window_ok` comment below for the
    per-ordered-pair derivation). In TOPOLOGICAL mode a bucket only
    sources candidates from units at its own level (matching
    NetworkBuilder.finalize's bucket-of-conn convention); PIPELINE's
    single bucket accepts a source at any level, since every live conn
    lives in one flat arena regardless of source level -- the level
    window itself is still consulted in both modes, only the destination
    bucket differs. With `dedupe` (the default for grid growth, opt-in for
    ProposeAddConn) candidates already present as a live edge in the bucket
    are masked out (each candidate's pair id binary-searched against the
    sorted live pair ids -- no num_units**2 occupancy grid), and repeated
    proposals within the step keep only their highest-scored copy, so growth
    never regrows an existing pair as a duplicate. Each bucket runs an independent
    top_k (static k) over its own scored, windowed candidates, with no
    cross-bucket sequencing. A candidate scored -inf is never committed -- the
    framework scores every invalid candidate -inf, and a growth policy returns
    -inf to veto one it must never grow (e.g. a non-deeper edge) -- so a bucket
    with more free slots than finite-scored candidates leaves the surplus empty
    rather than back-filling with vetoed edges.

    Free slots are claimed by a prefix-sum scan over each bucket's own
    `dead` mask: the scan turns dead-row rank into a slot assignment, so a
    committed candidate lands in the position of the rank-th free dead
    slot. A candidate that is invalid or for which the bucket has no free
    slot scatters to one past the bucket's valid range, which the
    scatter's default drop mode discards rather than mis-writing a live
    slot.

    Overflow is a real (growable, top-k-selected) candidate for which its
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

    # Scheme-A sharding: the conn arena is split across `num_shards` devices on
    # its capacity axis, so the slot claim below runs over each shard's own
    # slice and coordinates growth across shards with collectives -- entirely
    # device-resident, no host round-trip. Both reduce to the single-device
    # identity (num_shards == 1, shard_axis is None) when unsharded.
    shard_axis = _shard_axis(static)
    num_shards = static.sharding.num_shards if static.sharding is not None else 1

    # Optional candidate reduction: an AddConn may declare `max_candidate_units`
    # (M) and an `importance(u, i, g)` method to shortlist the M most important
    # units each step and draw candidates only from that M x M grid, cutting
    # the per-step cost from O(num_units^2) to O(num_units + M^2). Absent (or M
    # >= num_units), the full grid is used -- the historical behavior, so
    # existing AddConn policies are unaffected.
    #
    # The shortlist is global (top-M over all units) by default; a policy may
    # also set `shortlist_per_level = True` to instead draw each bucket its own
    # M x M grid -- top-M sources at that bucket's source level x top-M deeper
    # destinations within the window -- so every transition of a layered net is
    # served. A global top-M can concentrate on one level and starve a bucket,
    # letting sparsity drift down; per-level is the fix. It is levels-based,
    # hence topological only (pipeline keeps the global shortlist), and forward
    # only (its destinations are strictly deeper), matching how growth policies
    # veto non-deeper edges anyway.
    # Proposal growth (ProposeAddConn) replaces the grid as the candidate
    # source: the policy emits `num_proposals` (src, dst, score) triples and
    # everything downstream -- routing, window, top_k, slot claim, init -- is
    # shared with the grid path. The live-edge duplicate check defaults on for
    # the grid and off for proposals (parallel edges allowed; see
    # ProposeAddConn).
    use_propose = isinstance(ac, ProposeAddConn)
    dedupe = bool(getattr(ac, "dedupe", not use_propose))
    num_proposals = ac.num_proposals if isinstance(ac, ProposeAddConn) else 0

    max_candidate_units: int | None = getattr(ac, "max_candidate_units", None)
    importance_fn = getattr(ac, "importance", None)
    use_shortlist = (
        not use_propose
        and max_candidate_units is not None
        and importance_fn is not None
        and 0 < max_candidate_units < num_units
    )
    use_per_level = (
        use_shortlist
        and bool(getattr(ac, "shortlist_per_level", False))
        and not is_pipeline
    )
    pool_side = max_candidate_units if use_shortlist else num_units
    assert pool_side is not None  # use_shortlist implies max_candidate_units set
    # Static (Python-int) candidate-pool bound: top_k requires k <= pool size,
    # and a small test network's pool can undercut a generous max_candidates.
    pool = num_proposals if use_propose else pool_side * pool_side
    k = max(0, min(ac.max_candidates, pool))

    # Every candidate grid is built inside the traced phase, never here: this
    # builder runs eagerly (outside jit), where a num_units^2 grid would be a
    # real allocation held by the phase closure -- even for a proposal or
    # shortlist policy that never reads it (4 * num_units^2 bytes per column,
    # terabytes at a million units). Traced, an unused grid is dead code.

    def importance_scores(u_view: UnitView, g: GS) -> jax.Array:
        """The per-unit importance vector (num_units,), for either shortlist."""
        assert importance_fn is not None  # only called when shortlisting
        unit_ids = jnp.arange(num_units, dtype=jnp.int32)

        def one(i: jax.Array) -> jax.Array:
            score = importance_fn(u_view, UnitIdx(i), g).astype(jnp.float32)
            return cast(jax.Array, score)

        return jax.vmap(one)(unit_ids)

    def candidate_grid(u_view: UnitView, g: GS) -> tuple[jax.Array, jax.Array]:
        """The global (flat_src, flat_dst) grid: full num_units^2 or top-M^2."""
        if not use_shortlist:
            unit_ids = jnp.arange(num_units, dtype=jnp.int32)
            full_src = jnp.repeat(unit_ids, num_units, total_repeat_length=num_units**2)
            full_dst = jnp.tile(unit_ids, num_units)
            return full_src, full_dst
        _, top = jax.lax.top_k(importance_scores(u_view, g), pool_side)
        src = jnp.broadcast_to(top[:, None], (pool_side, pool_side)).reshape(-1)
        dst = jnp.broadcast_to(top[None, :], (pool_side, pool_side)).reshape(-1)
        return src, dst

    def per_level_grid(
        imp: jax.Array, unit_level: jax.Array, bucket_idx: int
    ) -> tuple[jax.Array, jax.Array]:
        """One bucket's grid: top-M sources at its level x top-M deeper dests.

        A source top_k that pulls in a wrong-level unit (fewer than M sit at the
        level) is harmless -- the bucket's own `src_ok` filter drops it.
        """
        src_imp = jnp.where(unit_level == bucket_idx, imp, -jnp.inf)
        _, src_top = jax.lax.top_k(src_imp, pool_side)
        deeper = (unit_level > bucket_idx) & (unit_level <= bucket_idx + neighbourhood)
        _, dst_top = jax.lax.top_k(jnp.where(deeper, imp, -jnp.inf), pool_side)
        src = jnp.broadcast_to(src_top[:, None], (pool_side, pool_side)).reshape(-1)
        dst = jnp.broadcast_to(dst_top[None, :], (pool_side, pool_side)).reshape(-1)
        return src, dst

    def add_conn_phase(
        state: NetworkState[GS], inputs: StepInputs
    ) -> tuple[NetworkState[GS], Float[Array, ""]]:
        del inputs
        units = state.units
        g = state.globals_
        u_view = UnitView(units)
        unit_level = units[LEVEL.name]
        # Per-level shortlisting draws each bucket its own grid in the loop
        # below; every other mode reuses this one global grid. Its per-bucket
        # window (`abs(gap) <= neighbourhood`, self-loops excluded -- the window
        # admits same-level and toward-shallower pairs, which a growth policy's
        # score vetoes if unwanted) is also computed in the loop, cheap on the
        # shared grid. The global grid is built unconditionally so the loop's two
        # branches both bind flat_src/flat_dst; when per-level it is the (small)
        # top-M grid and goes unused, dead-code-eliminated -- never the
        # num_units^2 full grid.
        imp = importance_scores(u_view, g) if use_per_level else None
        if isinstance(ac, ProposeAddConn):
            # The proposals are this step's global candidate list: computed
            # once, filtered per bucket in the loop. An out-of-range id is
            # vetoed and clamped to 0 so every gather below stays in bounds.
            def propose_one(
                j: jax.Array,
            ) -> tuple[jax.Array, jax.Array, jax.Array]:
                s_, d_, score_ = ac.propose(u_view, j, g)
                return (
                    jnp.asarray(s_, jnp.int32),
                    jnp.asarray(d_, jnp.int32),
                    jnp.asarray(score_, jnp.float32),
                )

            p_src, p_dst, p_score = jax.vmap(propose_one)(
                jnp.arange(num_proposals, dtype=jnp.int32)
            )
            in_range = (
                (p_src >= 0) & (p_src < num_units) & (p_dst >= 0) & (p_dst < num_units)
            )
            global_src = jnp.where(in_range, p_src, jnp.int32(0))
            global_dst = jnp.where(in_range, p_dst, jnp.int32(0))
        else:
            global_src, global_dst = candidate_grid(u_view, g)

        def scored(s: jax.Array, d: jax.Array, ok: jax.Array) -> jax.Array:
            assert isinstance(ac, AddConn)  # the grid path only
            raw = ac.score(u_view, UnitIdx(s), UnitIdx(d), g)
            return jnp.where(ok, raw.astype(jnp.float32), jnp.float32(-jnp.inf))

        def init_one(s: jax.Array, d: jax.Array) -> dict[str, jax.Array]:
            # ConnWrite is not pytree-registered (views.py); unwrap .fields
            # to a plain dict before vmap, matching _build_conn_update /
            # _apply_masked's UnitWrite handling in sweep.py.
            write = ac.init(u_view, UnitIdx(s), UnitIdx(d), g)
            return dict(write.fields)

        def not_live_duplicate(
            bucket_conns: Columns, cand_src: jax.Array, cand_dst: jax.Array
        ) -> jax.Array:
            """Candidates that are not already a live edge of this bucket.

            A sort of the live pairs plus a binary search per candidate,
            O((P + cap) * log cap) with no num_units**2 occupancy grid, so a
            shortlisted or proposal phase stays free of any num_units**2 term.
            """
            not_duplicate = ~live_pair_member(
                bucket_conns[FROM_ID.name],
                bucket_conns[TO_ID.name],
                bucket_conns[DEAD.name],
                cand_src,
                cand_dst,
                num_units,
            )
            if shard_axis is None:
                return not_duplicate
            # Under Scheme-A the live edges are split across shards, so the
            # binary search above only sees THIS shard's slice. A candidate
            # already live on any other shard must count as a duplicate
            # everywhere -- otherwise shards would score a different candidate
            # set, top_k differently, and disagree on the global slot
            # assignment below. All-reduce the local duplicate mask (pmax ==
            # boolean OR) so valid/scores/top_k are identical on every shard.
            dup_any = monoid.max_.collective(
                (~not_duplicate).astype(jnp.int32), shard_axis
            )
            nowhere_live: jax.Array = dup_any == jnp.int32(0)
            return nowhere_live

        new_conns: list[Columns] = []
        overflow = jnp.bool_(False)
        reassigning = jnp.bool_(False)
        for bucket_idx in range(num_buckets):
            bucket_conns = state.conns[bucket_idx]
            capacity_b = static.level_capacities[bucket_idx]
            if use_per_level:
                assert imp is not None  # use_per_level implies importance is set
                flat_src, flat_dst = per_level_grid(imp, unit_level, bucket_idx)
            else:
                flat_src, flat_dst = global_src, global_dst
            src_level = unit_level[flat_src]
            window_ok = (jnp.abs(unit_level[flat_dst] - src_level) <= neighbourhood) & (
                flat_src != flat_dst
            )
            src_ok = (
                jnp.ones_like(src_level, dtype=jnp.bool_)
                if is_pipeline
                else src_level == bucket_idx
            )
            valid = window_ok & src_ok
            if use_propose:
                valid = valid & in_range
            if dedupe:
                valid = valid & not_live_duplicate(bucket_conns, flat_src, flat_dst)
            if use_propose:
                flat_scores = jnp.where(valid, p_score, jnp.float32(-jnp.inf))
                if dedupe:
                    # Proposals, unlike grid cells, can repeat within a step.
                    # Keep each pair's highest-scored copy and veto the rest
                    # BEFORE top_k, so repeats never take a bucket's k slots.
                    by_score = jnp.argsort(-flat_scores, stable=True)
                    repeat = (
                        jnp.zeros_like(valid)
                        .at[by_score]
                        .set(repeats_earlier(flat_src[by_score], flat_dst[by_score]))
                    )
                    flat_scores = jnp.where(repeat, jnp.float32(-jnp.inf), flat_scores)
            else:
                flat_scores = jax.vmap(scored)(flat_src, flat_dst, valid)
            _, top_idx = jax.lax.top_k(flat_scores, k)
            top_src = flat_src[top_idx]
            top_dst = flat_dst[top_idx]
            top_valid = valid[top_idx]
            # A candidate is growable only if its score is finite. `scored`
            # sends every framework-invalid candidate (out-of-window, wrong
            # source level, or a duplicate of a live edge) to -inf, and a growth
            # policy likewise returns -inf to *veto* a candidate it must never
            # grow (e.g. a non-deeper edge that would break leveling) rather
            # than merely rank it last. Gating commitment on finiteness makes
            # -inf a hard veto: without it, a bucket whose free slots outnumber
            # its finite-scored candidates would back-fill the surplus with
            # vetoed edges, since top_k still surfaces them and `has_room`
            # alone would admit them.
            top_growable = top_valid & jnp.isfinite(flat_scores[top_idx])

            # Prefix-sum slot claim, sharding-aware. Under Scheme-A the runtime
            # dead mask is this shard's capacity slice (size capacity_b //
            # num_shards; power-of-two capacities keep it exact), so the claim
            # runs over the LOCAL slice and is coordinated across shards: each
            # growable candidate takes a GLOBAL free-slot rank and lands on the
            # one shard that owns it. Because shard g holds arena positions
            # [g*local_capacity, (g+1)*local_capacity), the global free-slot
            # order (shard 0's free slots, then shard 1's, ...) is exactly the
            # single-device position order -- so a candidate lands where the
            # single-device add_conn would. local_capacity == capacity_b and
            # offset == 0 when unsharded, leaving that path byte-identical.
            local_capacity = capacity_b // num_shards
            dead_b = bucket_conns[DEAD.name]
            # Per-block free counts over this shard's slice: one reduction
            # reading the dead mask, instead of a capacity-sized cumsum.
            free_blocks, block_len = count_free_blocks(dead_b)
            local_free = free_blocks[-1]
            # This shard's offset into the global free-slot space, and the total
            # free count. The offset is an exclusive prefix of the per-shard
            # free counts (an all-gather -- a prefix is not a plain all-reduce)
            # and feeds only the per-shard placement below. total_free is a
            # psum, not sum(all_gather): it flows into `overflow` and
            # `needs_resort`, which shard_map requires be provably replicated,
            # and psum is the all-reduce it recognizes as replicating. Both are
            # device-resident collectives.
            if shard_axis is not None:
                all_free = jax.lax.all_gather(local_free, shard_axis)
                my_index = jax.lax.axis_index(shard_axis)
                offset = jnp.sum(
                    jnp.where(jnp.arange(num_shards) < my_index, all_free, 0)
                )
                total_free = monoid.sum_.collective(local_free, shard_axis)
            else:
                offset = jnp.int32(0)
                total_free = local_free
            # growth_rank[i] = candidate i's rank among the growable top-k --
            # its global free-slot index. The -inf-scored candidates sort to
            # the tail of top_k, so top_growable is a contiguous prefix and
            # growth_rank[i] == i there: identical to the old per-index claim
            # when unsharded. A candidate whose rank exceeds the total free
            # slots is overflow -- dropped, flag raised.
            growth_rank = jnp.cumsum(top_growable.astype(jnp.int32)) - 1
            committed = top_growable & (growth_rank < total_free)
            overflow = overflow | jnp.any(top_growable & (growth_rank >= total_free))
            # A committed candidate belongs to THIS shard iff its global rank
            # falls in [offset, offset + local_free); place it at that shard-
            # local free slot, else scatter to `local_capacity` (out of this
            # slice's range), dropped by the scatter's drop mode.
            local_rank = growth_rank - offset
            mine = committed & (local_rank >= jnp.int32(0)) & (local_rank < local_free)
            safe_rank = jnp.where(mine, local_rank, jnp.int32(0))
            free_slot = nth_free_slot(dead_b, free_blocks, block_len, safe_rank)
            target_slot = jnp.where(mine, free_slot, jnp.int32(local_capacity))

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
