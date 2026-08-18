"""Phase builders: each returns a pure state->state function for one Do*
phase, or None when the trait slot is absent (trace-time elision, rung0
design section 2). Phase order matches plastix.hpp's 11-phase step for the
v1 subset: forward, loss, backward, update_conn, prune_conn, add_conn,
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
# contribution -- Deviation (IMPLEMENTATION_PLAN.md M2): globals_ is a fully
# opaque user pytree (GS may be None, plain dict, ...), so the loss phase's
# reduced scalar has no generic slot to land in inside NetworkState; thread
# it out as a second return so make_step can fold it into StepResult.loss
# (sibling to `overflow`, itself a framework-computed, state-external signal)
# instead of assuming globals_ has a loss field. Non-loss phases return 0.0.
type Phase[GS] = Callable[
    [NetworkState[GS], StepInputs], tuple[NetworkState[GS], Float[Array, ""]]  # noqa: F722
]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class StepInputs:
    """Clamped inputs + targets for one step; fixed pytree structure.

    inputs: (num_inputs,) values scattered to input unit ids (static tuple)
    targets: (num_outputs,) for the loss phase, or None when loss is absent

    Registered as a dataclass pytree (Deviation, IMPLEMENTATION_PLAN.md M2):
    the stub was a bare annotated class, not constructible/traceable. `None`
    is a valid data field (jax/_src/tree_util.py:1034) -- an empty subtree,
    not a leaf -- so `targets=None` flattens to zero leaves for a loss-less
    net, matching "fixed structure for a given net" (structure is static per
    net.loss presence, never per-call).
    """

    inputs: Float[Array, " num_inputs"]  # noqa: F722  jaxtyping named-axis string
    targets: Float[Array, " num_outputs"] | None  # noqa: F722  jaxtyping named-axis


def build_phases[GS](
    net: type[Network[GS]],
    static: NetworkStatic,
    *,
    overflow_sink: list[Bool[Array, ""]] | None = None,  # noqa: F722
) -> tuple[Phase[GS], ...]:
    """Assemble only the present phases; absent slots contribute nothing to
    the trace. Topological forward walks buckets 1..L (Python loop, static
    slices); backward walks L..1; pipeline is the 1-bucket flat sweep.

    Presence is a Python-level `if`, so an absent phase is never traced --
    zero equations, never lax.cond [D:2], which is exactly what
    test_phases_elision checks. Phase order (module docstring): forward,
    loss, backward, update_conn, prune_conn, add_conn, reset_global.

    `overflow_sink` (Deviation, IMPLEMENTATION_PLAN.md M4a -- absent from
    the rung0 sketch): an optional out-parameter, a length-1 list that
    `build_add_conn_phase` overwrites with its computed overflow flag on
    every call. `Phase[GS]`'s 2-tuple return `(state, loss_contribution)`
    has no slot for a THIRD, add_conn-only signal; widening it would ripple
    into every phase builder plus every existing direct caller
    (test_phases_elision.py's `_trace_phases`, test_update_conn.py's direct
    `build_update_conn_phase` calls). A mutable sink threaded only to the
    one phase that needs it keeps `Phase` and this function's own return
    type unchanged for every other caller, at the cost of one internal
    plumbing parameter. `step.py`'s `_cached_make_step` is the only real
    caller: the mutation happens once, at trace time (`jax.jit` traces the
    Python body of `step` exactly once), so the sunk value flows into the
    jaxpr as an ordinary data dependency, not a stale Python-side read.
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


def _build_forward_phase[GS](
    net: type[Network[GS]], static: NetworkStatic
) -> Phase[GS]:
    if net.propagation is Propagation.PIPELINE:
        # level_capacities is a 1-tuple (rung0 design section 3) -- the
        # single flat bucket is state.conns[0]. indices_are_sorted=True
        # holds from NetworkBuilder.finalize's per-bucket (dead, to_id) sort.
        sweep = build_forward_sweep(
            net.forward_pass,
            num_units=static.num_units,
            indices_are_sorted=True,
            input_ids=static.input_ids,
        )

        def forward_phase(
            state: NetworkState[GS], inputs: StepInputs
        ) -> tuple[NetworkState[GS], Float[Array, ""]]:  # noqa: F722
            del inputs
            new_units = sweep(state.units, state.conns[0], state.globals_)
            return dataclasses.replace(state, units=new_units), jnp.float32(0.0)

        return forward_phase

    return _build_forward_topological_phase(net, static)


def _build_forward_topological_phase[GS](
    net: type[Network[GS]], static: NetworkStatic
) -> Phase[GS]:
    """Level walk, source-level buckets 0..num_levels-1 in order
    (dispatch_cpu.hpp:41-67's `L = 1..NumLevels`, reindexed to plastax's
    0-based `conns` tuple: C++ `Ranges[L-1]` is plastax `conns[L-1]`; unit
    LEVEL values themselves need no reindexing, they already match the
    oracle's 1-based-from-inputs numbering).

    One accumulate call per bucket, combined into a carried accumulator
    (sweep.build_forward_accumulate) so a unit's contributions from EVERY
    earlier bucket survive even if its incoming edges are not all in bucket
    level-1 (a skip connection sources from an earlier level still). A
    unit is only finalized (sweep.build_forward_apply, write + accumulator
    reset) once every bucket that could feed it has been accumulated --
    which for a level-`level_idx+1` unit is exactly buckets `0..level_idx`,
    i.e. right after bucket `level_idx` is processed, since no edge sources
    from a level >= its own destination's level (the leveling invariant).
    """
    num_units = static.num_units
    num_levels = len(static.level_capacities)
    fp = net.forward_pass
    accumulate = build_forward_accumulate(
        fp, num_units=num_units, indices_are_sorted=True
    )
    apply = build_forward_apply(fp, num_units=num_units)
    not_input = ~unit_id_mask(static.input_ids, num_units)

    def forward_phase(
        state: NetworkState[GS], inputs: StepInputs
    ) -> tuple[NetworkState[GS], Float[Array, ""]]:  # noqa: F722
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
        # dispatch_cpu.hpp:390-411: no level structure, one flat bucket,
        # every unit Applied unconditionally (build_backward_sweep takes no
        # input_ids -- see its docstring). indices_are_sorted=False: backward
        # indexes segments by FROM_ID, but finalize sorts each bucket by
        # (dead, TO_ID), so those indices are not sorted -- correct on CPU
        # either way, honest for GPU/TPU, matching the topological backward.
        sweep = build_backward_sweep(
            bp, num_units=static.num_units, indices_are_sorted=False
        )

        def backward_phase(
            state: NetworkState[GS], inputs: StepInputs
        ) -> tuple[NetworkState[GS], Float[Array, ""]]:  # noqa: F722
            del inputs
            new_units = sweep(state.units, state.conns[0], state.globals_)
            return dataclasses.replace(state, units=new_units), jnp.float32(0.0)

        return backward_phase

    return _build_backward_topological_phase(net, static)


def _build_backward_topological_phase[GS](
    net: type[Network[GS]], static: NetworkStatic
) -> Phase[GS]:
    """Reverse level walk (dispatch_cpu.hpp:232-258, `L = NumLevels..1`).

    Bucket `level_idx` holds edges SOURCED at level_idx; backward
    accumulates into the source, so accumulating bucket `level_idx` is
    exactly what completes a level-`level_idx` unit's accumulator (every
    outgoing edge of a level-`level_idx` unit sources at level_idx, by
    definition of the bucketing -- unlike forward, there is no cross-bucket
    spread on the finalizing side). Walking buckets high-to-low is what
    guarantees a bucket's Map (which reads destination-side state, at a
    strictly higher level) only ever runs after that destination has
    already been finalized.

    The top level (== num_levels) has no source-level bucket of its own --
    no edge sources from the deepest level, since that would need a
    destination one level deeper still (dispatch_cpu.hpp:328-333 makes this
    explicit: "no edge has source level MaxLevels anywhere") -- so it is
    primed directly from the identity accumulator before the bucket loop,
    picking up only whatever an earlier phase (e.g. loss) wrote into unit
    columns Map/Apply itself read.

    The loop stops at bucket 1, never touching bucket 0 (input units' own
    outgoing edges): input units are excluded from `finalize` exactly like
    forward (dispatch_cpu.hpp:250 bounds Apply at NumInput same as :59), so
    accumulating bucket 0 would only feed an accumulator that is never
    read -- matching the oracle loop, which structurally never reaches
    L=0 either (`for (L = NumLevels; L >= 1; --L)`).
    """
    num_units = static.num_units
    num_levels = len(static.level_capacities)
    bp = net.backward_pass
    assert bp is not None  # build_phases only calls this when set
    accumulate = build_backward_accumulate(
        bp, num_units=num_units, indices_are_sorted=False
    )
    apply = build_backward_apply(bp, num_units=num_units)
    not_input = ~unit_id_mask(static.input_ids, num_units)

    def backward_phase(
        state: NetworkState[GS], inputs: StepInputs
    ) -> tuple[NetworkState[GS], Float[Array, ""]]:  # noqa: F722
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
    ) -> tuple[NetworkState[GS], Float[Array, ""]]:  # noqa: F722
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
    ) -> tuple[NetworkState[GS], Float[Array, ""]]:  # noqa: F722
        del inputs
        new_globals = reset_global.reset(state.globals_)
        return dataclasses.replace(state, globals_=new_globals), jnp.float32(0.0)

    return reset_global_phase


def build_update_conn_phase[GS](
    net: type[Network[GS]], static: NetworkStatic
) -> Phase[GS]:
    """Two full passes over every live conn in every bucket -- incoming
    then outgoing (dispatch_cpu.hpp:450-469). `state.conns` is swept
    unconditionally bucket-by-bucket for each pass (a 1-tuple in PIPELINE,
    one per source level in TOPOLOGICAL): unlike forward/backward,
    DoUpdateConn takes no Ranges/NumLevels -- it has no level structure of
    its own, it just walks every live conn twice.

    The two passes are sequenced across ALL buckets (every bucket's
    incoming write is merged before any bucket's outgoing pass reads conn
    state) so an edge's `outgoing` call observes that same edge's
    `incoming` write already landed, matching the oracle's single flat
    two-loop sweep over its one unbucketed conn arena. UpdateConn writes
    only ConnWrite (never a UnitWrite, traits.py's Protocol), so no edge's
    write is ever visible to a DIFFERENT edge's callback regardless of
    bucketing -- the cross-bucket sequencing only matters for an edge
    observing its own prior write, which per-bucket sequencing alone would
    already guarantee; kept global (all incoming buckets, then all
    outgoing buckets) for a literal 1:1 shape with the oracle's two loops.
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
    """Pure tombstone write (rung0 design section 5): vmap
    `prune_conn.predicate` over every conn row of every bucket, OR the
    result into that bucket's own `dead` column. Static shapes, no resort,
    no counter -- live counts stay derived (`state.live_conn_count`,
    `sum(~dead)`), matching dispatch_cpu.hpp:538-558's `DoPruneConnections`
    (the ConnPrune-only half; AddUnit/PruneUnit are out of scope, `plan.md`
    Scope contract, so the `HasUnitPrune` branch has no plastax analogue).

    dispatch_cpu.hpp:541-542 skips already-dead rows (`if (DeadTag) continue`)
    before calling `ShouldPrune`, both to save work and because a dead row's
    ids may be stale. vmap cannot skip (data-dependent control flow), so
    every row's predicate is evaluated unconditionally instead -- harmless:
    `dead[i] | predicate(...)` is `True` regardless of the (possibly
    meaningless) predicate result whenever `dead[i]` already is, so the OR
    merge is equivalent to the skip-then-OR the oracle performs.
    """
    pc = net.prune_conn
    assert pc is not None  # build_phases only calls this when set

    def prune_conn_phase(
        state: NetworkState[GS], inputs: StepInputs
    ) -> tuple[NetworkState[GS], Float[Array, ""]]:  # noqa: F722
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
    overflow_sink: list[Bool[Array, ""]] | None = None,  # noqa: F722
) -> Phase[GS]:
    """K-bounded candidates from the neighbourhood window; lax.top_k
    (static k, stable, lax.py:3563); per-bucket prefix-sum slot claim;
    overflow -> dropped scatter + flag; level-preserving adds must NOT set
    needs_resort (rung0 design section 5).

    Candidate window (M4b Deviation, IMPLEMENTATION_PLAN.md -- supersedes
    M4a's narrower one): matches dispatch_cpu.hpp:811-822's rolling
    level-PAIR window in full, `abs(level[dst] - level[src]) <=
    net.neighbourhood` with self-loops excluded (see the `window_ok`
    comment below for the unordered-level-pair derivation that makes this
    the oracle-faithful per-ordered-pair form -- dispatch_cpu.hpp's window
    visits every unordered pair {La, Lb} with `|Lb - La| <= Neighbourhood`,
    INCLUDING La == Lb, trying both edge directions between them). M4a
    restricted this to `0 < gap <= neighbourhood` (dst strictly ahead)
    because `topo.recompute_levels`/`topo.resort` were still
    `NotImplementedError` then, so there was no host mechanism to react to
    a relevel request; M4b implements both, so a same-level or
    behind-src candidate can now be genuinely accepted, letting
    `needs_resort` (below) actually fire -- the natural resort trigger the
    retrace-count tests exercise. `needs_resort` was already real, general
    code under M4a (not a hardcoded constant), so this widening is a pure
    enabling change to `window_ok`, not to the reassignment logic itself.

    One bucket is one top_k + prefix-sum claim, independent of every other
    bucket (no cross-bucket sequencing, unlike UpdateConn): in TOPOLOGICAL
    mode bucket `b` sources candidates only from units at level `b`
    (matching NetworkBuilder.finalize's `bucket_of_conn = levels[src_arr]`
    convention); PIPELINE's single bucket (level_capacities is a 1-tuple,
    rung0 design section 3) accepts a source at ANY level, since every
    live conn lives in that one flat arena regardless of source level --
    the level WINDOW itself is still consulted in both modes (native's
    AddConnections has no Pipeline/Topological distinction at all,
    dispatch_cpu.hpp:699-762 is driven purely by Kahn levels; only the
    destination BUCKET differs here). Design section 5's "in pipeline mode
    adds never resort at all" described a CONSEQUENCE of M4a's
    strictly-ahead window (every accepted candidate was level-preserving
    by construction, in either mode), not a mode-specific carve-out of
    `reassigning`'s formula below -- under this wider window a
    pipeline-mode commit can set `needs_resort` exactly like a
    topological-mode one; `topo.resort` keeps PIPELINE at a single bucket
    (its own docstring), so this stays correct, just no longer vacuous.
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
    # dynamically-sized candidate list [D:2 -- static shapes throughout].
    unit_ids = jnp.arange(num_units, dtype=jnp.int32)
    flat_src = jnp.broadcast_to(unit_ids[:, None], (num_units, num_units)).reshape(-1)
    flat_dst = jnp.broadcast_to(unit_ids[None, :], (num_units, num_units)).reshape(-1)
    not_self = flat_src != flat_dst

    def add_conn_phase(
        state: NetworkState[GS], inputs: StepInputs
    ) -> tuple[NetworkState[GS], Float[Array, ""]]:  # noqa: F722
        del inputs
        units = state.units
        g = state.globals_
        u_view = UnitView(units)
        unit_level = units[LEVEL.name]
        src_level = unit_level[flat_src]
        dst_level = unit_level[flat_dst]
        level_gap = dst_level - src_level
        # Oracle-faithful window (module docstring above): dispatch_cpu.hpp
        # :811-822's rolling window visits every unordered level pair {La,
        # Lb} with |Lb - La| <= Neighbourhood -- INCLUDING La == Lb -- and
        # tries both edge directions between them (TryPair's own `if (La !=
        # Lb)` branch adds the reverse-direction perspective; when La == Lb
        # the (Ai, Bi) double loop over that one level already visits both
        # orderings on its own). Per ordered (src, dst) pair that collapses
        # to exactly this: `abs(gap) <= neighbourhood`, self-loops excluded
        # (TryPair's own `if (U == V) return` -- gap == 0 no longer implies
        # src == dst now that same-level pairs are admitted, so `not_self`
        # must be checked explicitly here).
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
            valid = window_ok & src_ok

            flat_scores = jax.vmap(scored)(flat_src, flat_dst, valid)
            _, top_idx = jax.lax.top_k(flat_scores, k)
            top_src = flat_src[top_idx]
            top_dst = flat_dst[top_idx]
            top_valid = valid[top_idx]

            # Prefix-sum slot claim over this bucket's OWN dead mask
            # (rung0 design section 5): rank[i] is dead-position i's 0-based
            # rank among this bucket's dead slots; scattering `positions` by
            # `rank` (dropping live positions via the always-OOB `sink_len`
            # index) inverts that into slot_for_rank[j] = the position of
            # the j-th free slot, `capacity_b` (never a real position) where
            # fewer than j+1 free slots exist. `sink_len = max(capacity_b,
            # k)` keeps both the scatter (indices < capacity_b <= sink_len,
            # in bounds) and the later static `[:k]` slice in bounds without
            # a runtime clamp, even if k > capacity_b.
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
            # Overflow (module docstring, rung0 design section 5): a REAL
            # (valid, top-k-selected) candidate for which this bucket ran
            # out of dead slots. Un-committed candidates (invalid OR no
            # room) scatter to `capacity_b`, one past this bucket's own
            # valid range -- FILL_OR_DROP (scatter.py:187, the implicit
            # default mode) drops them rather than mis-writing a live slot.
            overflow = overflow | jnp.any(top_valid & ~has_room)
            target_slot = jnp.where(committed, free_slot, jnp.int32(capacity_b))

            # needs_resort (module docstring above): under M4a's
            # strictly-ahead window this was always False by construction;
            # the wider M4b window can now genuinely commit a same-level or
            # behind-src candidate, so this fires for real -- unchanged
            # code, since it was already computed rather than hardcoded.
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
                    # Not touched by ac.init: reset to the FieldSpec default
                    # rather than inheriting whatever a PREVIOUS tenant (a
                    # conn tombstoned by this same step's prune_conn pass,
                    # or the builder's initial padding) left behind --
                    # mirrors native's SOAAllocator.Allocate() placement-new
                    # default-constructing every field on claim
                    # (alloc.hpp:171-188), applied to plastax's dead-slot
                    # recycling in place of native's bump allocation.
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
