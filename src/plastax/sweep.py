"""Edge-sweep core: gather -> vmapped map -> segment reduce -> apply.

One bucket at a time; the topological level loop and the pipeline flat
sweep are both compositions of this (rung0 design sections 3-4). Dead
slots use the null-slot trick: destination index replaced by num_units,
dropped by scatter mode FILL_OR_DROP (jax/_src/ops/scatter.py:187).

M3 (Deviation, IMPLEMENTATION_PLAN.md M3): `build_forward_sweep` was
originally a single monolithic accumulate-then-apply pass, correct only for
a one-bucket sweep (pipeline). The topological level walk needs the
accumulator to persist across several buckets before a unit's own level is
reached (a level-l unit's incoming edges may sit in any bucket < l, not
just l-1 -- e.g. a skip connection), so the sweep is split into two
composable primitives: `*_accumulate` (gather + map + segment_reduce,
combined into a carried-in accumulator via `Monoid.combine_pairwise`) and
`*_apply` (vmapped apply, masked merge + accumulator reset per finalized
unit). `build_forward_sweep`/`build_backward_sweep` keep their original
signatures and behaviour as one-shot compositions of these two primitives
(identity accumulator in, finalize every live unit); phases.py's
topological walk composes them directly across the bucket loop instead.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool

from plastax._types import DEAD, FROM_ID, TO_ID, ConnIdx, FieldSpec, UnitIdx
from plastax.monoid import Monoid, MonoidTree
from plastax.state import Columns
from plastax.traits import BackwardPass, ForwardPass
from plastax.views import ConnView, UnitView, UnitWrite

# Module-scoped (not PEP 695) so it stays free inside the type aliases below;
# the builders each shadow it with their own PEP 695 [GS] locally.
GS = TypeVar("GS")

# (units, bucket_conns, globals) -> updated units
BucketSweep = Callable[[Columns, Columns, GS], Columns]
# (units, bucket_conns, acc, globals) -> acc combined with this bucket's edges
AccumulateFn = Callable[[Columns, Columns, Any, GS], Any]
type FinalizeMask = (
    Bool[Array, " num_units"]  # type: ignore[name-defined]  # noqa: F722
)
# (units, acc, globals, finalize_mask) -> (updated units, acc with finalized
# units reset to identity)
ApplyFn = Callable[[Columns, Any, GS, FinalizeMask], tuple[Columns, Any]]


def unit_id_mask(ids: tuple[int, ...], num_units: int) -> Bool[Array, " num_units"]:  # type: ignore[name-defined]  # noqa: F722
    """Static unit-id tuple -> (num_units,) boolean mask via scatter, never
    a shape-changing gather. Shared by forward's input-skip and
    topological backward's input-skip (dispatch_cpu.hpp:59/250 both bound
    their Apply loop at `NumInput`; plastax generalizes the assumed
    `[0, NumInput)` prefix to an arbitrary id tuple, M2 Deviation)."""
    return (
        jnp.zeros((num_units,), dtype=jnp.bool_)
        .at[jnp.asarray(ids, dtype=jnp.int32)]
        .set(True)
    )


def identity_accumulator(combine: MonoidTree, num_units: int) -> Any:
    """(num_units,)-per-leaf accumulator tree matching `combine`'s
    structure, filled with each leaf's identity -- the JAX analogue of
    `UAcc = Acc{}` (dispatch_cpu.hpp:41-67) as a carried-in starting value
    rather than a struct field. Leaves are float32 (matches
    materialize_acc_columns; v1's Acc leaves are always float, per the
    ForwardPass/BackwardPass usages in tests and the design)."""
    return jax.tree_util.tree_map(
        lambda m: jnp.full((num_units,), m.identity_for(jnp.float32)), combine
    )


def _accumulate_into[GS](
    map_fn: Callable[[UnitView, UnitIdx, UnitIdx, ConnView, ConnIdx, GS], Any],
    combine: MonoidTree,
    *,
    num_units: int,
    indices_are_sorted: bool,
    target_col: FieldSpec[Any],
    other_col: FieldSpec[Any],
) -> AccumulateFn[GS]:
    """Shared by forward (target=TO_ID, other=FROM_ID) and backward
    (target=FROM_ID, other=TO_ID, dispatch_cpu.hpp:232-258 direction
    reversal): gather + vmapped map + segment_reduce this bucket's live
    edges into `target_col`, then fold the result into the carried-in `acc`
    via `Monoid.combine_pairwise` so contributions from earlier buckets
    survive (the level-walk correctness requirement: a unit's accumulator
    is not read by Apply until every bucket that can write to it has run).

    `map_fn`'s first unit-id argument is always the accumulator target,
    matching the oracle's own calling convention (`FP::Map(UnitAlloc,
    ToId...)` at :56 vs `BP::Map(UnitAlloc, FromId...)` at :247).
    """

    def accumulate(units: Columns, bucket_conns: Columns, acc: Any, g: GS) -> Any:
        u_view = UnitView(units)
        c_view = ConnView(bucket_conns)
        target_id = bucket_conns[target_col.name]
        other_id = bucket_conns[other_col.name]
        dead = bucket_conns[DEAD.name]
        conn_ids = jnp.arange(target_id.shape[0])

        def per_edge(target: jax.Array, other: jax.Array, cid: jax.Array) -> Any:
            return map_fn(
                u_view, UnitIdx(target), UnitIdx(other), c_view, ConnIdx(cid), g
            )

        per_edge_acc = jax.vmap(per_edge)(target_id, other_id, conn_ids)

        # Null-slot trick (rung0 design section 3): a dead conn's target is
        # pushed to num_units (one past the end), so segment_reduce's
        # FILL_OR_DROP mode (scatter.py:187) drops its contribution instead
        # of a shape-changing masked gather [D:6].
        null_target_id = jnp.where(dead, num_units, target_id)

        def combine_leaf(
            monoid: Monoid[Any], prev: jax.Array, data: jax.Array
        ) -> jax.Array:
            bucket_reduced = monoid.segment_reduce(
                data, null_target_id, num_units, indices_are_sorted=indices_are_sorted
            )
            return monoid.combine_pairwise(prev, bucket_reduced)

        # combine (Monoid leaves), acc, and per_edge_acc (Array leaves) share
        # tree structure by the ForwardPass/BackwardPass contract, so
        # tree_map zips them leaf-by-leaf; a bare (non-container) Monoid is
        # itself the sole leaf, so this also covers the common scalar-Acc
        # case.
        return jax.tree_util.tree_map(combine_leaf, combine, acc, per_edge_acc)

    return accumulate


def _apply_masked[GS](
    apply_fn: Callable[[UnitView, UnitIdx, GS, Any], UnitWrite],
    combine: MonoidTree,
    *,
    num_units: int,
) -> ApplyFn[GS]:
    """Shared by forward and backward: apply is computed uniformly for
    every unit under vmap (static shapes, no dynamic-size gather over a
    unit subset), `mask` selects which units are finalized -- written into
    `units` and reset to identity in `acc` -- this call. Units outside the
    mask keep their previous `units` value and carry their in-progress
    `acc` to a later call, which is exactly what lets the accumulator
    persist across topological buckets.
    """

    def apply(units: Columns, acc: Any, g: GS, mask: jax.Array) -> tuple[Columns, Any]:
        u_view = UnitView(units)
        unit_ids = jnp.arange(num_units)

        def per_unit_apply(i: jax.Array, acc_i: Any) -> dict[str, jax.Array]:
            # UnitWrite is not pytree-registered (views.py); unwrap .fields
            # to a plain dict before returning, so vmap never has to batch
            # the wrapper object itself, only its array-valued contents.
            write = apply_fn(u_view, UnitIdx(i), g, acc_i)
            return dict(write.fields)

        batched_writes = jax.vmap(per_unit_apply)(unit_ids, acc)

        new_units: Columns = dict(units)
        for name, written in batched_writes.items():
            new_units[name] = jnp.where(mask, written, units[name])

        def reset_leaf(monoid: Monoid[Any], leaf: jax.Array) -> jax.Array:
            return jnp.where(mask, monoid.identity_for(leaf.dtype), leaf)

        new_acc = jax.tree_util.tree_map(reset_leaf, combine, acc)
        return new_units, new_acc

    return apply


def build_forward_accumulate[GS](
    fp: ForwardPass[object, GS], *, num_units: int, indices_are_sorted: bool
) -> AccumulateFn[GS]:
    """One bucket's contribution folded into a persisted per-unit
    accumulator indexed by TO_ID (dispatch_cpu.hpp:41-67)."""
    return _accumulate_into(
        fp.map,
        fp.combine,
        num_units=num_units,
        indices_are_sorted=indices_are_sorted,
        target_col=TO_ID,
        other_col=FROM_ID,
    )


def build_forward_apply[GS](
    fp: ForwardPass[object, GS], *, num_units: int
) -> ApplyFn[GS]:
    """Vmapped apply over destination units (dispatch_cpu.hpp:41-67)."""
    return _apply_masked(fp.apply, fp.combine, num_units=num_units)


def build_forward_sweep[GS](
    fp: ForwardPass[object, GS],
    *,
    num_units: int,
    indices_are_sorted: bool,
    input_ids: tuple[int, ...],
) -> BucketSweep[GS]:
    """One-shot forward sweep over a single bucket: identity accumulator in,
    every non-input unit finalized (dispatch_cpu.hpp:202-223 pipeline
    semantics -- no level structure, one hop per call). The topological
    level walk (phases.py) instead composes `build_forward_accumulate` and
    `build_forward_apply` directly across the bucket loop so the
    accumulator can persist across buckets; this function is their
    one-bucket-and-done composition, unchanged from M2's behaviour.

    `input_ids` (Deviation, IMPLEMENTATION_PLAN.md M2 -- the stub signature
    had no such parameter): dispatch_cpu.hpp:217-222 only Applies over `I in
    [NumInput, NumUnits)`, leaving input units' activation exactly as the
    step's input-scatter wrote it. plastax input units are an arbitrary id
    tuple (NetworkBuilder.mark_input), not necessarily the prefix `[0,
    NumInput)` C++ assumes, so the skip is a static boolean mask rather than
    a slice offset.
    """
    accumulate = build_forward_accumulate(
        fp, num_units=num_units, indices_are_sorted=indices_are_sorted
    )
    apply = build_forward_apply(fp, num_units=num_units)
    is_input = unit_id_mask(input_ids, num_units)

    def sweep(units: Columns, bucket_conns: Columns, g: GS) -> Columns:
        acc = identity_accumulator(fp.combine, num_units)
        acc = accumulate(units, bucket_conns, acc, g)
        new_units, _ = apply(units, acc, g, ~is_input)
        return new_units

    return sweep


def build_backward_accumulate[GS](
    bp: BackwardPass[object, GS], *, num_units: int, indices_are_sorted: bool
) -> AccumulateFn[GS]:
    """One bucket's contribution folded into a persisted per-unit
    accumulator indexed by FROM_ID -- direction reversal
    (dispatch_cpu.hpp:232-258): the accumulator target is the edge's
    SOURCE, not its destination."""
    return _accumulate_into(
        bp.map,
        bp.combine,
        num_units=num_units,
        indices_are_sorted=indices_are_sorted,
        target_col=FROM_ID,
        other_col=TO_ID,
    )


def build_backward_apply[GS](
    bp: BackwardPass[object, GS], *, num_units: int
) -> ApplyFn[GS]:
    """Vmapped apply over source units (dispatch_cpu.hpp:232-258)."""
    return _apply_masked(bp.apply, bp.combine, num_units=num_units)


def build_backward_sweep[GS](
    bp: BackwardPass[object, GS], *, num_units: int, indices_are_sorted: bool
) -> BucketSweep[GS]:
    """Accumulates into FROM_ID sources; apply runs on sources
    (dispatch_cpu.hpp:232-258), one bucket and done -- the pipeline
    (dispatch_cpu.hpp:390-411) shape of the backward sweep, mirroring
    build_forward_sweep. No `input_ids` parameter (unlike
    build_forward_sweep): DoBackwardPipeline's NumInput parameter is
    declared but unused (dispatch_cpu.hpp:391-392, `size_t /*NumInput*/`)
    and its Apply loop runs unconditionally over `[0, NumUnits)`
    (dispatch_cpu.hpp:405-410) -- pipeline-mode backward really does apply
    to every unit, inputs included, so this sweep takes no skip mask. The
    topological level walk (phases.py) DOES skip input units on apply
    (dispatch_cpu.hpp:250, same bound as forward) by composing
    build_backward_accumulate/build_backward_apply directly with its own
    mask, same split rationale as the forward sweep above.
    """
    accumulate = build_backward_accumulate(
        bp, num_units=num_units, indices_are_sorted=indices_are_sorted
    )
    apply = build_backward_apply(bp, num_units=num_units)

    def sweep(units: Columns, bucket_conns: Columns, g: GS) -> Columns:
        acc = identity_accumulator(bp.combine, num_units)
        acc = accumulate(units, bucket_conns, acc, g)
        finalize_all = jnp.ones((num_units,), dtype=jnp.bool_)
        new_units, _ = apply(units, acc, g, finalize_all)
        return new_units

    return sweep


def materialize_acc_columns(combine: MonoidTree, num_units: int) -> Columns:
    """One accumulator column per monoid leaf, initialized to identity;
    reset in the sweep epilogue (UAcc = Acc{} analogue).

    `combine` is walked like any pytree (a bare Monoid is itself an atomic
    leaf since Monoid is an unregistered dataclass, matching the tree_map
    zip in build_forward_sweep above); leaves are named by their key path
    (jax.tree_util.keystr) so a struct accumulator (dict/tuple MonoidTree,
    rung0 design section 3) gets one distinctly-named column per field, and
    the common bare-Monoid case collapses to a single "acc" column.
    """
    leaves_with_paths, _ = jax.tree_util.tree_flatten_with_path(combine)
    columns: Columns = {}
    for path, monoid in leaves_with_paths:
        key = jax.tree_util.keystr(path) or "acc"
        if key in columns:
            raise ValueError(
                f"materialize_acc_columns: MonoidTree paths collide at {key!r}"
            )
        columns[key] = jnp.full(
            (num_units,), monoid.identity_for(jnp.float32), dtype=jnp.float32
        )
    return columns
