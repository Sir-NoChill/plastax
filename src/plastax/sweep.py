"""Edge-sweep core: gather -> vmapped map -> segment reduce -> apply.

One bucket at a time; the topological level loop and the pipeline flat
sweep are both compositions of this (rung0 design sections 3-4). Dead
slots use the null-slot trick: destination index replaced by num_units,
dropped by scatter mode FILL_OR_DROP (jax/_src/ops/scatter.py:187).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

import jax
import jax.numpy as jnp

from plastax._types import DEAD, FROM_ID, TO_ID, ConnIdx, UnitIdx
from plastax.monoid import Monoid, MonoidTree
from plastax.state import Columns
from plastax.traits import BackwardPass, ForwardPass
from plastax.views import ConnView, UnitView

# Module-scoped (not PEP 695) so it stays free inside the BucketSweep alias;
# the two builders below shadow it with their own PEP 695 [GS] locally.
GS = TypeVar("GS")

# (units, bucket_conns, globals) -> updated units
BucketSweep = Callable[[Columns, Columns, GS], Columns]


def build_forward_sweep[GS](
    fp: ForwardPass[object, GS],
    *,
    num_units: int,
    indices_are_sorted: bool,
    input_ids: tuple[int, ...],
) -> BucketSweep[GS]:
    """Accumulates into TO_ID targets, then applies over destination units
    (dispatch_cpu.hpp:41-67 semantics).

    One bucket == the pipeline flat sweep (design section 3) or one
    topological level's arena (M3). `map`/`apply` see UnitView/ConnView, not
    raw columns [D:2]: the "gather" is `u[SPEC, idx]`/`c[SPEC, idx]` inside
    the user's policy, executed under vmap so it lowers to a real gather.

    `input_ids` (Deviation, IMPLEMENTATION_PLAN.md M2 -- the stub signature
    had no such parameter): dispatch_cpu.hpp:217-222 only Applies over `I in
    [NumInput, NumUnits)`, leaving input units' activation exactly as the
    step's input-scatter wrote it. plastax input units are an arbitrary id
    tuple (NetworkBuilder.mark_input), not necessarily the prefix `[0,
    NumInput)` C++ assumes, so the skip is a static boolean mask rather than
    a slice offset.
    """

    def sweep(units: Columns, bucket_conns: Columns, g: GS) -> Columns:
        u_view = UnitView(units)
        c_view = ConnView(bucket_conns)
        to_id = bucket_conns[TO_ID.name]
        from_id = bucket_conns[FROM_ID.name]
        dead = bucket_conns[DEAD.name]
        conn_ids = jnp.arange(to_id.shape[0])

        def per_edge(dst: jax.Array, src: jax.Array, cid: jax.Array) -> Any:
            return fp.map(u_view, UnitIdx(dst), UnitIdx(src), c_view, ConnIdx(cid), g)

        per_edge_acc = jax.vmap(per_edge)(to_id, from_id, conn_ids)

        # Null-slot trick (rung0 design section 3): a dead conn's
        # destination is pushed to num_units (one past the end), so
        # segment_reduce's FILL_OR_DROP mode (scatter.py:187) drops its
        # contribution instead of a shape-changing masked gather [D:6].
        null_to_id = jnp.where(dead, num_units, to_id)

        def reduce_leaf(monoid: Monoid[Any], data: jax.Array) -> jax.Array:
            return monoid.segment_reduce(
                data, null_to_id, num_units, indices_are_sorted=indices_are_sorted
            )

        # fp.combine (Monoid leaves) and per_edge_acc (Array leaves) share
        # tree structure by the ForwardPass contract, so tree_map zips them
        # leaf-by-leaf; a bare (non-container) Monoid is itself the sole
        # leaf, so this also covers the common scalar-Acc case.
        acc_per_unit = jax.tree_util.tree_map(reduce_leaf, fp.combine, per_edge_acc)

        unit_ids = jnp.arange(num_units)

        def per_unit_apply(i: jax.Array, acc: Any) -> dict[str, jax.Array]:
            # UnitWrite is not pytree-registered (views.py); unwrap .fields
            # to a plain dict before returning, so vmap never has to batch
            # the wrapper object itself, only its array-valued contents.
            write = fp.apply(u_view, UnitIdx(i), g, acc)
            return dict(write.fields)

        batched_writes = jax.vmap(per_unit_apply)(unit_ids, acc_per_unit)

        # Input units are never Apply'd (dispatch_cpu.hpp:217-222): mask the
        # merge so their activation stays exactly what the step's input
        # scatter wrote, not the (typically-identity, since inputs have no
        # incoming edges) accumulator this vmap computed for them regardless
        # -- computing it uniformly for all units and masking on merge keeps
        # every shape static, no dynamic-size gather over a unit subset.
        is_input = (
            jnp.zeros((num_units,), dtype=jnp.bool_)
            .at[jnp.asarray(input_ids, dtype=jnp.int32)]
            .set(True)
        )

        # Epilogue: merge writes into unit columns (dispatch_cpu.hpp:41-67
        # `UAcc = Acc{}` has no JAX analogue here -- the accumulator above is
        # per-sweep scratch, never persisted in NetworkState). Fields the
        # policy didn't write (e.g. `level`) pass through unchanged.
        new_units: Columns = dict(units)
        for name, written in batched_writes.items():
            new_units[name] = jnp.where(is_input, units[name], written)
        return new_units

    return sweep


def build_backward_sweep[GS](
    bp: BackwardPass[object, GS], *, num_units: int, indices_are_sorted: bool
) -> BucketSweep[GS]:
    """Accumulates into FROM_ID sources; apply runs on sources
    (dispatch_cpu.hpp:232-258)."""
    raise NotImplementedError


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
