"""Edge-sweep core: gather -> vmapped map -> segment reduce -> apply.

One bucket at a time; the topological level loop and the pipeline flat
sweep are both compositions of this (rung0 design sections 3-4). Dead
slots use the null-slot trick: destination index replaced by num_units,
dropped by scatter mode FILL_OR_DROP (jax/_src/ops/scatter.py:187).
"""
from __future__ import annotations

from typing import Callable, TypeVar

from plastax.monoid import MonoidTree
from plastax.state import Columns, NetworkState
from plastax.traits import BackwardPass, ForwardPass

GS = TypeVar("GS")

# (units, bucket_conns, globals) -> updated units
BucketSweep = Callable[[Columns, Columns, GS], Columns]


def build_forward_sweep(
    fp: ForwardPass[object, GS], *, num_units: int, indices_are_sorted: bool
) -> BucketSweep[GS]:
    """Accumulates into TO_ID targets, then applies over destination units
    (dispatch_cpu.hpp:41-67 semantics)."""
    raise NotImplementedError


def build_backward_sweep(
    bp: BackwardPass[object, GS], *, num_units: int, indices_are_sorted: bool
) -> BucketSweep[GS]:
    """Accumulates into FROM_ID sources; apply runs on sources
    (dispatch_cpu.hpp:232-258)."""
    raise NotImplementedError


def materialize_acc_columns(combine: MonoidTree, num_units: int) -> Columns:
    """One accumulator column per monoid leaf, initialized to identity;
    reset in the sweep epilogue (UAcc = Acc{} analogue)."""
    raise NotImplementedError
