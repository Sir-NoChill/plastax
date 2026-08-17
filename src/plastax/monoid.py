"""Declared combine monoids (rung0 design, section 3).

A product of monoids is a monoid: struct-valued accumulators are declared
as a pytree of Monoid leaves matching the Acc pytree structure.
"""
from __future__ import annotations

import dataclasses
import enum
from typing import Callable, Generic, TypeAlias, TypeVar

from jaxtyping import Array, Int32, Shaped

Acc = TypeVar("Acc")


class _Named(enum.Enum):
    SUM = "sum"
    PROD = "prod"
    MAX = "max"
    MIN = "min"


@dataclasses.dataclass(frozen=True)
class Monoid(Generic[Acc]):
    """Associative combine with identity.

    v1 lowers only named monoids (segment_sum/prod/max/min); the generic
    (op, identity) constructor is reserved for v2's segmented-scan path and
    must raise UnsupportedMonoidError at lowering time in v1.
    """

    named: _Named | None
    op: Callable[[Acc, Acc], Acc] | None = None
    identity: Acc | None = None

    def segment_reduce(
        self,
        data: Shaped[Array, " n"],
        segment_ids: Int32[Array, " n"],
        num_segments: int,
        *,
        indices_are_sorted: bool,
    ) -> Shaped[Array, " num_segments"]:
        """Named-monoid lowering via jax.ops.segment_* (scatter.py:221)."""
        raise NotImplementedError


sum_: Monoid[Array] = Monoid(_Named.SUM)
prod: Monoid[Array] = Monoid(_Named.PROD)
max_: Monoid[Array] = Monoid(_Named.MAX)
min_: Monoid[Array] = Monoid(_Named.MIN)

# Acc pytrees pair with monoid pytrees of identical structure.
MonoidTree: TypeAlias = "Monoid[Array] | dict[str, MonoidTree] | tuple[MonoidTree, ...]"


class UnsupportedMonoidError(NotImplementedError):
    """Raised when a generic Monoid reaches the v1 lowering."""
