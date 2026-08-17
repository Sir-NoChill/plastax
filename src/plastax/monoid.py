"""Declared combine monoids (rung0 design, section 3).

A product of monoids is a monoid: struct-valued accumulators are declared
as a pytree of Monoid leaves matching the Acc pytree structure.
"""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Callable

from jaxtyping import Array, Int32, Shaped


class _Named(enum.Enum):
    SUM = "sum"
    PROD = "prod"
    MAX = "max"
    MIN = "min"


@dataclasses.dataclass(frozen=True)
class Monoid[Acc]:
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
        # named-axis strings; mypy resolves bare " n" as a forward ref to an
        # undefined name "n" (jaxtyping friction -- TOOLING.md), ruff's F722
        # flags the same quoted shapes as broken forward-annotation syntax.
        data: Shaped[Array, " n"],  # type: ignore[name-defined]  # noqa: F722
        segment_ids: Int32[Array, " n"],  # type: ignore[name-defined]  # noqa: F722
        num_segments: int,
        *,
        indices_are_sorted: bool,
    ) -> Shaped[Array, " num_segments"]:  # type: ignore[name-defined]  # noqa: F722
        """Named-monoid lowering via jax.ops.segment_* (scatter.py:221)."""
        raise NotImplementedError


sum_: Monoid[Array] = Monoid(_Named.SUM)
prod: Monoid[Array] = Monoid(_Named.PROD)
max_: Monoid[Array] = Monoid(_Named.MAX)
min_: Monoid[Array] = Monoid(_Named.MIN)

# Acc pytrees pair with monoid pytrees of identical structure. PEP 695 `type`
# aliases are lazily evaluated, so the recursive reference needs no string.
type MonoidTree = Monoid[Array] | dict[str, MonoidTree] | tuple[MonoidTree, ...]


class UnsupportedMonoidError(NotImplementedError):
    """Raised when a generic Monoid reaches the v1 lowering."""
