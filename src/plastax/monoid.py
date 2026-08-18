"""Declared combine monoids.

A product of monoids is a monoid: struct-valued accumulators are declared
as a pytree of Monoid leaves matching the Acc pytree structure.
"""

from __future__ import annotations

import dataclasses
import enum
import math
from collections.abc import Callable

import jax.numpy as jnp
import jax.ops
from jaxtyping import Array, DTypeLike, Int32, Shaped


class _Named(enum.Enum):
    SUM = "sum"
    PROD = "prod"
    MAX = "max"
    MIN = "min"


# jax.ops.segment_* dispatch table and each op's pre-fill identity.
_SEGMENT_REDUCERS: dict[_Named, Callable[..., Array]] = {
    _Named.SUM: jax.ops.segment_sum,
    _Named.PROD: jax.ops.segment_prod,
    _Named.MAX: jax.ops.segment_max,
    _Named.MIN: jax.ops.segment_min,
}
_NAMED_IDENTITY: dict[_Named, float] = {
    _Named.SUM: 0.0,
    _Named.PROD: 1.0,
    _Named.MAX: -math.inf,
    _Named.MIN: math.inf,
}

# Elementwise binary op underlying each named monoid's segment_reduce --
# folds a persisted accumulator across topological-mode buckets
# (segment_sum's pairwise op is `+`, etc.).
_NAMED_PAIRWISE: dict[_Named, Callable[[Array, Array], Array]] = {
    _Named.SUM: jnp.add,
    _Named.PROD: jnp.multiply,
    _Named.MAX: jnp.maximum,
    _Named.MIN: jnp.minimum,
}


@dataclasses.dataclass(frozen=True)
class Monoid[Acc]:
    """Associative combine with identity.

    v1 lowers only named monoids (segment_sum/prod/max/min); the generic
    (op, identity) constructor is reserved and raises UnsupportedMonoidError
    at lowering time.

    Type Args:
        Acc: the accumulator type combined by this monoid (a pytree of
            leaves, or a single array leaf).

    Attributes:
        named: The built-in monoid to lower to, or None for a generic one.
        op: Associative binary op for a generic monoid (reserved for v2).
        identity: Identity element for a generic monoid (reserved for v2).
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
        """Reduce `data` per segment via the named monoid's jax.ops.segment_*.

        `num_segments` is static and `mode` is left at its default
        (FILL_OR_DROP), which gives the null-slot trick: a dead conn's
        segment_id set to num_units (one past the end, by the caller) falls
        outside [0, num_segments) and is dropped rather than accumulated.

        Args:
            data: Values to reduce, one per input index.
            segment_ids: Segment index for each entry in `data`.
            num_segments: Static number of output segments.
            indices_are_sorted: Whether `segment_ids` is already sorted.

        Returns:
            One reduced value per segment.

        Raises:
            UnsupportedMonoidError: If this is a generic (non-named) monoid.
        """
        if self.named is None:
            raise UnsupportedMonoidError(
                "generic Monoid(op, identity) has no v1 lowering (rung0 design "
                "section 3); only named monoids (sum/prod/max/min) are supported"
            )
        segment_fn = _SEGMENT_REDUCERS[self.named]
        return segment_fn(
            data,
            segment_ids,
            num_segments=num_segments,
            indices_are_sorted=indices_are_sorted,
        )

    def combine_pairwise(
        self,
        a: Shaped[Array, " n"],
        b: Shaped[Array, " n"],
    ) -> Shaped[Array, " n"]:
        """Elementwise associative combine of two values.

        This is the same binary op segment_reduce folds over a whole
        segment, applied to just two values -- used to fold a bucket's
        segment_reduce result into an accumulator persisted across
        topological-mode buckets, run once per bucket rather than once per
        edge.

        Args:
            a: Left operand.
            b: Right operand.

        Returns:
            The combined value, `a` op `b`.

        Raises:
            UnsupportedMonoidError: If this is a generic (non-named) monoid.
        """
        if self.named is None:
            raise UnsupportedMonoidError(
                "generic Monoid(op, identity) has no v1 lowering (rung0 design "
                "section 3); only named monoids (sum/prod/max/min) are supported"
            )
        return _NAMED_PAIRWISE[self.named](a, b)

    def identity_for(self, dtype: DTypeLike) -> Shaped[Array, ""]:
        """Build the concrete 0-d identity element at `dtype`.

        Args:
            dtype: Target dtype for the identity element.

        Returns:
            A 0-d array holding the monoid's identity value.

        Raises:
            UnsupportedMonoidError: If this is a generic (non-named) monoid
                with no declared identity.
        """
        if self.named is None:
            if self.identity is None:
                raise UnsupportedMonoidError(
                    "generic Monoid(op, identity) has no v1 lowering (rung0 "
                    "design section 3)"
                )
            return jnp.asarray(self.identity, dtype=dtype)
        return jnp.asarray(_NAMED_IDENTITY[self.named], dtype=dtype)


sum_: Monoid[Array] = Monoid(_Named.SUM)
prod: Monoid[Array] = Monoid(_Named.PROD)
max_: Monoid[Array] = Monoid(_Named.MAX)
min_: Monoid[Array] = Monoid(_Named.MIN)

# Acc pytrees pair with monoid pytrees of identical structure. PEP 695 `type`
# aliases are lazily evaluated, so the recursive reference needs no string.
type MonoidTree = Monoid[Array] | dict[str, MonoidTree] | tuple[MonoidTree, ...]


class UnsupportedMonoidError(NotImplementedError):
    """Raised when a generic Monoid reaches the v1 lowering."""
