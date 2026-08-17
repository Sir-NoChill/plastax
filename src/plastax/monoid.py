"""Declared combine monoids (rung0 design, section 3).

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


# jax.ops.segment_* dispatch table and each op's pre-fill identity (mirrors
# jax's own table, jax/_src/ops/scatter.py:153-172 _get_identity).
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

# Elementwise binary op underlying each named monoid's segment_reduce (M3:
# folds a persisted accumulator across topological-mode buckets, rung0
# design section 4 level walk -- segment_sum's pairwise op is `+`, etc.).
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
    ) -> Shaped[Array, " num_segments"]:  # type: ignore[name-defined]  # noqa: F722
        """Named-monoid lowering via jax.ops.segment_* (scatter.py:221).

        `num_segments` static and `mode` left at its default (FILL_OR_DROP,
        scatter.py:187) is exactly the null-slot trick: a dead conn's
        segment_id set to num_units (one past the end, by the caller) falls
        outside [0, num_segments) and is dropped rather than accumulated.
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
        a: Shaped[Array, " n"],  # type: ignore[name-defined]  # noqa: F722
        b: Shaped[Array, " n"],  # type: ignore[name-defined]  # noqa: F722
    ) -> Shaped[Array, " n"]:  # type: ignore[name-defined]  # noqa: F722
        """Elementwise associative combine, the binary op segment_reduce
        folds over a whole segment (scatter.py:221) applied to just two
        values. M3 uses this to fold a bucket's segment_reduce result into
        an accumulator persisted across topological-mode buckets, the JAX
        analogue of `UAcc = Combine(UAcc, Map(...))`
        (dispatch_cpu.hpp:55-56, :246-247) run once per bucket rather than
        once per edge.
        """
        if self.named is None:
            raise UnsupportedMonoidError(
                "generic Monoid(op, identity) has no v1 lowering (rung0 design "
                "section 3); only named monoids (sum/prod/max/min) are supported"
            )
        return _NAMED_PAIRWISE[self.named](a, b)

    def identity_for(self, dtype: DTypeLike) -> Shaped[Array, ""]:  # noqa: F722
        """Concrete 0-d identity element at `dtype` (sweep.py's
        materialize_acc_columns: `UAcc = Acc{}` analogue,
        dispatch_cpu.hpp:41-67)."""
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
