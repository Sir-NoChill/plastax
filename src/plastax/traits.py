"""User-facing traits surface: policy Protocols + Network base class.

Python analogue of the C++ policy concepts (traits.hpp:357-395). Static
checking via ty / mypy --strict (TOOLING.md); runtime concept check in
__init_subclass__.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
from jaxtyping import Array, Bool, Float

from plastax._types import ConnIdx, FieldSpec, Propagation, UnitIdx
from plastax.monoid import MonoidTree
from plastax.views import ConnView, ConnWrite, UnitView, UnitWrite


@runtime_checkable
class ForwardPass[Acc, GS](Protocol):
    combine: MonoidTree

    def map(
        self,
        u: UnitView,
        dst: UnitIdx,
        src: UnitIdx,
        c: ConnView,
        cid: ConnIdx,
        g: GS,
    ) -> Acc: ...

    def apply(self, u: UnitView, i: UnitIdx, g: GS, acc: Acc) -> UnitWrite: ...


@runtime_checkable
class BackwardPass[Acc, GS](Protocol):
    """Same shape as ForwardPass; accumulates into the SOURCE unit
    (dispatch_cpu.hpp:232-258 reverses direction)."""

    combine: MonoidTree

    def map(
        self,
        u: UnitView,
        dst: UnitIdx,
        src: UnitIdx,
        c: ConnView,
        cid: ConnIdx,
        g: GS,
    ) -> Acc: ...

    def apply(self, u: UnitView, i: UnitIdx, g: GS, acc: Acc) -> UnitWrite: ...


@runtime_checkable
class Loss[GS](Protocol):
    # jaxtyping scalar-shape strings below read as broken forward refs to
    # ruff's F722 (TOOLING.md: jaxtyping friction).
    def per_output(
        self,
        u: UnitView,
        i: UnitIdx,
        target: Float[Array, ""],  # noqa: F722
        g: GS,
    ) -> tuple[Float[Array, ""], UnitWrite]: ...  # noqa: F722


@runtime_checkable
class UpdateConn[GS](Protocol):
    """Two full passes, incoming then outgoing (dispatch_cpu.hpp:450-469)."""

    def incoming(
        self,
        u: UnitView,
        dst: UnitIdx,
        src: UnitIdx,
        c: ConnView,
        cid: ConnIdx,
        g: GS,
    ) -> ConnWrite: ...

    def outgoing(
        self,
        u: UnitView,
        src: UnitIdx,
        dst: UnitIdx,
        c: ConnView,
        cid: ConnIdx,
        g: GS,
    ) -> ConnWrite: ...


@runtime_checkable
class PruneConn[GS](Protocol):
    def predicate(
        self, u: UnitView, c: ConnView, cid: ConnIdx, g: GS
    ) -> Bool[Array, ""]: ...  # noqa: F722  jaxtyping scalar shape


@runtime_checkable
class AddConn[GS](Protocol):
    """K-bounded growth (rung0 design section 5; GrowFanout analogue)."""

    max_candidates: int

    def score(
        self, u: UnitView, src: UnitIdx, dst: UnitIdx, g: GS
    ) -> Float[Array, ""]: ...  # noqa: F722  jaxtyping scalar shape

    def init(self, u: UnitView, src: UnitIdx, dst: UnitIdx, g: GS) -> ConnWrite: ...


@runtime_checkable
class ResetGlobal[GS](Protocol):
    def reset(self, g: GS) -> GS: ...


class Network[GS]:
    """Subclass and set class attributes; absent phases are elided at trace
    time. Mirrors DefaultNetworkTraits<> slot-for-slot for the v1 scope."""

    forward_pass: ForwardPass[object, GS]
    backward_pass: BackwardPass[object, GS] | None = None
    loss: Loss[GS] | None = None
    update_conn: UpdateConn[GS] | None = None
    prune_conn: PruneConn[GS] | None = None
    add_conn: AddConn[GS] | None = None
    reset_global: ResetGlobal[GS] | None = None

    extra_unit_fields: tuple[FieldSpec[np.generic], ...] = ()
    extra_conn_fields: tuple[FieldSpec[np.generic], ...] = ()
    propagation: Propagation = Propagation.TOPOLOGICAL
    kahn_max_depth: int | None = None
    neighbourhood: int = 1

    def __init_subclass__(cls) -> None:
        _validate_traits(cls)


def _validate_traits(cls: type[Network[Any]]) -> None:
    """Runtime concept check: protocol conformance, monoid tree structure
    matches a probe Acc, field name uniqueness, no reserved names."""
    raise NotImplementedError
