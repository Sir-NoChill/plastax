"""User-facing traits surface: policy Protocols + Network base class.

Python analogue of the C++ policy concepts (traits.hpp:357-395). Static
checking via ty / mypy --strict (TOOLING.md); runtime concept check in
__init_subclass__.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
from jaxtyping import Array, Bool, Float

from plastax._types import (
    ACTIVATION,
    DEAD,
    FROM_ID,
    LEVEL,
    TO_ID,
    WEIGHT,
    ConnIdx,
    FieldSpec,
    Propagation,
    UnitIdx,
)
from plastax.monoid import Monoid, MonoidTree
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


_RESERVED_FIELD_NAMES = frozenset(
    {FROM_ID.name, TO_ID.name, DEAD.name, WEIGHT.name, ACTIVATION.name, LEVEL.name}
)


def _validate_monoid_tree(
    tree: object, cls: type[Network[Any]], attr_name: str
) -> None:
    """Recursively check `tree` is a well-formed MonoidTree: a Monoid leaf,
    or a non-empty dict/tuple of well-formed MonoidTrees (rung0 design
    section 3: a product of monoids is a monoid)."""
    if isinstance(tree, Monoid):
        return
    if isinstance(tree, dict):
        if not tree:
            raise ValueError(
                f"{cls.__name__}.{attr_name}.combine: dict MonoidTree is empty"
            )
        for leaf in tree.values():
            _validate_monoid_tree(leaf, cls, attr_name)
        return
    if isinstance(tree, tuple):
        if not tree:
            raise ValueError(
                f"{cls.__name__}.{attr_name}.combine: tuple MonoidTree is empty"
            )
        for leaf in tree:
            _validate_monoid_tree(leaf, cls, attr_name)
        return
    raise TypeError(
        f"{cls.__name__}.{attr_name}.combine is not a well-formed MonoidTree "
        f"(Monoid | dict[str, MonoidTree] | tuple[MonoidTree, ...]); got {tree!r}"
    )


def _validate_field_names(cls: type[Network[Any]]) -> None:
    """Field-name uniqueness across builtin+extra unit and conn fields; user
    extra fields must not collide with reserved builtin names."""
    all_extra = (*cls.extra_unit_fields, *cls.extra_conn_fields)
    seen: set[str] = set()
    for spec in all_extra:
        if spec.name in _RESERVED_FIELD_NAMES:
            raise ValueError(
                f"{cls.__name__}: extra field {spec.name!r} collides with a reserved "
                f"builtin name ({sorted(_RESERVED_FIELD_NAMES)})"
            )
        if spec.name in seen:
            raise ValueError(
                f"{cls.__name__}: duplicate extra field name {spec.name!r}"
            )
        seen.add(spec.name)


def _validate_traits(cls: type[Network[Any]]) -> None:
    """Runtime concept check: protocol conformance, monoid tree structure
    matches a probe Acc, field name uniqueness, no reserved names."""
    forward_pass = getattr(cls, "forward_pass", None)
    if forward_pass is None:
        raise TypeError(f"{cls.__name__}.forward_pass is required and was not set")
    if not isinstance(forward_pass, ForwardPass):
        raise TypeError(
            f"{cls.__name__}.forward_pass must satisfy ForwardPass "
            f"(combine, map, apply); got {forward_pass!r}"
        )
    _validate_monoid_tree(forward_pass.combine, cls, "forward_pass")

    if cls.backward_pass is not None:
        if not isinstance(cls.backward_pass, BackwardPass):
            raise TypeError(
                f"{cls.__name__}.backward_pass must satisfy BackwardPass "
                f"(combine, map, apply); got {cls.backward_pass!r}"
            )
        _validate_monoid_tree(cls.backward_pass.combine, cls, "backward_pass")

    if cls.loss is not None and not isinstance(cls.loss, Loss):
        raise TypeError(f"{cls.__name__}.loss must satisfy Loss; got {cls.loss!r}")

    if cls.update_conn is not None and not isinstance(cls.update_conn, UpdateConn):
        raise TypeError(
            f"{cls.__name__}.update_conn must satisfy UpdateConn; "
            f"got {cls.update_conn!r}"
        )

    if cls.prune_conn is not None and not isinstance(cls.prune_conn, PruneConn):
        raise TypeError(
            f"{cls.__name__}.prune_conn must satisfy PruneConn; got {cls.prune_conn!r}"
        )

    if cls.add_conn is not None and not isinstance(cls.add_conn, AddConn):
        raise TypeError(
            f"{cls.__name__}.add_conn must satisfy AddConn; got {cls.add_conn!r}"
        )

    if cls.reset_global is not None and not isinstance(cls.reset_global, ResetGlobal):
        raise TypeError(
            f"{cls.__name__}.reset_global must satisfy ResetGlobal; "
            f"got {cls.reset_global!r}"
        )

    _validate_field_names(cls)
