"""User-facing traits surface: policy Protocols + Network base class.

Python analogue of the C++ policy concepts; static checking via ty / mypy
--strict; runtime concept check in __init_subclass__.
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
    ShardSpec,
    UnitIdx,
)
from plastax.monoid import Monoid, MonoidTree
from plastax.views import ConnView, ConnWrite, UnitView, UnitWrite


@runtime_checkable
class ForwardPass[Acc, GS](Protocol):
    """Forward propagation policy: map-reduce over incoming edges per unit.

    Type Args:
        Acc: the per-edge accumulator type combined by `combine`.
        GS: the global state type threaded through the network.

    Attributes:
        combine: the monoid tree used to reduce per-edge Acc values.
    """

    combine: MonoidTree

    def map(
        self,
        u: UnitView,
        dst: UnitIdx,
        src: UnitIdx,
        c: ConnView,
        cid: ConnIdx,
        g: GS,
    ) -> Acc:
        """Compute the accumulator contribution of one incoming edge.

        Args:
            u: the unit view.
            dst: index of the destination unit.
            src: index of the source unit.
            c: the connection view.
            cid: index of the connection.
            g: the global state.

        Returns:
            The per-edge accumulator contribution.
        """
        ...

    def apply(self, u: UnitView, i: UnitIdx, g: GS, acc: Acc) -> UnitWrite:
        """Combine the reduced accumulator into a write for one unit.

        Args:
            u: the unit view.
            i: index of the unit.
            g: the global state.
            acc: the reduced accumulator for this unit.

        Returns:
            The UnitWrite for that unit.
        """
        ...


@runtime_checkable
class BackwardPass[Acc, GS](Protocol):
    """Backward propagation policy.

    Same shape as ForwardPass but accumulates into the SOURCE unit.

    Type Args:
        Acc: the per-edge accumulator type combined by `combine`.
        GS: the global state type threaded through the network.

    Attributes:
        combine: the monoid tree used to reduce per-edge Acc values.
    """

    combine: MonoidTree

    def map(
        self,
        u: UnitView,
        src: UnitIdx,
        dst: UnitIdx,
        c: ConnView,
        cid: ConnIdx,
        g: GS,
    ) -> Acc:
        """Compute the accumulator contribution of one outgoing edge.

        The first unit-id argument is the ACCUMULATOR TARGET, as in
        `ForwardPass.map` -- but backward accumulates into the edge's SOURCE, so
        it is `src` here where forward has `dst`. The argument ORDER is the same
        in both directions (target first); only which endpoint that is differs.
        `dst` is the edge's destination, whose value the reverse level walk has
        already finalized, and is therefore the one a backward map reads.

        Args:
            u: the unit view.
            src: index of the source unit -- this pass's accumulator target.
            dst: index of the destination unit, already finalized.
            c: the connection view.
            cid: index of the connection.
            g: the global state.

        Returns:
            The per-edge accumulator contribution.
        """
        ...

    def apply(self, u: UnitView, i: UnitIdx, g: GS, acc: Acc) -> UnitWrite:
        """Combine the reduced accumulator into a write for one unit.

        Args:
            u: the unit view.
            i: index of the unit.
            g: the global state.
            acc: the reduced accumulator for this unit.

        Returns:
            The UnitWrite for that unit.
        """
        ...


@runtime_checkable
class Loss[GS](Protocol):
    """Per-output loss policy producing a loss contribution and a unit write.

    Type Args:
        GS: the global state type threaded through the network.
    """

    def per_output(
        self,
        u: UnitView,
        i: UnitIdx,
        target: Float[Array, ""],
        g: GS,
    ) -> tuple[Float[Array, ""], UnitWrite]:
        """Compute the loss contribution and gradient write for one output.

        Args:
            u: the unit view.
            i: index of the output unit.
            target: the target value for this output.
            g: the global state.

        Returns:
            A (loss-contribution, UnitWrite) pair.
        """
        ...


@runtime_checkable
class UpdateConn[GS](Protocol):
    """Connection update policy: two full passes, incoming then outgoing.

    Type Args:
        GS: the global state type threaded through the network.
    """

    def incoming(
        self,
        u: UnitView,
        dst: UnitIdx,
        src: UnitIdx,
        c: ConnView,
        cid: ConnIdx,
        g: GS,
    ) -> ConnWrite:
        """Update one connection from the destination unit's perspective.

        Args:
            u: the unit view.
            dst: index of the destination unit.
            src: index of the source unit.
            c: the connection view.
            cid: index of the connection.
            g: the global state.

        Returns:
            The ConnWrite for this connection.
        """
        ...

    def outgoing(
        self,
        u: UnitView,
        src: UnitIdx,
        dst: UnitIdx,
        c: ConnView,
        cid: ConnIdx,
        g: GS,
    ) -> ConnWrite:
        """Update one connection from the source unit's perspective.

        Args:
            u: the unit view.
            src: index of the source unit.
            dst: index of the destination unit.
            c: the connection view.
            cid: index of the connection.
            g: the global state.

        Returns:
            The ConnWrite for this connection.
        """
        ...


@runtime_checkable
class PruneConn[GS](Protocol):
    """Connection pruning policy: tombstone connections by predicate.

    Type Args:
        GS: the global state type threaded through the network.
    """

    def predicate(
        self, u: UnitView, c: ConnView, cid: ConnIdx, g: GS
    ) -> Bool[Array, ""]:
        """Decide whether to tombstone one connection.

        Args:
            u: the unit view.
            c: the connection view.
            cid: index of the connection.
            g: the global state.

        Returns:
            A scalar bool; True to tombstone the connection.
        """
        ...


@runtime_checkable
class AddConn[GS](Protocol):
    """Connection growth policy: K-bounded growth by scored candidates.

    An implementation may optionally declare two extra members to shortlist
    growth candidates instead of scoring the full num_units^2 grid: an integer
    attribute `max_candidate_units` (M) and a method
    `importance(u, i, g) -> Float[Array, ""]`. When both are present (and
    0 < M < num_units), the add-conn phase draws candidates only from the M x M
    grid of that step's top-M most important units -- O(num_units + M^2) instead
    of O(num_units^2). They are read structurally (getattr), so omitting them
    keeps the exhaustive grid; they are not part of the required protocol.

    Type Args:
        GS: the global state type threaded through the network.

    Attributes:
        max_candidates: the maximum number of candidate connections
            considered per growth step.
    """

    max_candidates: int

    def score(self, u: UnitView, src: UnitIdx, dst: UnitIdx, g: GS) -> Float[Array, ""]:
        """Score a candidate connection for growth.

        Args:
            u: the unit view.
            src: index of the candidate source unit.
            dst: index of the candidate destination unit.
            g: the global state.

        Returns:
            The candidate score.
        """
        ...

    def init(self, u: UnitView, src: UnitIdx, dst: UnitIdx, g: GS) -> ConnWrite:
        """Initialize a new connection selected for growth.

        Args:
            u: the unit view.
            src: index of the source unit.
            dst: index of the destination unit.
            g: the global state.

        Returns:
            The ConnWrite for the new edge.
        """
        ...


@runtime_checkable
class ResetGlobal[GS](Protocol):
    """Global-state reset policy invoked between episodes or runs.

    Type Args:
        GS: the global state type threaded through the network.
    """

    def reset(self, g: GS) -> GS:
        """Reset the global state.

        Args:
            g: the current global state.

        Returns:
            The reset global state.
        """
        ...


class Network[GS]:
    """Base configuration surface for a network's traits.

    Subclass and set class attributes; absent phases are elided at trace
    time.

    Type Args:
        GS: the global state type threaded through the network.

    Attributes:
        forward_pass: the forward propagation policy.
        backward_pass: the backward propagation policy, or None to elide it.
        loss: the loss policy, or None to elide it.
        update_conn: the connection update policy, or None to elide it.
        prune_conn: the connection pruning policy, or None to elide it.
        add_conn: the connection growth policy, or None to elide it.
        reset_global: the global-state reset policy, or None to elide it.
        extra_unit_fields: extra per-unit fields beyond the builtin ones.
        extra_conn_fields: extra per-connection fields beyond the builtin ones.
        propagation: the propagation strategy used to schedule updates.
        kahn_max_depth: max depth for Kahn-order propagation, or None if unbounded.
        neighbourhood: the neighbourhood radius used by the propagation strategy.
        sharding: Scheme-A sharding config, or None for a single device.
    """

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
    sharding: ShardSpec | None = None

    def __init_subclass__(cls) -> None:
        """Validate the trait slots when a Network subclass is defined."""
        _validate_traits(cls)


_RESERVED_FIELD_NAMES = frozenset(
    {FROM_ID.name, TO_ID.name, DEAD.name, WEIGHT.name, ACTIVATION.name, LEVEL.name}
)


def _validate_monoid_tree(
    tree: object, cls: type[Network[Any]], attr_name: str
) -> None:
    """Recursively check that `tree` is a well-formed MonoidTree.

    A well-formed MonoidTree is a Monoid leaf, or a non-empty dict/tuple of
    well-formed MonoidTrees -- a product of monoids is itself a monoid.

    Args:
        tree: the candidate MonoidTree to validate.
        cls: the Network subclass being validated, used for error messages.
        attr_name: the name of the trait attribute `tree` belongs to.

    Raises:
        ValueError: if a dict or tuple node in `tree` is empty.
        TypeError: if `tree` is not a Monoid, dict, or tuple.
    """
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
    """Check field-name uniqueness and reject reserved builtin names.

    Verifies uniqueness across builtin and extra unit/conn fields, and that
    user extra fields do not collide with reserved builtin names.

    Args:
        cls: the Network subclass whose extra fields are validated.

    Raises:
        ValueError: if an extra field name collides with a reserved builtin
            name, or if two extra field names collide with each other.
    """
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
    """Run the runtime concept check for a Network subclass.

    Checks protocol conformance of each configured trait, delegating monoid
    tree structure and field-name checks to `_validate_monoid_tree` and
    `_validate_field_names` respectively.

    Args:
        cls: the Network subclass to validate.

    Raises:
        TypeError: if forward_pass is missing, or a configured trait does
            not satisfy its protocol.
        ValueError: if a delegated check fails -- a malformed combine
            MonoidTree, or a field-name collision or duplicate.
    """
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
