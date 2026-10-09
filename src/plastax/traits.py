"""User-facing traits surface: policy Protocols + Network base class.

Python analogue of the C++ policy concepts; static checking via ty / mypy
--strict; runtime concept check in __init_subclass__.
"""

from __future__ import annotations

import inspect
from typing import Any, NamedTuple, Protocol, runtime_checkable

import numpy as np
from jaxtyping import Array, Bool, Float, Int32

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
from plastax.rng import Rng
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

    Under a batched step (`make_step(..., batch_size=B)`) the update is reduced
    over the batch. By default each sample's change to every floating
    connection column is averaged (exact for rules linear in the per-sample
    term, e.g. SGD; unwritten columns stay untouched), and a non-floating
    column takes sample 0's write. A policy may instead
    declare, structurally (read with getattr, not part of this Protocol), the
    exact pair: `per_sample(u, dst, src, c, cid, g) -> pytree`, evaluated per
    sample and averaged over the batch, and `incoming_batched(u, dst, src, c,
    cid, g, stat) -> ConnWrite`, applied once with that average (and the
    batch-mean unit view) in place of `incoming`. Every `plastax.optim`
    bundle declares it, so a batched optimizer step is one step on the
    batch-mean gradient.

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

    A candidate that is already a live edge is excluded (`dedupe`, read
    structurally, defaults to True here), so this path never grows a parallel
    edge. For growth whose cost follows the churn rather than the arena, see
    `ProposeAddConn`.

    The growth window is the rule's own `max_level_gap` (int, read
    structurally, default 1): a candidate is in-window when
    `abs(level[dst] - level[src]) <= max_level_gap` and `src != dst`.

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


class Proposal(NamedTuple):
    """One growth proposal: a directed candidate edge and its priority.

    Attributes:
        src: proposed source unit id (int32 scalar).
        dst: proposed destination unit id (int32 scalar).
        score: the candidate's priority; ``-inf`` vetoes it.
    """

    src: Int32[Array, ""]
    dst: Int32[Array, ""]
    score: Float[Array, ""]


# The namedtuple field descriptors carry "Alias for field number N" docstrings
# that autodoc would document on top of the Attributes entries above
# (duplicate object descriptions under sphinx -W). The Attributes section is
# the documentation of record; silence the aliases.
for _field in Proposal._fields:
    getattr(Proposal, _field).__doc__ = None


@runtime_checkable
class ProposeAddConn[GS](Protocol):
    """Connection growth policy: bounded growth from sampled proposals.

    The counterpart of `AddConn` whose cost follows the churn, not the arena.
    Instead of scoring a candidate grid, the policy emits proposals through
    `propose`, called `proposals_per_proposer` times (index `j`) for each
    proposing site. Who proposes is the rule's `proposer` (read structurally,
    default ``"per_unit"``):

    - ``"per_unit"``: every unit proposes; `propose(u, i, j, g, rng)` with `i`
      the proposing unit. Candidate order index is ``i * P + j``.
    - ``"per_connection"``: every live connection proposes;
      `propose(u, c, cid, j, g, rng)` with `cid` the proposing connection's
      arena slot. Candidate order index is ``r * P + j`` with ``r`` the
      connection's rank in ascending ``(src, dst, occurrence)`` order over the
      live connections (occurrence counts parallel edges in ascending slot
      order). A unit with no live connections proposes nothing.
    - ``"global"``: one proposer; `propose(u, j, g, rng)`. Order index ``j``.

    Each proposal site receives its own counter-based `Rng`
    (see `plastax.rng`), keyed by the network seed, the step counter, the
    growth stream, the proposer (unit id, connection key, or 0) and `j` --
    so proposal streams vary per step and replay exactly. A rule is free to
    ignore `rng` and derive proposals from state instead.

    Downstream, every proposal passes the shared pipeline: the level-gap
    window (the rule's own `max_level_gap`, read structurally, default 1:
    `abs(level[dst] - level[src]) <= max_level_gap`, no self-loops), the
    opt-in dedupe stages, and each bucket's `max_candidates`-bounded
    selection in the deterministic total candidate order. A score of -inf
    vetoes a proposal, as do ids outside [0, num_units).

    **Duplicates.** By default nothing checks a proposal against the live
    edges or this step's other proposals: a proposal equal to a live pair
    grows a *parallel edge* (the network becomes a multigraph; parallel edges
    contribute independently, so a weighted-sum forward sees their weights
    add), and two equal proposals in one step grow two edges. A policy that
    must never grow a duplicate either proposes only absent pairs by
    construction (the fast route) or sets `dedupe_live = True` (veto
    proposals equal to a live edge; costs a sort of the live keys every
    growth step) and/or `dedupe_step = True` (keep only the first copy, in
    the total candidate order, of equal proposals within the step).

    Type Args:
        GS: the global state type threaded through the network.

    Attributes:
        max_candidates: the maximum number of connections grown per bucket
            per step.
        proposals_per_proposer: how many proposals each proposing site emits
            per step, a static int >= 1.
    """

    max_candidates: int
    proposals_per_proposer: int

    def propose(
        self,
        u: UnitView,
        i: UnitIdx,
        j: Int32[Array, ""],
        g: GS,
        rng: Rng,
    ) -> Proposal:
        """Emit proposal `j` of proposing unit `i` (the per-unit shape).

        The ``"per_connection"`` and ``"global"`` proposers use the
        signatures documented on the class; the declared `proposer` selects
        which shape is validated and called.

        Args:
            u: the unit view.
            i: the proposing unit (per-unit proposer).
            j: the proposal index, in [0, proposals_per_proposer).
            g: the global state.
            rng: this site's draw stream.

        Returns:
            The proposed edge and its priority. -inf vetoes it; ids outside
            [0, num_units) are vetoed too.
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
        sharding: Scheme-A sharding config, or None for a single device.
        seed: the network seed keying the framework's counter-based RNG
            (`plastax.rng`); identical seeds give identical draw streams.
    """

    forward_pass: ForwardPass[object, GS]
    backward_pass: BackwardPass[object, GS] | None = None
    loss: Loss[GS] | None = None
    update_conn: UpdateConn[GS] | None = None
    prune_conn: PruneConn[GS] | None = None
    add_conn: AddConn[GS] | ProposeAddConn[GS] | None = None
    reset_global: ResetGlobal[GS] | None = None

    extra_unit_fields: tuple[FieldSpec[np.generic], ...] = ()
    extra_conn_fields: tuple[FieldSpec[np.generic], ...] = ()
    propagation: Propagation = Propagation.TOPOLOGICAL
    kahn_max_depth: int | None = None
    sharding: ShardSpec | None = None
    seed: int = 0

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
    if "neighbourhood" in vars(cls):
        raise TypeError(
            f"{cls.__name__}.neighbourhood is no longer a Network attribute: "
            "the growth window moved onto the growth rule. Set "
            "`max_level_gap` (int, default 1) on the add_conn policy instead."
        )
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

    if cls.add_conn is not None:
        grid = isinstance(cls.add_conn, AddConn)
        proposed = isinstance(cls.add_conn, ProposeAddConn)
        if grid == proposed:
            raise TypeError(
                f"{cls.__name__}.add_conn must satisfy exactly one of AddConn "
                f"(score) or ProposeAddConn (propose); got {cls.add_conn!r}"
            )
        if isinstance(cls.add_conn, ProposeAddConn) and not grid:
            _validate_propose_rule(cls, cls.add_conn)

    if cls.reset_global is not None and not isinstance(cls.reset_global, ResetGlobal):
        raise TypeError(
            f"{cls.__name__}.reset_global must satisfy ResetGlobal; "
            f"got {cls.reset_global!r}"
        )

    _validate_field_names(cls)


_PROPOSE_ARITY = {
    # positional parameters of `propose` after self: (names, shape hint)
    "per_unit": (5, "propose(self, u, i, j, g, rng)"),
    "per_connection": (6, "propose(self, u, c, cid, j, g, rng)"),
    "global": (4, "propose(self, u, j, g, rng)"),
}


def _validate_propose_rule(cls: type[Network[Any]], ac: ProposeAddConn[Any]) -> None:
    """Check a ProposeAddConn's knobs and `propose` shape at class definition.

    Args:
        cls: the Network subclass being validated (for error messages).
        ac: the declared propose rule.

    Raises:
        TypeError: on a removed attribute (`num_proposals`, `dedupe`), a bad
            `proposer` / `proposals_per_proposer`, or a `propose` whose
            positional arity does not match the declared proposer.
    """
    if hasattr(ac, "num_proposals"):
        raise TypeError(
            f"{cls.__name__}.add_conn.num_proposals was renamed: declare "
            "`proposals_per_proposer` (int >= 1) on the propose rule instead."
        )
    if hasattr(ac, "dedupe"):
        raise TypeError(
            f"{cls.__name__}.add_conn.dedupe does not apply to propose rules: "
            "set `dedupe_live` and/or `dedupe_step` (bool, default False) "
            "on the rule instead."
        )
    n: object = getattr(ac, "proposals_per_proposer", None)
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise TypeError(
            f"{cls.__name__}.add_conn.proposals_per_proposer must be an "
            f"int >= 1; got {n!r}"
        )
    proposer = getattr(ac, "proposer", "per_unit")
    if not isinstance(proposer, str) or proposer not in _PROPOSE_ARITY:
        raise TypeError(
            f"{cls.__name__}.add_conn.proposer must be one of "
            f"{sorted(_PROPOSE_ARITY)}; got {proposer!r}"
        )
    want, shape = _PROPOSE_ARITY[proposer]
    sig = inspect.signature(ac.propose)
    if any(
        prm.kind is inspect.Parameter.VAR_POSITIONAL for prm in sig.parameters.values()
    ):
        return  # *args adapters are arity-unverifiable; the call site decides
    got = sum(
        1
        for prm in sig.parameters.values()
        if prm.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )
    if got != want:
        raise TypeError(
            f"{cls.__name__}.add_conn: a {proposer!r} proposer's propose "
            f"takes {want} arguments after self -- expected `{shape}`, got {got}."
        )
