"""User-facing traits surface: policy Protocols + Network base class.

Python analogue of the C++ policy concepts; static checking via ty / mypy
--strict; runtime concept check in __init_subclass__.
"""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Callable
from typing import Any, NamedTuple, Protocol, runtime_checkable

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int32

from plastax._types import (
    ACTIVATION,
    DEAD,
    FROM_ID,
    LEVEL,
    PRUNED,
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
    """Whole-output loss policy: one call sees every output unit.

    `calculate_loss` reads the output units (any field, through the unit view)
    and the targets, and returns the scalar loss together with the gradient
    seed dL/d(output) of every output. The framework writes the seed into the
    unit column the policy declares as `seed_field`, at the output ids only;
    the loss writes nothing else. A whole-output signature is what makes losses
    that couple the outputs expressible, e.g. softmax cross-entropy
    (`SoftmaxCrossEntropyLoss`).

    The backward accumulator is not writable by any policy: it reaches a rule
    only as `BackwardPass.apply`'s `acc` argument, read-only. The backward pass
    picks the seed up from `seed_field` (the output level's own `acc` is the
    identity, since no edge sources from the deepest level).

    Type Args:
        GS: the global state type threaded through the network.

    Attributes:
        seed_field: the float unit column the gradient seed is written to; one
            of the network's unit columns (normally an `extra_unit_fields`
            entry).
    """

    seed_field: FieldSpec[np.float32]

    def calculate_loss(
        self,
        u: UnitView,
        outputs: Int32[Array, " num_outputs"],
        targets: Float[Array, " num_outputs"],
        g: GS,
    ) -> tuple[Float[Array, ""], Float[Array, " num_outputs"]]:
        """Compute the scalar loss and the gradient seed over every output.

        Args:
            u: the unit view.
            outputs: the output unit ids, in the builder's output order;
                `targets[k]` is the target of unit `outputs[k]`.
            targets: the target value of every output.
            g: the global state.

        Returns:
            A (loss, seed) pair: the scalar loss and the `(num_outputs,)`
            gradient seed, aligned with `outputs`.
        """
        ...


@dataclasses.dataclass(frozen=True)
class SoftmaxCrossEntropyLoss:
    """Softmax over the output activations, cross-entropy against the targets.

    The targets are a distribution over the outputs (one-hot or soft). With
    logits ``a``, ``m = max(a)``, ``z = a - m`` and ``s = sum(exp(z))``:

    - loss ``L = sum(t * (log(s) - z))``, the max-subtracted log-sum-exp form
      of ``-sum(t * log(softmax(a)))``, finite for any finite logits;
    - seed ``dL/da = exp(z) / s - t``, written to `seed_field`.

    Matches plastax-cpp's `SoftmaxCrossEntropyLoss` operation for operation in
    float32 (max, then the shifted exponentials and their sum, then the
    per-output quotient and the target-weighted log-sum-exp).

    Attributes:
        seed_field: the unit column the gradient seed is written to.
    """

    seed_field: FieldSpec[np.float32]

    def calculate_loss(
        self,
        u: UnitView,
        outputs: Int32[Array, " num_outputs"],
        targets: Float[Array, " num_outputs"],
        g: object,
    ) -> tuple[Float[Array, ""], Float[Array, " num_outputs"]]:
        """Compute the cross-entropy and the softmax-minus-target seed.

        Args:
            u: the unit view.
            outputs: the output unit ids.
            targets: the target distribution over the outputs.
            g: the global state (unused).

        Returns:
            The (loss, seed) pair.
        """
        del g
        logits = u.gather(ACTIVATION, outputs)
        z = logits - jnp.max(logits)
        shifted = jnp.exp(z)
        total = jnp.sum(shifted)
        seed = shifted / total - targets
        loss = jnp.sum(targets * (jnp.log(total) - z))
        return loss, seed


@runtime_checkable
class UpdateUnit[GS](Protocol):
    """Unit update policy: one write per live unit, inputs and outputs included.

    Runs after the backward pass and before the connection update, so it
    reads this step's forward, loss and backward values and the connection
    update reads its writes. A slot holding no live unit (see
    `Network.unit_capacity`) is skipped. Under a batched step it runs on every
    sample's unit state, like forward and backward.

    Type Args:
        GS: the global state type threaded through the network.
    """

    def update(self, u: UnitView, i: UnitIdx, g: GS) -> UnitWrite:
        """Compute the update of one unit.

        Args:
            u: the unit view.
            i: index of the unit.
            g: the global state.

        Returns:
            The UnitWrite for that unit.
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
class ScoreAddConn[GS](Protocol):
    """Connection growth policy: scored candidates through the shared pipeline.

    The rule scores candidate pairs; the framework selects and commits them
    through the deterministic growth pipeline (validity window, optional
    dedupe stages, the total candidate order, per-source-level selection, and
    the in-order slot claim). Candidate production is the rule's
    ``candidates`` attribute (read structurally):

    - ``"exhaustive"`` (default): every ordered unit pair.
    - ``"shortlist"``: the M x M grid over the step's top-M most important
      units (one global importance ranking), M = ``shortlist_size``; requires
      an ``importance(u, i, g) -> Float[Array, ""]`` method. Importance ties
      break by ascending unit id. O(num_units + M^2) instead of
      O(num_units^2).
    - ``"shortlist_per_level"``: each source level draws its own M x M grid --
      top-M sources at that level x top-M destinations inside that level's
      gap window, both by importance. Topological mode only.

    Selection is the rule's ``selection`` (read structurally):

    - ``"top_k"`` (default): each source level's first ``max_new_per_level``
      finite candidates in the total order.
    - ``"threshold"``: those with score >= ``threshold(g)`` (a method, read
      per step), at most ``max_new_per_level``.
    - ``"all"``: every finite candidate.

    ``max_new_per_step`` (int, default None) then caps the step's total
    across levels, level-ascending. The validity window is the rule's
    ``max_level_gap`` (default 1), ``direction`` (``"any"`` default,
    ``"deeper"``, ``"same_or_deeper"``) and ``allow_self_loops`` (default
    False). A score of -inf (or NaN, or any non-finite) vetoes a candidate.

    **Duplicates.** Nothing is deduplicated by default: a candidate equal to
    a live edge grows a *parallel edge*, and two equal candidates in one step
    grow two edges. Set ``dedupe_live = True`` to veto candidates equal to a
    live edge (costs a sort of the live keys every growth step) and/or
    ``dedupe_step = True`` to keep only the first copy, in the total
    candidate order, of equal candidates within the step.

    ``trigger`` gates the phase: ``"every_step"`` (default), ``("every", n)``
    (fires when ``step % n == 0``), ``"on_units_added"`` (fires when this
    step added units), or ``"when"`` (fires when the rule's ``when(g)``
    returns True). ``on_overflow`` is ``"flag"`` (default: dropped commits
    raise the state's ``overflow`` flag) or ``"error"`` (additionally raise
    at runtime). For growth whose cost follows the churn rather than the
    arena, see `ProposeAddConn`.

    Type Args:
        GS: the global state type threaded through the network.
    """

    def score(self, u: UnitView, src: UnitIdx, dst: UnitIdx, g: GS) -> Float[Array, ""]:
        """Score a candidate connection for growth.

        Args:
            u: the unit view.
            src: index of the candidate source unit.
            dst: index of the candidate destination unit.
            g: the global state.

        Returns:
            The candidate score; -inf vetoes the candidate.
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
      arena slot (under Scheme-A sharding, its slot within the shard's band
      of `c`). Candidate order index is ``r * P + j`` with ``r`` the
      connection's rank in ascending ``(src, dst, occurrence)`` order over the
      live connections (occurrence counts parallel edges in ascending slot
      order). Rank and occurrence are global across shards, so a sharded
      step proposes and commits exactly what a single-device step does. A
      unit with no live connections proposes nothing.
    - ``"global"``: one proposer; `propose(u, j, g, rng)`. Order index ``j``.

    Each proposal site receives its own counter-based `Rng`
    (see `plastax.rng`), keyed by the network seed, the step counter, the
    growth stream, the proposer (unit id, connection key, or 0) and `j` --
    so proposal streams vary per step and replay exactly. A rule is free to
    ignore `rng` and derive proposals from state instead.

    Downstream, every proposal passes the shared pipeline: the level-gap
    window (the rule's own `max_level_gap`, read structurally, default 1:
    `abs(level[dst] - level[src]) <= max_level_gap`, no self-loops), the
    opt-in dedupe stages, and each source level's `max_new_per_level`-bounded
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

    Selection, the validity window (``max_level_gap``, ``direction``,
    ``allow_self_loops``), ``max_new_per_level`` / ``max_new_per_step``,
    ``trigger`` and ``on_overflow`` are the same rule attributes
    `ScoreAddConn` documents; proposals feed the same pipeline.

    Attributes:
        proposals_per_proposer: how many proposals each proposing site emits
            per step, a static int >= 1 (required).
    """

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
        update_unit: the unit update policy, or None to elide it.
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
        structural_interval: run the structural phases (connection pruning
            and growth) only every this many steps -- ``step % n == 0`` fires
            them. Default 1 (every step, the historical behavior). The
            growth rule's own ``trigger`` composes on top: both gates must
            pass for growth to run.
        unit_capacity: the number of unit slots, or None (the default) for
            exactly the built unit count. A capacity sizes every unit column
            to that many slots and adds the built-in `PRUNED` column: the
            built units are live and the slots above them are free (marked
            pruned). A slot that holds no live unit is skipped by every pass's
            apply and by connection growth, and keeps its field defaults.
            Input and output units are always built units and are never
            pruned.
        max_levels: the unit-level bound: ``max_levels - 1`` is the deepest
            level unit addition may assign. Default 1024, the C++ library's
            bound.
    """

    forward_pass: ForwardPass[object, GS]
    backward_pass: BackwardPass[object, GS] | None = None
    loss: Loss[GS] | None = None
    update_unit: UpdateUnit[GS] | None = None
    update_conn: UpdateConn[GS] | None = None
    prune_conn: PruneConn[GS] | None = None
    add_conn: ScoreAddConn[GS] | ProposeAddConn[GS] | None = None
    reset_global: ResetGlobal[GS] | None = None

    extra_unit_fields: tuple[FieldSpec[np.generic], ...] = ()
    extra_conn_fields: tuple[FieldSpec[np.generic], ...] = ()
    propagation: Propagation = Propagation.TOPOLOGICAL
    kahn_max_depth: int | None = None
    sharding: ShardSpec | None = None
    seed: int = 0
    structural_interval: int = 1
    unit_capacity: int | None = None
    max_levels: int = 1024

    def __init_subclass__(cls) -> None:
        """Validate the trait slots when a Network subclass is defined."""
        _validate_traits(cls)


_RESERVED_FIELD_NAMES = frozenset(
    {
        FROM_ID.name,
        TO_ID.name,
        DEAD.name,
        WEIGHT.name,
        ACTIVATION.name,
        LEVEL.name,
        PRUNED.name,
    }
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


def _validate_loss(cls: type[Network[Any]], loss: object) -> None:
    """Check the loss policy's shape and its declared seed field.

    Args:
        cls: the Network subclass being validated.
        loss: the candidate loss policy.

    Raises:
        TypeError: if `loss` still has the per-output signature, does not
            satisfy Loss, or declares a seed field that is not a float unit
            column of `cls`.
    """
    if hasattr(loss, "per_output") and not hasattr(loss, "calculate_loss"):
        raise TypeError(
            f"{cls.__name__}.loss: the per-output loss signature was replaced by "
            "the whole-output one. Declare `seed_field` (the unit column the "
            "gradient seed is written to) and implement "
            "`calculate_loss(u, outputs, targets, g) -> (loss, seed)`."
        )
    if not isinstance(loss, Loss):
        raise TypeError(
            f"{cls.__name__}.loss must satisfy Loss (seed_field, calculate_loss); "
            f"got {loss!r}"
        )
    seed: object = loss.seed_field
    unit_fields = (ACTIVATION, *cls.extra_unit_fields)
    if not isinstance(seed, FieldSpec) or seed not in unit_fields:
        raise TypeError(
            f"{cls.__name__}.loss.seed_field must be one of the network's unit "
            f"columns (ACTIVATION or an extra_unit_fields entry); got {seed!r}"
        )
    if not np.issubdtype(seed.dtype, np.floating):
        raise TypeError(
            f"{cls.__name__}.loss.seed_field {seed.name!r} must be a float "
            f"column; got dtype {seed.dtype}"
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

    if cls.loss is not None:
        _validate_loss(cls, cls.loss)

    if cls.update_unit is not None and not isinstance(cls.update_unit, UpdateUnit):
        raise TypeError(
            f"{cls.__name__}.update_unit must satisfy UpdateUnit (update); "
            f"got {cls.update_unit!r}"
        )

    if cls.update_conn is not None and not isinstance(cls.update_conn, UpdateConn):
        raise TypeError(
            f"{cls.__name__}.update_conn must satisfy UpdateConn; "
            f"got {cls.update_conn!r}"
        )

    if cls.prune_conn is not None and not isinstance(cls.prune_conn, PruneConn):
        raise TypeError(
            f"{cls.__name__}.prune_conn must satisfy PruneConn; got {cls.prune_conn!r}"
        )

    interval: object = getattr(cls, "structural_interval", 1)
    if not isinstance(interval, int) or isinstance(interval, bool) or interval < 1:
        raise TypeError(
            f"{cls.__name__}.structural_interval must be an int >= 1; got {interval!r}"
        )

    _validate_unit_slots(cls)

    if cls.add_conn is not None:
        grid = isinstance(cls.add_conn, ScoreAddConn)
        proposed = isinstance(cls.add_conn, ProposeAddConn)
        if grid == proposed:
            raise TypeError(
                f"{cls.__name__}.add_conn must satisfy exactly one of "
                f"ScoreAddConn (score) or ProposeAddConn (propose); "
                f"got {cls.add_conn!r}"
            )
        if isinstance(cls.add_conn, ProposeAddConn):
            _validate_propose_rule(cls, cls.add_conn)
        else:
            _validate_score_rule(cls, cls.add_conn)
        _validate_growth_knobs(cls, cls.add_conn)

    if cls.reset_global is not None and not isinstance(cls.reset_global, ResetGlobal):
        raise TypeError(
            f"{cls.__name__}.reset_global must satisfy ResetGlobal; "
            f"got {cls.reset_global!r}"
        )

    _validate_field_names(cls)


def _validate_unit_slots(cls: type[Network[Any]]) -> None:
    """Check `unit_capacity` and `max_levels` at class definition.

    Args:
        cls: the Network subclass being validated (for error messages).

    Raises:
        TypeError: if `unit_capacity` is neither None nor an int >= 1, or
            `max_levels` is not an int >= 2.
    """
    capacity: object = getattr(cls, "unit_capacity", None)
    if capacity is not None and (
        not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 1
    ):
        raise TypeError(
            f"{cls.__name__}.unit_capacity must be None or an int >= 1; "
            f"got {capacity!r}"
        )
    max_levels: object = getattr(cls, "max_levels", 1024)
    if (
        not isinstance(max_levels, int)
        or isinstance(max_levels, bool)
        or max_levels < 2
    ):
        raise TypeError(
            f"{cls.__name__}.max_levels must be an int >= 2; got {max_levels!r}"
        )


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


_SELECTIONS = ("top_k", "threshold", "all")
_DIRECTIONS = ("any", "deeper", "same_or_deeper")
_CANDIDATES = ("exhaustive", "shortlist", "shortlist_per_level")


def _validate_growth_knobs(cls: type[Network[Any]], ac: object) -> None:
    """Check the growth-rule knobs shared by both strategies.

    Args:
        cls: the Network subclass being validated (for error messages).
        ac: the declared growth rule.

    Raises:
        TypeError: on a removed attribute name, an unknown knob value, a
            missing required companion (`max_new_per_level`, `threshold`,
            `when`), or a mis-typed knob.
    """
    for removed, repl in (
        ("dedupe", "set `dedupe_live` and/or `dedupe_step` (bool, default False)"),
        ("max_candidates", "declare `max_new_per_level` (int >= 1)"),
        (
            "max_candidate_units",
            'declare `candidates = "shortlist"` and `shortlist_size` (int >= 1)',
        ),
        ("shortlist_per_level", 'declare `candidates = "shortlist_per_level"`'),
    ):
        if hasattr(ac, removed):
            raise TypeError(
                f"{cls.__name__}.add_conn.{removed} was removed: {repl} "
                "on the growth rule instead."
            )
    selection = getattr(ac, "selection", "top_k")
    if selection not in _SELECTIONS:
        raise TypeError(
            f"{cls.__name__}.add_conn.selection must be one of "
            f"{_SELECTIONS}; got {selection!r}"
        )
    mnpl: object = getattr(ac, "max_new_per_level", None)
    if selection != "all" or mnpl is not None:
        if not isinstance(mnpl, int) or isinstance(mnpl, bool) or mnpl < 1:
            raise TypeError(
                f"{cls.__name__}.add_conn.max_new_per_level must be an "
                f'int >= 1 (required unless selection = "all"); got {mnpl!r}'
            )
    if selection == "threshold" and not callable(getattr(ac, "threshold", None)):
        raise TypeError(
            f'{cls.__name__}.add_conn: selection = "threshold" requires a '
            "`threshold(g)` method on the rule."
        )
    mnps: object = getattr(ac, "max_new_per_step", None)
    if mnps is not None and (
        not isinstance(mnps, int) or isinstance(mnps, bool) or mnps < 1
    ):
        raise TypeError(
            f"{cls.__name__}.add_conn.max_new_per_step must be an int >= 1 "
            f"or None; got {mnps!r}"
        )
    direction = getattr(ac, "direction", "any")
    if direction not in _DIRECTIONS:
        raise TypeError(
            f"{cls.__name__}.add_conn.direction must be one of "
            f"{_DIRECTIONS}; got {direction!r}"
        )
    trigger: object = getattr(ac, "trigger", "every_step")
    ok = trigger in ("every_step", "on_units_added", "when") or (
        isinstance(trigger, tuple)
        and len(trigger) == 2
        and trigger[0] == "every"
        and isinstance(trigger[1], int)
        and not isinstance(trigger[1], bool)
        and trigger[1] >= 1
    )
    if not ok:
        raise TypeError(
            f"{cls.__name__}.add_conn.trigger must be 'every_step', "
            f"('every', n >= 1), 'on_units_added' or 'when'; got {trigger!r}"
        )
    if trigger == "when" and not callable(getattr(ac, "when", None)):
        raise TypeError(
            f"{cls.__name__}.add_conn: trigger = 'when' requires a "
            "`when(g)` method on the rule."
        )
    on_overflow = getattr(ac, "on_overflow", "flag")
    if on_overflow not in ("flag", "error"):
        raise TypeError(
            f"{cls.__name__}.add_conn.on_overflow must be 'flag' or "
            f"'error'; got {on_overflow!r}"
        )
    for flag in ("dedupe_live", "dedupe_step", "allow_self_loops"):
        v = getattr(ac, flag, False)
        if not isinstance(v, bool):
            raise TypeError(f"{cls.__name__}.add_conn.{flag} must be a bool; got {v!r}")


def _validate_score_rule(cls: type[Network[Any]], ac: ScoreAddConn[Any]) -> None:
    """Check a ScoreAddConn's candidate-production knobs at class definition.

    Args:
        cls: the Network subclass being validated (for error messages).
        ac: the declared score rule.

    Raises:
        TypeError: on an unknown `candidates`, a shortlist without
            `shortlist_size` / `importance`, or a shortlist size that is not
            a positive int.
    """
    candidates = getattr(ac, "candidates", "exhaustive")
    if candidates not in _CANDIDATES:
        raise TypeError(
            f"{cls.__name__}.add_conn.candidates must be one of "
            f"{_CANDIDATES}; got {candidates!r}"
        )
    if candidates != "exhaustive":
        size: object = getattr(ac, "shortlist_size", None)
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise TypeError(
                f"{cls.__name__}.add_conn.shortlist_size must be an int >= 1 "
                f"for candidates = {candidates!r}; got {size!r}"
            )
        if not callable(getattr(ac, "importance", None)):
            raise TypeError(
                f"{cls.__name__}.add_conn: candidates = {candidates!r} "
                "requires an `importance(u, i, g)` method on the rule."
            )


def predicate_add_conn[GS](
    should_add: Callable[[UnitView, UnitIdx, UnitIdx, GS], Bool[Array, ""]],
    init: Callable[[UnitView, UnitIdx, UnitIdx, GS], ConnWrite],
    **params: Any,
) -> ScoreAddConn[GS]:
    """Adapt a boolean predicate to a `ScoreAddConn`.

    The predicate becomes a score of 0.0 (grow) or -inf (veto), with
    ``selection = "all"`` and ``dedupe_step = True`` -- every distinct pair
    the predicate admits is committed once per step, capacity allowing, which
    is the natural reading of a boolean growth rule. Both defaults (and any
    other growth-rule knob) can be overridden through ``params``.

    Type Args:
        GS: the global state type threaded through the network.

    Args:
        should_add: the predicate over (u, src, dst, g).
        init: the new-edge initializer, as `ScoreAddConn.init`.
        **params: growth-rule attributes set on the adapted rule
            (e.g. ``max_level_gap=2``, ``dedupe_live=True``).

    Returns:
        A rule satisfying `ScoreAddConn`.
    """

    class _PredicateRule:
        selection = "all"
        dedupe_step = True

        def score(
            self, u: UnitView, src: UnitIdx, dst: UnitIdx, g: GS
        ) -> Float[Array, ""]:
            grow = should_add(u, src, dst, g)
            out: Float[Array, ""] = jnp.where(
                grow, jnp.float32(0.0), jnp.float32(-jnp.inf)
            )
            return out

        def init(self, u: UnitView, src: UnitIdx, dst: UnitIdx, g: GS) -> ConnWrite:
            return init(u, src, dst, g)

    for key, value in params.items():
        setattr(_PredicateRule, key, value)
    _PredicateRule.__name__ = "PredicateAddConn"
    return _PredicateRule()
