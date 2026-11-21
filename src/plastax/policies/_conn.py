"""Policies written for one connection."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple, Protocol, runtime_checkable

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int32

from plastax._types import (
    ConnIdx,
    UnitIdx,
)
from plastax.rng import Rng
from plastax.views import ConnView, ConnWrite, UnitView


@runtime_checkable
class UpdateConnPolicy[GS](Protocol):
    """Connection update policy: two full passes, incoming then outgoing.

    Under a batched step (`make_step(..., batch_size=B)`) the update is reduced
    over the batch. By default each sample's change to every floating
    connection column is averaged (exact for rules linear in the per-sample
    term, e.g. SGD; unwritten columns stay untouched), and a non-floating
    column takes sample 0's write. A policy may instead
    declare, structurally (read with getattr, not part of this Protocol), the
    exact pair: `per_sample(u, dst, src, c, cid, g) -> pytree`, evaluated per
    sample and averaged over the batch, and `incoming_batched(u, dst, src, c,
    cid, g, stat) -> ConnWrite`, applied once with that average (and the unit
    view reduced by the network's `batch_reduction`) in place of `incoming`.
    Every `plastax.optim` bundle declares it, so a batched optimizer step is
    one step on the batch-mean gradient.

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
class PruneConnPolicy[GS](Protocol):
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
class ScoreAddConnPolicy[GS](Protocol):
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

    ``level_cap_scope`` (read structurally) scopes ``max_new_per_level``:
    ``"source_level"`` (default) applies it to each source level in turn, in
    both propagation models; ``"step"`` applies it once to the whole step --
    the first ``max_new_per_level`` qualifying candidates of the total order,
    whatever their source levels. Either way the winners are listed levels
    ascending, the total order within a level, and each claims a slot for its
    own source level. ``max_new_per_step`` (int, default None) then caps that
    list, level-ascending. The validity window is the rule's
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
    arena, see `ProposeAddConnPolicy`.

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
class ProposeAddConnPolicy[GS](Protocol):
    """Connection growth policy: bounded growth from sampled proposals.

    The counterpart of `ScoreAddConnPolicy` whose cost follows the churn, not the
    arena, and the default growth for networks with unit addition: a
    per-unit proposer lets every live unit, one spawned this step included,
    propose its own edges, num_units x P candidates instead of a
    num_units^2 grid.
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
    ``allow_self_loops``), ``max_new_per_level`` / ``level_cap_scope`` /
    ``max_new_per_step``,
    ``trigger`` and ``on_overflow`` are the same rule attributes
    `ScoreAddConnPolicy` documents; proposals feed the same pipeline.

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


def predicate_add_conn[GS](
    should_add: Callable[[UnitView, UnitIdx, UnitIdx, GS], Bool[Array, ""]],
    init: Callable[[UnitView, UnitIdx, UnitIdx, GS], ConnWrite],
    **params: Any,
) -> ScoreAddConnPolicy[GS]:
    """Adapt a boolean predicate to a `ScoreAddConnPolicy`.

    The predicate becomes a score of 0.0 (grow) or -inf (veto), with
    ``selection = "all"`` and ``dedupe_step = True`` -- every distinct pair
    the predicate admits is committed once per step, capacity allowing, which
    is the natural reading of a boolean growth rule. Both defaults (and any
    other growth-rule knob) can be overridden through ``params``.

    Type Args:
        GS: the global state type threaded through the network.

    Args:
        should_add: the predicate over (u, src, dst, g).
        init: the new-edge initializer, as `ScoreAddConnPolicy.init`.
        **params: growth-rule attributes set on the adapted rule
            (e.g. ``max_level_gap=2``, ``dedupe_live=True``).

    Returns:
        A rule satisfying `ScoreAddConnPolicy`.
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
