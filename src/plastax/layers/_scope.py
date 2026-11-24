"""Where each update and prune policy of a network applies.

A network's policy is the default and a layer may have its own, or none. In one
phase every element then has one policy or none, and the phase is computed over
the elements that have it: runs of unit ids, and runs of slots in each bucket.
`scopes` works those runs out from the resolved table and the layout.

Two different policies in one phase are not run yet. `clashes` finds them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from plastax.layers._module import BuildError
from plastax.layers._resolve import Override, PolicyTable, _same
from plastax.state import NetworkStatic, Span

if TYPE_CHECKING:
    from plastax.network import Network

# The phases a layer may differ from its network in, so far.
SCOPED_PHASES: tuple[Literal["update", "prune"], ...] = ("update", "prune")


@dataclass(frozen=True)
class UnitScope:
    """The one unit policy of a phase, and the units it applies to.

    Attributes:
        policy: The policy.
        spans: The unit ids it applies to, as runs `(start, stop)` in ascending
            order with adjacent runs joined, or None for every unit.
    """

    policy: Any
    spans: tuple[Span, ...] | None


@dataclass(frozen=True)
class ConnScope:
    """The one connection policy of a phase, and the connections it applies to.

    Attributes:
        policy: The policy.
        spans: For each bucket, the slots it applies to, as runs
            `(start, stop)` in ascending order with adjacent runs joined; a
            bucket it does not apply in has none. None for every slot of every
            bucket.
    """

    policy: Any
    spans: tuple[tuple[Span, ...], ...] | None


@dataclass(frozen=True)
class Scopes:
    """The scope of each phase a layer may differ in, or None for a phase no one has.

    Attributes:
        update_unit: The unit update.
        update_conn: The connection update.
        prune_unit: The unit pruning.
        prune_conn: The connection pruning.
    """

    update_unit: UnitScope | None
    update_conn: ConnScope | None
    prune_unit: UnitScope | None
    prune_conn: ConnScope | None

    @property
    def whole(self) -> bool:
        """Whether every policy applies to the whole network."""
        cells = (self.update_unit, self.update_conn, self.prune_unit, self.prune_conn)
        return all(scope is None or scope.spans is None for scope in cells)


def _distinct(cells: list[object]) -> list[object]:
    """The different policies among some cells, None left out."""
    found: list[object] = []
    for cell in cells:
        if cell is not None and not any(_same(cell, seen) for seen in found):
            found.append(cell)
    return found


def clashes(table: PolicyTable) -> list[Override]:
    """The cells of a table that put two different policies in one phase.

    A phase may have one policy on some layers and none on the others. Where
    the layers and the network have two or more different policies between
    them, every layer whose policy is not the network's is returned.

    Args:
        table: The resolved policies.

    Returns:
        The overriding cells of every update or prune phase that has more than
        one policy, in the order of `PolicyTable.overrides`.
    """
    crowded: set[tuple[str, str]] = set()
    rows: tuple[tuple[str, Any, tuple[Any, ...]], ...] = (
        ("connections", table.network_conns, table.conns),
        ("units", table.network_units, table.units),
    )
    for kind, network, blocks in rows:
        for phase in SCOPED_PHASES:
            cells = [getattr(network, phase), *(getattr(b, phase) for b in blocks)]
            if len(_distinct(cells)) > 1:
                crowded.add((kind, phase))
    return [
        o
        for o in table.overrides()
        if (o.kind, o.phase) in crowded and o.policy is not None
    ]


def _joined(spans: list[Span]) -> tuple[Span, ...]:
    """Sort runs and join those that touch; empty runs are dropped."""
    out: list[Span] = []
    for start, stop in sorted(span for span in spans if span[1] > span[0]):
        if out and out[-1][1] == start:
            out[-1] = (out[-1][0], stop)
        else:
            out.append((start, stop))
    return tuple(out)


def _one(phase: str, network: object, blocks: list[object]) -> object | None:
    """The one policy of a phase, or None when no element has one.

    Raises:
        BuildError: If the phase has two different policies.
    """
    found = _distinct([network, *blocks])
    if len(found) > 1:
        names = " and ".join(type(policy).__name__ for policy in found)
        raise BuildError(
            f"the {phase} phase has more than one policy ({names}); two "
            f"different policies in one phase are not run yet (layer policies, "
            f"stage 4)"
        )
    return found[0] if found else None


def scopes(net: type[Network[Any]], static: NetworkStatic) -> Scopes:
    """Work out where each update and prune policy of a network applies.

    A network with no layout (one built from edge arrays, or one that grows
    connections of its own) has no layers to differ: each of its policies
    applies everywhere. Otherwise the network's topology is traced again, which
    computes nothing, and each policy's runs are read from the layout.

    A unit that is in no block (an input unit, a free unit slot) has the
    network's policy. The spare slots of a bucket hold no connection and are in
    no scope that is narrower than the whole network.

    Args:
        net: The network class.
        static: The static configuration the network was built into.

    Returns:
        The scope of each phase.

    Raises:
        BuildError: If a phase has two different policies, or `static` was not
            built from this network's topology.
    """
    layout = static.layout
    if layout is None or net.topology is None:
        resolved = net.resolved()

        def everywhere[S: (UnitScope, ConnScope)](
            kind: type[S], policy: object
        ) -> S | None:
            return None if policy is None else kind(policy, None)

        return Scopes(
            update_unit=everywhere(UnitScope, resolved.update_unit),
            update_conn=everywhere(ConnScope, resolved.update_conn),
            prune_unit=everywhere(UnitScope, resolved.prune_unit),
            prune_conn=everywhere(ConnScope, resolved.prune_conn),
        )

    table = net.policy_table()
    if len(table.units) != len(layout.unit_runs) or len(table.conns) != len(
        layout.conn_runs
    ):
        raise BuildError(
            f"this static configuration was not built from the topology of "
            f"{net.__name__}: it has {len(layout.unit_runs)} blocks of units and "
            f"{len(layout.conn_runs)} of connections, and the topology has "
            f"{len(table.units)} and {len(table.conns)}"
        )

    # The unit ids of no block: the network's.
    in_blocks = _joined(list(layout.unit_runs))
    edges = [0, *(edge for span in in_blocks for edge in span), static.num_units]
    outside = [(edges[k], edges[k + 1]) for k in range(0, len(edges), 2)]
    num_buckets = len(static.level_capacities)

    def units(phase: Literal["update", "prune"]) -> UnitScope | None:
        network = getattr(table.network_units, phase)
        blocks = [getattr(block, phase) for block in table.units]
        policy = _one(f"unit {phase}", network, blocks)
        if policy is None:
            return None
        spans = [
            run
            for run, mine in zip(layout.unit_runs, blocks, strict=True)
            if mine is not None
        ]
        if network is not None:
            spans += outside
        joined = _joined(spans)
        everything = joined == ((0, static.num_units),) or static.num_units == 0
        return UnitScope(policy, None if everything else joined)

    def conns(phase: Literal["update", "prune"]) -> ConnScope | None:
        network = getattr(table.network_conns, phase)
        blocks = [getattr(block, phase) for block in table.conns]
        policy = _one(f"connection {phase}", network, blocks)
        if policy is None:
            return None
        if network is not None and all(mine is not None for mine in blocks):
            return ConnScope(policy, None)
        per_bucket: list[list[Span]] = [[] for _ in range(num_buckets)]
        for runs, mine in zip(layout.conn_runs, blocks, strict=True):
            if mine is not None:
                for run in runs:
                    per_bucket[run.bucket].append((run.start, run.stop))
        return ConnScope(policy, tuple(_joined(spans) for spans in per_bucket))

    return Scopes(
        update_unit=units("update"),
        update_conn=conns("update"),
        prune_unit=units("prune"),
        prune_conn=conns("prune"),
    )
