"""Where each policy of a network applies: forward, backward, update and prune.

A network's policy is the default and a layer may have its own, or none. In one
phase every element then has one policy or none, and the phase is computed over
the elements that have it: runs of unit ids, and runs of slots in each bucket.
`scopes` works those runs out from the resolved table, the build record and the
layout.

Two different policies in one phase are not run yet. `clashes` finds them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from plastax.layers._module import BuildError
from plastax.layers._resolve import Override, PolicyTable, _same, resolve
from plastax.state import NetworkStatic, Span

if TYPE_CHECKING:
    from plastax.network import Network

type ScopedPhase = Literal["forward", "backward", "update", "prune"]

# The phases a layer may differ from its network in, so far: all but growth.
SCOPED_PHASES: tuple[ScopedPhase, ...] = ("forward", "backward", "update", "prune")
# Those of them that still take one policy, or none: the two passes. An update
# or a pruning may have several, each on its own layers.
SINGLE_PHASES: tuple[ScopedPhase, ...] = ("forward", "backward")


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
    """The scope of each policy of each phase a layer may differ in.

    A pass has one policy for each kind of element, or none (None). An update
    or a pruning may have several policies, each on its own layers: one scope
    for each, over disjoint elements, and none (an empty tuple) for a phase no
    element has.

    Attributes:
        forward_unit: The unit half of the forward pass: the units that reduce
            what arrives and are written.
        forward_conn: The connection half of the forward pass: the connections
            that send a value to their destination.
        backward_unit: The unit half of the backward pass.
        backward_conn: The connection half of the backward pass: the
            connections that send a value to their source.
        update_unit: The unit updates.
        update_conn: The connection updates.
        prune_unit: The unit prunings.
        prune_conn: The connection prunings.
    """

    forward_unit: UnitScope | None
    forward_conn: ConnScope | None
    backward_unit: UnitScope | None
    backward_conn: ConnScope | None
    update_unit: tuple[UnitScope, ...]
    update_conn: tuple[ConnScope, ...]
    prune_unit: tuple[UnitScope, ...]
    prune_conn: tuple[ConnScope, ...]

    @property
    def whole(self) -> bool:
        """Whether every policy applies to the whole network."""
        passes = (
            self.forward_unit,
            self.forward_conn,
            self.backward_unit,
            self.backward_conn,
        )
        several = (self.update_unit, self.update_conn, self.prune_unit, self.prune_conn)
        return all(scope is None or scope.spans is None for scope in passes) and all(
            scope.spans is None for scopes_ in several for scope in scopes_
        )


def _distinct(cells: list[object]) -> list[object]:
    """The different policies among some cells, None left out."""
    found: list[object] = []
    for cell in cells:
        if cell is not None and not any(_same(cell, seen) for seen in found):
            found.append(cell)
    return found


def clashes(table: PolicyTable) -> list[Override]:
    """The cells of a table that put two different policies in one pass.

    The forward and the backward pass may each have one policy on some layers
    and none on the others. Where the layers and the network have two or more
    different policies between them, every layer whose policy is not the
    network's is returned. An update or a pruning may have several policies and
    never clashes.

    Args:
        table: The resolved policies.

    Returns:
        The overriding cells of every pass that has more than one policy, in
        the order of `PolicyTable.overrides`.
    """
    crowded: set[tuple[str, str]] = set()
    rows: tuple[tuple[str, Any, tuple[Any, ...]], ...] = (
        ("connections", table.network_conns, table.conns),
        ("units", table.network_units, table.units),
    )
    for kind, network, blocks in rows:
        for phase in SINGLE_PHASES:
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
            f"the {phase} pass has more than one policy ({names}); two "
            f"different policies in one pass are not run yet (layer policies, "
            f"stage 4)"
        )
    return found[0] if found else None


def scopes(net: type[Network[Any]], static: NetworkStatic) -> Scopes:
    """Work out where each policy of a network applies.

    A network with no layout (one built from edge arrays, or one that grows
    connections of its own) has no layers to differ: each of its policies
    applies everywhere. Otherwise the network's topology is traced again, which
    computes nothing, and each policy's runs are read from the layout.

    A unit that is in no block (an input unit, a free unit slot) has the
    network's policy. The spare slots of a bucket hold no connection and are in
    no scope that is narrower than the whole network.

    Going forward a connection sends a value to its destination, and going
    backward to its source. A block none of whose receiving units has a unit
    half in that pass is left out of the pass's connection scope: what it sent
    would be reduced by nothing and read by nothing.

    Args:
        net: The network class.
        static: The static configuration the network was built into.

    Returns:
        The scope of each phase.

    Raises:
        BuildError: If a pass has two different policies, or `static` was not
            built from this network's topology.
    """
    layout = static.layout
    if layout is None or net.topology is None:
        resolved = net.resolved()
        forward_conn, forward_unit = net.forward.halves()
        backward = net.backward
        backward_conn, backward_unit = (
            (None, None) if backward is None else backward.halves()
        )

        def everywhere[S: (UnitScope, ConnScope)](
            kind: type[S], policy: object
        ) -> S | None:
            return None if policy is None else kind(policy, None)

        def each[S: (UnitScope, ConnScope)](
            kind: type[S], policy: object
        ) -> tuple[S, ...]:
            return () if policy is None else (kind(policy, None),)

        return Scopes(
            forward_unit=everywhere(UnitScope, forward_unit),
            forward_conn=everywhere(ConnScope, forward_conn),
            backward_unit=everywhere(UnitScope, backward_unit),
            backward_conn=everywhere(ConnScope, backward_conn),
            update_unit=each(UnitScope, resolved.update_unit),
            update_conn=each(ConnScope, resolved.update_conn),
            prune_unit=each(UnitScope, resolved.prune_unit),
            prune_conn=each(ConnScope, resolved.prune_conn),
        )

    record = net._trace()
    table = resolve(net, record)
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

    def unit_cells(phase: ScopedPhase) -> tuple[object, list[object]]:
        return (
            getattr(table.network_units, phase),
            [getattr(block, phase) for block in table.units],
        )

    def conn_cells(phase: ScopedPhase) -> tuple[object, list[object]]:
        return (
            getattr(table.network_conns, phase),
            [getattr(block, phase) for block in table.conns],
        )

    def units_of(policy: object, network: object, blocks: list[object]) -> UnitScope:
        """The units that have one policy: its layers', and the network's if it is."""
        spans = [
            run
            for run, mine in zip(layout.unit_runs, blocks, strict=True)
            if mine is not None and _same(mine, policy)
        ]
        if network is not None and _same(network, policy):
            spans += outside
        joined = _joined(spans)
        everything = joined == ((0, static.num_units),) or static.num_units == 0
        return UnitScope(policy, None if everything else joined)

    def conns_of(
        policy: object,
        network: object,
        blocks: list[object],
        received_by: tuple[UnitScope | None, str] | None = None,
    ) -> ConnScope:
        """The connections that have one policy, as runs in each bucket."""
        mine = network is not None and _same(network, policy)
        sends = [block is not None and _same(block, policy) for block in blocks]
        if received_by is not None:
            # A pass: drop the blocks none of whose receiving units reduces.
            receivers, end = received_by
            reduces = np.zeros(static.num_units, bool)
            if receivers is not None:
                for start, stop in receivers.spans or ((0, static.num_units),):
                    reduces[start:stop] = True
            sends = [
                send and bool(reduces[getattr(block, end)].any())
                for send, block in zip(sends, record.blocks, strict=True)
            ]
        if mine and all(sends):
            return ConnScope(policy, None)
        per_bucket: list[list[Span]] = [[] for _ in range(num_buckets)]
        for runs, send in zip(layout.conn_runs, sends, strict=True):
            if send:
                for run in runs:
                    per_bucket[run.bucket].append((run.start, run.stop))
        return ConnScope(policy, tuple(_joined(spans) for spans in per_bucket))

    def one_unit(phase: ScopedPhase) -> UnitScope | None:
        network, blocks = unit_cells(phase)
        policy = _one(f"unit {phase}", network, blocks)
        return None if policy is None else units_of(policy, network, blocks)

    def one_conn(
        phase: ScopedPhase, received_by: tuple[UnitScope | None, str]
    ) -> ConnScope | None:
        network, blocks = conn_cells(phase)
        policy = _one(f"connection {phase}", network, blocks)
        if policy is None:
            return None
        return conns_of(policy, network, blocks, received_by)

    def each_unit(phase: ScopedPhase) -> tuple[UnitScope, ...]:
        network, blocks = unit_cells(phase)
        return tuple(
            units_of(policy, network, blocks)
            for policy in _distinct([network, *blocks])
        )

    def each_conn(phase: ScopedPhase) -> tuple[ConnScope, ...]:
        network, blocks = conn_cells(phase)
        return tuple(
            conns_of(policy, network, blocks)
            for policy in _distinct([network, *blocks])
        )

    forward_units, backward_units = one_unit("forward"), one_unit("backward")
    return Scopes(
        forward_unit=forward_units,
        forward_conn=one_conn("forward", (forward_units, "dst")),
        backward_unit=backward_units,
        backward_conn=one_conn("backward", (backward_units, "src")),
        update_unit=each_unit("update"),
        update_conn=each_conn("update"),
        prune_unit=each_unit("prune"),
        prune_conn=each_conn("prune"),
    )
