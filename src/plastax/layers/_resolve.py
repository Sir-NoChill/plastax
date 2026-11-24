"""Resolving the policies of a traced topology against its network.

A network's policies are the default. A layer may be given its own, phase by
phase, and a phase it does not name is the network's. `resolve` works that out for
every block of a build record and gives the result as a table.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Literal

from plastax.layers._module import _show
from plastax.layers._policies import PHASES, ConnPolicies, UnitPolicies
from plastax.layers._record import Record
from plastax.policies import INHERIT

if TYPE_CHECKING:
    from plastax.network import Network


def _name(policy: object) -> str:
    """How a cell is printed: the policy's class, or `-` for none."""
    return "-" if policy is None else type(policy).__name__


def _same(a: object, b: object) -> bool:
    """Whether two cells hold the same policy."""
    return a is b or bool(a == b)


@dataclass(frozen=True)
class Override:
    """One cell in which a block's policy is not the network's.

    Attributes:
        path: The path of the layer that made the block.
        kind: Whether the block is units or connections.
        phase: The phase: `forward`, `backward`, `update`, `prune` or `grow`.
        policy: The block's policy in that phase, or None for none.
        network: The network's policy in that phase, or None for none.
    """

    path: tuple[str, ...]
    kind: Literal["units", "connections"]
    phase: str
    policy: object | None
    network: object | None


@dataclass(frozen=True)
class PolicyTable:
    """Every block's policy in every phase, with nothing left to inherit.

    Printed (`str(table)`), it is a Markdown table: the network's two rows, then
    one row for each block that differs from the network. In a block's row `.`
    is the network's policy and `-` is none.

    Attributes:
        network_units: The network's unit policies: a policy or None in each
            phase.
        network_conns: The network's connection policies, likewise.
        unit_paths: The path of each block of units, in the record's order.
        units: The policies of each block of units, in the same order.
        conn_paths: The path of each block of connections, in the record's
            order.
        conns: The policies of each block of connections, in the same order.
    """

    network_units: UnitPolicies
    network_conns: ConnPolicies
    unit_paths: tuple[tuple[str, ...], ...]
    units: tuple[UnitPolicies, ...]
    conn_paths: tuple[tuple[str, ...], ...]
    conns: tuple[ConnPolicies, ...]

    def overrides(self) -> list[Override]:
        """The cells in which a block differs from the network.

        Returns:
            One entry per differing cell: the connection blocks in the record's
            order, then the unit blocks, each in the order of the phases.
        """
        found: list[Override] = []
        rows: tuple[tuple[Any, Any, Any, Literal["units", "connections"]], ...] = (
            (self.conn_paths, self.conns, self.network_conns, "connections"),
            (self.unit_paths, self.units, self.network_units, "units"),
        )
        for paths, blocks, network, kind in rows:
            for path, block in zip(paths, blocks, strict=True):
                for phase in PHASES:
                    mine, theirs = getattr(block, phase), getattr(network, phase)
                    if not _same(mine, theirs):
                        found.append(Override(path, kind, phase, mine, theirs))
        return found

    def __str__(self) -> str:
        """The table as Markdown."""
        header = ("layer", "elements", *PHASES)
        rows: list[tuple[str, ...]] = [
            (
                "(network)",
                "connections",
                *(_name(getattr(self.network_conns, phase)) for phase in PHASES),
            ),
            (
                "(network)",
                "units",
                *(_name(getattr(self.network_units, phase)) for phase in PHASES),
            ),
        ]
        blocks: tuple[tuple[Any, Any, Any, str], ...] = (
            (self.conn_paths, self.conns, self.network_conns, "connections"),
            (self.unit_paths, self.units, self.network_units, "units"),
        )
        for paths, policies, network, kind in blocks:
            for path, block in zip(paths, policies, strict=True):
                cells = tuple(
                    "."
                    if _same(getattr(block, phase), getattr(network, phase))
                    else _name(getattr(block, phase))
                    for phase in PHASES
                )
                if any(cell != "." for cell in cells):
                    rows.append((_show(path), kind, *cells))
        widths = [max(len(row[k]) for row in (header, *rows)) for k in range(7)]

        def line(row: tuple[str, ...]) -> str:
            cells = (c.ljust(w) for c, w in zip(row, widths, strict=True))
            return "| " + " | ".join(cells) + " |"

        rule = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
        return "\n".join([line(header), rule, *(line(row) for row in rows)])


def _fill[P: (UnitPolicies, ConnPolicies)](given: P, network: P) -> P:
    """Replace each `INHERIT` cell of a block with the network's.

    Args:
        given: The block's policies as its layer was given them.
        network: The network's policies for the same kind of element.

    Returns:
        The block's policies with a policy or None in every cell.
    """
    return type(given)(
        **{
            f.name: (
                getattr(network, f.name)
                if getattr(given, f.name) is INHERIT
                else getattr(given, f.name)
            )
            for f in fields(given)
        }
    )


def resolve(net: type[Network[Any]], record: Record) -> PolicyTable:
    """Work out the policy of every block of a record in every phase.

    A block's own policy applies where its layer gave one, or gave None; the
    network's applies where the layer gave nothing. The network's update is its
    optimizer's or its `update`'s, as the network resolves it; a layer's `update`
    replaces that for the layer's own elements.

    Args:
        net: The network class the record was traced for.
        record: The traced topology.

    Returns:
        The resolved table.
    """
    resolved = net.resolved()
    forward_conn, forward_unit = net.forward.halves()
    backward = net.backward
    backward_conn, backward_unit = (
        (None, None) if backward is None else backward.halves()
    )
    network_conns = ConnPolicies(
        forward=forward_conn,
        backward=backward_conn,
        update=resolved.update_conn,
        prune=resolved.prune_conn,
        grow=resolved.grow_conn,
    )
    network_units = UnitPolicies(
        forward=forward_unit,
        backward=backward_unit,
        update=resolved.update_unit,
        prune=resolved.prune_unit,
        grow=resolved.grow_unit,
    )
    return PolicyTable(
        network_units=network_units,
        network_conns=network_conns,
        unit_paths=tuple(block.path for block in record.unit_blocks),
        units=tuple(_fill(b.policies, network_units) for b in record.unit_blocks),
        conn_paths=tuple(block.path for block in record.blocks),
        conns=tuple(_fill(b.policies, network_conns) for b in record.blocks),
    )
