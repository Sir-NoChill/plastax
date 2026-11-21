"""The pairs a network declares its behaviour with: one per phase.

Behaviour is a grid of phase by element. A pair is one phase's row: the policy
for a connection and the policy for a unit.
"""

from __future__ import annotations

import dataclasses

from plastax.policies._conn import (
    BackwardConnPolicy,
    ForwardConnPolicy,
    ProposeAddConnPolicy,
    PruneConnPolicy,
    ScoreAddConnPolicy,
    UpdateConnPolicy,
)
from plastax.policies._unit import (
    AddUnitPolicy,
    BackwardUnitPolicy,
    ForwardUnitPolicy,
    PruneUnitPolicy,
    UpdateUnitPolicy,
)


def _require_a_half(pair: object, conn: object, unit: object) -> None:
    """Refuse a pair with neither half set.

    Args:
        pair: the pair being built, for the message.
        conn: its connection half.
        unit: its unit half.

    Raises:
        ValueError: if both halves are None.
    """
    if conn is None and unit is None:
        name = type(pair).__name__
        raise ValueError(
            f"{name}() has neither a conn nor a unit policy; set one, or leave "
            f"the network's field None"
        )


@dataclasses.dataclass(frozen=True, kw_only=True)
class Forward[GS]:
    """The forward pass: a value per connection, reduced and applied per unit.

    One object that has `map`, `combine` and `apply` may be given to a network
    in place of a `Forward`; it is read as the pair with that object in both
    halves.

    Type Args:
        GS: the global state type threaded through the network.

    Attributes:
        conn: the connection half, which maps one connection to a value.
        unit: the unit half, which reduces the values arriving at a unit and
            writes it.
    """

    conn: ForwardConnPolicy[object, GS]
    unit: ForwardUnitPolicy[object, GS]


@dataclasses.dataclass(frozen=True, kw_only=True)
class Backward[GS]:
    """The backward pass: a value per connection, reduced at its source.

    One object that has `map`, `combine` and `apply` may be given to a network
    in place of a `Backward`, as for `Forward`.

    Type Args:
        GS: the global state type threaded through the network.

    Attributes:
        conn: the connection half, which maps one connection to a value.
        unit: the unit half, which reduces the values arriving at a
            connection's source and writes that unit.
    """

    conn: BackwardConnPolicy[object, GS]
    unit: BackwardUnitPolicy[object, GS]


@dataclasses.dataclass(frozen=True, kw_only=True)
class Update[GS]:
    """The update phase: a unit update, a connection update, or both.

    The unit update runs first, so the connection update reads its writes.

    Type Args:
        GS: the global state type threaded through the network.

    Attributes:
        conn: the connection update policy, or None.
        unit: the unit update policy, or None.
    """

    conn: UpdateConnPolicy[GS] | None = None
    unit: UpdateUnitPolicy[GS] | None = None

    def __post_init__(self) -> None:
        """Refuse a pair with neither half set."""
        _require_a_half(self, self.conn, self.unit)


@dataclasses.dataclass(frozen=True, kw_only=True)
class Prune[GS]:
    """Pruning: of units, of connections, or of both.

    Type Args:
        GS: the global state type threaded through the network.

    Attributes:
        conn: the connection pruning policy, or None.
        unit: the unit pruning policy, or None. Requires
            `Network.unit_capacity`.
    """

    conn: PruneConnPolicy[GS] | None = None
    unit: PruneUnitPolicy[GS] | None = None

    def __post_init__(self) -> None:
        """Refuse a pair with neither half set."""
        _require_a_half(self, self.conn, self.unit)


@dataclasses.dataclass(frozen=True, kw_only=True)
class Grow[GS]:
    """Growth: of units, of connections, or of both.

    Type Args:
        GS: the global state type threaded through the network.

    Attributes:
        conn: the connection growth policy, a `ScoreAddConnPolicy` or a
            `ProposeAddConnPolicy`, or None.
        unit: the unit addition policy, or None. Requires
            `Network.unit_capacity`.
    """

    conn: ScoreAddConnPolicy[GS] | ProposeAddConnPolicy[GS] | None = None
    unit: AddUnitPolicy[GS] | None = None

    def __post_init__(self) -> None:
        """Refuse a pair with neither half set."""
        _require_a_half(self, self.conn, self.unit)
