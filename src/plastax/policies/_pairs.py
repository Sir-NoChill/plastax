"""The pairs behaviour is declared with: one per phase.

Behaviour is a grid of phase by element. A pair is one phase's row: the policy
for a connection and the policy for a unit. A network declares its pairs as
class attributes; a layer is given pairs as keywords, and a half it does not
give is the network's (`INHERIT`).
"""

from __future__ import annotations

import dataclasses
import enum

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


class Inherit(enum.Enum):
    """The type of `INHERIT`, the marker for a policy that was not given.

    Attributes:
        INHERIT: the one marker.
    """

    INHERIT = "inherit"


INHERIT = Inherit.INHERIT
"""A half or a keyword that was not given: the network's policy applies.

It is different from None, which says there is no policy. On a network there
is nothing to inherit from, so there it reads as None.
"""


def _require_a_half(pair: object, conn: object, unit: object) -> None:
    """Refuse a pair that gives neither half.

    Args:
        pair: the pair being built, for the message.
        conn: its connection half.
        unit: its unit half.

    Raises:
        ValueError: if both halves are `INHERIT`.
    """
    if conn is INHERIT and unit is INHERIT:
        name = type(pair).__name__
        raise ValueError(
            f"{name}() has neither a conn nor a unit policy, so it says "
            f"nothing; give a half, or pass None in place of the pair for no "
            f"policy at all"
        )


def _missing_half(pair: object, conn: object, unit: object) -> str:
    """The message for reading a pass as two policies when a half is missing.

    Args:
        pair: the pair being read, for the message.
        conn: its connection half.
        unit: its unit half.

    Returns:
        The message, naming the first half that was not given or is None.
    """
    name, half = ("conn", conn) if conn is INHERIT or conn is None else ("unit", unit)
    said = "was not given" if half is INHERIT else "is None"
    return (
        f"this {type(pair).__name__} has no `{name}` policy (it {said}); a "
        f"network's pass needs both halves, because a network has nothing to "
        f"inherit from"
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class Forward[GS]:
    """The forward pass: a value per connection, reduced and applied per unit.

    One object that has `map`, `combine` and `apply` may be given in place of
    a `Forward`; it is read as the pair with that object in both halves.

    A network gives both halves. A layer may give one, and the other is the
    network's; None in a half says the layer's connections, or its units, have
    no forward policy.

    Type Args:
        GS: the global state type threaded through the network.

    Attributes:
        conn: the connection half, which maps one connection to a value.
        unit: the unit half, which reduces the values arriving at a unit and
            writes it.
    """

    conn: ForwardConnPolicy[object, GS] | Inherit | None = INHERIT
    unit: ForwardUnitPolicy[object, GS] | Inherit | None = INHERIT

    def __post_init__(self) -> None:
        """Refuse a pair with neither half given."""
        _require_a_half(self, self.conn, self.unit)

    def halves(
        self,
    ) -> tuple[ForwardConnPolicy[object, GS], ForwardUnitPolicy[object, GS]]:
        """The two policies of a pass that has both.

        Returns:
            The connection half and the unit half.

        Raises:
            TypeError: if a half was not given or is None.
        """
        conn, unit = self.conn, self.unit
        if conn is None or unit is None:
            raise TypeError(_missing_half(self, conn, unit))
        if isinstance(conn, Inherit) or isinstance(unit, Inherit):
            raise TypeError(_missing_half(self, conn, unit))
        return conn, unit


@dataclasses.dataclass(frozen=True, kw_only=True)
class Backward[GS]:
    """The backward pass: a value per connection, reduced at its source.

    One object that has `map`, `combine` and `apply` may be given in place of
    a `Backward`, as for `Forward`. The halves are given or left out as a
    `Forward`'s are.

    Type Args:
        GS: the global state type threaded through the network.

    Attributes:
        conn: the connection half, which maps one connection to a value.
        unit: the unit half, which reduces the values arriving at a
            connection's source and writes that unit.
    """

    conn: BackwardConnPolicy[object, GS] | Inherit | None = INHERIT
    unit: BackwardUnitPolicy[object, GS] | Inherit | None = INHERIT

    def __post_init__(self) -> None:
        """Refuse a pair with neither half given."""
        _require_a_half(self, self.conn, self.unit)

    def halves(
        self,
    ) -> tuple[BackwardConnPolicy[object, GS], BackwardUnitPolicy[object, GS]]:
        """The two policies of a pass that has both.

        Returns:
            The connection half and the unit half.

        Raises:
            TypeError: if a half was not given or is None.
        """
        conn, unit = self.conn, self.unit
        if conn is None or unit is None:
            raise TypeError(_missing_half(self, conn, unit))
        if isinstance(conn, Inherit) or isinstance(unit, Inherit):
            raise TypeError(_missing_half(self, conn, unit))
        return conn, unit


@dataclasses.dataclass(frozen=True, kw_only=True)
class Update[GS]:
    """The update phase: a unit update, a connection update, or both.

    The unit update runs first, so the connection update reads its writes.

    Type Args:
        GS: the global state type threaded through the network.

    Attributes:
        conn: the connection update policy, None for none, or `INHERIT`.
        unit: the unit update policy, likewise.
    """

    conn: UpdateConnPolicy[GS] | Inherit | None = INHERIT
    unit: UpdateUnitPolicy[GS] | Inherit | None = INHERIT

    def __post_init__(self) -> None:
        """Refuse a pair with neither half given."""
        _require_a_half(self, self.conn, self.unit)


@dataclasses.dataclass(frozen=True, kw_only=True)
class Prune[GS]:
    """Pruning: of units, of connections, or of both.

    Type Args:
        GS: the global state type threaded through the network.

    Attributes:
        conn: the connection pruning policy, None for none, or `INHERIT`.
        unit: the unit pruning policy, likewise. Requires
            `Network.unit_capacity`.
    """

    conn: PruneConnPolicy[GS] | Inherit | None = INHERIT
    unit: PruneUnitPolicy[GS] | Inherit | None = INHERIT

    def __post_init__(self) -> None:
        """Refuse a pair with neither half given."""
        _require_a_half(self, self.conn, self.unit)


@dataclasses.dataclass(frozen=True, kw_only=True)
class Grow[GS]:
    """Growth: of units, of connections, or of both.

    Type Args:
        GS: the global state type threaded through the network.

    Attributes:
        conn: the connection growth policy, a `ScoreAddConnPolicy` or a
            `ProposeAddConnPolicy`; None for none, or `INHERIT`.
        unit: the unit addition policy, likewise. Requires
            `Network.unit_capacity`.
    """

    conn: ScoreAddConnPolicy[GS] | ProposeAddConnPolicy[GS] | Inherit | None = INHERIT
    unit: AddUnitPolicy[GS] | Inherit | None = INHERIT

    def __post_init__(self) -> None:
        """Refuse a pair with neither half given."""
        _require_a_half(self, self.conn, self.unit)
