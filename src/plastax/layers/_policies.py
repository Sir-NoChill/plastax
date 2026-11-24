"""The policies a layer is given: what a block holds, and the checks at the layer.

A layer takes the network's five keywords: `forward`, `backward`, `update`,
`prune`, `grow`. A primitive creates one kind of element and takes that half
directly. A helper that creates both takes pairs and splits them. A keyword that
is not given is `INHERIT`: the network's policy applies. None says the layer has
none.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from plastax.policies import (
    INHERIT,
    AddUnitPolicy,
    Backward,
    BackwardConnPolicy,
    BackwardUnitPolicy,
    Forward,
    ForwardConnPolicy,
    ForwardUnitPolicy,
    Grow,
    Inherit,
    ProposeAddConnPolicy,
    Prune,
    PruneConnPolicy,
    PruneUnitPolicy,
    ScoreAddConnPolicy,
    Update,
    UpdateConnPolicy,
    UpdateUnitPolicy,
)

type Given[P] = P | Inherit | None
"""A keyword's value: a policy, None for no policy, or `INHERIT`."""

PHASES = ("forward", "backward", "update", "prune", "grow")
"""The five keywords, in the order a step runs them."""


@dataclass(frozen=True)
class UnitPolicies:
    """The unit half of each phase, for one block of units.

    In a build record a cell may be `INHERIT`. In a resolved table every cell is
    a policy or None.

    Attributes:
        forward: reduces the values arriving at a unit and writes it.
        backward: the same for the values sent back to a unit.
        update: the unit update.
        prune: the unit pruning policy.
        grow: the unit addition policy.
    """

    forward: Given[ForwardUnitPolicy[Any, Any]] = INHERIT
    backward: Given[BackwardUnitPolicy[Any, Any]] = INHERIT
    update: Given[UpdateUnitPolicy[Any]] = INHERIT
    prune: Given[PruneUnitPolicy[Any]] = INHERIT
    grow: Given[AddUnitPolicy[Any]] = INHERIT


@dataclass(frozen=True)
class ConnPolicies:
    """The connection half of each phase, for one block of connections.

    In a build record a cell may be `INHERIT`. In a resolved table every cell is
    a policy or None.

    Attributes:
        forward: maps one connection to a value for its destination.
        backward: maps one connection to a value for its source.
        update: the connection update.
        prune: the connection pruning policy.
        grow: the connection growth policy.
    """

    forward: Given[ForwardConnPolicy[Any, Any]] = INHERIT
    backward: Given[BackwardConnPolicy[Any, Any]] = INHERIT
    update: Given[UpdateConnPolicy[Any]] = INHERIT
    prune: Given[PruneConnPolicy[Any]] = INHERIT
    grow: Given[ScoreAddConnPolicy[Any] | ProposeAddConnPolicy[Any]] = INHERIT


# For each keyword: the protocols a value may satisfy, and the members that make
# one, for the message.
_UNIT_TRAITS: dict[str, tuple[tuple[type, ...], str]] = {
    "forward": ((ForwardUnitPolicy,), "ForwardUnitPolicy (combine, apply)"),
    "backward": ((BackwardUnitPolicy,), "BackwardUnitPolicy (combine, apply)"),
    "update": ((UpdateUnitPolicy,), "UpdateUnitPolicy (update)"),
    "prune": ((PruneUnitPolicy,), "PruneUnitPolicy (predicate)"),
    "grow": ((AddUnitPolicy,), "AddUnitPolicy (spawn, init)"),
}
_CONN_TRAITS: dict[str, tuple[tuple[type, ...], str]] = {
    "forward": ((ForwardConnPolicy,), "ForwardConnPolicy (map)"),
    "backward": ((BackwardConnPolicy,), "BackwardConnPolicy (map)"),
    "update": ((UpdateConnPolicy,), "UpdateConnPolicy (incoming, outgoing)"),
    "prune": ((PruneConnPolicy,), "PruneConnPolicy (predicate)"),
    "grow": (
        (ScoreAddConnPolicy, ProposeAddConnPolicy),
        "ScoreAddConnPolicy (score, init) or ProposeAddConnPolicy (propose, init)",
    ),
}
_PAIRS: dict[str, type[Any]] = {
    "forward": Forward,
    "backward": Backward,
    "update": Update,
    "prune": Prune,
    "grow": Grow,
}


def _check(
    layer: str,
    kind: str,
    traits: dict[str, tuple[tuple[type, ...], str]],
    given: dict[str, object],
) -> None:
    """Check each keyword a layer was given against the protocol of its phase.

    Args:
        layer: The layer's class name, for the message.
        kind: "unit" or "connection", for the message.
        traits: The protocols of each phase for this kind of element.
        given: The five keywords as the layer received them.

    Raises:
        TypeError: If a value is not a policy of its phase, None or `INHERIT`.
    """
    for phase in PHASES:
        value = given[phase]
        if value is INHERIT or value is None:
            continue
        protocols, wanted = traits[phase]
        if isinstance(value, _PAIRS[phase]):
            raise TypeError(
                f"{layer}({phase}=...) was given a {type(value).__name__} pair. "
                f"{layer} creates {kind}s only, so it takes the {kind} policy "
                f"itself: a {wanted}"
            )
        if not isinstance(value, protocols):
            raise TypeError(
                f"{layer}({phase}=...) takes a {kind} policy: a {wanted}, None "
                f"for no policy, or nothing to take the network's. "
                f"Got {value!r}"
            )


def unit_policies(layer: str, **given: Any) -> UnitPolicies:
    """Check the keywords of a layer that creates units, and hold them.

    Args:
        layer: The layer's class name, for the message.
        **given: The five keywords: `forward`, `backward`, `update`, `prune`,
            `grow`.

    Returns:
        The keywords as given.
    """
    _check(layer, "unit", _UNIT_TRAITS, given)
    return UnitPolicies(**given)


def conn_policies(layer: str, **given: Any) -> ConnPolicies:
    """Check the keywords of a layer that creates connections, and hold them.

    Args:
        layer: The layer's class name, for the message.
        **given: The five keywords: `forward`, `backward`, `update`, `prune`,
            `grow`.

    Returns:
        The keywords as given.
    """
    _check(layer, "connection", _CONN_TRAITS, given)
    return ConnPolicies(**given)


def split_pairs(layer: str, **given: Any) -> tuple[ConnPolicies, UnitPolicies]:
    """Split the pairs a helper layer was given into its two kinds of element.

    The `conn` half of each pair goes to the layer's connections and the `unit`
    half to its units. A keyword that is None gives None to both, and one that is
    not given leaves both to the network. One object that has `map`, `combine`
    and `apply` may be given as `forward` or `backward`, as on a network.

    Args:
        layer: The layer's class name, for the message.
        **given: The five keywords, each a pair of its phase, None or `INHERIT`.

    Returns:
        The connection halves and the unit halves, each checked against the
        protocol of its phase.

    Raises:
        TypeError: If a keyword is not a pair of its phase, None or `INHERIT`.
    """
    conn: dict[str, object] = {}
    unit: dict[str, object] = {}
    for phase in PHASES:
        value = given[phase]
        pair = _PAIRS[phase]
        if value is INHERIT or value is None:
            conn[phase] = unit[phase] = value
        elif isinstance(value, pair):
            conn[phase], unit[phase] = value.conn, value.unit
        elif (
            phase in ("forward", "backward")
            and isinstance(value, _CONN_TRAITS[phase][0])
            and isinstance(value, _UNIT_TRAITS[phase][0])
        ):
            conn[phase] = unit[phase] = value
        else:
            both = (
                ", or one object with map, combine and apply"
                if phase in ("forward", "backward")
                else ""
            )
            raise TypeError(
                f"{layer}({phase}=...) creates connections and units, so it "
                f"takes a pair: plastax.policies.{pair.__name__}(conn=, unit=)"
                f"{both}; None for no policy, or nothing to take the network's. "
                f"Got {value!r}"
            )
    return conn_policies(layer, **conn), unit_policies(layer, **unit)
