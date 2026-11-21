"""Policies written for one unit."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from jaxtyping import Array, Bool, Int32

from plastax._types import (
    UnitIdx,
)
from plastax.views import UnitView, UnitWrite


@runtime_checkable
class UpdateUnitPolicy[GS](Protocol):
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
class PruneUnitPolicy[GS](Protocol):
    """Unit pruning policy: permanently remove units by predicate.

    The predicate is evaluated, on the state as it stands before the phase,
    for every live unit except the input and output units, which are never
    pruned. Each unit it selects is marked `PRUNED` for good: its columns
    reset to their declared defaults except `LEVEL`, which it keeps, and every
    connection incident to it is tombstoned in the same phase. A slot holding
    no live unit is skipped by every later phase. The phase runs every step,
    after the connection update and before connection pruning, and under a
    batched step it runs once on the reduced unit state. It requires
    `Network.unit_capacity`.

    Type Args:
        GS: the global state type threaded through the network.
    """

    def predicate(self, u: UnitView, i: UnitIdx, g: GS) -> Bool[Array, ""]:
        """Decide whether to prune one unit.

        Args:
            u: the unit view.
            i: index of the unit.
            g: the global state.

        Returns:
            A scalar bool; True to prune the unit.
        """
        ...


@runtime_checkable
class AddUnitPolicy[GS](Protocol):
    """Unit addition policy: live units spawn children into free slots.

    `spawn` is evaluated, on the state as it stands before the phase, for
    every unit live at the start of the phase, inputs and outputs included.
    Each parent spawns at most one child per step, and the parents that spawn
    are served in ascending id order: the i-th one receives the i-th lowest
    free slot id. The free slots are every slot holding no live unit, so the
    ids pruned earlier in the same step are reused before the never-allocated
    ones above them. A spawn with no free slot left is dropped and raises
    `NetworkState.unit_overflow`. Children are not parents in the step that
    creates them.

    A child slot starts at its declared column defaults, live, at level
    ``clamp(level(parent) + offset, 1, max_levels - 1)``; `init` then writes
    its fields. The child has no connections until growth adds them, and its
    level is provisional until the next resort recomputes the levels from the
    live edges. The phase runs after connection pruning and before growth,
    gated by `Network.structural_interval`, and under a batched step it runs
    once on the reduced unit state. It requires `Network.unit_capacity`.

    Type Args:
        GS: the global state type threaded through the network.
    """

    def spawn(
        self, u: UnitView, parent: UnitIdx, g: GS
    ) -> tuple[Bool[Array, ""], Int32[Array, ""]]:
        """Decide whether one live unit spawns a child.

        Args:
            u: the unit view, before the phase.
            parent: index of the candidate parent.
            g: the global state.

        Returns:
            A scalar bool, True to spawn, and the child's level offset from
            the parent's level.
        """
        ...

    def init(self, u: UnitView, child: UnitIdx, parent: UnitIdx, g: GS) -> UnitWrite:
        """Write a new child's fields.

        Args:
            u: the unit view, with the child's slot at its column defaults
                and its assigned level.
            child: index of the child's slot.
            parent: index of the parent.
            g: the global state.

        Returns:
            The child's fields; unwritten fields keep their defaults.
        """
        ...
