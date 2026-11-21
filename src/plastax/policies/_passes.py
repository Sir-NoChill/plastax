"""The bundled forward and backward pass traits.

Temporary: cut into connection and unit halves by the pass split.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from plastax._types import (
    ConnIdx,
    UnitIdx,
)
from plastax.monoid import MonoidTree
from plastax.views import ConnView, UnitView, UnitWrite


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
