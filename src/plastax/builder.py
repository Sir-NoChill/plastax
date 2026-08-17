"""Host-side network construction (pre-jit, eager, plain numpy).

Mirrors the manual construction path of the C++ examples: add units and
connections imperatively, then finalize into (NetworkStatic, NetworkState).
"""

from __future__ import annotations

from collections.abc import Callable

from jaxtyping import PRNGKeyArray

from plastax.state import NetworkState, NetworkStatic
from plastax.topology import Topology
from plastax.traits import Network


class NetworkBuilder[GS]:
    def __init__(self, net: type[Network[GS]], globals_: GS) -> None:
        raise NotImplementedError

    @classmethod
    def from_topology(
        cls,
        net: type[Network[GS]],
        topology_fn: Callable[[PRNGKeyArray], Topology],
        key: PRNGKeyArray,
        *,
        globals_: GS,
    ) -> tuple[NetworkStatic, NetworkState[GS]]:
        """Expand a plastax.topology spec into arenas: add units, mark the
        first/last blocks as inputs/outputs, bulk add_conn the edge set,
        finalize."""
        raise NotImplementedError

    def add_unit(self, **field_values: float | int | bool) -> int:
        """Returns the new unit's global id (dense, 0-based)."""
        raise NotImplementedError

    def add_conn(self, src: int, dst: int, **field_values: float | int | bool) -> None:
        raise NotImplementedError

    def mark_input(self, unit_id: int) -> None:
        raise NotImplementedError

    def mark_output(self, unit_id: int) -> None:
        raise NotImplementedError

    def finalize(self) -> tuple[NetworkStatic, NetworkState[GS]]:
        """Computes initial levels (topo.initial_levels), buckets conns by
        source level with capacity_policy headroom, allocates arenas, and
        freezes the static config."""
        raise NotImplementedError
