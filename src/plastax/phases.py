"""Phase builders: each returns a pure state->state function for one Do*
phase, or None when the trait slot is absent (trace-time elision, rung0
design section 2). Phase order matches plastix.hpp's 11-phase step for the
v1 subset: forward, loss, backward, update_conn, prune_conn, add_conn,
reset_global.
"""
from __future__ import annotations

from typing import Callable, TypeAlias, TypeVar

from jaxtyping import Array, Float

from plastax.state import NetworkState, NetworkStatic
from plastax.traits import Network

GS = TypeVar("GS")
Phase: TypeAlias = Callable[["NetworkState[GS]", "StepInputs"], "NetworkState[GS]"]


class StepInputs:
    """Clamped inputs + targets for one step; fixed pytree structure.

    inputs: (num_inputs,) values scattered to input unit ids (static tuple)
    targets: (num_outputs,) for the loss phase, or None when loss is absent
    """

    inputs: Float[Array, " num_inputs"]
    targets: Float[Array, " num_outputs"] | None


def build_phases(
    net: type[Network[GS]], static: NetworkStatic
) -> tuple[Phase[GS], ...]:
    """Assemble only the present phases; absent slots contribute nothing to
    the trace. Topological forward walks buckets 1..L (Python loop, static
    slices); backward walks L..1; pipeline is the 1-bucket flat sweep."""
    raise NotImplementedError


def build_add_conn_phase(
    net: type[Network[GS]], static: NetworkStatic
) -> Phase[GS]:
    """K-bounded candidates from the neighbourhood window; lax.top_k
    (static k, stable); per-bucket prefix-sum slot claim; overflow ->
    dropped scatter + flag; level-preserving adds must NOT set
    needs_resort (rung0 design section 5)."""
    raise NotImplementedError
