"""Step assembly: the monomorphization point.

One jit cache entry per (Network subclass, NetworkStatic); donation on the
whole state pytree (donate_argnums=0). Cached with the weakref_lru_cache
pattern.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable
from typing import Any, TypeVar, cast

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from plastax._types import ACTIVATION
from plastax.phases import StepInputs, build_phases
from plastax.state import NetworkState, NetworkStatic
from plastax.traits import Network

# Module-scoped (not PEP 695) so it stays free inside the StepFn alias below;
# StepResult/make_step below shadow it with their own PEP 695 [GS] locally.
GS = TypeVar("GS")


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class StepResult[GS]:
    """One step's output: the new state plus framework-computed signals.

    Type Args:
        GS: the user's global-state pytree, opaque to the framework.

    Attributes:
        state: The network state after the step.
        overflow: AddConn overflow flag for the step.
        loss: Reduced per-output loss for the step; 0.0 when the net has no
            loss phase.
    """

    state: NetworkState[GS]
    overflow: Bool[Array, ""]
    loss: Float[Array, ""]


StepFn = Callable[[NetworkState[GS], StepInputs], StepResult[GS]]


def make_step[GS](net: type[Network[GS]], static: NetworkStatic) -> StepFn[GS]:
    """Assemble the present phases and jit them with donate_argnums=0.

    The returned callable must be shape-preserving on the state pytree so
    every leaf donates (CI promotes the donation warning to an error).

    Type Args:
        GS: the user's global-state pytree, opaque to the framework.

    Args:
        net: The network subclass to assemble phases for.
        static: The network's static configuration.

    Returns:
        A jitted step function for the given network and static config.
    """
    # mypy false positive: a parameterized generic base class fails the
    # structural Hashable check, though a class is always hashable by
    # identity; hence the cast.
    return cast(StepFn[GS], _cached_make_step(net, static))  # type: ignore[arg-type]


# jax.util.weakref_lru_cache is not cleanly importable off the pinned jax
# floor, so functools.cache is used instead: it gives the same hash/eq-keyed
# reuse, at the cost of strong (rather than weak) references to (net,
# static) and the cached StepFn -- benign for v1's low-cardinality,
# process-lifetime pairs.
@functools.cache
def _cached_make_step(net: type[Network[Any]], static: NetworkStatic) -> StepFn[Any]:
    # overflow_sink (see build_phases): a length-1 out-parameter
    # build_add_conn_phase (when net.add_conn is set) overwrites on every
    # call; stays [False] otherwise. Created once here (mirrors `phases`
    # itself), mutated once at trace time, read below into StepResult --
    # jax.jit traces step's body exactly once, so this is an ordinary data
    # dependency in the resulting jaxpr, not a stale Python-side read.
    overflow_sink: list[Bool[Array, ""]] = [jnp.bool_(False)]
    phases = build_phases(net, static, overflow_sink=overflow_sink)
    input_ids = jnp.asarray(static.input_ids, dtype=jnp.int32)

    def step(state: NetworkState[Any], inputs: StepInputs) -> StepResult[Any]:
        # Step input scatter, before any phase: StepInputs.inputs onto
        # units[ACTIVATION] at the static input_ids.
        activation = state.units[ACTIVATION.name].at[input_ids].set(inputs.inputs)
        state = dataclasses.replace(
            state, units={**state.units, ACTIVATION.name: activation}
        )

        total_loss = jnp.float32(0.0)
        for phase in phases:
            state, contribution = phase(state, inputs)
            total_loss = total_loss + contribution

        return StepResult(state=state, overflow=overflow_sink[0], loss=total_loss)

    # jax.jit's return type is opaque under follow_imports="skip" (pyproject,
    # jax.* -> Any); step's own signature is the true (and already checked)
    # contract, so cast rather than let strict mypy's no-any-return fire.
    return cast(StepFn[Any], jax.jit(step, donate_argnums=0))
