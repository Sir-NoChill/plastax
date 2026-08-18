"""Step assembly: the monomorphization point (rung0 design section 2).

One jit cache entry per (Network subclass, NetworkStatic); donation on the
whole state pytree (donate_argnums=0). Cached with the weakref_lru_cache
pattern of jax/_src/pjit.py:612.
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
    """Registered as a dataclass pytree (Deviation, IMPLEMENTATION_PLAN.md
    M2): the stub's plain @dataclasses.dataclass cannot cross a jit boundary
    as a return value (confirmed empirically: unregistered dataclasses trace
    fine but jit raises "not a valid JAX type" on return); make_step's
    jitted `step` returns StepResult directly, so it must be registered.
    """

    state: NetworkState[GS]
    overflow: Bool[Array, ""]  # noqa: F722  AddConn overflow flag, jaxtyping shape
    # Reduced (summed) per_output loss -- Deviation, IMPLEMENTATION_PLAN.md
    # M2 loss-target question (phases.py's Phase docstring has the full
    # reasoning): globals_ is a fully opaque user pytree with no framework-
    # known "loss slot" (Network[None] is a valid, exercised instantiation),
    # so the reduced scalar lands here instead, sibling to `overflow`.
    # jnp.float32(0.0) when net.loss is None, mirroring overflow's "always a
    # constant in M2" convention.
    loss: Float[Array, ""]  # noqa: F722  jaxtyping scalar shape


StepFn = Callable[[NetworkState[GS], StepInputs], StepResult[GS]]


def make_step[GS](net: type[Network[GS]], static: NetworkStatic) -> StepFn[GS]:
    """Assemble present phases, jit with donate_argnums=0. The returned
    callable must be shape-preserving on the state pytree so every leaf
    donates (CI promotes the donation warning to an error)."""
    # mypy false positive (confirmed via a minimal repro isolated from this
    # module): `type[Network[GS]]` -- a parameterized GENERIC base class --
    # fails mypy's structural Hashable check on object.__hash__'s synthesized
    # self-type ("def __hash__(self: object)" vs the Hashable protocol's
    # "def __hash__()"), even though the identical check passes for a
    # concrete (non-generic-parameterized) Network subclass, and NetworkStatic
    # -- itself frozen/hashable -- already proves value-equal-instances collapse
    # to one cache entry (test_pytree.py's
    # test_networkstatic_value_equality_drives_cache_reuse). `net` is in fact
    # always hashable (a class, by default object identity).
    return cast(StepFn[GS], _cached_make_step(net, static))  # type: ignore[arg-type]


# jax.util.weakref_lru_cache is not cleanly importable off the pinned floor
# (Deviation, IMPLEMENTATION_PLAN.md M2, per the module docstring's citation
# of jax/_src/pjit.py:612): confirmed empirically against jax==0.11.0
# (pyproject `jax>=0.11.0`) -- `jax.util` raises AttributeError (the
# compatibility shim the rung0 design's citation assumes is gone) and
# `import jax.util` raises ModuleNotFoundError. `jax._src.util.weakref_lru_cache`
# is reachable, but reaching into `_src` is exactly the fragility the task's
# "if not cleanly importable" clause is guarding against, not a substitute
# for it. functools.cache (an unbounded functools.lru_cache, ruff UP033)
# gives the same hash/eq-keyed reuse -- proven against NetworkStatic by
# test_pytree.py's test_networkstatic_value_equality_drives_cache_reuse --
# at the cost of strong (not weak) references to (net, static) and the
# cached StepFn; benign for v1's process-lifetime, low-cardinality (net,
# static) pairs.
@functools.cache
def _cached_make_step(net: type[Network[Any]], static: NetworkStatic) -> StepFn[Any]:
    # overflow_sink (phases.py's build_phases docstring, IMPLEMENTATION_PLAN.md
    # M4a Deviation): a length-1 out-parameter build_add_conn_phase (when
    # net.add_conn is set) overwrites on every call; stays [False] otherwise,
    # matching the pre-M4a constant. Created once here (mirrors `phases`
    # itself), mutated once at trace time, read below into StepResult --
    # jax.jit traces step's body exactly once, so this is an ordinary data
    # dependency in the resulting jaxpr, not a stale Python-side read.
    overflow_sink: list[Bool[Array, ""]] = [jnp.bool_(False)]
    phases = build_phases(net, static, overflow_sink=overflow_sink)
    input_ids = jnp.asarray(static.input_ids, dtype=jnp.int32)

    def step(state: NetworkState[Any], inputs: StepInputs) -> StepResult[Any]:
        # Step input scatter (IMPLEMENTATION_PLAN.md M2), before any phase:
        # StepInputs.inputs onto units[ACTIVATION] at the static input_ids.
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
