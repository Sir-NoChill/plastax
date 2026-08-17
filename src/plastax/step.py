"""Step assembly: the monomorphization point (rung0 design section 2).

One jit cache entry per (Network subclass, NetworkStatic); donation on the
whole state pytree (donate_argnums=0). Cached with the weakref_lru_cache
pattern of jax/_src/pjit.py:612.
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Generic, TypeVar

from jaxtyping import Array, Bool

from plastax.phases import StepInputs
from plastax.state import NetworkState, NetworkStatic
from plastax.traits import Network

GS = TypeVar("GS")


@dataclasses.dataclass(frozen=True)
class StepResult(Generic[GS]):
    state: NetworkState[GS]
    overflow: Bool[Array, ""]     # per-step AddConn bucket overflow flag


StepFn = Callable[[NetworkState[GS], StepInputs], StepResult[GS]]


def make_step(net: type[Network[GS]], static: NetworkStatic) -> StepFn[GS]:
    """Assemble present phases, jit with donate_argnums=0. The returned
    callable must be shape-preserving on the state pytree so every leaf
    donates (CI promotes the donation warning to an error)."""
    raise NotImplementedError
