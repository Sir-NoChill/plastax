"""Initial-structure front end: layer specs expand to explicit edge lists
at builder time (host-side, numpy). No lowering is involved: an FC layer
is a dense bipartite edge set, a conv layer a sparse structured one, and
the existing sweep is their forward pass once they land in the arenas.

Weight sharing decision (2026-08-17): conv kernels are UNROLLED — each
(position, tap) pair is its own edge initialized from the shared kernel
value, and plasticity evolves them independently thereafter. Convolution
is an initialization prior on structure, not a maintained constraint.

Initial weights reuse jax.nn.initializers: any (key, shape) -> array
callable is accepted.
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Protocol, TypeAlias

import jax
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray

Initializer: TypeAlias = Callable[[PRNGKeyArray, tuple[int, ...]], Float[Array, "..."]]


@dataclasses.dataclass(frozen=True)
class EdgeSet:
    """Host-side edge list: the sole output currency of all generators."""

    from_ids: np.ndarray      # (E,) int32, block-local unit ids
    to_ids: np.ndarray        # (E,) int32
    weights: np.ndarray       # (E,) float32, already initialized


class Block(Protocol):
    """A unit block plus the edges it contributes; blocks compose by
    concatenating unit id spaces (sequential offsets them)."""

    num_units: int

    def edges(self, key: PRNGKeyArray, offset_in: int, offset_out: int) -> EdgeSet: ...


@dataclasses.dataclass(frozen=True)
class Topology:
    """Finalized structure handed to NetworkBuilder.from_topology."""

    num_units: int
    input_ids: tuple[int, ...]
    output_ids: tuple[int, ...]
    edges: EdgeSet


def input_units(n: int) -> Block:
    raise NotImplementedError


def dense(
    n_in: int,
    n_out: int,
    *,
    init: Initializer = jax.nn.initializers.glorot_uniform(),
) -> Block:
    """Fully connected bipartite block: n_in * n_out edges."""
    raise NotImplementedError


def conv2d(
    in_shape: tuple[int, int, int],       # (H, W, C_in)
    kernel: tuple[int, int, int],         # (kH, kW, C_out)
    *,
    stride: int = 1,
    init: Initializer = jax.nn.initializers.he_normal(),
) -> Block:
    """Receptive-field edge enumeration with unrolled (non-shared) weights;
    index arithmetic mirrors lax.conv_general_dilated shape logic."""
    raise NotImplementedError


def sequential(*blocks: Block) -> Callable[[PRNGKeyArray], Topology]:
    """Compose blocks by offsetting unit id spaces; first block's units are
    inputs, last block's are outputs."""
    raise NotImplementedError
