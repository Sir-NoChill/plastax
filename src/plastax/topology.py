"""Expand layer specs into explicit edge lists at build time.

Building happens host-side, using numpy; no lowering is involved. An FC
layer is a dense bipartite edge set, a conv layer a sparse structured
one, and the existing sweep is their forward pass once they land in the
arenas.

Weight sharing decision (2026-08-17): conv kernels are UNROLLED — each
(position, tap) pair is its own edge initialized from the shared kernel
value, and plasticity evolves them independently thereafter. Convolution
is an initialization prior on structure, not a maintained constraint.

Initial weights reuse jax.nn.initializers: any (key, shape) -> array
callable is accepted.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Protocol, runtime_checkable

import jax
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray

type Initializer = Callable[[PRNGKeyArray, tuple[int, ...]], Float[Array, "..."]]

# Module-level singletons, not calls in argument defaults (ruff B008): a
# default evaluated in the signature is a single object either way, but a
# named singleton makes the one-time construction explicit.
_GLOROT_UNIFORM: Initializer = jax.nn.initializers.glorot_uniform()
_HE_NORMAL: Initializer = jax.nn.initializers.he_normal()


@dataclasses.dataclass(frozen=True)
class EdgeSet:
    """Host-side edge list: the sole output currency of all generators.

    Attributes:
        from_ids: (E,) int32 global source unit ids (offsets applied by block).
        to_ids: (E,) int32 global destination unit ids.
        weights: (E,) float32 initial edge weights.
    """

    from_ids: np.ndarray
    to_ids: np.ndarray
    weights: np.ndarray


@runtime_checkable
class Block(Protocol):
    """A unit block plus the edges it contributes.

    Blocks compose by concatenating unit id spaces; `sequential` offsets
    them accordingly.

    `@runtime_checkable` is required because pytest's jaxtyping+beartype
    instrumentation wraps every function in this module, including ones
    returning or accepting `Block`, and beartype cannot build a checker
    for a Protocol that isn't runtime-checkable — it would raise at
    import time.
    """

    num_units: int

    def edges(self, key: PRNGKeyArray, offset_in: int, offset_out: int) -> EdgeSet:
        """Compute this block's contributed edges, offset into global ids.

        Args:
            key: PRNG key for weight initialization.
            offset_in: Global unit id offset for this block's source units.
            offset_out: Global unit id offset for this block's output units.

        Returns:
            The edge set wiring into this block, with globally-offset ids.
        """
        ...


@dataclasses.dataclass(frozen=True)
class Topology:
    """Finalized structure handed to NetworkBuilder.from_topology.

    Attributes:
        num_units: Total unit count across all blocks.
        input_ids: Global ids of the input units (the first block).
        output_ids: Global ids of the output units (the last block).
        edges: The full host-side edge list.
    """

    num_units: int
    input_ids: tuple[int, ...]
    output_ids: tuple[int, ...]
    edges: EdgeSet


@dataclasses.dataclass  # not frozen: Block.num_units is a plain (writable) attribute,
# and a frozen dataclass's fields are structurally read-only against that Protocol.
class _EdgeBlock:
    """Concrete Block: a fixed unit count plus an edge-computing closure.

    `dense`, `conv2d`, and `input_units` all return one of these;
    `sequential` is the only caller of `.edges`.
    """

    num_units: int
    _make_edges: Callable[[PRNGKeyArray, int, int], EdgeSet]

    def edges(self, key: PRNGKeyArray, offset_in: int, offset_out: int) -> EdgeSet:
        """Delegate to the stored edge-computing closure.

        Args:
            key: PRNG key for weight initialization.
            offset_in: Global unit id offset for this block's source units.
            offset_out: Global unit id offset for this block's output units.

        Returns:
            The edge set wiring into this block, with globally-offset ids.
        """
        return self._make_edges(key, offset_in, offset_out)


def input_units(n: int) -> Block:
    """Build a block of n input units that contributes no edges.

    Args:
        n: Number of input units.

    Returns:
        A Block of n input units with an empty edge set.
    """

    def make_edges(key: PRNGKeyArray, offset_in: int, offset_out: int) -> EdgeSet:
        del key, offset_in, offset_out
        return EdgeSet(
            from_ids=np.zeros((0,), dtype=np.int32),
            to_ids=np.zeros((0,), dtype=np.int32),
            weights=np.zeros((0,), dtype=np.float32),
        )

    return _EdgeBlock(num_units=n, _make_edges=make_edges)


def dense(
    n_in: int,
    n_out: int,
    *,
    init: Initializer = _GLOROT_UNIFORM,
) -> Block:
    """Build a fully connected bipartite block with n_in * n_out edges.

    Args:
        n_in: Number of source units.
        n_out: Number of output units.
        init: Weight initializer, called as init(key, (n_in, n_out)).

    Returns:
        A Block wiring every input unit to every output unit.
    """

    def make_edges(key: PRNGKeyArray, offset_in: int, offset_out: int) -> EdgeSet:
        # fan_in/fan_out convention for a (n_in, n_out) weight matrix, per
        # jax.nn.initializers._compute_fans default in_axis=-2, out_axis=-1.
        weights = np.asarray(init(key, (n_in, n_out)), dtype=np.float32)
        src, dst = np.meshgrid(
            np.arange(n_in, dtype=np.int32),
            np.arange(n_out, dtype=np.int32),
            indexing="ij",
        )
        return EdgeSet(
            from_ids=(src + offset_in).reshape(-1).astype(np.int32),
            to_ids=(dst + offset_out).reshape(-1).astype(np.int32),
            weights=weights.reshape(-1),
        )

    return _EdgeBlock(num_units=n_out, _make_edges=make_edges)


def conv2d(
    in_shape: tuple[int, int, int],  # (H, W, C_in)
    kernel: tuple[int, int, int],  # (kH, kW, C_out)
    *,
    stride: int = 1,
    init: Initializer = _HE_NORMAL,
) -> Block:
    """Enumerate receptive-field edges with unrolled (non-shared) weights.

    Index arithmetic mirrors lax.conv_general_dilated's shape logic.

    Args:
        in_shape: Input shape (H, W, C_in).
        kernel: Kernel shape (kH, kW, C_out).
        stride: Convolution stride.
        init: Weight initializer, called as init(key, (kH, kW, C_in, C_out)).

    Returns:
        A Block wiring each output unit to its receptive field, with one
        edge per (position, tap) pair.
    """
    h, w, c_in = in_shape
    kh, kw, c_out = kernel
    # VALID padding, no dilation: matches lax.conv_general_dilated's output
    # spatial size for dimension_numbers=("NHWC", "HWIO", "NHWC").
    out_h = (h - kh) // stride + 1
    out_w = (w - kw) // stride + 1

    def make_edges(key: PRNGKeyArray, offset_in: int, offset_out: int) -> EdgeSet:
        # HWIO kernel layout: fan_in = c_in * kh * kw via the same default
        # axes as dense (_compute_fans treats non in/out axes as receptive
        # field), matching lax.conv_general_dilated's expected kernel shape.
        weights = np.asarray(init(key, (kh, kw, c_in, c_out)), dtype=np.float32)
        oh, ow, co, th, tw, ci = np.meshgrid(
            np.arange(out_h, dtype=np.int32),
            np.arange(out_w, dtype=np.int32),
            np.arange(c_out, dtype=np.int32),
            np.arange(kh, dtype=np.int32),
            np.arange(kw, dtype=np.int32),
            np.arange(c_in, dtype=np.int32),
            indexing="ij",
        )
        ih = oh * stride + th
        iw = ow * stride + tw
        in_local = (ih * w + iw) * c_in + ci
        out_local = (oh * out_w + ow) * c_out + co
        return EdgeSet(
            from_ids=(in_local + offset_in).reshape(-1).astype(np.int32),
            to_ids=(out_local + offset_out).reshape(-1).astype(np.int32),
            weights=weights[th, tw, ci, co].reshape(-1),
        )

    return _EdgeBlock(num_units=out_h * out_w * c_out, _make_edges=make_edges)


def sequential(*blocks: Block) -> Callable[[PRNGKeyArray], Topology]:
    """Compose blocks by offsetting unit id spaces into one topology.

    The first block's units are inputs and the last block's are outputs;
    it contributes no edges of its own. Each later block's `.edges` is
    called with offset_in fixed at the previous block's base id and
    offset_out at its own, wiring it directly to its predecessor.

    Args:
        *blocks: Blocks to connect in sequence, first to last.

    Returns:
        A callable that takes a PRNG key and builds the finalized Topology.

    Raises:
        ValueError: If no blocks are given.
    """
    if not blocks:
        raise ValueError("sequential: at least one block is required")

    def build(key: PRNGKeyArray) -> Topology:
        offsets: list[int] = []
        total = 0
        for block in blocks:
            offsets.append(total)
            total += block.num_units

        keys = jax.random.split(key, len(blocks))
        from_chunks: list[np.ndarray] = []
        to_chunks: list[np.ndarray] = []
        weight_chunks: list[np.ndarray] = []
        for i in range(1, len(blocks)):
            edge_set = blocks[i].edges(keys[i], offsets[i - 1], offsets[i])
            from_chunks.append(edge_set.from_ids)
            to_chunks.append(edge_set.to_ids)
            weight_chunks.append(edge_set.weights)

        edges = EdgeSet(
            from_ids=np.concatenate(from_chunks)
            if from_chunks
            else np.zeros((0,), dtype=np.int32),
            to_ids=np.concatenate(to_chunks)
            if to_chunks
            else np.zeros((0,), dtype=np.int32),
            weights=np.concatenate(weight_chunks)
            if weight_chunks
            else np.zeros((0,), dtype=np.float32),
        )
        input_ids = tuple(range(blocks[0].num_units))
        output_ids = tuple(range(offsets[-1], total))
        return Topology(
            num_units=total, input_ids=input_ids, output_ids=output_ids, edges=edges
        )

    return build
