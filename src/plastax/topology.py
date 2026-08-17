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
    """Host-side edge list: the sole output currency of all generators."""

    from_ids: np.ndarray  # (E,) int32, global unit ids (offsets applied by block)
    to_ids: np.ndarray  # (E,) int32
    weights: np.ndarray  # (E,) float32, already initialized


@runtime_checkable
class Block(Protocol):
    """A unit block plus the edges it contributes; blocks compose by
    concatenating unit id spaces (sequential offsets them).

    @runtime_checkable (Deviation, IMPLEMENTATION_PLAN.md): the stub omitted
    it, but pytest's jaxtyping+beartype instrumentation (pyproject addopts)
    wraps every function in this module, including ones returning/accepting
    `Block`, and beartype cannot build a return-type checker for a plain
    (non-runtime_checkable) Protocol -- it raises at import time.
    """

    num_units: int

    def edges(self, key: PRNGKeyArray, offset_in: int, offset_out: int) -> EdgeSet: ...


@dataclasses.dataclass(frozen=True)
class Topology:
    """Finalized structure handed to NetworkBuilder.from_topology."""

    num_units: int
    input_ids: tuple[int, ...]
    output_ids: tuple[int, ...]
    edges: EdgeSet


@dataclasses.dataclass  # not frozen: Block.num_units is a plain (writable) attribute,
# and a frozen dataclass's fields are structurally read-only against that Protocol.
class _EdgeBlock:
    """Concrete Block: a fixed unit count plus a closure computing its
    (already globally-offset) edges. dense/conv2d/input_units all return
    one of these; `sequential` is the only caller of `.edges`."""

    num_units: int
    _make_edges: Callable[[PRNGKeyArray, int, int], EdgeSet]

    def edges(self, key: PRNGKeyArray, offset_in: int, offset_out: int) -> EdgeSet:
        return self._make_edges(key, offset_in, offset_out)


def input_units(n: int) -> Block:
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
    """Fully connected bipartite block: n_in * n_out edges."""

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
    """Receptive-field edge enumeration with unrolled (non-shared) weights;
    index arithmetic mirrors lax.conv_general_dilated shape logic."""
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
    """Compose blocks by offsetting unit id spaces; first block's units are
    inputs, last block's are outputs.

    The first block contributes no edges of its own (it is the input
    layer); each later block's `.edges` is called with offset_in fixed at
    the previous block's base id and offset_out at its own, so it wires
    directly to its predecessor -- "connects consecutive blocks".
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
