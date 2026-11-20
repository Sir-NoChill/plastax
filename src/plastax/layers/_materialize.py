"""Materialization: a build record becomes a state that can be stepped.

A bridge onto today's `Network` class and today's builder, so that a topology written
as a `Module` runs before the policy redesign lands. It is replaced when
`Network.init` is built, and does as little as it can: it checks the record, gives
constants their value, draws each block's weights, and hands whole arrays to
`NetworkBuilder.from_edges`.

Plan: `plans/20261119T201041Z-step6-materialize.md` at the workspace root.
"""

from __future__ import annotations

import zlib

import jax
import numpy as np
from jaxtyping import PRNGKeyArray

from plastax import init as px_init
from plastax._types import ACTIVATION
from plastax.builder import NetworkBuilder
from plastax.init import Fans
from plastax.layers._module import BuildError
from plastax.layers._record import ConnBlock, Record
from plastax.state import NetworkState, NetworkStatic
from plastax.traits import Network


def _name(block: ConnBlock) -> str:
    """Write a block's path as it is shown in an error."""
    return "/".join(block.path)


def _on_a_cycle(src: np.ndarray, dst: np.ndarray, num_units: int) -> np.ndarray:
    """Find the connections that lie on a cycle or between two cycles.

    A connection whose source has nothing coming in cannot be on a cycle, nor can one
    whose destination has nothing going out. Both kinds are set aside until none is
    left; what remains, if anything, is held up by a cycle.

    Args:
        src: The source of each connection.
        dst: The destination of each connection.
        num_units: How many units there are.

    Returns:
        A boolean mask over the connections. All False when there is no cycle.
    """
    held = np.ones(src.shape, bool)
    while True:
        fed = np.bincount(dst[held], minlength=num_units) > 0
        read = np.bincount(src[held], minlength=num_units) > 0
        still = held & fed[src] & read[dst]
        if still.sum() == held.sum():
            return held
        held = still


def _check(record: Record, src: np.ndarray, dst: np.ndarray, check_cycle: bool) -> None:
    """Refuse a record that today's engine cannot run as it was written.

    Args:
        record: The record.
        src: The source of every connection, block after block.
        dst: The destination of every connection, block after block.
        check_cycle: Whether a cycle is refused here.

    Raises:
        BuildError: If a connection leads into a constant unit, or, with
            `check_cycle`, if the connections close a cycle.
    """
    constants = record.constants
    for block in record.blocks:
        into = np.isin(block.dst, constants)
        if into.any():
            k = int(np.flatnonzero(into)[0])
            raise BuildError(
                f"block {_name(block)} connects unit {int(block.src[k])} into unit "
                f"{int(block.dst[k])}, which is a constant; a unit that is fed is "
                f"recomputed every step and would not keep its value"
            )

    if not check_cycle:
        return
    held = _on_a_cycle(src, dst, record.num_units)
    if held.any():
        ends = np.cumsum([block.src.size for block in record.blocks])
        starts = ends - [block.src.size for block in record.blocks]
        names = [
            _name(block)
            for block, start, end in zip(record.blocks, starts, ends, strict=True)
            if held[start:end].any()
        ]
        raise BuildError(
            f"the connections close a cycle, through {', '.join(names)}; "
            f"topological propagation needs a network without one. "
            f"`check_cycle=False` skips this check"
        )


def _weights(
    block: ConnBlock,
    key: PRNGKeyArray,
    in_degree: np.ndarray,
    out_degree: np.ndarray,
) -> np.ndarray:
    """Draw one block's weights from its own key.

    Args:
        block: The block.
        key: The key given to `materialize`.
        in_degree: For each unit, the connections into it that are counted in a
            fan-in.
        out_degree: For each unit, the connections out of it.

    Returns:
        The weights as a flat float32 array, in block order.

    Raises:
        BuildError: If the initializer cannot be called with `(key, fans)`, refuses
            the block, or returns anything but an array of the block's shape.
    """
    # crc32 and not `hash`, which differs from one process to the next. A block's
    # key depends on its path only, so adding a layer leaves the others' weights.
    block_key = jax.random.fold_in(key, zlib.crc32(_name(block).encode()))
    fans = Fans(
        block.shape,
        in_degree[block.dst].reshape(block.shape),
        out_degree[block.src].reshape(block.shape),
    )
    initializer = px_init.lecun_normal() if block.init is None else block.init
    try:
        drawn = np.asarray(initializer(block_key, fans))
    except TypeError as error:
        raise BuildError(
            f"the initializer of block {_name(block)} could not be called with "
            f"(key, fans): {error}. An initializer of px.init is written with its "
            f"call, as in `px.init.zeros()`"
        ) from error
    except ValueError as error:
        raise BuildError(f"block {_name(block)}: {error}") from error

    numbers = np.issubdtype(drawn.dtype, np.number)
    if drawn.shape != block.shape or not numbers:
        got = (
            f"an array of shape {drawn.shape}"
            if numbers
            else "something that is not an array of numbers"
        )
        raise BuildError(
            f"the initializer of block {_name(block)} returned {got}, not weights of "
            f"the block's shape {block.shape}. An initializer of px.init is written "
            f"with its call, as in `px.init.zeros()`"
        )
    return drawn.astype(np.float32).reshape(-1)


def materialize[GS](
    record: Record,
    net: type[Network[GS]],
    key: PRNGKeyArray,
    *,
    globals_: GS,
    check_cycle: bool = True,
    capacity_headroom: float = 0.0,
) -> tuple[NetworkStatic, NetworkState[GS]]:
    """Turn a traced topology into a state that `px.make_step` can run.

    Record ids are the state's unit ids, unchanged. A step's `inputs` holds the
    record's inputs flattened in row-major order, one after another in the order
    they were given to `trace`; its outputs are the handles the topology returned,
    flattened and joined the same way.

    A constant unit starts at its value and keeps it, because nothing feeds it. That
    holds under topological propagation only.

    Each block's weights are drawn by its initializer, `lecun_normal()` when it has
    none, from a key derived from `key` and the block's path. The same key and the
    same topology give the same weights in every run, and adding a layer does not
    change the weights of the others. The fans an initializer is given are counted
    over the whole record: the fan-in leaves out connections from constant units, so
    a bias does not count, and the fan-out counts every connection.

    Two connections between the same pair of units, written by two blocks, become two
    connections, each with its own weight.

    A `BuildError` is raised if a connection leads into a constant unit, if the
    connections close a cycle and `check_cycle` is on, or if an initializer cannot be
    called or does not return weights of its block's shape.

    Args:
        record: The traced topology.
        net: The network class, as today's builder takes it: its policies and fields.
        key: The key the weights are drawn from.
        globals_: The network's globals value.
        check_cycle: Whether to refuse a record whose connections close a cycle,
            naming the blocks involved. Off, the check is left to the builder, which
            refuses a cycle for a topological network and accepts one for a pipeline
            network.
        capacity_headroom: Extra room for connections, as `from_edges` takes it.

    Returns:
        The `(static, state)` pair, as `NetworkBuilder.from_edges` returns it.
    """
    if record.blocks:
        src = np.concatenate([block.src for block in record.blocks])
        dst = np.concatenate([block.dst for block in record.blocks])
    else:
        src = dst = np.zeros(0, np.int32)
    _check(record, src, dst, check_cycle)

    activation = np.full(record.num_units, ACTIVATION.default, ACTIVATION.dtype)
    for units in record.unit_blocks:
        if units.constant is not None:
            size = int(np.prod(units.shape, dtype=np.int64))
            activation[units.start : units.start + size] = units.constant

    counted = ~np.isin(src, record.constants)
    in_degree = np.bincount(dst[counted], minlength=record.num_units)
    out_degree = np.bincount(src, minlength=record.num_units)
    drawn = [_weights(block, key, in_degree, out_degree) for block in record.blocks]
    weights = np.concatenate([np.zeros(0, np.float32), *drawn])

    inputs = [ids.reshape(-1) for ids in record.inputs.values()]
    outputs = [ids.reshape(-1) for ids in record.outputs]
    return NetworkBuilder.from_edges(
        net,
        record.num_units,
        src,
        dst,
        weights=weights,
        input_ids=tuple(int(i) for ids in inputs for i in ids),
        output_ids=tuple(int(i) for ids in outputs for i in ids),
        globals_=globals_,
        extra_unit_columns={ACTIVATION.name: activation},
        capacity_headroom=capacity_headroom,
    )
