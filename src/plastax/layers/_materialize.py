"""Materialization: a build record becomes a state that can be stepped.

A bridge onto today's `Network` class and today's builder, so that a topology written
as a `Module` runs before the policy redesign lands. It is replaced when
`Network.init` is built, and does as little as it can: it checks the record, gives
constants their value, draws each block's weights, and hands whole arrays to
`NetworkBuilder.from_edges`.

Plan: `plans/20261119T201041Z-step6-materialize.md` at the workspace root.
"""

from __future__ import annotations

import dataclasses
import zlib

import jax
import numpy as np
from jaxtyping import PRNGKeyArray

from plastax import init as px_init
from plastax import topo
from plastax._types import ACTIVATION, Capacity, Propagation
from plastax.builder import BlockSpan, NetworkBuilder
from plastax.init import Fans
from plastax.layers._module import BuildError, _show
from plastax.layers._record import ConnBlock, Record
from plastax.network import Network
from plastax.state import NetworkState, NetworkStatic


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


def _spans[GS](
    record: Record, net: type[Network[GS]], src: np.ndarray, dst: np.ndarray
) -> list[BlockSpan] | None:
    """Give the builder the record's blocks, or None when it cannot keep them.

    A network that grows connections of its own puts a new connection in any
    free slot of its bucket and may re-sort its buckets, so its blocks do not
    stay where they were put. It is built as one sorted bucket per level, with
    no layout, and a capacity on one of its layers is refused.

    Args:
        record: The record.
        net: The network class.
        src: The source of every connection, block after block.
        dst: The destination of every connection, block after block.

    Returns:
        One span per block of the record, in order, or None.

    Raises:
        BuildError: If a layer is given a number as its capacity and the network
            grows connections of its own, the layer has no connections, or its
            sources are on more than one level.
    """
    counted = [b for b in record.blocks if not isinstance(b.capacity, Capacity)]
    grow = net.resolved().grow_conn
    if grow is not None:
        if counted:
            block = counted[0]
            raise BuildError(
                f"{_show(block.path)} is given capacity={block.capacity}, and "
                f"{net.__name__} grows connections of its own "
                f"({type(grow).__name__}), which may take any free slot. A layer's "
                f"own slots beside the network's growth are not supported yet "
                f"(layer policies, stage 5); leave the capacity out"
            )
        return None

    if counted and net.propagation is not Propagation.PIPELINE:
        # A run of slots is in one bucket, and a bucket holds the connections
        # whose sources are on one level.
        levels = topo.initial_levels(
            record.num_units,
            np.stack([src, dst], axis=1),
            allow_cycles=False,
            input_ids=tuple(
                int(i) for ids in record.inputs.values() for i in ids.reshape(-1)
            ),
        )
        for block in counted:
            on = np.unique(levels[block.src])
            if on.size == 0:
                raise BuildError(
                    f"{_show(block.path)} is given capacity={block.capacity} and "
                    f"creates no connections, so there is no level to put its "
                    f"slots on; leave the capacity out"
                )
            if on.size > 1:
                raise BuildError(
                    f"{_show(block.path)} is given capacity={block.capacity}, and "
                    f"its sources are on {on.size} levels "
                    f"({', '.join(str(int(level)) for level in on)}). A capacity "
                    f"is one run of slots, and a layer with sources on several "
                    f"levels has one run for each; how a capacity is divided "
                    f"between them is not decided yet. Leave the capacity out, "
                    f"or give each level's sources a layer of its own"
                )
    return [
        BlockSpan(
            int(block.src.size),
            None if isinstance(block.capacity, Capacity) else int(block.capacity),
        )
        for block in record.blocks
    ]


def materialize[GS](
    record: Record,
    net: type[Network[GS]],
    key: PRNGKeyArray,
    *,
    globals_: GS,
    check_cycle: bool = True,
    capacity_headroom: float = 0.0,
    capacity_align: int | None = None,
) -> tuple[NetworkStatic, NetworkState[GS]]:
    """Turn a traced topology into a state that `px.make_step` can run.

    Record ids are the state's unit ids, unchanged. A record with one input declares
    that input's shape as `static.input_shape`, and a step's `inputs` are given in
    it; likewise one returned handle and `static.output_shape`, for a step's
    `targets` and outputs. With several inputs, a step's `inputs` is one flat vector:
    the inputs flattened in row-major order, one after another in the order they
    were given to `trace`. Several returned handles are flattened and joined the
    same way.

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

    The connections are stored block by block. In each bucket (the connections whose
    sources are on one level) a block has one run of slots, as long as its capacity,
    with its connections sorted by source and then destination; the runs follow one
    another in the order the blocks were written, and the rest of the bucket is
    spare. `static.layout` records the runs of every block, and the unit ids of every
    block of units. A network that grows connections of its own is the exception: it
    is stored as one sorted bucket per level and has no layout.

    A `BuildError` is raised if a connection leads into a constant unit, if the
    connections close a cycle and `check_cycle` is on, if an initializer cannot be
    called or does not return weights of its block's shape, or if a layer is given a
    number as its capacity and it cannot be honoured: the network grows connections
    of its own, the layer has no connections, or its sources are on several levels.

    Args:
        record: The traced topology.
        net: The network class, as today's builder takes it: its policies and fields.
        key: The key the weights are drawn from.
        globals_: The network's globals value.
        check_cycle: Whether to refuse a record whose connections close a cycle,
            naming the blocks involved. Off, the check is left to the builder, which
            refuses a cycle for a topological network and accepts one for a pipeline
            network.
        capacity_headroom: Extra room in each bucket, beyond its blocks, as a
            fraction of them (`from_edges`).
        capacity_align: What a bucket's capacity is rounded up to a multiple of,
            or None for the next power of two (`from_edges`).

    Returns:
        The `(static, state)` pair, as `NetworkBuilder.from_edges` returns it.
    """
    if record.blocks:
        src = np.concatenate([block.src for block in record.blocks])
        dst = np.concatenate([block.dst for block in record.blocks])
    else:
        src = dst = np.zeros(0, np.int32)
    _check(record, src, dst, check_cycle)
    spans = _spans(record, net, src, dst)

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
    static, state = NetworkBuilder.from_edges(
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
        capacity_align=capacity_align,
        blocks=spans,
    )
    shapes = [ids.shape for ids in record.inputs.values()]
    returned = [ids.shape for ids in record.outputs]
    layout = static.layout
    if layout is not None:
        unit_runs = tuple(
            (units.start, units.start + int(np.prod(units.shape, dtype=np.int64)))
            for units in record.unit_blocks
        )
        layout = dataclasses.replace(layout, unit_runs=unit_runs)
    static = dataclasses.replace(
        static,
        input_shape=shapes[0] if len(shapes) == 1 else None,
        output_shape=returned[0] if len(returned) == 1 else None,
        layout=layout,
    )
    return static, state
