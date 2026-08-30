"""Multi-controller (jax.distributed) placement for Scheme-A sharding.

Under a single controller, `make_step`'s `shard_map` slices a host-built state
implicitly: the whole state is one process's `jax.Array`, and XLA distributes
it across that process's local devices. Under true multi-controller --
`jax.distributed`, one process per node/device-group -- no single process holds
every device, so the state cannot be fed to the sharded step as host-local
data. It must first be assembled into a GLOBAL `jax.Array` spanning every
process's devices; `distribute_state` does that assembly, and `scheme_a_mesh`
exposes the same device mesh the step wraps its `shard_map` around.

Once distributed, the state behaves as one global array: the sharded step's
per-shard collectives all-reduce across processes (NCCL on GPU, gloo on CPU),
and the driver's host-side scalar reads (`live_conn_count`, `overflow`,
`needs_resort`) reduce globally, so every process's Python control flow agrees.
"""

from __future__ import annotations

from typing import Any, cast

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from plastax.state import Columns, NetworkState, NetworkStatic


def scheme_a_mesh(static: NetworkStatic) -> Mesh:
    """Build the Scheme-A device mesh: the first `num_shards` global devices.

    `jax.devices()` returns every device across every process, so this mesh is
    identical under a single controller and under multi-controller; it is the
    same mesh `make_step` wraps the sharded step around, kept here as the single
    source of truth for both placement and the step.

    Args:
        static: static config whose ShardSpec names the axis and shard count.

    Returns:
        A 1-D device mesh over `static.sharding.num_shards` devices, with the
        axis named by `static.sharding.axis_name`.

    Raises:
        ValueError: if `static` has no sharding, or fewer than `num_shards`
            devices are visible.
    """
    sharding = static.sharding
    if sharding is None:
        raise ValueError("scheme_a_mesh: static.sharding is None (not a Scheme-A net)")
    devices = jax.devices()
    if len(devices) < sharding.num_shards:
        raise ValueError(
            f"scheme_a_mesh: need >= {sharding.num_shards} devices for "
            f"{sharding.num_shards}-way sharding, got {len(devices)}"
        )
    return Mesh(np.asarray(devices[: sharding.num_shards]), (sharding.axis_name,))


def distribute_state[GS](
    static: NetworkStatic, state: NetworkState[GS]
) -> NetworkState[GS]:
    """Place a host-built state onto the Scheme-A mesh as a global `jax.Array`.

    Matches `make_step`'s `shard_map` in_specs: connection columns are sharded
    on their capacity axis (`PartitionSpec(axis_name)`); units, user globals,
    and `needs_resort` are replicated on every device. Returns `state`
    unchanged when `static.sharding` is None (single device).

    Every leaf is supplied to `make_array_from_process_local_data` as its full
    array with `global_shape == local_data.shape`, so each device looks up its
    own slice: a sharded leaf's device gets its capacity-axis band, a replicated
    leaf's device gets the whole column. This requires the passed `state` to be
    identical on every process -- the intended source is a deterministic build
    (same seed) run independently per process. (Building only each process's own
    shard, to avoid every process materializing the full arenas, is a separate
    memory-scaling extension; this placement is the correctness path.)

    Type Args:
        GS: the user's global-state pytree, opaque to the framework.

    Args:
        static: static config carrying the ShardSpec; its `level_capacities`
            must each be divisible by the shard count (Scheme-A capacities are
            powers of two, so this holds for shard counts in {1, 2, 4, ...}).
        state: host-built network state, identical across processes.

    Returns:
        The state with every leaf replaced by a global `jax.Array` sharded or
        replicated to match the sharded step; ready to pass to `make_step`.

    Raises:
        ValueError: if a bucket capacity is not divisible by the shard count.
    """
    sharding = static.sharding
    if sharding is None:
        return state

    mesh = scheme_a_mesh(static)
    # PartitionSpec is untyped in jax's stubs (mirrors step._shard_map_step).
    conn_pspec: Any = PartitionSpec(sharding.axis_name)  # type: ignore[no-untyped-call]
    repl_pspec: Any = PartitionSpec()  # type: ignore[no-untyped-call]
    conn_sharding = NamedSharding(mesh, conn_pspec)
    repl_sharding = NamedSharding(mesh, repl_pspec)

    for level, capacity in enumerate(static.level_capacities):
        if capacity % sharding.num_shards != 0:
            raise ValueError(
                f"distribute_state: bucket {level} capacity {capacity} is not "
                f"divisible by num_shards {sharding.num_shards}"
            )

    def place(leaf: jax.Array, leaf_sharding: NamedSharding) -> jax.Array:
        full = np.asarray(leaf)
        placed: Any = jax.make_array_from_process_local_data(  # type: ignore[no-untyped-call]
            leaf_sharding, full, full.shape
        )
        return cast(jax.Array, placed)

    units: Columns = {
        name: place(col, repl_sharding) for name, col in state.units.items()
    }
    conns: tuple[Columns, ...] = tuple(
        {name: place(col, conn_sharding) for name, col in bucket.items()}
        for bucket in state.conns
    )
    globals_ = jax.tree_util.tree_map(
        lambda leaf: place(leaf, repl_sharding), state.globals_
    )
    needs_resort = place(state.needs_resort, repl_sharding)

    return NetworkState(
        units=units, conns=conns, globals_=globals_, needs_resort=needs_resort
    )
