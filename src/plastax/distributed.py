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

`distribute_state` takes a fully-materialised state and slices each leaf to the
mesh. The construction path (`NetworkBuilder.from_edges(..., sharding=)`) instead
builds only each process's shard directly and uses the same placement helpers
here, so no process ever materialises the full arena.
"""

from __future__ import annotations

from typing import Any, cast

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from plastax._types import ShardSpec
from plastax.state import Columns, NetworkState, NetworkStatic


def _mesh_for_spec(spec: ShardSpec) -> Mesh:
    """Build the 1-D device mesh for a ShardSpec: the first `num_shards` devices.

    `jax.devices()` returns every device across every process, so the mesh is
    identical under a single controller and under multi-controller.
    """
    devices = jax.devices()
    if len(devices) < spec.num_shards:
        raise ValueError(
            f"scheme_a_mesh: need >= {spec.num_shards} devices for "
            f"{spec.num_shards}-way sharding, got {len(devices)}"
        )
    return Mesh(np.asarray(devices[: spec.num_shards]), (spec.axis_name,))


def _shardings_for_spec(
    spec: ShardSpec,
) -> tuple[Mesh, NamedSharding, NamedSharding]:
    """Return (mesh, conn sharding, replicated sharding) for a ShardSpec.

    The connection sharding partitions the capacity axis over `axis_name`; the
    replicated sharding places a full copy on every device. Both match
    `make_step`'s `shard_map` in_specs.
    """
    mesh = _mesh_for_spec(spec)
    # PartitionSpec is untyped in jax's stubs (mirrors step._shard_map_step).
    conn_pspec: Any = PartitionSpec(spec.axis_name)  # type: ignore[no-untyped-call]
    repl_pspec: Any = PartitionSpec()  # type: ignore[no-untyped-call]
    return mesh, NamedSharding(mesh, conn_pspec), NamedSharding(mesh, repl_pspec)


def _place(
    host: np.ndarray | jax.Array,
    sharding: NamedSharding,
    global_shape: tuple[int, ...],
) -> jax.Array:
    """Assemble a global `jax.Array` from this process's host-local data.

    `host` is the data for this process's addressable shards: the full array
    when `global_shape == host.shape` (each device then slices its own band or
    takes the whole copy), or just this process's window when it is smaller.
    """
    arr = np.asarray(host)
    placed: Any = jax.make_array_from_process_local_data(  # type: ignore[no-untyped-call]
        sharding, arr, global_shape
    )
    return cast(jax.Array, placed)


def _addressable_window(sharding: NamedSharding, size: int) -> tuple[int, int]:
    """The contiguous ``[lo, hi)`` band of a sharded axis this process supplies.

    For the 1-D capacity-axis sharding each process owns a contiguous band of
    positions (its local devices' shards). Returns that band so the construction
    path can materialise only it. Raises if the addressable shards are not
    contiguous (an unsupported device ordering).
    """
    idx_map = sharding.addressable_devices_indices_map((size,))
    spans: list[tuple[int, int]] = []
    for index in idx_map.values():
        if index is None:
            raise ValueError("distribute: sharding has no addressable index band")
        axis = index[0]
        spans.append((axis.start or 0, size if axis.stop is None else axis.stop))
    spans.sort()
    cursor = spans[0][0]
    for start, stop in spans:
        if start != cursor:
            raise ValueError(
                "distribute: addressable shards are non-contiguous; per-shard "
                "construction needs a contiguous device band per process"
            )
        cursor = stop
    return spans[0][0], cursor


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
    if static.sharding is None:
        raise ValueError("scheme_a_mesh: static.sharding is None (not a Scheme-A net)")
    return _mesh_for_spec(static.sharding)


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
    (same seed) run independently per process. To avoid every process
    materialising the full arenas, build with `NetworkBuilder.from_edges(...,
    sharding=)` instead, which places only each process's shard.

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

    _, conn_sharding, repl_sharding = _shardings_for_spec(sharding)
    for level, capacity in enumerate(static.level_capacities):
        if capacity % sharding.num_shards != 0:
            raise ValueError(
                f"distribute_state: bucket {level} capacity {capacity} is not "
                f"divisible by num_shards {sharding.num_shards}"
            )

    def place(leaf: jax.Array, leaf_sharding: NamedSharding) -> jax.Array:
        return _place(leaf, leaf_sharding, np.asarray(leaf).shape)

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
