"""Multi-CONTROLLER per-shard construction: from_edges(sharding=) across procs.

`distribute_state` slices a fully-built state; this exercises the construction
path that never builds the full arena. N genuinely separate processes launched
via `jax.distributed` (gloo on CPU, one device each) each call
`from_edges(..., sharding=)`, which materialises ONLY that process's
capacity-axis band per bucket and assembles a global `jax.Array`.

Asserts, per process:
  * the per-process window is exactly `capacity / num_shards` -- the memory
    property (no process ever holds the full padded column);
  * the assembled global state is byte-identical to building the full state and
    slicing it with `distribute_state` (gathered back to host);
  * a sharded forward step on the per-shard-built state matches single-device.

Launcher/worker re-exec; each worker forces one CPU device. Prints
MC CONSTRUCT EQUIVALENCE PASS.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys

N_SHARDS = 4
_NUM_UNITS = 48


def _edges() -> tuple[list[int], list[int]]:
    """A deterministic 3-layer DAG: [0,16) -> [16,36) -> [36,48)."""
    import numpy as np

    rng = np.random.default_rng(0)
    froms: list[int] = []
    tos: list[int] = []
    for lo_s, hi_s, lo_d, hi_d in [(0, 16, 16, 36), (16, 36, 36, 48)]:
        for dst in range(lo_d, hi_d):
            for src in rng.choice(range(lo_s, hi_s), size=6, replace=False):
                froms.append(int(src))
                tos.append(dst)
    return froms, tos


def _worker(pid: int, port: int) -> None:
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

    import dataclasses

    import jax
    import jax.numpy as jnp
    import numpy as np

    jax.distributed.initialize(
        coordinator_address=f"localhost:{port}",
        num_processes=N_SHARDS,
        process_id=pid,
    )
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    import plastax as px
    from plastax.distributed import _addressable_window, _shardings_for_spec
    from plastax.views import UnitWrite

    class _SumForward(px.ForwardPass):
        combine = px.monoid.sum_

        def map(
            self,
            u: px.UnitView,
            dst: px.UnitIdx,
            src: px.UnitIdx,
            c: px.ConnView,
            cid: px.ConnIdx,
            g: None,
        ) -> jax.Array:
            return c[px.WEIGHT, cid] * u[px.ACTIVATION, src]

        def apply(
            self, u: px.UnitView, i: px.UnitIdx, g: None, acc: jax.Array
        ) -> UnitWrite:
            return UnitWrite.of((px.ACTIVATION, acc))

    class _Net(px.Network[None]):
        forward_pass = _SumForward()
        propagation = px.Propagation.TOPOLOGICAL

    froms, tos = _edges()
    from_ids = np.asarray(froms, np.int32)
    to_ids = np.asarray(tos, np.int32)
    weights = np.linspace(0.1, 1.0, len(froms)).astype(np.float32)
    shard = px.ShardSpec("shard", N_SHARDS)
    kw: dict[str, object] = {
        "input_ids": tuple(range(16)),
        "output_ids": tuple(range(36, 48)),
        "globals_": None,
    }

    def log(msg: str) -> None:
        if pid == 0:
            print(msg, flush=True)

    # full build + slice (reference) vs per-shard build (under test)
    static_f, state_f = px.NetworkBuilder.from_edges(
        _Net, _NUM_UNITS, from_ids, to_ids, weights=weights, **kw
    )
    ref = px.distribute_state(dataclasses.replace(static_f, sharding=shard), state_f)
    static_s, state_s = px.NetworkBuilder.from_edges(
        _Net, _NUM_UNITS, from_ids, to_ids, weights=weights, sharding=shard, **kw
    )

    if static_s.sharding != shard:
        raise AssertionError(f"sharding not set: {static_s.sharding}")
    if static_s.level_capacities != static_f.level_capacities:
        raise AssertionError("per-shard capacities differ from full build")

    # memory property: this process built only its cap/num_shards band.
    _, conn_sharding, _ = _shardings_for_spec(shard)
    for capacity in static_s.level_capacities:
        lo, hi = _addressable_window(conn_sharding, capacity)
        if hi - lo != capacity // N_SHARDS:
            raise AssertionError(
                f"process window {hi - lo} != cap/shards {capacity // N_SHARDS}"
            )
    log(f"OK per-process window == cap/{N_SHARDS} for every bucket")

    repl = NamedSharding(px.scheme_a_mesh(static_s), P())

    def gather(x: jax.Array) -> np.ndarray:
        return np.asarray(jax.jit(lambda a: a, out_shardings=repl)(x))

    for bucket_r, bucket_s in zip(ref.conns, state_s.conns, strict=True):
        for name in bucket_r:
            if not np.array_equal(gather(bucket_r[name]), gather(bucket_s[name])):
                raise AssertionError(f"conn column {name!r} differs per-shard vs full")
    for name in ref.units:
        if not np.array_equal(gather(ref.units[name]), gather(state_s.units[name])):
            raise AssertionError(f"unit column {name!r} differs per-shard vs full")
    log("OK per-shard global state == full build sliced (every column)")

    # the per-shard-built state runs a sharded step matching single-device.
    si = px.StepInputs(
        inputs=jnp.arange(16, dtype=jnp.float32) * 0.1 + 0.1, targets=None
    )
    single = px.make_step(_Net, static_f)(state_f, si)
    multi = px.make_step(_Net, static_s)(state_s, si)
    if not np.allclose(
        np.asarray(single.state.units[px.ACTIVATION.name]),
        gather(multi.state.units[px.ACTIVATION.name]),
        atol=1e-5,
    ):
        raise AssertionError("forward on per-shard-built state differs from single")
    log("OK sharded forward on the per-shard-built state matches single-device")

    if pid == 0:
        print("MC CONSTRUCT EQUIVALENCE PASS", flush=True)
    jax.distributed.shutdown()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return int(s.getsockname()[1])


def main() -> None:
    """Spawn N worker processes and exit non-zero if any fails."""
    port = _free_port()
    env = {**os.environ, "PLX_MC_PORT": str(port)}
    procs = [
        subprocess.Popen([sys.executable, __file__], env={**env, "PLX_MC_PID": str(i)})
        for i in range(N_SHARDS)
    ]
    codes = [p.wait() for p in procs]
    if any(c != 0 for c in codes):
        raise SystemExit(f"worker exit codes: {codes}")


if __name__ == "__main__":
    _pid = os.environ.get("PLX_MC_PID")
    if _pid is None:
        main()
    else:
        _worker(int(_pid), int(os.environ["PLX_MC_PORT"]))
