"""Multi-CONTROLLER Driver structural events: overflow -> grow, and resort.

sharding_driver_equiv.py pins these under single-controller shard_map, where one
process owns every device. This is the true multi-controller case: N genuinely
separate processes launched via jax.distributed (gloo on CPU), one device each --
the local stand-in for a one-process-per-node Narval run.

It shows the "multi-controller resort/overflow unhandled" gap is closed. Once
`distribute_state` makes the state a global jax.Array, the Driver's eager
host-side transforms -- grow_bucket's pad-and-retrace, resort's recompute +
compacting scatter + sort -- execute as collective SPMD ops: their scalar reads
(`live_conn_count`, overflow) reduce globally, and their vector reads (resort's
per-level live histogram) come back replicated because the reduced axis is not
the sharded one. So the grown / resorted arena is byte-identical to single-device
AND a subsequent sharded step runs correctly on the resorted state.

Launcher/worker re-exec; each worker forces one CPU device so the mesh spans the
processes. Prints MC DRIVER EQUIVALENCE PASS.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path

N_SHARDS = 4
_LAYERS = (33, 128, 128, 10)
_BUDGETS = (512, 512, 128)


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
    from jax.experimental import multihost_utils
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
    from dst_sparse import _one_hot, _sample, build_sparse_mlp, make_net, teacher_task
    from mlp_xor import GradPreAct

    import plastax as px
    from plastax import topo

    opt = px.optim.adam(0.05, GradPreAct)
    classes = _LAYERS[-1]
    train_net = make_net(opt, method="set", mode="train")

    static, base = build_sparse_mlp(train_net, _LAYERS, _BUDGETS, seed=0)
    static_s = dataclasses.replace(static, sharding=px.ShardSpec("shard", N_SHARDS))
    mesh = px.scheme_a_mesh(static_s)
    repl = NamedSharding(mesh, P())

    def log(msg: str) -> None:
        if pid == 0:
            print(msg, flush=True)

    def _copy(state: px.NetworkState[None]) -> px.NetworkState[None]:
        return jax.tree_util.tree_map(lambda x: jnp.array(x), state)

    def _gather(x: jax.Array) -> np.ndarray:
        """Sharded/replicated global array -> full host array on every process."""
        return np.asarray(jax.jit(lambda a: a, out_shardings=repl)(x))

    def _edge_set(state: px.NetworkState[None], sharded: bool) -> set[tuple[int, int]]:
        pairs: set[tuple[int, int]] = set()
        for b in state.conns:
            dead = _gather(b["dead"]) if sharded else np.asarray(b["dead"])
            fr = _gather(b["from_id"]) if sharded else np.asarray(b["from_id"])
            to = _gather(b["to_id"]) if sharded else np.asarray(b["to_id"])
            live = ~dead
            pairs.update(zip(fr[live].tolist(), to[live].tolist(), strict=True))
        return pairs

    def _conns_match(
        single: px.NetworkState[None], sharded: px.NetworkState[None]
    ) -> bool:
        for bkt_s, bkt_h in zip(single.conns, sharded.conns, strict=True):
            for name in bkt_s:
                if not np.allclose(
                    np.asarray(bkt_s[name]), _gather(bkt_h[name]), atol=1e-5
                ):
                    return False
        return True

    # Warm single-device so the churn/resort operate on a realistic state.
    teacher, rng = teacher_task(_LAYERS[0] - 1, classes, 0)
    step_ref = px.make_step(train_net, static)
    warmed = base
    for _ in range(6):
        inp, label = _sample(teacher, rng)
        warmed = step_ref(
            warmed, px.StepInputs(inputs=inp, targets=_one_hot(label, classes))
        ).state
    warmed = _copy(warmed)

    # --- overflow -> grow_bucket -> retrace --------------------------------
    def run_growth(
        st: px.NetworkStatic, state: px.NetworkState[None]
    ) -> px.Driver[None]:
        # grow-heavy but bounded: a couple of grow_bucket rounds exercise the
        # retrace path without many recompiles (mirrors sharding_driver_equiv).
        churn = make_net(opt, method="set", mode="churn", zeta=0.1, max_candidates=256)
        drv = px.Driver(churn, st, state)
        sp = px.StepInputs(inputs=jnp.zeros((_LAYERS[0],), jnp.float32), targets=None)
        for _ in range(3):
            drv.step(sp)
        return drv

    initial_caps = tuple(static.level_capacities)
    s_drv = run_growth(static, _copy(warmed))
    s_live = int(px.state.live_conn_count(s_drv.state))
    s_caps = tuple(s_drv.static.level_capacities)
    s_edges = _edge_set(s_drv.state, sharded=False)

    h_drv = run_growth(static_s, px.distribute_state(static_s, _copy(warmed)))
    h_live = int(px.state.live_conn_count(h_drv.state))
    h_caps = tuple(h_drv.static.level_capacities)
    h_edges = _edge_set(h_drv.state, sharded=True)

    if s_caps == initial_caps:
        raise AssertionError(
            f"overflow: no grow_bucket fired (caps stayed {initial_caps}); "
            "the scenario no longer exercises the overflow path -- retune it"
        )
    if s_live != h_live:
        raise AssertionError(f"overflow: live single={s_live} multi={h_live}")
    if s_caps != h_caps:
        raise AssertionError(f"overflow: caps single={s_caps} multi={h_caps}")
    if s_edges != h_edges:
        raise AssertionError("overflow: grown edge set differs multi vs single")
    lives = np.asarray(
        multihost_utils.process_allgather(jnp.array([h_live], jnp.int32))
    ).ravel()
    if not bool((lives == h_live).all()):
        raise AssertionError(f"overflow: live disagrees across processes: {lives}")
    log(f"OK overflow -> grow_bucket -> retrace (live {h_live}, caps {h_caps})")

    # --- topo.resort, then a sharded step on the resorted state ------------
    rin, rlabel = _sample(teacher, rng)
    si_r = px.StepInputs(inputs=rin, targets=_one_hot(rlabel, classes))

    s_rs = px.make_step(train_net, static)(_copy(warmed), si_r).state
    r_static, r_state = topo.resort(static, s_rs)

    step_s = px.make_step(train_net, static_s)
    h_rs = step_s(px.distribute_state(static_s, _copy(warmed)), si_r).state
    h_static, h_state = topo.resort(static_s, h_rs)

    if r_static.level_capacities != h_static.level_capacities:
        raise AssertionError(
            f"resort: caps single={r_static.level_capacities} "
            f"multi={h_static.level_capacities}"
        )
    if not np.array_equal(
        np.asarray(r_state.units[px.LEVEL.name]), _gather(h_state.units[px.LEVEL.name])
    ):
        raise AssertionError("resort: recomputed levels differ multi vs single")
    if _edge_set(r_state, sharded=False) != _edge_set(h_state, sharded=True):
        raise AssertionError("resort: edge set differs multi vs single")
    if not _conns_match(r_state, h_state):
        raise AssertionError("resort: conn columns differ multi vs single")

    # The resorted global state must be usable by a subsequent sharded step.
    post_single = px.make_step(train_net, static)(r_state, si_r)
    post_multi = step_s(h_state, si_r)
    if not np.allclose(
        np.asarray(post_single.loss), _gather(post_multi.loss), atol=1e-5
    ):
        raise AssertionError("resort: post-resort sharded step loss differs")
    log("OK resort (caps + levels + conns) and a sharded step on the result")

    if pid == 0:
        print("MC DRIVER EQUIVALENCE PASS", flush=True)
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
