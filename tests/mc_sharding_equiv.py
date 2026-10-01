"""Multi-CONTROLLER Scheme-A equivalence: distribute_state across real processes.

Single-controller `shard_map` (sharding_churn_equiv.py) slices a host-built
state implicitly -- one process owns every device. This is the true
multi-controller case: N genuinely separate processes launched via
`jax.distributed` (gloo collectives on CPU), one device each, exactly the
one-process-per-node model a Narval multi-node run uses. It is the local
stand-in for that run.

Each worker builds the same net deterministically, assembles the state as a
GLOBAL `jax.Array` with `plastax.distribute_state`, and checks that:

  * a full TRAIN step, a SET CHURN step, and a RigL CHURN step all match a
    single-device reference (loss, activations, and every conn column, gathered
    back to host with a replicating reshard so sharded arrays are addressable);
  * the driver's host-side read `live_conn_count` -- an eager reduction over the
    sharded conn arenas -- returns the SAME value on every process
    (process_allgather), which is the property multi-controller control flow
    depends on: each process's Python decisions stay consistent.

Launcher/worker re-exec: `python tests/mc_sharding_equiv.py` spawns N workers;
each re-invokes this file with PLX_MC_PID set and forces one CPU device so the
mesh distributes one shard per process. Prints MC SHARDING EQUIVALENCE PASS.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path

N_SHARDS = 4
_LAYERS = (17, 64, 4)
_BUDGETS = (256, 64)  # bucket capacities, both divisible by N_SHARDS


def _worker(pid: int, port: int) -> None:
    # One CPU device per process, so the N-device mesh spans the N processes
    # (override any inherited --xla_force_host_platform_device_count). Must be
    # set before jax initialises its backend.
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

    opt = px.optim.adam(0.05, GradPreAct)
    classes = _LAYERS[-1]
    train_net = make_net(opt, method="set", mode="train")

    static, base = build_sparse_mlp(train_net, _LAYERS, _BUDGETS, seed=0)
    static_s = dataclasses.replace(static, sharding=px.ShardSpec("shard", N_SHARDS))
    mesh = px.scheme_a_mesh(static_s)
    repl = NamedSharding(mesh, P())

    def _copy(state: px.NetworkState[None]) -> px.NetworkState[None]:
        return jax.tree_util.tree_map(lambda x: jnp.array(x), state)

    def _gather(x: jax.Array) -> np.ndarray:
        """Sharded/replicated global array -> full host array on every process.

        A replicating reshard (out_shardings=repl) all-gathers the shards so the
        result is fully addressable; np.asarray of a still-sharded global array
        would raise. Replicated inputs pass through unchanged.
        """
        gathered = jax.jit(lambda a: a, out_shardings=repl)(x)
        return np.asarray(gathered)

    # Warm a few train steps single-device so RigL churn reads a real, non-zero
    # grad_pre_act; the sharded and single runs then start from this same state.
    teacher, rng = teacher_task(_LAYERS[0] - 1, classes, 0)
    seq = [_sample(teacher, rng) for _ in range(4)]
    step_ref = px.make_step(train_net, static)
    warmed = base
    for inp, label in seq[:-1]:
        warmed = step_ref(
            warmed, px.StepInputs(inputs=inp, targets=_one_hot(label, classes))
        ).state
    warmed = _copy(warmed)  # detach from the donated loop state

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

    def log(msg: str) -> None:
        if pid == 0:
            print(msg, flush=True)

    # --- TRAIN -------------------------------------------------------------
    si = px.StepInputs(inputs=seq[-1][0], targets=_one_hot(seq[-1][1], classes))
    single = px.make_step(train_net, static)(_copy(warmed), si)
    sharded = px.make_step(train_net, static_s)(
        px.distribute_state(static_s, _copy(warmed)), si
    )
    if not np.allclose(np.asarray(single.loss), _gather(sharded.loss), atol=1e-5):
        raise AssertionError("train: loss differs multi-controller vs single")
    if not np.allclose(
        np.asarray(single.state.units[px.ACTIVATION.name]),
        _gather(sharded.state.units[px.ACTIVATION.name]),
        atol=1e-5,
    ):
        raise AssertionError("train: activations differ multi-controller vs single")
    if not _conns_match(single.state, sharded.state):
        raise AssertionError("train: conn columns differ multi-controller vs single")
    log("OK train step (loss + activations + conn/opt columns)")

    # --- SET and RigL CHURN ------------------------------------------------
    sp = px.StepInputs(inputs=jnp.zeros((_LAYERS[0],), jnp.float32), targets=None)
    for method in ("set", "rigl"):
        churn_net = make_net(
            opt, method=method, mode="churn", zeta=0.3, max_candidates=max(_BUDGETS)
        )
        single = px.make_step(churn_net, static)(_copy(warmed), sp).state
        sharded = px.make_step(churn_net, static_s)(
            px.distribute_state(static_s, _copy(warmed)), sp
        ).state

        s_live = int(px.state.live_conn_count(single))
        h_live = int(px.state.live_conn_count(sharded))
        if s_live != h_live:
            raise AssertionError(
                f"churn:{method}: live differs single={s_live} sharded={h_live}"
            )
        if not _conns_match(single, sharded):
            raise AssertionError(f"churn:{method}: conn columns differ vs single")

        # The host read must be identical on EVERY process, not just correct on
        # process 0 -- this is what keeps multi-controller Python control flow
        # consistent across processes.
        lives = np.asarray(
            multihost_utils.process_allgather(jnp.array([h_live], jnp.int32))
        ).ravel()
        if not bool((lives == h_live).all()):
            raise AssertionError(
                f"churn:{method}: live_conn_count disagrees across processes: {lives}"
            )
        log(f"OK churn:{method}: live {h_live} conserved, agrees across procs")

    if pid == 0:
        print("MC SHARDING EQUIVALENCE PASS", flush=True)
    jax.distributed.shutdown()


def _free_port() -> int:
    """Grab an ephemeral port for the coordinator (closed before workers bind)."""
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
