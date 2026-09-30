"""Per-phase cost of one churn step on a synthetic three-layer sparse net (GPU).

Three layers of `--width` units; `--edges` live edges split evenly over the two
source-level buckets. Each step prunes about `--k` edges per bucket (a hash of
the pair and the step) and grows up to `--k` per bucket, so the live count
stays near `--edges`. Four net variants share one built state, so the phase
costs difference out: forward only, + prune, + add, and the full churn step.

Run on a GPU venv (see TOOLING.md), e.g.:

    XLA_PYTHON_CLIENT_PREALLOCATE=false .venv-gpu/bin/python \\
        examples/benchmarks/churn_probe.py --width 158114 --edges 50000000
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import plastax as px

Globals = dict[str, jax.Array]


def hash01(a: jax.Array, b: jax.Array, c: jax.Array) -> jax.Array:
    """Stateless integer hash of three int scalars to a float in [0, 1)."""
    h = (a.astype(jnp.uint32) + jnp.uint32(0x9E3779B1)) * jnp.uint32(0x85EBCA77)
    h = (h ^ b.astype(jnp.uint32)) * jnp.uint32(0xC2B2AE3D)
    h = (h ^ c.astype(jnp.uint32)) * jnp.uint32(0x27D4EB2F)
    h = h ^ (h >> 15)
    h = h * jnp.uint32(0x2C1B3C6D)
    h = h ^ (h >> 13)
    return (h >> jnp.uint32(8)).astype(jnp.float32) / jnp.float32(1 << 24)


class TanhForward(px.ForwardPass):
    combine = px.monoid.sum_

    def map(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: Globals,
    ) -> jax.Array:
        del dst, g
        return c[px.WEIGHT, cid] * u[px.ACTIVATION, src]

    def apply(
        self, u: px.UnitView, i: px.UnitIdx, g: Globals, acc: jax.Array
    ) -> px.UnitWrite:
        del u, i, g
        return px.UnitWrite.of((px.ACTIVATION, jnp.tanh(acc)))


class HashPrune(px.PruneConn):
    def __init__(self, kill_p: float) -> None:
        self.kill_p = kill_p

    def predicate(
        self, u: px.UnitView, c: px.ConnView, cid: px.ConnIdx, g: Globals
    ) -> jax.Array:
        del u
        h = hash01(c[px.FROM_ID, cid], c[px.TO_ID, cid], g["step"])
        return h < jnp.float32(self.kill_p)


class HashGrow(px.AddConn):
    """Grid growth over a per-level top-M shortlist with a random score."""

    shortlist_per_level = True

    def __init__(self, k: int, m: int) -> None:
        self.max_candidates = k
        self.max_candidate_units = m

    def importance(self, u: px.UnitView, i: px.UnitIdx, g: Globals) -> jax.Array:
        del u
        return hash01(i, g["step"], jnp.int32(7))

    def score(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: Globals
    ) -> jax.Array:
        deeper = u[px.LEVEL, dst] > u[px.LEVEL, src]
        return jnp.where(deeper, hash01(src, dst, g["step"]), -jnp.inf)

    def init(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: Globals
    ) -> px.ConnWrite:
        del u, src, dst, g
        return px.ConnWrite.of((px.WEIGHT, jnp.float32(0.01)))


class ProposeGrow(px.ProposeAddConn):
    """Uniform proposals: 4k per bucket, a random (src, dst) between layers."""

    def __init__(self, k: int, width: int, *, dedupe: bool) -> None:
        self.max_candidates = k
        self.num_proposals = 2 * 4 * k  # two buckets, 4x oversampled
        self.width = width
        self.dedupe = dedupe

    def propose(
        self, u: px.UnitView, j: jax.Array, g: Globals
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        del u
        layer = j % 2
        w = jnp.float32(self.width)
        src = layer * self.width + (hash01(j, g["step"], jnp.int32(1)) * w).astype(
            jnp.int32
        )
        dst = (layer + 1) * self.width + (
            hash01(j, g["step"], jnp.int32(2)) * w
        ).astype(jnp.int32)
        return src, dst, hash01(j, g["step"], jnp.int32(3))

    def init(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: Globals
    ) -> px.ConnWrite:
        del u, src, dst, g
        return px.ConnWrite.of((px.WEIGHT, jnp.float32(0.01)))


class Tick(px.ResetGlobal):
    def reset(self, g: Globals) -> Globals:
        return {"step": g["step"] + 1}


def make_net(
    *, prune: px.PruneConn | None, add: px.AddConn | px.ProposeAddConn | None
) -> type[px.Network[Globals]]:
    """A churn-net variant; every variant shares the same field layout."""

    class Net(px.Network[Globals]):
        forward_pass = TanhForward()
        prune_conn = prune
        add_conn = add
        reset_global = Tick()
        propagation = px.Propagation.TOPOLOGICAL

    return Net


def random_layers(
    width: int, edges: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Distinct random (src, dst) pairs, half between each adjacent layer pair."""
    src, dst = [], []
    per = edges // 2
    for layer in range(2):
        ids = np.unique(rng.integers(0, width * width, int(per * 1.3), np.int64))
        ids = rng.permutation(ids)[:per]
        src.append(layer * width + ids // width)
        dst.append((layer + 1) * width + ids % width)
    return np.concatenate(src).astype(np.int32), np.concatenate(dst).astype(np.int32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--width", type=int, default=16384)
    ap.add_argument("--edges", type=int, default=5_400_000)
    ap.add_argument("--k", type=int, default=64, help="churned edges per bucket")
    ap.add_argument("--m", type=int, default=64, help="shortlist side")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--headroom", type=float, default=0.05)
    ap.add_argument("--grow", choices=("grid", "propose"), default="grid")
    ap.add_argument("--dedupe", action="store_true", help="propose: exact dedupe")
    ap.add_argument("--json", help="append one JSON line of results here")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    frm, to = random_layers(args.width, args.edges, rng)
    prune = HashPrune(args.k / (args.edges / 2))
    grow: px.AddConn | px.ProposeAddConn = (
        HashGrow(args.k, args.m)
        if args.grow == "grid"
        else ProposeGrow(args.k, args.width, dedupe=args.dedupe)
    )
    full = make_net(prune=prune, add=grow)
    static, state0 = px.NetworkBuilder.from_edges(
        full,
        3 * args.width,
        frm,
        to,
        weights=(rng.standard_normal(frm.shape[0]) * 0.05).astype(np.float32),
        input_ids=list(range(args.width)),
        output_ids=list(range(2 * args.width, 3 * args.width)),
        globals_={"step": jnp.int32(0)},
        capacity_headroom=args.headroom,
    )
    del frm, to
    state_gb = sum(a.nbytes for a in jax.tree.leaves(state0)) / 1e9
    print(
        f"width={args.width} edges={args.edges} k={args.k} m={args.m} "
        f"grow={args.grow}{'+dedupe' if args.dedupe else ''} "
        f"caps={static.level_capacities} state={state_gb:.2f} GB",
        flush=True,
    )
    x = jnp.asarray(rng.standard_normal(args.width).astype(np.float32))
    inputs = px.StepInputs(inputs=x, targets=None)

    def run(net: type[px.Network[Globals]]) -> float:
        step = px.make_step(net, static)
        state = jax.tree.map(lambda a: a.copy(), state0)
        state = step(state, inputs).state  # compile + one warm step
        jax.block_until_ready(state)
        t0 = time.perf_counter()
        for _ in range(args.steps):
            state = step(state, inputs).state
        jax.block_until_ready(state)
        return (time.perf_counter() - t0) / args.steps * 1e3

    variants: dict[str, type[px.Network[Globals]]] = {
        "fwd": make_net(prune=None, add=None),
        "fwd+prune": make_net(prune=prune, add=None),
        "fwd+add": make_net(prune=None, add=grow),
        "churn": full,
    }
    result: dict[str, Any] = {**vars(args), "state_gb": state_gb}
    for name, net in variants.items():
        result[name] = run(net)
        print(f"  {name:10s} {result[name]:9.3f} ms/step", flush=True)
    if args.json:
        with open(args.json, "a") as f:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
