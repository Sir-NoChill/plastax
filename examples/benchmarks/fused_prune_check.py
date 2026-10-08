"""Check plastax's fused forward + prune Triton kernel against the two-pass step.

Needs an NVIDIA GPU and the `triton` extra. Builds a layered net whose forward
marks this step's scheduled units and whose prune kills their edges (the
plastax-cpp synthetic benchmark's churn step), or a hashed-edge prune, grows
replacements, and runs `make_step(fuse_prune="auto")` (one Triton kernel per bucket) and
`fuse_prune="off"` side by side over many churn steps: every tombstone, slot
claim and integer column must match exactly, floats to summation order. Then
times both steps.

    .venv-gpu/bin/python examples/benchmarks/fused_prune_check.py --edges 5000000
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import plastax as px
from plastax import phases

G = dict[str, jax.Array]
MARKED = px.FieldSpec.int32("check/marked")


def _hash01(a: jax.Array, b: jax.Array, c: jax.Array) -> jax.Array:
    h = (a.astype(jnp.uint32) + jnp.uint32(0x9E3779B1)) * jnp.uint32(0x85EBCA77)
    h = (h ^ b.astype(jnp.uint32)) * jnp.uint32(0xC2B2AE3D)
    h = (h ^ c.astype(jnp.uint32)) * jnp.uint32(0x27D4EB2F)
    h = h ^ (h >> 15)
    return (h >> jnp.uint32(8)).astype(jnp.float32) / jnp.float32(1 << 24)


class Forward(px.ForwardPass):
    """Weighted sum; also flags the units on this step's prune row."""

    combine = px.monoid.sum_
    linear_input = px.ACTIVATION

    def map(self, u: Any, dst: Any, src: Any, c: Any, cid: Any, g: G) -> jax.Array:
        del dst, g
        return c[px.WEIGHT, cid] * u[px.ACTIVATION, src]

    def apply(self, u: Any, i: Any, g: G, acc: jax.Array) -> px.UnitWrite:
        del u
        ids = g["prune"][g["step"] % g["prune"].shape[0]]
        pos = jnp.minimum(
            jnp.searchsorted(ids, i, method="scan_unrolled"), ids.shape[0] - 1
        )
        return px.UnitWrite.of(
            (px.ACTIVATION, acc), (MARKED, (ids[pos] == i).astype(jnp.int32))
        )


class MarkedPrune(px.PruneConn):
    """Kill every edge touching a unit the forward marked."""

    def predicate(self, u: Any, c: Any, cid: Any, g: G) -> jax.Array:
        del g
        return (u[MARKED, c[px.FROM_ID, cid]] | u[MARKED, c[px.TO_ID, cid]]) == 1


class HashPrune(px.PruneConn):
    """Kill a hashed ~1% of edges (uint32 arithmetic, shifts, a float compare)."""

    def predicate(self, u: Any, c: Any, cid: Any, g: G) -> jax.Array:
        del u
        return _hash01(c[px.FROM_ID, cid], c[px.TO_ID, cid], g["step"]) < 0.01


class Propose(px.ProposeAddConn):
    """Grow `num_proposals` random next-layer edges per step."""

    def __init__(self, num_units: int, width: int, num_proposals: int) -> None:
        self.num_units, self.width = num_units, width
        self.num_proposals = self.max_candidates = num_proposals

    def propose(self, u: Any, j: jax.Array, g: G) -> tuple[Any, Any, Any]:
        src = (
            _hash01(j, g["step"], jnp.int32(1)) * (self.num_units - self.width)
        ).astype(jnp.int32)
        off = (_hash01(j, g["step"], jnp.int32(2)) * self.width).astype(jnp.int32)
        dst = (src // self.width + 1) * self.width + off
        return src, dst, jnp.float32(0.0)

    def init(self, u: Any, src: Any, dst: Any, g: G) -> px.ConnWrite:
        del u
        return px.ConnWrite.of((px.WEIGHT, _hash01(src, dst, g["step"]) - 0.5))


class Tick(px.ResetGlobal):
    def reset(self, g: G) -> G:
        return {**g, "step": g["step"] + 1}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--edges", type=int, default=5_000_000)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--predicate", choices=("marked", "hash"), default="marked")
    args = ap.parse_args()
    if not phases.nvidia_triton_available():
        raise SystemExit("needs an NVIDIA GPU with jax_triton installed")
    rng = np.random.default_rng(0)
    nl = args.layers
    per = args.edges // (nl - 1)
    width = int(np.sqrt(per / 0.001))  # density 0.1 % per layer pair
    n = width * nl
    frm = np.concatenate(
        [k * width + rng.integers(0, width, per) for k in range(nl - 1)]
    ).astype(np.int32)
    to = np.concatenate(
        [(k + 1) * width + rng.integers(0, width, per) for k in range(nl - 1)]
    ).astype(np.int32)
    rows = np.sort(rng.integers(width, n, (64, 64)), axis=1).astype(np.int32)
    prune = MarkedPrune() if args.predicate == "marked" else HashPrune()

    class Net(px.Network[G]):
        forward_pass = Forward()
        prune_conn = prune
        add_conn = Propose(n, width, 4096)
        reset_global = Tick()
        extra_unit_fields = (MARKED,)

    static, state = px.NetworkBuilder.from_edges(
        Net, n, frm, to,
        weights=rng.standard_normal(frm.size).astype(np.float32),
        input_ids=list(range(width)),
        output_ids=list(range((nl - 1) * width, n)),
        globals_={"step": jnp.int32(0), "prune": jnp.asarray(rows)},
        capacity_headroom=0.05, capacity_align=256,
    )  # fmt: skip
    del frm, to
    fused = px.make_step(Net, static, fuse_prune="auto")
    plain = px.make_step(Net, static, fuse_prune="off")
    a = jax.tree.map(jnp.copy, state)
    b = state
    x = jnp.asarray(rng.standard_normal(width).astype(np.float32))
    inp = px.StepInputs(inputs=x, targets=None)
    for _ in range(args.steps):
        a, b = fused(a, inp).state, plain(b, inp).state
    plan = fused.prune_fusion.plan  # type: ignore[attr-defined]
    print(f"plan: {plan}")
    for (path, la), lb in zip(
        jax.tree_util.tree_flatten_with_path(a)[0], jax.tree.leaves(b), strict=True
    ):
        la, lb = np.asarray(la), np.asarray(lb)
        if np.issubdtype(la.dtype, np.floating):
            ok = np.allclose(la, lb, rtol=1e-4, atol=1e-4)
        else:
            ok = np.array_equal(la, lb)
        if not ok:
            raise SystemExit(f"MISMATCH at {jax.tree_util.keystr(path)}")
    print(f"OK: {args.steps} churn steps identical; live {px.state.live_conn_count(a)}")

    def timed(step: Any, st: Any) -> tuple[float, Any]:
        ts = []
        for _ in range(100):
            t0 = time.perf_counter()
            st = step(st, inp).state
            jax.block_until_ready(st)
            ts.append(time.perf_counter() - t0)
        return float(np.median(ts) * 1e3), st

    tf, a = timed(fused, a)
    tp, b = timed(plain, b)
    print(f"step: fused {tf:.3f} ms, two-pass {tp:.3f} ms ({tp / tf:.2f}x)")


if __name__ == "__main__":
    main()
