"""Check and time plastax's jax_triton edge kernel against the XLA product (GPU).

Both directions: the forward (accumulate into TO_ID, random within the
source-major bucket) and the backward (into FROM_ID, which comes in runs).
`--dead` is the tombstoned fraction of slots. Needs an NVIDIA GPU and the
`triton` extra:

    .venv-gpu/bin/python examples/benchmarks/triton_check.py --edges 25000000
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from plastax import phases
from plastax._types import DEAD, FROM_ID, TO_ID, WEIGHT


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--units", type=int, default=316228)
    ap.add_argument("--edges", type=int, default=25_000_000)
    ap.add_argument("--batches", default="1,2,8,32,128")
    ap.add_argument("--dead", type=float, default=0.05)
    ap.add_argument("--reps", type=int, default=50)
    args = ap.parse_args()
    if not phases.nvidia_triton_available():
        raise SystemExit("needs an NVIDIA GPU with jax_triton installed")
    rng = np.random.default_rng(0)
    n, e = args.units, args.edges
    bucket = {
        FROM_ID.name: jnp.asarray(np.sort(rng.integers(0, n, e, dtype=np.int32))),
        TO_ID.name: jnp.asarray(rng.integers(0, n, e, dtype=np.int32)),
        WEIGHT.name: jnp.asarray(rng.standard_normal(e).astype(np.float32)),
        DEAD.name: jnp.asarray(rng.random(e) < args.dead),
    }

    def timed(f: object, *a: object) -> tuple[float, jax.Array]:
        """Median over 7 rounds of the mean ms per call over `--reps` calls."""
        r = f(*a)  # type: ignore[operator]
        jax.block_until_ready(r)
        rounds = []
        for _ in range(7):
            t0 = time.perf_counter()
            for _ in range(args.reps):
                r = f(*a)  # type: ignore[operator]
            jax.block_until_ready(r)
            rounds.append((time.perf_counter() - t0) / args.reps * 1e3)
        return float(np.median(rounds)), r

    directions = (("forward", TO_ID, FROM_ID), ("backward", FROM_ID, TO_ID))
    for b in (int(v) for v in args.batches.split(",")):
        x = jnp.asarray(rng.standard_normal((n, b)).astype(np.float32))
        for name, rows, cols in directions:
            tri = jax.jit(
                lambda bk, x, rows=rows, cols=cols: phases.triton_bucket_product(
                    bk, x, n, rows=rows, cols=cols
                )
            )
            xla = jax.jit(
                lambda bk, x, rows=rows, cols=cols: phases.xla_bucket_product(
                    bk, x, n, rows=rows, cols=cols
                )
            )
            tt, rt = timed(tri, bucket, x)
            tx, rx = timed(xla, bucket, x)
            err = float(jnp.max(jnp.abs(rt - rx)) / jnp.max(jnp.abs(rx)))
            print(
                f"edges={e} dead={args.dead:.2f} {name:8s} B={b:4d}: "
                f"triton {tt:8.3f} ms  xla {tx:8.3f} ms ({tx / tt:.2f}x)  "
                f"max rel diff {err:.1e}",
                flush=True,
            )


if __name__ == "__main__":
    main()
