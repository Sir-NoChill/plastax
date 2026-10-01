"""Check and time plastax's jax_triton edge kernel against the XLA product (GPU).

Needs an NVIDIA GPU and the `triton` extra:

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
    ap.add_argument("--batches", default="1,8,32,128")
    args = ap.parse_args()
    if not phases.nvidia_triton_available():
        raise SystemExit("needs an NVIDIA GPU with jax_triton installed")
    rng = np.random.default_rng(0)
    n, e = args.units, args.edges
    bucket = {
        FROM_ID.name: jnp.asarray(np.sort(rng.integers(0, n, e, dtype=np.int32))),
        TO_ID.name: jnp.asarray(rng.integers(0, n, e, dtype=np.int32)),
        WEIGHT.name: jnp.asarray(rng.standard_normal(e).astype(np.float32)),
        DEAD.name: jnp.asarray(rng.random(e) < 0.05),
    }

    def timed(f: object, *a: object) -> tuple[float, jax.Array]:
        r = f(*a)  # type: ignore[operator]
        jax.block_until_ready(r)
        t0 = time.perf_counter()
        for _ in range(20):
            r = f(*a)  # type: ignore[operator]
        jax.block_until_ready(r)
        return (time.perf_counter() - t0) / 20 * 1e3, r

    for b in (int(v) for v in args.batches.split(",")):
        x = jnp.asarray(rng.standard_normal((n, b)).astype(np.float32))
        tri = jax.jit(
            lambda bk, x: phases.triton_bucket_product(
                bk, x, n, rows=TO_ID, cols=FROM_ID
            )
        )
        xla = jax.jit(
            lambda bk, x: phases.xla_bucket_product(bk, x, n, rows=TO_ID, cols=FROM_ID)
        )
        tt, rt = timed(tri, bucket, x)
        tx, rx = timed(xla, bucket, x)
        err = float(jnp.max(jnp.abs(rt - rx)))
        print(
            f"edges={e} B={b:4d}: triton {tt:8.3f} ms  xla {tx:8.3f} ms "
            f"({tx / tt:.2f}x)  max|diff| {err:.1e}"
        )


if __name__ == "__main__":
    main()
