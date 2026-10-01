"""One `width x width` sparse layer in each JAX layout: forward and rebuild (GPU).

Layouts: COO `segment_sum` (plastax's edge list, unsorted and sorted), BCOO and
BCSR from `jax.experimental.sparse` (cuSPARSE with `jax_bcoo_cusparse_lowering`
on, which it is here -- without it both fall back to generic kernels 30-80x
slower), and a dense matmul when the dense matrix fits. Each forward is timed at
batch 1 and batch `--batch`. The rebuild column is the edge-list -> CSR cost:
a single-key radix sort by row, then gathers and a bincount/cumsum indptr.

    XLA_PYTHON_CLIENT_PREALLOCATE=false .venv-gpu/bin/python \\
        examples/benchmarks/layouts_probe.py --width 158114 --nnz 25000000
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse as jsp

jax.config.update("jax_bcoo_cusparse_lowering", True)


def timed(f: Callable[..., jax.Array], *a: object, n: int = 20) -> float:
    """Mean ms per call after one warm-up (compile) call."""
    jax.block_until_ready(f(*a))
    t0 = time.perf_counter()
    for _ in range(n):
        out = f(*a)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / n * 1e3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--width", type=int, default=16384)
    ap.add_argument("--nnz", type=int, default=2_700_000)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()
    width, nnz, batch = args.width, args.nnz, args.batch

    rng = np.random.default_rng(0)
    ids = np.unique(rng.integers(0, width * width, int(nnz * 1.3), np.int64))
    ids = rng.permutation(ids)[:nnz]
    row = jnp.asarray((ids // width).astype(np.int32))  # destination
    col = jnp.asarray((ids % width).astype(np.int32))  # source
    val = jnp.asarray(rng.standard_normal(nnz).astype(np.float32))
    x = jnp.asarray(rng.standard_normal(width).astype(np.float32))
    xb = jnp.asarray(rng.standard_normal((width, batch)).astype(np.float32))
    del ids
    print(f"width={width} nnz={nnz} density={nnz / width / width:.2e}")

    def line(name: str, one: float, many: float, extra: str = "") -> None:
        print(f"  {name:26s} B=1 {one:8.3f} ms  B={batch} {many:9.3f} ms  {extra}")

    def coo(r: jax.Array, c: jax.Array, v: jax.Array, x: jax.Array) -> jax.Array:
        return jax.ops.segment_sum(v * x[c], r, width)

    def coo_b(r: jax.Array, c: jax.Array, v: jax.Array, x: jax.Array) -> jax.Array:
        return jax.ops.segment_sum(v[:, None] * x[c], r, width)

    line(
        "COO segment_sum unsorted",
        timed(jax.jit(coo), row, col, val, x),
        timed(jax.jit(coo_b), row, col, val, xb),
    )

    @jax.jit
    def rebuild(
        r: jax.Array, c: jax.Array, v: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        _, perm = jax.lax.sort_key_val(r, jnp.arange(r.shape[0], dtype=jnp.int32))
        rs = r[perm]
        indptr = jnp.concatenate(
            [jnp.zeros(1, jnp.int32), jnp.cumsum(jnp.bincount(rs, length=width))]
        ).astype(jnp.int32)
        return rs, c[perm], v[perm], indptr

    rebuild_ms = timed(rebuild, row, col, val, n=5)
    rs, cs, vs, indptr = rebuild(row, col, val)

    def coo_s(r: jax.Array, c: jax.Array, v: jax.Array, x: jax.Array) -> jax.Array:
        return jax.ops.segment_sum(v * x[c], r, width, indices_are_sorted=True)

    line(
        "COO segment_sum sorted",
        timed(jax.jit(coo_s), rs, cs, vs, x),
        timed(jax.jit(coo_b), rs, cs, vs, xb),
        f"(rebuild {rebuild_ms:.3f} ms)",
    )

    def matvec(a: jsp.JAXSparse | jax.Array, x: jax.Array) -> jax.Array:
        return a @ x

    mv = jax.jit(matvec)
    bcoo = jsp.BCOO(
        (vs, jnp.stack([rs, cs], 1)),
        shape=(width, width),
        indices_sorted=True,
        unique_indices=True,
    )
    line("BCOO @", timed(mv, bcoo, x), timed(mv, bcoo, xb))
    bcsr = jsp.BCSR(
        (vs, cs, indptr), shape=(width, width), indices_sorted=True, unique_indices=True
    )
    cusparse = "cusparse" in jax.jit(matvec).lower(bcsr, x).compile().as_text().lower()
    line(
        "BCSR @",
        timed(mv, bcsr, x),
        timed(mv, bcsr, xb),
        f"(cuSPARSE: {cusparse})",
    )
    err = float(jnp.max(jnp.abs(mv(bcsr, x) - coo(row, col, val, x))))
    print(f"  BCSR vs COO max |diff| {err:.2e}")

    dense_gb = width * width * 4 / 1e9
    if dense_gb < 12:
        d = jnp.zeros((width, width), jnp.float32).at[row, col].set(val)
        line("dense matmul", timed(mv, d, x), timed(mv, d, xb), f"({dense_gb:.1f} GB)")
    else:
        print(f"  dense: {dense_gb:.0f} GB, does not fit")


if __name__ == "__main__":
    main()
