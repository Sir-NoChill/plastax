"""How the sort is written decides whether XLA uses a radix sort (GPU).

A multi-operand `lax.sort` over (row, col, perm) with two keys lowers to a
comparison sort; two stable single-key `sort_key_val` passes (col, then row)
give the same lexicographic order via radix sorts; a row-only single-key pass
is what a CSR rebuild needs.

    .venv-gpu/bin/python examples/benchmarks/sort_probe.py --width 158114 --n 25000000
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import jax
import jax.numpy as jnp


def timed(f: Callable[..., object], *a: object, n: int = 5) -> float:
    """Mean ms per call after one warm-up (compile) call."""
    jax.block_until_ready(f(*a))
    t0 = time.perf_counter()
    for _ in range(n):
        out = f(*a)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / n * 1e3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--width", type=int, default=158114)
    ap.add_argument("--n", type=int, default=25_000_000)
    args = ap.parse_args()
    n = args.n
    r = jax.random.randint(jax.random.PRNGKey(0), (n,), 0, args.width)
    c = jax.random.randint(jax.random.PRNGKey(1), (n,), 0, args.width)
    perm0 = jnp.arange(n, dtype=jnp.int32)

    @jax.jit
    def two_key(r: jax.Array, c: jax.Array) -> jax.Array:
        return jax.lax.sort((r, c, perm0), num_keys=2)[2]

    @jax.jit
    def two_pass(r: jax.Array, c: jax.Array) -> jax.Array:
        _, p = jax.lax.sort_key_val(c, perm0, is_stable=True)
        _, p = jax.lax.sort_key_val(r[p], p, is_stable=True)
        return p

    @jax.jit
    def row_only(r: jax.Array, c: jax.Array) -> jax.Array:
        del c
        return jax.lax.sort_key_val(r, perm0)[1]

    print(
        f"n={n}: two-key comparison sort {timed(two_key, r, c):.2f} ms | "
        f"two stable radix passes {timed(two_pass, r, c):.2f} ms | "
        f"row-only radix {timed(row_only, r, c):.2f} ms"
    )


if __name__ == "__main__":
    main()
