"""Time one growth point three ways, for the px vs cx GPU gap analysis.

Builds one point of ``growth_bench.py`` (same network and rules), compiles
the add_conn phase, runs ``GAP_WARM`` warm-up calls (default 30: px needs
about 15 calls to reach steady state, see benchmarks/gpu_gap_analysis.md),
then reports, over ``reps`` calls:

- ``synced``: ``perf_counter`` around each call plus ``block_until_ready``,
  the growth bench's method (host dispatch + device + sync);
- ``dispatch``: the time for the call to return, without a sync;
- ``pipelined``: ``max(reps, 50)`` calls back to back and one sync at the end,
  divided by the call count (device time, with dispatch overlapped).

``GAP_SORT`` picks how `plastax.phases.total_order` sorts: ``auto`` (the
default: the library's own choice), ``comparison`` (always the four-key
comparison sort) or ``radix`` (always the three stable one-key passes, which
XLA lowers to CUB radix sorts on a GPU). Both give the same total order; the
printed digest (a hash of every connection column of every bucket after the
last call) lets two runs be compared. ``GAP_NVTX=1`` then
runs 10 more synced calls inside NVTX ranges named ``grow_call``, for
``nsys profile -t cuda,nvtx --cuda-graph-trace=node`` (needs the ``nvtx``
package).

Usage (GPU venv):

    XLA_PYTHON_CLIENT_PREALLOCATE=false \\
    XLA_FLAGS=--xla_disable_hlo_passes=constant_folding \\
        python examples/benchmarks/gpu_gap_probe.py per_unit 65536 65536 4 31
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import plastax as px
import plastax.phases as phases
from plastax.phases import build_add_conn_phase

sys.path.insert(0, str(Path(__file__).resolve().parent))
import growth_bench as gb  # noqa: E402


def main() -> None:
    strategy = sys.argv[1]
    n, c, p = (int(a) for a in sys.argv[2:5])
    reps = int(sys.argv[5]) if len(sys.argv) > 5 else 21
    sort = os.environ.get("GAP_SORT", "auto")
    if sort != "auto":
        phases.RADIX_TOTAL_ORDER_MIN = {"radix": 0, "comparison": 1 << 62}[sort]
    width = n // gb.LEVELS
    net = gb.make_net(gb.make_rule(strategy, p, width))
    frm, to = gb.layered_edges(n, c)
    static, state = px.NetworkBuilder.from_edges(
        net,
        n,
        frm,
        to,
        weights=np.full(frm.shape, 0.5, np.float32),
        input_ids=list(range(width)),
        output_ids=list(range(n - width, n)),
        globals_={},
        capacity_headroom=0.05,
    )
    phase = build_add_conn_phase(
        net, static, growth=os.environ.get("GAP_GROWTH", "xla")
    )
    inputs = px.StepInputs(inputs=jnp.zeros((width,), jnp.float32), targets=None)

    def call(st: px.NetworkState[Any]) -> px.NetworkState[Any]:
        new, _ = phase(st, inputs)
        return dataclasses.replace(new, step=new.step + 1)

    compiled = jax.jit(call, donate_argnums=0).lower(state).compile()
    for _ in range(int(os.environ.get("GAP_WARM", "30"))):
        state = jax.block_until_ready(compiled(state))
    synced = []
    for _ in range(reps):
        t0 = time.perf_counter()
        state = jax.block_until_ready(compiled(state))
        synced.append((time.perf_counter() - t0) * 1e3)
    dispatch = []
    for _ in range(reps):
        jax.block_until_ready(state)
        t0 = time.perf_counter()
        state = compiled(state)
        dispatch.append((time.perf_counter() - t0) * 1e3)
    jax.block_until_ready(state)
    calls = max(reps, 50)
    t0 = time.perf_counter()
    for _ in range(calls):
        state = compiled(state)
    jax.block_until_ready(state)
    pipelined = (time.perf_counter() - t0) * 1e3 / calls
    h = hashlib.sha256()
    for bucket in state.conns:
        for name in sorted(bucket):
            h.update(np.asarray(bucket[name]).tobytes())
    digest = h.hexdigest()[:16]
    print(
        f"RESULT sort={sort} {strategy} N={n} C={c} P={p} "
        f"synced_med_ms={statistics.median(synced):.4f} "
        f"dispatch_med_ms={statistics.median(dispatch):.4f} "
        f"pipelined_ms={pipelined:.4f} grown={int(state.grown)} digest={digest}"
    )
    if os.environ.get("GAP_NVTX"):
        import nvtx

        for _ in range(10):
            with nvtx.annotate("grow_call"):
                state = jax.block_until_ready(compiled(state))


if __name__ == "__main__":
    main()
