"""Compile plastax steps for a TPU without a TPU: ahead-of-time, from a topology.

libtpu (the `tpu` extra) compiles against a topology description, so every
step type can be lowered and compiled for TPU v4 / v5e / v5p / v6e on any host.
This checks that each path lowers on TPU and reports XLA's memory and cost
analysis plus the scatter/gather/sort ops in the optimized HLO. Nothing runs:
timings need hardware.

    UV_PROJECT_ENVIRONMENT=.venv-tpu uv sync --extra tpu
    JAX_PLATFORMS=cpu .venv-tpu/bin/python examples/benchmarks/tpu_aot_check.py \\
        --topology v5p:2x2x1 --width 16384 --edges 5400000

Topologies need a whole host (e.g. v5e:2x2, v6e:2x2, v5p:2x2x1, v4:2x2x1).
"""

from __future__ import annotations

import argparse
import collections
import re
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import topologies
from jax.sharding import SingleDeviceSharding

import plastax as px

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import churn_probe as cp  # noqa: E402
import mlp_xor  # noqa: E402


def compile_report(label: str, step: Any, state: Any, inputs: Any, sh: Any) -> None:
    """Compile one step for the target device and print its analysis."""

    def to_sds(tree: Any) -> Any:
        return jax.tree.map(
            lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype, sharding=sh), tree
        )

    try:
        compiled = step.trace(to_sds(state), to_sds(inputs)).lower().compile()
    except Exception as e:  # noqa: BLE001 - report every failure, keep going
        print(f"{label:30s} FAILED {type(e).__name__}: {str(e)[:240]}")
        return
    mem = compiled.memory_analysis()
    cost = compiled.cost_analysis() or {}
    cost = cost[0] if isinstance(cost, list) else cost
    hlo = compiled.as_text()
    ops = collections.Counter(re.findall(r"= \S+ (sort|scatter|gather|while)\(", hlo))
    print(
        f"{label:30s} OK  args {mem.argument_size_in_bytes / 1e6:8.1f} MB  "
        f"temp {mem.temp_size_in_bytes / 1e6:8.1f} MB  "
        f"est. bytes {cost.get('bytes accessed', 0) / 1e9:7.1f} GB  {dict(ops)}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--topology", default="v5p:2x2x1")
    ap.add_argument("--width", type=int, default=16384)
    ap.add_argument("--edges", type=int, default=5_400_000)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    topo = topologies.get_topology_desc(topology_name=args.topology, platform="tpu")
    dev = topo.devices[0]
    sh = SingleDeviceSharding(dev)
    print(
        f"target {dev.device_kind} ({args.topology}); width {args.width}, "
        f"edges {args.edges}"
    )
    rng = np.random.default_rng(0)
    w, n = args.width, 3 * args.width
    frm, to = cp.random_layers(w, args.edges, rng)
    weights = np.full(frm.size, 0.01, np.float32)
    io = {"input_ids": list(range(w)), "output_ids": list(range(2 * w, 3 * w))}

    prune = cp.HashPrune(64 / (args.edges / 2))
    for name, add in (
        ("churn, proposal growth", cp.ProposeGrow(64, w, dedupe=False)),
        ("churn, proposal + dedupe", cp.ProposeGrow(64, w, dedupe=True)),
        ("churn, grid growth", cp.HashGrow(64, 64)),
    ):
        net = cp.make_net(prune=prune, add=add)
        static, state = px.NetworkBuilder.from_edges(
            net,
            n,
            frm,
            to,
            weights=weights,
            globals_={"step": jnp.int32(0)},
            capacity_headroom=0.05,
            capacity_align=256,
            **io,
        )
        x = px.StepInputs(inputs=jnp.zeros((w,), jnp.float32), targets=None)
        compile_report(name, px.make_step(net, static), state, x, sh)

    opt = px.optim.adam(0.01, mlp_xor.GradPreAct)

    class MLP(px.Network[None]):
        forward_pass = mlp_xor.SigmoidForward()
        backward_pass = mlp_xor.SigmoidBackward()
        loss = mlp_xor.MSELoss()
        update_conn = opt.update_conn()
        extra_unit_fields = (mlp_xor.GradPreAct, mlp_xor.LossGrad)
        extra_conn_fields = opt.state_fields
        propagation = px.Propagation.TOPOLOGICAL
        batch_reduction = px.MeanFloatFirstRest()

    static, state = px.NetworkBuilder.from_edges(
        MLP, n, frm, to, weights=weights, globals_=None, capacity_align=256, **io
    )
    b = args.batch
    xb = px.StepInputs(
        inputs=jnp.zeros((b, w), jnp.float32), targets=jnp.zeros((b, w), jnp.float32)
    )
    x1 = px.StepInputs(
        inputs=jnp.zeros((w,), jnp.float32), targets=jnp.zeros((w,), jnp.float32)
    )
    compile_report("train adam, streaming", px.make_step(MLP, static), state, x1, sh)
    for layout in ("auto", "edge_list", "csr"):
        step = px.make_step(MLP, static, batch_size=b, layout=layout)
        compile_report(f"train adam, B={b} {layout}", step, state, xb, sh)


if __name__ == "__main__":
    main()
