"""Sparse tanh reservoir on Scheme-A multi-GPU sharding.

A feedforward (DAG) sparse tanh network whose connections are sharded
edge-wise across the device mesh while units are replicated -- Scheme A. This
is the feedforward precursor to a recurrent Echo State Network: plastax's
builder currently levels the graph with Kahn's algorithm and rejects cycles,
so a truly recurrent reservoir does not build yet (a cyclic-graph builder mode
is the natural next step). Run directly: `python examples/sharded_reservoir.py`.
It uses four fake CPU devices so it runs anywhere; on a real multi-GPU host the
same ShardSpec shards across the physical devices.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax.numpy as jnp
import numpy as np

import plastax as px
from plastax.views import UnitWrite

N_INPUTS = 4
N_HIDDEN = 48
N_OUTPUT = 8
N_SHARDS = 4


class _TanhForward(px.ForwardPass):
    """Layer update: pre-activation is a weighted sum, apply is tanh."""

    combine = px.monoid.sum_

    def map(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: None,
    ) -> jnp.ndarray:
        return c[px.WEIGHT, cid] * u[px.ACTIVATION, src]

    def apply(
        self, u: px.UnitView, i: px.UnitIdx, g: None, acc: jnp.ndarray
    ) -> UnitWrite:
        return UnitWrite.of((px.ACTIVATION, jnp.tanh(acc)))


class ShardedReservoir(px.Network[None]):
    """Feedforward sparse tanh reservoir, sharded edge-wise across the mesh."""

    forward_pass = _TanhForward()
    propagation = px.Propagation.TOPOLOGICAL
    sharding = px.ShardSpec("shard", N_SHARDS)


def _connect(
    b: px.NetworkBuilder[None],
    rng: np.random.Generator,
    src: range,
    dst: range,
    density: float,
) -> None:
    scale = 1.0 / np.sqrt(max(1.0, density * len(src)))
    for s in src:
        for d in dst:
            if rng.random() < density:
                b.add_conn(s, d, weight=float(scale * rng.standard_normal()))


def _build() -> tuple[px.NetworkStatic, px.NetworkState[None]]:
    rng = np.random.default_rng(0)
    b = px.NetworkBuilder(ShardedReservoir, None)
    for _ in range(N_INPUTS + N_HIDDEN + N_OUTPUT):
        b.add_unit()
    inputs = range(N_INPUTS)
    hidden = range(N_INPUTS, N_INPUTS + N_HIDDEN)
    outputs = range(N_INPUTS + N_HIDDEN, N_INPUTS + N_HIDDEN + N_OUTPUT)
    for i in inputs:
        b.mark_input(i)
    for o in outputs:
        b.mark_output(o)
    _connect(b, rng, inputs, hidden, density=1.0)
    _connect(b, rng, hidden, outputs, density=0.25)
    return b.finalize()


def main() -> None:
    """Drive the sharded reservoir for a few steps and print its output norm."""
    static, state = _build()
    step = px.make_step(ShardedReservoir, static)
    for t in range(5):
        drive = jnp.asarray(np.sin(0.4 * t + np.arange(N_INPUTS)), jnp.float32)
        result = step(state, px.StepInputs(inputs=drive, targets=None))
        state = result.state
        norm = float(jnp.linalg.norm(state.units[px.ACTIVATION.name]))
        print(f"step {t}: activation norm = {norm:.4f}")


if __name__ == "__main__":
    main()
