"""Standalone Scheme-A equivalence check (no jaxtyping instrumentation).

Run directly (`python tests/sharding_equiv.py`) or via the subprocess wrapper
in test_sharding.py, and usable to validate on real multi-GPU too. Kept out of
pytest's jaxtyping instrumentation on purpose: shard_map reconstructs the
registered NetworkState pytree with placeholder leaves internally, which
beartype rejects -- a test-only artifact, not a production issue.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import dataclasses

import jax
import jax.numpy as jnp

import plastax as px
from plastax.views import UnitWrite

N_SHARDS = 4
_EDGES = (
    (0, 2, 0.5),
    (1, 2, -0.25),
    (2, 3, 2.0),
    (3, 4, 0.3),
    (2, 4, -1.0),
    (4, 5, 1.5),
)


class _SumForward(px.ForwardPass):
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
        return UnitWrite.of((px.ACTIVATION, acc))


class _PipelineNet(px.Network[None]):
    forward_pass = _SumForward()
    propagation = px.Propagation.PIPELINE


class _TopoNet(px.Network[None]):
    forward_pass = _SumForward()
    propagation = px.Propagation.TOPOLOGICAL


def _build(
    net: type[px.Network[None]],
) -> tuple[px.NetworkStatic, px.NetworkState[None]]:
    b = px.NetworkBuilder(net, None)
    for _ in range(6):
        b.add_unit()
    b.mark_input(0)
    b.mark_input(1)
    b.mark_output(5)
    for s, d, w in _EDGES:
        b.add_conn(s, d, weight=w)
    return b.finalize()


def check(net: type[px.Network[None]]) -> None:
    """Assert the sharded step matches the single-device step for `net`."""
    inputs = px.StepInputs(inputs=jnp.asarray([1.0, 2.0], jnp.float32), targets=None)
    static, state = _build(net)
    single = px.make_step(net, static)(state, inputs)

    static_s, state_s = _build(net)
    static_s = dataclasses.replace(static_s, sharding=px.ShardSpec("shard", N_SHARDS))
    sharded = px.make_step(net, static_s)(state_s, inputs)

    for name in (px.ACTIVATION.name, px.LEVEL.name):
        if not bool(
            jnp.array_equal(
                jnp.asarray(single.state.units[name]),
                jnp.asarray(sharded.state.units[name]),
            )
        ):
            raise AssertionError(f"{net.__name__}: unit field {name!r} differs")
    if not bool(jnp.allclose(jnp.asarray(single.loss), jnp.asarray(sharded.loss))):
        raise AssertionError(f"{net.__name__}: loss differs")


def main() -> None:
    """Run the equivalence check on every propagation mode."""
    if len(jax.devices()) < N_SHARDS:
        raise SystemExit(f"need >= {N_SHARDS} devices, got {len(jax.devices())}")
    for net in (_PipelineNet, _TopoNet):
        check(net)
        print(f"OK {net.__name__}")
    print("SHARDING EQUIVALENCE PASS")


if __name__ == "__main__":
    main()
