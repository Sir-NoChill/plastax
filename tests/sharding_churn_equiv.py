"""Scheme-A across the DST phases: train and prune shard, growth does not.

Checked in a clean subprocess (no jaxtyping instrumentation -- shard_map is
incompatible with it; see test_sharding.py). Complements the forward-only
sharding_equiv.py by covering the phases a dynamic-sparse run actually uses:

  * a full TRAIN step (forward + loss + backward + adam update_conn) under
    Scheme-A must match the single-device step numerically, incl. every
    per-edge optimizer column -- this is the path the memory/perf timing runs
    execute on a static sharded topology;
  * a magnitude PRUNE step shards too (edge-local tombstone);
  * ADD_CONN growth does NOT shard: build_add_conn_phase's prefix-sum slot
    claim mixes the static full bucket capacity with the per-shard capacity
    slice, so it raises under shard_map. Pinned here so the limitation is
    explicit and can't regress silently -- and it is exactly why the scaling
    experiment times a static sharded topology and characterizes churn
    (rewiring) single-node.

Run directly (`python tests/sharding_churn_equiv.py`) or via the subprocess
wrapper in test_sharding_churn.py; usable on real multi-GPU too.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import dataclasses
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

# The DST example is the real churn/train code the experiment uses; load it by
# path (examples/ is not importable by default), matching how the acceptance
# tests exercise examples.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

from dst_sparse import (
    _EXTRA_UNIT_FIELDS,
    MagnitudeStats,
    SetPrune,
    _one_hot,
    _sample,
    build_sparse_mlp,
    make_net,
    teacher_task,
)
from mlp_xor import GradPreAct

import plastax as px

N_SHARDS = 4
_LAYERS = (17, 64, 4)
_BUDGETS = (256, 64)  # bucket capacities 256 and 64 -- both divisible by N_SHARDS
_OPT = px.optim.adam(0.05, GradPreAct)


def _copy(state: px.NetworkState[None]) -> px.NetworkState[None]:
    """Independent copy so one step's donation can't consume the other run's."""
    return jax.tree_util.tree_map(lambda x: jnp.array(x), state)


def _conns_allclose(a: px.NetworkState[None], b: px.NetworkState[None]) -> bool:
    """Every conn column (from/to/dead/weight + optimizer state) matches."""
    for bucket_a, bucket_b in zip(a.conns, b.conns, strict=True):
        for name in bucket_a:
            if not bool(
                jnp.allclose(
                    jnp.asarray(bucket_a[name]), jnp.asarray(bucket_b[name]), atol=1e-5
                )
            ):
                return False
    return True


def _check_train_step_shards(
    static: px.NetworkStatic, static_s: px.NetworkStatic, state: px.NetworkState[None]
) -> None:
    """A full train step must be numerically identical sharded vs single."""
    train_net = make_net(_OPT, method="set", mode="train")
    teacher, rng = teacher_task(_LAYERS[0] - 1, _LAYERS[-1], 0)
    inp, label = _sample(teacher, rng)
    si = px.StepInputs(inputs=inp, targets=_one_hot(label, _LAYERS[-1]))

    single = px.make_step(train_net, static)(_copy(state), si)
    sharded = px.make_step(train_net, static_s)(_copy(state), si)

    if not bool(
        jnp.allclose(
            single.state.units[px.ACTIVATION.name],
            sharded.state.units[px.ACTIVATION.name],
            atol=1e-5,
        )
    ):
        raise AssertionError("train: activations differ sharded vs single")
    if not bool(jnp.allclose(single.loss, sharded.loss, atol=1e-5)):
        raise AssertionError("train: loss differs sharded vs single")
    if not _conns_allclose(single.state, sharded.state):
        raise AssertionError("train: conn columns (weights/opt state) differ")


def _check_prune_step_shards(
    static: px.NetworkStatic, static_s: px.NetworkStatic, state: px.NetworkState[None]
) -> None:
    """Magnitude prune is edge-local, so it must shard identically."""

    class _PruneNet(px.Network[None]):
        forward_pass = MagnitudeStats(0.3)
        prune_conn = SetPrune()
        extra_unit_fields = _EXTRA_UNIT_FIELDS
        extra_conn_fields = _OPT.state_fields
        propagation = px.Propagation.TOPOLOGICAL

    sp = px.StepInputs(inputs=jnp.zeros((_LAYERS[0],), jnp.float32), targets=None)
    single = px.make_step(_PruneNet, static)(_copy(state), sp).state
    sharded = px.make_step(_PruneNet, static_s)(_copy(state), sp).state

    if int(px.state.live_conn_count(single)) != int(px.state.live_conn_count(sharded)):
        raise AssertionError("prune: live-edge count differs sharded vs single")
    if not _conns_allclose(single, sharded):
        raise AssertionError("prune: conn columns differ sharded vs single")


def _check_add_conn_does_not_shard(
    static_s: px.NetworkStatic, state: px.NetworkState[None]
) -> None:
    """add_conn growth is single-device-only; assert it raises under sharding."""
    churn_net = make_net(
        _OPT, method="set", mode="churn", zeta=0.3, max_candidates=max(_BUDGETS)
    )
    sp = px.StepInputs(inputs=jnp.zeros((_LAYERS[0],), jnp.float32), targets=None)
    try:
        px.make_step(churn_net, static_s)(_copy(state), sp)
    except ValueError:
        return  # expected: prefix-sum slot claim mismatches the sharded slice
    raise AssertionError(
        "add_conn under sharding did not raise -- the limitation changed; "
        "revisit whether growth is now shardable and update the docs/design"
    )


def main() -> None:
    """Run every DST-phase sharding check and print the pass sentinel."""
    if len(jax.devices()) < N_SHARDS:
        raise SystemExit(f"need >= {N_SHARDS} devices, got {len(jax.devices())}")

    train_net = make_net(_OPT, method="set", mode="train")
    static, state = build_sparse_mlp(train_net, _LAYERS, _BUDGETS, seed=0)
    static_s = dataclasses.replace(static, sharding=px.ShardSpec("shard", N_SHARDS))

    _check_train_step_shards(static, static_s, state)
    print("OK train step shards (forward + loss + backward + adam update_conn)")
    _check_prune_step_shards(static, static_s, state)
    print("OK prune step shards")
    _check_add_conn_does_not_shard(static_s, state)
    print("OK add_conn growth is single-device-only (raises under sharding)")
    print("CHURN SHARDING CHECK PASS")


if __name__ == "__main__":
    main()
