"""Scheme-A shards every DST phase: train, prune, and churn (growth).

Checked in a clean subprocess (no jaxtyping instrumentation -- shard_map is
incompatible with it; see test_sharding.py). Complements the forward-only
sharding_equiv.py by covering the phases a dynamic-sparse run actually uses:

  * a full TRAIN step (forward + loss + backward + adam update_conn) under
    Scheme-A matches the single-device step numerically, incl. every per-edge
    optimizer column -- the path the memory/perf timing runs execute on a
    static sharded topology;
  * a magnitude PRUNE step shards (edge-local tombstone);
  * a full SET CHURN step (prune + add_conn growth) shards byte-identically:
    add_conn coordinates its slot claim across shards entirely on device (an
    all-reduce dedup so all shards agree on the candidate set, plus an
    all-gathered global free-slot rank that sends each new edge to the one
    shard owning its slot), so every rewired edge lands in the same arena
    position as single-device. This is what lets churn run under Scheme-A, not
    just the static-topology timing path.

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
import numpy as np

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


def _check_churn_step_shards(
    static: px.NetworkStatic, static_s: px.NetworkStatic, state: px.NetworkState[None]
) -> None:
    """A full SET churn step (prune + device-resident add_conn) shards exactly.

    add_conn's slot claim is coordinated across shards (all-reduce dedup +
    all-gathered global free-slot rank), and because shard g owns arena
    positions [g*cap/G, (g+1)*cap/G) the global free-slot order equals the
    single-device position order -- so the rewired arena is byte-identical.
    """
    churn_net = make_net(
        _OPT, method="set", mode="churn", zeta=0.3, max_candidates=max(_BUDGETS)
    )
    sp = px.StepInputs(inputs=jnp.zeros((_LAYERS[0],), jnp.float32), targets=None)
    single = px.make_step(churn_net, static)(_copy(state), sp).state
    sharded = px.make_step(churn_net, static_s)(_copy(state), sp).state

    if int(px.state.live_conn_count(single)) != int(px.state.live_conn_count(sharded)):
        raise AssertionError("churn: live-edge count differs sharded vs single")
    if not _conns_allclose(single, sharded):
        raise AssertionError("churn: rewired conn columns differ sharded vs single")


class _HashPropose(px.ProposeAddConn[None]):
    """Two random deeper partners per unit (plastix's sampled GrowFanout)."""

    def __init__(self, num_units: int, *, dedupe: bool) -> None:
        self.num_units = num_units
        self.num_proposals = 2 * num_units
        self.max_candidates = 16
        self.dedupe = dedupe

    def propose(
        self, u: px.UnitView, j: jax.Array, g: None
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        del g
        src = j // 2
        h = (j.astype(jnp.uint32) * jnp.uint32(0x9E3779B1)) ^ jnp.uint32(0x85EBCA77)
        h = (h ^ (h >> 13)) * jnp.uint32(0xC2B2AE3D)
        dst = (h % jnp.uint32(self.num_units)).astype(jnp.int32)
        deeper = u[px.LEVEL, px.UnitIdx(dst)] > u[px.LEVEL, px.UnitIdx(src)]
        score = (h >> jnp.uint32(8)).astype(jnp.float32)
        return src, dst, jnp.where(deeper, score, -jnp.inf)

    def init(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: None
    ) -> px.ConnWrite:
        del u, src, dst, g
        return px.ConnWrite.of((px.WEIGHT, jnp.float32(0.01)))


def _check_propose_churn_shards(
    static: px.NetworkStatic, static_s: px.NetworkStatic, state: px.NetworkState[None]
) -> None:
    """Prune + ProposeAddConn growth shards byte-identically, dedupe on and off.

    Proposals read only replicated units/globals, so every shard proposes the
    same candidates; the live-duplicate mask is all-reduced as on the grid path.
    """
    for dedupe in (False, True):

        class _ProposeNet(px.Network[None]):
            forward_pass = MagnitudeStats(0.3)
            prune_conn = SetPrune()
            add_conn = _HashPropose(sum(_LAYERS), dedupe=dedupe)
            extra_unit_fields = _EXTRA_UNIT_FIELDS
            extra_conn_fields = _OPT.state_fields
            propagation = px.Propagation.TOPOLOGICAL

        sp = px.StepInputs(inputs=jnp.zeros((_LAYERS[0],), jnp.float32), targets=None)
        single = px.make_step(_ProposeNet, static)(_copy(state), sp).state
        sharded = px.make_step(_ProposeNet, static_s)(_copy(state), sp).state
        if int(px.state.live_conn_count(single)) != int(
            px.state.live_conn_count(sharded)
        ):
            raise AssertionError(f"propose(dedupe={dedupe}): live count differs")
        if not _conns_allclose(single, sharded):
            raise AssertionError(f"propose(dedupe={dedupe}): conn columns differ")
        grown = sum(
            int(((b[px.WEIGHT.name] == 0.01) & ~b[px.DEAD.name]).sum())
            for b in single.conns
        )
        if grown == 0:
            raise AssertionError(f"propose(dedupe={dedupe}): nothing grew")


class _HashPrune(px.PruneConn):
    """Tombstone a hashed fraction of edges: reads only edge columns."""

    def predicate(
        self, u: px.UnitView, c: px.ConnView, cid: px.ConnIdx, g: None
    ) -> jax.Array:
        del u, g
        a = c[px.FROM_ID, cid].astype(jnp.uint32) * jnp.uint32(0x9E3779B1)
        h = (a ^ c[px.TO_ID, cid].astype(jnp.uint32)) * jnp.uint32(0x85EBCA77)
        return (h >> jnp.uint32(24)) < jnp.uint32(40)


def _check_fused_prune_shards(
    static: px.NetworkStatic, static_s: px.NetworkStatic, state: px.NetworkState[None]
) -> None:
    """The prune-into-forward fusion (XLA lowering) runs under Scheme-A.

    A sharded step with `fuse_prune="xla"` must match the single-device
    two-pass step: the fused predicate sees each shard's own edge slice, like
    the separate prune sweep.
    """

    class _FusedNet(px.Network[None]):
        forward_pass = MagnitudeStats(0.3)
        prune_conn = _HashPrune()
        add_conn = _HashPropose(sum(_LAYERS), dedupe=False)
        extra_unit_fields = _EXTRA_UNIT_FIELDS
        extra_conn_fields = _OPT.state_fields
        propagation = px.Propagation.TOPOLOGICAL

    sp = px.StepInputs(inputs=jnp.zeros((_LAYERS[0],), jnp.float32), targets=None)
    single = px.make_step(_FusedNet, static, fuse_prune="off")(_copy(state), sp).state
    step_s = px.make_step(_FusedNet, static_s, fuse_prune="xla")
    sharded = step_s(_copy(state), sp).state
    plan = step_s.prune_fusion.plan  # type: ignore[attr-defined]
    if plan is None or not plan.fused:
        raise AssertionError(f"fused prune: not fused under Scheme-A ({plan})")
    if int(px.state.live_conn_count(single)) != int(px.state.live_conn_count(sharded)):
        raise AssertionError("fused prune: live-edge count differs sharded vs single")
    if not _conns_allclose(single, sharded):
        raise AssertionError("fused prune: conn columns differ sharded vs single")


def _check_batched_train_step_shards(
    static: px.NetworkStatic, static_s: px.NetworkStatic, state: px.NetworkState[None]
) -> None:
    """A batched (B = 5) train step shards like the streaming one."""
    train_net = make_net(_OPT, method="set", mode="train")
    rng = np.random.default_rng(0)
    xs = jnp.asarray(rng.standard_normal((5, _LAYERS[0])).astype(np.float32))
    ys = jax.nn.one_hot(jnp.asarray(rng.integers(0, _LAYERS[-1], 5)), _LAYERS[-1])
    sp = px.StepInputs(inputs=xs, targets=ys)
    for layout in ("edge_list", "csr", "triton"):
        single = px.make_step(train_net, static, batch_size=5, layout=layout)(
            _copy(state), sp
        )
        sharded = px.make_step(train_net, static_s, batch_size=5, layout=layout)(
            _copy(state), sp
        )
        if not _conns_allclose(single.state, sharded.state):
            raise AssertionError(f"batched {layout}: conns differ sharded vs single")


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
    _check_churn_step_shards(static, static_s, state)
    print("OK churn step shards (prune + device-resident add_conn growth)")
    _check_propose_churn_shards(static, static_s, state)
    print("OK proposal churn step shards (dedupe off and on)")
    _check_fused_prune_shards(static, static_s, state)
    print("OK fused prune step shards (prune predicate in the forward sweep)")
    _check_batched_train_step_shards(static, static_s, state)
    print("OK batched train step shards")
    print("CHURN SHARDING CHECK PASS")


if __name__ == "__main__":
    main()
