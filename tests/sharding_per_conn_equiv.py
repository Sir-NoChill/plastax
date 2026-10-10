"""Per-connection proposals under Scheme-A match single-device bit for bit.

Checked in a clean subprocess (no jaxtyping instrumentation -- shard_map is
incompatible with it; see test_sharding.py). A per-connection proposer is the
one growth strategy whose proposers are themselves sharded: each live
connection proposes, its candidate index is `rank * P + j` with `rank` its
position in ascending `(src, dst, occurrence)` order over every live
connection, and its rng is keyed by `conn_key(src, dst, occurrence)`. Both
rank and occurrence must be global across shards for the sharded step to
equal the single-device one.

For 2 and 4 shards, over several churn steps (hash prune + per-connection
growth with keyed rng draws, under capacity pressure) this checks:

  * the rank-ordered candidate arrays a sharded `candidates_propose_per_conn`
    returns equal the single-device ones (src, dst, score bits, validity) --
    which pins the global rank, since the candidate index only breaks ties
    among otherwise identical candidates in the committed state;
  * every connection column, the grown count and the overflow flag of the
    sharded step equal the single-device step's;
  * the check is not vacuous: some step grows edges, some step overflows,
    the winning proposers sit on more than one shard, and some live parallel
    edge has a shard-local occurrence different from its global one.

Run directly (`python tests/sharding_per_conn_equiv.py`) or via the
subprocess wrapper in test_sharding_per_conn.py.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import dataclasses
from collections import Counter
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec

import plastax as px
from plastax import phases
from sharding_equiv import assert_conns_sharded

SHARD_COUNTS = (2, 4)
STEPS = 6
_P = 2  # proposals per connection
_INPUTS, _HIDDEN, _OUTPUTS = 4, 8, 4
_NUM_UNITS = _INPUTS + _HIDDEN + _OUTPUTS


class _Forward(px.ForwardPass):
    """Weighted-sum forward; present only so the network has a forward pass."""

    combine = px.monoid.sum_

    def map(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: None,
    ) -> jax.Array:
        del dst, g
        return c[px.WEIGHT, cid] * u[px.ACTIVATION, src]

    def apply(
        self, u: px.UnitView, i: px.UnitIdx, g: None, acc: jax.Array
    ) -> px.UnitWrite:
        del u, i, g
        return px.UnitWrite.of((px.ACTIVATION, acc))


class _HashPrune(px.PruneConn):
    """Tombstone a hashed subset of edges, so every step frees slots."""

    def predicate(
        self, u: px.UnitView, c: px.ConnView, cid: px.ConnIdx, g: None
    ) -> jax.Array:
        del u, g
        a = c[px.FROM_ID, cid].astype(jnp.uint32) * jnp.uint32(0x9E3779B1)
        h = (a ^ c[px.TO_ID, cid].astype(jnp.uint32)) * jnp.uint32(0x85EBCA77)
        return (h >> jnp.uint32(24)) < jnp.uint32(70)


class _PerConnPropose(px.ProposeAddConn[None]):
    """Each live edge proposes P edges from its endpoints to a drawn unit."""

    proposer = "per_connection"
    proposals_per_proposer = _P
    max_new_per_level = 64
    max_level_gap = 2
    direction = "deeper"

    def propose(
        self,
        u: px.UnitView,
        c: px.ConnView,
        cid: px.ConnIdx,
        j: jax.Array,
        g: None,
        rng: px.rng.Rng,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        del u, g
        src = jnp.where(j == 0, c[px.FROM_ID, cid], c[px.TO_ID, cid])
        dst = rng.uniform_int(_NUM_UNITS).astype(jnp.int32)
        # Coarse scores, so selection also exercises its (src, dst) tiebreak.
        score = jnp.floor(rng.uniform() * jnp.float32(8.0))
        return src.astype(jnp.int32), dst, score

    def init(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: None
    ) -> px.ConnWrite:
        del u, g
        w = (src * 7 + dst).astype(jnp.float32) / jnp.float32(64.0)
        return px.ConnWrite.of((px.WEIGHT, w))


class _ChurnNet(px.Network[None]):
    forward_pass = _Forward()
    prune_conn = _HashPrune()
    add_conn = _PerConnPropose()
    propagation = px.Propagation.TOPOLOGICAL


class _PruneNet(px.Network[None]):
    forward_pass = _Forward()
    prune_conn = _HashPrune()
    propagation = px.Propagation.TOPOLOGICAL


def _build() -> tuple[px.NetworkStatic, px.NetworkState[None]]:
    """A 3-level net with parallel edges, its buckets nearly full."""
    rng = np.random.default_rng(7)
    hidden = np.arange(_INPUTS, _INPUTS + _HIDDEN)
    outputs = np.arange(_INPUTS + _HIDDEN, _NUM_UNITS)
    src0 = rng.integers(0, _INPUTS, 28)
    dst0 = rng.choice(hidden, 28)
    src1 = rng.choice(hidden, 13)
    dst1 = rng.choice(outputs, 13)
    # A few pairs repeated, so parallel-edge runs straddle shard boundaries.
    src = np.concatenate([src0, [0, 0, 0, 1, 1], src1, [4, 4, 4]])
    dst = np.concatenate([dst0, [4, 4, 4, 5, 5], dst1, [12, 12, 12]])
    return px.NetworkBuilder.from_edges(
        _ChurnNet,
        _NUM_UNITS,
        src.astype(np.int32),
        dst.astype(np.int32),
        input_ids=list(range(_INPUTS)),
        output_ids=[int(o) for o in outputs],
        globals_=None,
    )


def _copy(state: px.NetworkState[None]) -> px.NetworkState[None]:
    """Independent copy so one step's donation can't consume the other run's."""
    return jax.tree_util.tree_map(lambda x: jnp.array(x), state)


def _assert_states_equal(
    a: px.NetworkState[None], b: px.NetworkState[None], what: str
) -> None:
    for level, (bucket_a, bucket_b) in enumerate(zip(a.conns, b.conns, strict=True)):
        for name in bucket_a:
            col_a, col_b = np.asarray(bucket_a[name]), np.asarray(bucket_b[name])
            if not np.array_equal(col_a.view(np.uint8), col_b.view(np.uint8)):
                raise AssertionError(f"{what}: bucket {level} column {name} differs")
    for name in ("grown", "overflow", "needs_resort"):
        if int(getattr(a, name)) != int(getattr(b, name)):
            raise AssertionError(f"{what}: {name} differs")


def _live_pairs(state: px.NetworkState[None]) -> Counter[tuple[int, int]]:
    pairs: Counter[tuple[int, int]] = Counter()
    for bucket in state.conns:
        live = ~np.asarray(bucket[px.DEAD.name])
        src = np.asarray(bucket[px.FROM_ID.name])[live]
        dst = np.asarray(bucket[px.TO_ID.name])[live]
        pairs.update(zip(src.tolist(), dst.tolist(), strict=True))
    return pairs


def _flat_edges(
    state: px.NetworkState[None], num_shards: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bucket-major (src, dst, dead, owning shard) over the whole arena."""
    src, dst, dead, shard = [], [], [], []
    for bucket in state.conns:
        cap = bucket[px.DEAD.name].shape[0]
        src.append(np.asarray(bucket[px.FROM_ID.name]))
        dst.append(np.asarray(bucket[px.TO_ID.name]))
        dead.append(np.asarray(bucket[px.DEAD.name]))
        shard.append(np.arange(cap) // (cap // num_shards))
    return tuple(np.concatenate(x) for x in (src, dst, dead, shard))  # type: ignore[return-value]


def _rank_order(src: np.ndarray, dst: np.ndarray, dead: np.ndarray) -> np.ndarray:
    """Arena slots of the live edges in ascending (src, dst, slot) order."""
    slots = np.flatnonzero(~dead)
    return slots[np.lexsort((slots, dst[slots], src[slots]))]


def _occurrences(src: np.ndarray, dst: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Occurrence index of each slot in `order` within its (src, dst) run."""
    occ = np.zeros(order.shape[0], dtype=np.int64)
    for r in range(1, order.shape[0]):
        same = (src[order[r]], dst[order[r]]) == (src[order[r - 1]], dst[order[r - 1]])
        occ[r] = occ[r - 1] + 1 if same else 0
    return occ


def _shard_local_occurrence_differs(
    state: px.NetworkState[None], num_shards: int
) -> bool:
    """Whether some live edge's per-shard occurrence differs from its global one."""
    src, dst, dead, shard = _flat_edges(state, num_shards)
    order = _rank_order(src, dst, dead)
    global_occ = dict(zip(order.tolist(), _occurrences(src, dst, order), strict=True))
    for g in range(num_shards):
        local = order[shard[order] == g]
        local_occ = _occurrences(src, dst, local)
        pairs = zip(local.tolist(), local_occ, strict=True)
        if any(global_occ[s] != o for s, o in pairs):
            return True
    return False


def _candidates(
    static: px.NetworkStatic,
    state: px.NetworkState[None],
    num_shards: int | None,
) -> tuple[np.ndarray, ...]:
    """The per-connection candidate arrays, single-device or Scheme-A sharded."""
    rule = _ChurnNet.add_conn
    assert isinstance(rule, _PerConnPropose)

    def run(
        units: px.state.Columns,
        conns: tuple[px.state.Columns, ...],
        step: jax.Array,
        axis: str | None,
        shards: int,
    ) -> tuple[jax.Array, ...]:
        src, dst, score, valid = phases.candidates_propose_per_conn(
            rule.propose,
            px.UnitView(units),
            conns,
            None,
            _P,
            _NUM_UNITS,
            seed=static.seed,
            step=step,
            shard_axis=axis,
            num_shards=shards,
        )
        return src, dst, jax.lax.bitcast_convert_type(score, jnp.int32), valid

    if num_shards is None:
        out = jax.jit(run, static_argnums=(3, 4))(
            state.units, state.conns, state.step, None, 1
        )
    else:
        spec = px.ShardSpec("shard", num_shards)
        static_s = dataclasses.replace(static, sharding=spec)
        repl: Any = PartitionSpec()  # type: ignore[no-untyped-call]
        conn: Any = PartitionSpec(spec.axis_name)  # type: ignore[no-untyped-call]
        conn_specs = tuple({name: conn for name in b} for b in state.conns)
        sharded: Any = jax.shard_map(
            lambda u, c, s: run(u, c, s, spec.axis_name, num_shards),
            mesh=px.distributed.scheme_a_mesh(static_s),
            in_specs=({name: repl for name in state.units}, conn_specs, repl),
            out_specs=(repl, repl, repl, repl),
        )
        out = jax.jit(sharded)(state.units, state.conns, state.step)
    return tuple(np.asarray(x) for x in out)


def _winning_shards(
    static: px.NetworkStatic,
    pre: px.NetworkState[None],
    post: px.NetworkState[None],
    num_shards: int,
) -> set[int]:
    """Shards owning a proposer whose candidate became one of this step's edges."""
    pruned = px.make_step(_PruneNet, static)(
        _copy(pre), px.StepInputs(inputs=jnp.zeros((_INPUTS,)), targets=None)
    ).state
    pruned = dataclasses.replace(pruned, step=pre.step)
    new_pairs = _live_pairs(post) - _live_pairs(pruned)
    src, dst, valid = (_candidates(static, pruned, None)[i] for i in (0, 1, 3))
    e_src, e_dst, e_dead, shard = _flat_edges(pruned, num_shards)
    order = _rank_order(e_src, e_dst, e_dead)
    shards = set()
    for idx in np.flatnonzero(valid):
        if (int(src[idx]), int(dst[idx])) in new_pairs:
            shards.add(int(shard[order[idx // _P]]))
    return shards


def main() -> None:
    """Run the per-connection Scheme-A equivalence and print the pass sentinel."""
    if len(jax.devices()) < max(SHARD_COUNTS):
        raise SystemExit(f"need >= {max(SHARD_COUNTS)} devices")
    static, state0 = _build()
    sp = px.StepInputs(inputs=jnp.zeros((_INPUTS,), jnp.float32), targets=None)
    single_step = px.make_step(_ChurnNet, static)

    for num_shards in SHARD_COUNTS:
        static_s = dataclasses.replace(
            static, sharding=px.ShardSpec("shard", num_shards)
        )
        sharded_step = px.make_step(_ChurnNet, static_s)
        single, sharded = _copy(state0), _copy(state0)
        total_grown = overflowed = 0
        winners: set[int] = set()
        occurrence_matters = False
        for step in range(STEPS):
            what = f"{num_shards} shards, step {step}"
            want = _candidates(static, single, None)
            got = _candidates(static, single, num_shards)
            names = ("src", "dst", "score", "valid")
            for name, a, b in zip(names, want, got, strict=True):
                if not np.array_equal(a, b):
                    raise AssertionError(f"{what}: candidate {name} differs")
            occurrence_matters |= _shard_local_occurrence_differs(single, num_shards)
            pre = _copy(single)
            single = single_step(single, sp).state
            sharded = sharded_step(sharded, sp).state
            assert_conns_sharded(sharded, num_shards, what)
            _assert_states_equal(single, sharded, what)
            total_grown += int(single.grown)
            overflowed += int(single.overflow)
            winners |= _winning_shards(static, pre, single, num_shards)
            live = int(px.state.live_conn_count(single))
            print(
                f"OK {what}: grown={int(single.grown)} "
                f"overflow={bool(single.overflow)} live={live}"
            )
        if total_grown == 0:
            raise AssertionError(f"{num_shards} shards: nothing grew")
        if overflowed == 0:
            raise AssertionError(f"{num_shards} shards: capacity never overflowed")
        if len(winners) < 2:
            raise AssertionError(
                f"{num_shards} shards: winning proposers on shards {winners} only"
            )
        if not occurrence_matters:
            raise AssertionError(
                f"{num_shards} shards: no parallel edge spans shards, so a "
                "shard-local occurrence would go unnoticed"
            )
        print(f"OK {num_shards} shards: winners on shards {sorted(winners)}")
    print("PER-CONNECTION SHARDING CHECK PASS")


if __name__ == "__main__":
    main()
