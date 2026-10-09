"""The add_conn free-slot claim: engines agree, and match the pre-kernel path.

Random churn over layered nets: a hashed prune predicate kills a slice of the
live edges each step and a growth policy (hashed proposals, or a scored grid)
refills them. Every config is run for many steps and the full state (every
connection column of every bucket, `needs_resort`, and the per-step overflow
flags) is digested. The digests are pinned to what the claim path produced
before the fused growth kernel (commit 7ff7cc6): the XLA claim must stay
bit-identical to it, and on an NVIDIA GPU with jax_triton the Triton kernel
must match the XLA claim step for step.

The configs cover overflow (tight capacities), dedupe on and off, the large-
and small-claim regimes, several levels and buckets, aligned and unaligned
capacities, and an extra written field plus an extra defaulted one. Weights are
small integers in float32 so the digests do not depend on float rounding.
"""

from __future__ import annotations

import hashlib
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import plastax as px
from plastax import phases

_AGE = px.FieldSpec.int32("test/age")  # written by init
_TRACE = px.FieldSpec.float32("test/trace", default=1.5)  # left at its default
_STEPS = 24


def _mix(x: jax.Array) -> jax.Array:
    """A uint32 integer hash (lowbias32)."""
    x = x.astype(jnp.uint32)
    x = x ^ (x >> 16)
    x = x * jnp.uint32(0x7FEB352D)
    x = x ^ (x >> 15)
    x = x * jnp.uint32(0x846CA68B)
    return x ^ (x >> 16)


def _h(*parts: jax.Array) -> jax.Array:
    acc = jnp.uint32(0x9E3779B9)
    for p in parts:
        acc = _mix(acc ^ jnp.asarray(p).astype(jnp.uint32))
    return acc


class _Forward(px.ForwardPass):
    combine = px.monoid.sum_

    def map(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: Any,
    ) -> jax.Array:
        del dst, g
        return c[px.WEIGHT, cid] * u[px.ACTIVATION, src]

    def apply(
        self, u: px.UnitView, i: px.UnitIdx, g: Any, acc: jax.Array
    ) -> px.UnitWrite:
        del u, i, g
        return px.UnitWrite.of((px.ACTIVATION, acc))


class _HashPrune(px.PruneConn):
    """Kill about `rate`/256 of the live edges, re-drawn every step."""

    def __init__(self, rate: int) -> None:
        self.rate = rate

    def predicate(
        self, u: px.UnitView, c: px.ConnView, cid: px.ConnIdx, g: Any
    ) -> jax.Array:
        del u
        h = _h(c[px.FROM_ID, cid], c[px.TO_ID, cid], g["t"], jnp.uint32(7))
        return (h & jnp.uint32(255)) < jnp.uint32(self.rate)


def _init(src: jax.Array, dst: jax.Array, g: Any) -> px.ConnWrite:
    weight = ((src * 3 + dst) % 17).astype(jnp.float32) - 8.0
    return px.ConnWrite.of((px.WEIGHT, weight), (_AGE, g["t"] + src))


class _HashProposals(px.ProposeAddConn):
    """Hashed proposals; a slice is vetoed, scores are coarse (many ties)."""

    def __init__(
        self, num_units: int, num_proposals: int, max_candidates: int, dedupe: bool
    ) -> None:
        self.num_units = num_units
        self.proposer = "global"
        self.proposals_per_proposer = num_proposals
        self.max_candidates = max_candidates
        # The pinned digests predate the split flags; the old single flag
        # meant both stages at once.
        self.dedupe_live = dedupe
        self.dedupe_step = dedupe

    def propose(
        self, u: px.UnitView, j: jax.Array, g: Any, rng: px.rng.Rng
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        del u, rng
        h = _h(j, g["t"], jnp.uint32(11))
        src = (h % jnp.uint32(self.num_units)).astype(jnp.int32)
        h2 = _h(h, jnp.uint32(13))
        dst = (h2 % jnp.uint32(self.num_units)).astype(jnp.int32)
        score = ((h2 >> 8) & jnp.uint32(7)).astype(jnp.float32)
        veto = (h >> 24) < jnp.uint32(40)
        return src, dst, jnp.where(veto, -jnp.inf, score)

    def init(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: Any
    ) -> px.ConnWrite:
        del u
        return _init(src, dst, g)


class _HashGrid(px.AddConn):
    """Grid growth scored by a hash, a slice vetoed; dedupe on (the default)."""

    def __init__(self, max_candidates: int) -> None:
        self.max_candidates = max_candidates

    def score(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: Any
    ) -> jax.Array:
        del u
        h = _h(src, dst, g["t"], jnp.uint32(17))
        score = (h & jnp.uint32(15)).astype(jnp.float32)
        return jnp.where((h >> 28) < jnp.uint32(3), -jnp.inf, score)

    def init(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: Any
    ) -> px.ConnWrite:
        del u
        return _init(src, dst, g)


class _Tick(px.ResetGlobal):
    def reset(self, g: Any) -> Any:
        return {"t": g["t"] + jnp.int32(1)}


_CONFIGS: dict[str, dict[str, Any]] = {
    # Three buckets, tight capacities (overflow), claims larger than the
    # bucket (the large-claim regime), parallel edges allowed.
    "propose_overflow": dict(
        widths=(12, 16, 16, 8),
        density=0.5,
        rate=48,
        grow="propose",
        proposals=128,
        k=128,
        dedupe=False,
        headroom=0.02,
        align=None,
    ),
    # The same with the live and within-step duplicate checks, and top_k
    # (k below the proposal count).
    "propose_dedupe": dict(
        widths=(12, 16, 16, 8),
        density=0.5,
        rate=48,
        grow="propose",
        proposals=160,
        k=48,
        dedupe=True,
        headroom=0.1,
        align=None,
    ),
    # Wide buckets, few candidates: the small-claim (two-level) regime, on
    # 256-aligned capacities.
    "propose_small_claim": dict(
        widths=(96, 128, 64),
        density=0.6,
        rate=4,
        grow="propose",
        proposals=24,
        k=4,
        dedupe=False,
        headroom=0.05,
        align=256,
    ),
    # Neighbourhood 2 (skip edges, same-level proposals -> needs_resort),
    # capacities not a multiple of 64 (the padded block count).
    "propose_window2": dict(
        widths=(10, 9, 11, 7),
        density=0.6,
        rate=40,
        grow="propose",
        proposals=48,
        k=48,
        dedupe=False,
        headroom=0.0,
        align=7,
        max_level_gap=2,
    ),
    # Grid growth (dedupe default on), top_k over the full grid.
    "grid": dict(
        widths=(8, 12, 6),
        density=0.4,
        rate=80,
        grow="grid",
        k=20,
        headroom=0.05,
        align=None,
    ),
}


def _build(cfg: dict[str, Any]) -> tuple[type[px.Network[Any]], Any, Any]:
    rng = np.random.default_rng(0)
    widths = cfg["widths"]
    offsets = np.concatenate([[0], np.cumsum(widths)])
    num_units = int(offsets[-1])
    frm, to = [], []
    for layer in range(len(widths) - 1):
        a = np.arange(offsets[layer], offsets[layer + 1])
        b = np.arange(offsets[layer + 1], offsets[layer + 2])
        src, dst = np.meshgrid(a, b, indexing="ij")
        keep = rng.random(src.shape) < cfg["density"]
        frm.append(src[keep])
        to.append(dst[keep])
    from_ids = np.concatenate(frm).astype(np.int32)
    to_ids = np.concatenate(to).astype(np.int32)
    if cfg["grow"] == "propose":
        policy: Any = _HashProposals(
            num_units, cfg["proposals"], cfg["k"], cfg["dedupe"]
        )
    else:
        policy = _HashGrid(cfg["k"])
    prune = _HashPrune(cfg["rate"])
    # the growth window is the rule's attribute, not the Network's
    policy.max_level_gap = cfg.get("max_level_gap", 1)

    class Net(px.Network[Any]):
        forward_pass = _Forward()
        prune_conn = prune
        add_conn = policy
        reset_global = _Tick()
        extra_conn_fields = (_AGE, _TRACE)
        propagation = px.Propagation.TOPOLOGICAL

    static, state = px.NetworkBuilder.from_edges(
        Net,
        num_units,
        from_ids,
        to_ids,
        weights=np.ones(from_ids.size, np.float32),
        input_ids=list(range(widths[0])),
        output_ids=list(range(int(offsets[-2]), num_units)),
        globals_={"t": jnp.int32(0)},
        capacity_headroom=cfg["headroom"],
        capacity_align=cfg["align"],
    )
    return Net, static, state


def _digest(state: Any, overflows: list[bool]) -> str:
    h = hashlib.sha256()
    for bucket in state.conns:
        for name in sorted(bucket):
            h.update(name.encode())
            h.update(np.ascontiguousarray(np.asarray(bucket[name])).tobytes())
    h.update(bytes([bool(state.needs_resort)]))
    h.update(bytes(overflows))
    return h.hexdigest()[:16]


def _run(
    name: str, growth: Literal["auto", "xla", "triton"]
) -> tuple[list[str], list[bool]]:
    """Per-step digests and overflow flags of `_STEPS` churn steps."""
    net, static, state = _build(_CONFIGS[name])
    step = px.make_step(net, static, growth=growth)
    inputs = px.StepInputs(
        inputs=jnp.ones((len(static.input_ids),), jnp.float32), targets=None
    )
    digests, overflows = [], []
    for _ in range(_STEPS):
        result = step(state, inputs)
        state = result.state
        overflows.append(bool(result.overflow))
        digests.append(_digest(state, overflows))
    return digests, overflows


# Final-state digests of the pre-kernel claim path (CPU). propose_dedupe and
# propose_small_claim were re-pinned when selection adopted the total candidate
# order (-score, src, dst, candidate_index): their hashed scores are coarse
# (deliberate ties), and ties now resolve by (src, dst) before candidate index.
# All five were re-pinned again when growth gained the deepest-level bucket:
# candidates sourced at the deepest level, previously dropped for lack of a
# bucket, now commit (verified: the grid config churns 64 live edges into the
# new bucket), and the arena hash covers the extra bucket itself.
_GOLDEN: dict[str, str] = {
    "grid": "4e8864fe12526877",
    "propose_dedupe": "e9b8e9998f5449b8",
    "propose_overflow": "052b11bb31cb9ce7",
    "propose_small_claim": "84b5c49d0d202d72",
    "propose_window2": "fecc2e9c83061b02",
}


@pytest.mark.parametrize("name", sorted(_CONFIGS))
def test_xla_claim_matches_the_pre_kernel_path(name: str) -> None:
    digests, overflows = _run(name, "xla")
    assert digests[-1] == _GOLDEN[name]


def test_the_configs_exercise_overflow_and_resort() -> None:
    _, overflows = _run("propose_overflow", "xla")
    assert any(overflows) and not all(overflows)
    net, static, state = _build(_CONFIGS["propose_small_claim"])
    k = _CONFIGS["propose_small_claim"]["k"]
    # The deepest (growth-only) bucket starts empty and small; the big-bucket
    # claim-path property is about the populated buckets.
    assert all(k * 1024 <= cap for cap in static.level_capacities[:-1])
    net, static, state = _build(_CONFIGS["propose_window2"])
    assert any(cap % 64 for cap in static.level_capacities)
    step = px.make_step(net, static, growth="xla")
    inputs = px.StepInputs(
        inputs=jnp.ones((len(static.input_ids),), jnp.float32), targets=None
    )
    assert bool(step(state, inputs).state.needs_resort)


@pytest.mark.skipif(
    not phases.nvidia_triton_available(), reason="needs an NVIDIA GPU + jax_triton"
)
@pytest.mark.parametrize("name", sorted(_CONFIGS))
def test_triton_claim_matches_xla_step_for_step(name: str) -> None:
    assert _run(name, "triton") == _run(name, "xla")


def _claim_case(
    rng: np.random.Generator, caps: tuple[int, ...], k: int, density: float
) -> tuple[list[dict[str, jax.Array]], list[phases.GrowthClaim]]:
    """Random buckets (uniform or clustered free slots) and candidates."""
    buckets, claims = [], []
    src = jnp.asarray(rng.integers(0, 1000, k, dtype=np.int32))
    dst = jnp.asarray(rng.integers(0, 1000, k, dtype=np.int32))
    weight = jnp.asarray(rng.standard_normal(k).astype(np.float32))
    age = jnp.asarray(rng.integers(0, 50, k, dtype=np.int32))
    for cap in caps:
        dead = rng.random(cap) < density
        if rng.random() < 0.5:  # a free tail, like a bucket's headroom
            dead[int(cap * rng.uniform(0.5, 1.0)) :] = True
        buckets.append(
            {
                px.DEAD.name: jnp.asarray(dead),
                px.FROM_ID.name: jnp.asarray(rng.integers(0, 9, cap, dtype=np.int32)),
                px.TO_ID.name: jnp.asarray(rng.integers(0, 9, cap, dtype=np.int32)),
                px.WEIGHT.name: jnp.asarray(rng.standard_normal(cap), jnp.float32),
                _AGE.name: jnp.zeros((cap,), jnp.int32),
            }
        )
        claims.append(
            phases.GrowthClaim(
                growable=jnp.asarray(rng.random(k) < rng.uniform(0.2, 1.0)),
                violating=jnp.asarray(rng.random(k) < 0.002),
                values={
                    px.FROM_ID.name: src,
                    px.TO_ID.name: dst,
                    px.WEIGHT.name: weight,
                    _AGE.name: age,
                },
            )
        )
    return buckets, claims


_CLAIM_CASES = [
    ((4096,), 100, 0.05),
    ((4096, 72, 100_000), 300, 0.05),  # a capacity not a multiple of 64
    ((1 << 16,), 3000, 0.9),
    ((1 << 16, 3 << 14), 2000, 0.01),  # sparse: overflow
    ((256 * 1001,), 700, 0.001),
    ((64, 128), 5, 0.5),
]


@pytest.mark.parametrize("seed", range(3))
def test_xla_claim_takes_the_first_free_slots_in_order(seed: int) -> None:
    rng = np.random.default_rng(seed)
    for caps, k, density in _CLAIM_CASES:
        buckets, claims = _claim_case(rng, caps, k, density)
        for bucket, claim in zip(buckets, claims, strict=True):
            new, overflowed, resort = phases.xla_claim(bucket, claim)
            dead = np.asarray(bucket[px.DEAD.name])
            grow = np.asarray(claim.growable)
            free = np.flatnonzero(dead)
            rank = np.cumsum(grow) - 1
            committed = grow & (rank < free.size)
            slots = free[rank[committed]]
            want_dead = dead.copy()
            want_dead[slots] = False
            np.testing.assert_array_equal(np.asarray(new[px.DEAD.name]), want_dead)
            for name, value in claim.values.items():
                want = np.asarray(bucket[name]).copy()
                want[slots] = np.asarray(value)[committed]
                np.testing.assert_array_equal(np.asarray(new[name]), want)
            np.testing.assert_array_equal(
                np.asarray(overflowed), grow & (rank >= free.size)
            )
            np.testing.assert_array_equal(
                np.asarray(resort), committed & np.asarray(claim.violating)
            )


@pytest.mark.skipif(
    not phases.nvidia_triton_available(), reason="needs an NVIDIA GPU + jax_triton"
)
@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("precomputed", [False, True])
def test_triton_claim_matches_xla_claim(seed: int, precomputed: bool) -> None:
    rng = np.random.default_rng(seed)

    def xla(buckets: Any, claims: Any) -> Any:
        out = [phases.xla_claim(b, c) for b, c in zip(buckets, claims, strict=True)]
        return (
            [o[0] for o in out],
            jnp.any(jnp.stack([o[1] for o in out])),
            jnp.any(jnp.stack([o[2] for o in out])),
        )

    def fused(buckets: Any, claims: Any) -> Any:
        counts = (
            [
                phases.free_block_counts(b[px.DEAD.name], phases.TRITON_CLAIM_BLOCK)
                for b in buckets
            ]
            if precomputed
            else None
        )
        return phases.triton_claim(buckets, claims, block_counts=counts)

    for caps, k, density in _CLAIM_CASES:
        buckets, claims = _claim_case(rng, caps, k, density)
        want = jax.jit(xla)(buckets, claims)
        got = jax.jit(fused)(buckets, claims)
        for w, g in zip(want[0], got[0], strict=True):
            for name in w:
                np.testing.assert_array_equal(np.asarray(g[name]), np.asarray(w[name]))
        assert bool(got[1]) == bool(want[1])
        assert bool(got[2]) == bool(want[2])


@pytest.mark.parametrize("cap", [4096, 1 << 16, 3 << 14, 256 * 1001, 100_000, 72, 64])
@pytest.mark.parametrize("max_block", [64, 128, 256, 1024])
def test_finer_free_counts_regroup_exactly(cap: int, max_block: int) -> None:
    # A fused prune sweep may count free slots in the Triton claim's 256-slot
    # blocks; the XLA claim regroups them into its own.
    rng = np.random.default_rng(cap + max_block)
    dead = jnp.asarray(rng.random(cap) < 0.3)
    fine = phases.free_block_counts(dead, max_block)
    block = phases.free_block_length(cap, max_block)
    np.testing.assert_array_equal(
        np.asarray(phases.regroup_free_counts(fine, block, cap)),
        np.asarray(phases.free_block_counts(dead)),
    )


@pytest.mark.parametrize("seed", range(2))
def test_xla_claim_takes_precomputed_triton_block_counts(seed: int) -> None:
    rng = np.random.default_rng(seed)
    for caps, k, density in _CLAIM_CASES:
        buckets, claims = _claim_case(rng, caps, k, density)
        for bucket, claim in zip(buckets, claims, strict=True):
            dead = bucket[px.DEAD.name]
            cap = dead.shape[0]
            counts = (
                phases.free_block_counts(dead, phases.TRITON_CLAIM_BLOCK),
                phases.free_block_length(cap, phases.TRITON_CLAIM_BLOCK),
            )
            want = phases.xla_claim(bucket, claim)
            got = phases.xla_claim(bucket, claim, free_counts=counts)
            for name in want[0]:
                np.testing.assert_array_equal(
                    np.asarray(got[0][name]), np.asarray(want[0][name])
                )
            np.testing.assert_array_equal(np.asarray(got[1]), np.asarray(want[1]))
            np.testing.assert_array_equal(np.asarray(got[2]), np.asarray(want[2]))
