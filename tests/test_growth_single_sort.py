"""add_conn's single total-order sort selects exactly what per-bucket sorts did.

`build_add_conn_phase` sorts a shared candidate list into the total order
once and takes each bucket's winners from its own source level's members.
`_reference_phase` below is the earlier formulation, kept only as this test's
oracle: every bucket re-validates and re-scores the whole list with the other
levels vetoed, sorts all of it, and takes the first k. The two must agree bit
for bit -- every column of every bucket, the overflow and resort flags, and
the grown count -- over randomized layered nets covering every candidate
source, every selection mode, the step cap, both dedupe stages, the window
knobs, unit capacities, PIPELINE, and buckets tight enough to overflow.
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import plastax as px
from plastax import phases
from plastax._types import DEAD, FROM_ID, LEVEL, TO_ID
from plastax.state import live_unit_mask
from plastax.views import UnitView

_KINDS = (
    "per_unit",
    "per_connection",
    "global",
    "exhaustive",
    "shortlist",
    "shortlist_per_level",
)
_SELECTIONS = ("top_k", "threshold", "all")
_SEEDS = (0, 1)
_STEPS = 3


def _mix(x: jax.Array) -> jax.Array:
    x = jnp.asarray(x).astype(jnp.uint32)
    x = x ^ (x >> 16)
    x = x * jnp.uint32(0x7FEB352D)
    x = x ^ (x >> 15)
    x = x * jnp.uint32(0x846CA68B)
    return x ^ (x >> 16)


def _score(h: jax.Array) -> jax.Array:
    """Coarse scores (many ties), with -inf vetoes and NaNs mixed in."""
    score = (h & jnp.uint32(7)).astype(jnp.float32) - 2.0
    score = jnp.where((h >> 8) % jnp.uint32(11) == 0, -jnp.inf, score)
    return jnp.where((h >> 12) % jnp.uint32(13) == 0, jnp.nan, score)


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


class _Knobs:
    """Init and the threshold shared by every rule."""

    def init(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: Any
    ) -> px.ConnWrite:
        del u, g
        weight = ((jnp.asarray(src) * 3 + jnp.asarray(dst)) % 17).astype(jnp.float32)
        return px.ConnWrite.of((px.WEIGHT, weight))

    def threshold(self, g: Any) -> jax.Array:
        del g
        return jnp.float32(1.0)


class _PerUnit(_Knobs, px.ProposeAddConn):
    proposer = "per_unit"

    def __init__(self, num_units: int) -> None:
        self.num_units = num_units

    def propose(
        self, u: px.UnitView, i: px.UnitIdx, j: jax.Array, g: Any, rng: px.rng.Rng
    ) -> px.Proposal:
        del u, j, g
        dst = rng.uniform_int(self.num_units).astype(jnp.int32)
        return px.Proposal(jnp.asarray(i), dst, _score(_mix(rng.uniform_int(1 << 20))))


class _PerConnection(_Knobs, px.ProposeAddConn):
    proposer = "per_connection"

    def __init__(self, num_units: int) -> None:
        self.num_units = num_units

    def propose(
        self,
        u: px.UnitView,
        c: px.ConnView,
        cid: px.ConnIdx,
        j: jax.Array,
        g: Any,
        rng: px.rng.Rng,
    ) -> px.Proposal:
        del u, j, g
        dst = rng.uniform_int(self.num_units).astype(jnp.int32)
        score = _score(_mix(rng.uniform_int(1 << 20)))
        return px.Proposal(c[px.FROM_ID, cid], dst, score)


class _Global(_Knobs, px.ProposeAddConn):
    proposer = "global"

    def __init__(self, num_units: int) -> None:
        self.num_units = num_units

    def propose(
        self, u: px.UnitView, j: jax.Array, g: Any, rng: px.rng.Rng
    ) -> px.Proposal:
        del u, j, g
        src = rng.uniform_int(self.num_units).astype(jnp.int32)
        dst = rng.uniform_int(self.num_units).astype(jnp.int32)
        return px.Proposal(src, dst, _score(_mix(rng.uniform_int(1 << 20))))


class _Scored(_Knobs, px.ScoreAddConn):
    def score(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: Any
    ) -> jax.Array:
        del u
        return _score(_mix(_mix(jnp.asarray(src) * 977 + jnp.asarray(dst)) ^ g["t"]))

    def importance(self, u: px.UnitView, i: px.UnitIdx, g: Any) -> jax.Array:
        del u
        return (_mix(jnp.asarray(i) ^ g["t"]) & jnp.uint32(15)).astype(jnp.float32)


class _Tick(px.ResetGlobal):
    def reset(self, g: Any) -> Any:
        return {"t": g["t"] + jnp.int32(1)}


def _config(kind: str, selection: str, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(
        [_KINDS.index(kind), _SELECTIONS.index(selection), seed]
    )
    pipeline = kind != "shortlist_per_level" and bool(rng.random() < 0.2)
    return dict(
        widths=tuple(int(w) for w in rng.integers(3, 9, size=int(rng.integers(2, 6)))),
        density=float(rng.uniform(0.3, 0.8)),
        headroom=float(rng.choice([0.0, 0.05, 0.5])),
        proposals=int(rng.integers(1, 6)),
        global_proposals=int(rng.integers(8, 96)),
        shortlist=int(rng.integers(3, 9)),
        k=int(rng.integers(1, 24)),
        max_new_per_step=None if rng.random() < 0.5 else int(rng.integers(1, 40)),
        dedupe_live=bool(rng.random() < 0.5),
        dedupe_step=bool(rng.random() < 0.5),
        direction=str(rng.choice(["any", "deeper", "same_or_deeper"])),
        max_level_gap=int(rng.integers(0, 4)),
        allow_self_loops=bool(rng.random() < 0.3),
        unit_capacity=bool(rng.random() < 0.3),
        pipeline=pipeline,
        kill=float(rng.uniform(0.1, 0.5)),
    )


def _build(
    kind: str, selection: str, cfg: dict[str, Any]
) -> tuple[type[px.Network[Any]], px.NetworkStatic, px.NetworkState[Any]]:
    rng = np.random.default_rng(1)
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

    rule: Any
    if kind == "per_unit":
        rule = _PerUnit(num_units)
        rule.proposals_per_proposer = cfg["proposals"]
    elif kind == "per_connection":
        rule = _PerConnection(num_units)
        rule.proposals_per_proposer = cfg["proposals"]
    elif kind == "global":
        rule = _Global(num_units)
        rule.proposals_per_proposer = cfg["global_proposals"]
    else:
        rule = _Scored()
        rule.candidates = kind if kind != "exhaustive" else "exhaustive"
        rule.shortlist_size = cfg["shortlist"]
    rule.selection = selection
    rule.max_new_per_level = cfg["k"]
    rule.max_new_per_step = cfg["max_new_per_step"]
    rule.dedupe_live = cfg["dedupe_live"]
    rule.dedupe_step = cfg["dedupe_step"]
    rule.direction = cfg["direction"]
    rule.max_level_gap = cfg["max_level_gap"]
    rule.allow_self_loops = cfg["allow_self_loops"]

    class Net(px.Network[Any]):
        forward_pass = _Forward()
        add_conn = rule
        reset_global = _Tick()
        propagation = (
            px.Propagation.PIPELINE if cfg["pipeline"] else px.Propagation.TOPOLOGICAL
        )
        unit_capacity = num_units + 3 if cfg["unit_capacity"] else None

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
    )
    return Net, static, state


def _reference_phase(net: type[px.Network[Any]], static: px.NetworkStatic) -> Any:
    """The per-bucket-sort add_conn phase (XLA claim, unsharded, ungated)."""
    ac: Any = net.add_conn
    num_units = static.num_units
    num_buckets = len(static.level_capacities)
    is_pipeline = net.propagation is px.Propagation.PIPELINE
    use_propose = isinstance(ac, px.ProposeAddConn)
    kind = "" if use_propose else str(getattr(ac, "candidates", "exhaustive"))
    m = getattr(ac, "shortlist_size", None)
    use_shortlist = kind != "exhaustive" and m is not None and 0 < m < num_units
    use_per_level = use_shortlist and kind == "shortlist_per_level"
    pool_side = int(m) if use_shortlist else num_units
    if use_propose:
        n_p = ac.proposals_per_proposer
        pool = {
            "per_unit": num_units * n_p,
            "per_connection": sum(static.level_capacities) * n_p,
            "global": n_p,
        }[ac.proposer]
    else:
        pool = pool_side * pool_side
    k = pool if ac.selection == "all" else max(0, min(ac.max_new_per_level, pool))

    def phase(state: px.NetworkState[Any]) -> px.NetworkState[Any]:
        units, g = state.units, state.globals_
        u_view = UnitView(units)
        level = units[LEVEL.name]
        live = live_unit_mask(units)

        def importance() -> jax.Array:
            imp = phases.importance_scores(ac.importance, u_view, g, num_units)
            return imp if live is None else jnp.where(live, imp, -jnp.inf)

        kw = dict(seed=static.seed, step=state.step)
        if use_propose:
            if ac.proposer == "per_unit":
                src_all, dst_all, p_score, in_range = (
                    phases.candidates_propose_per_unit(
                        ac.propose, u_view, g, n_p, num_units, **kw
                    )
                )
                if live is not None:
                    in_range = in_range & jnp.repeat(live, n_p)
            elif ac.proposer == "per_connection":
                src_all, dst_all, p_score, in_range = (
                    phases.candidates_propose_per_conn(
                        ac.propose, u_view, state.conns, g, n_p, num_units, **kw
                    )
                )
            else:
                src_all, dst_all, p_score, in_range = phases.candidates_propose(
                    ac.propose, u_view, g, n_p, num_units, **kw
                )
        elif use_shortlist and not use_per_level:
            src_all, dst_all = phases.candidates_shortlist(importance(), pool_side)
        else:
            src_all, dst_all = phases.candidates_grid(num_units)

        def init_one(s: jax.Array, d: jax.Array) -> dict[str, jax.Array]:
            return dict(ac.init(u_view, px.UnitIdx(s), px.UnitIdx(d), g).fields)

        claims = []
        for b in range(num_buckets):
            if use_per_level:
                src, dst = phases.candidates_per_level(
                    importance(), level, b, pool_side, ac.max_level_gap, ac.direction
                )
            else:
                src, dst = src_all, dst_all
            valid = phases.apply_validity(
                src,
                dst,
                level,
                b,
                ac.max_level_gap,
                is_pipeline,
                ac.direction,
                ac.allow_self_loops,
            )
            if use_propose:
                valid = valid & in_range
            if live is not None:
                valid = valid & live[src] & live[dst]
            if ac.dedupe_live:
                valid = valid & phases.dedupe_live(
                    state.conns[b], src, dst, num_units, None
                )
            if use_propose:
                scores = jnp.where(valid, p_score, -jnp.inf)
            else:
                raw = jax.vmap(
                    lambda s, d: ac.score(u_view, px.UnitIdx(s), px.UnitIdx(d), g)
                )(src, dst)
                scores = jnp.where(valid, raw.astype(jnp.float32), -jnp.inf)
            if ac.dedupe_step:
                scores = phases.dedupe_step(scores, src, dst)
            top = phases.select(scores, src, dst, k)
            growable = valid[top] & jnp.isfinite(scores[top])
            if ac.selection == "threshold":
                growable = growable & (scores[top] >= ac.threshold(g))
            init = jax.vmap(init_one)(src[top], dst[top])
            values = {}
            for spec in static.conn_fields:
                if spec.name == FROM_ID.name:
                    values[spec.name] = src[top].astype(spec.dtype)
                elif spec.name == TO_ID.name:
                    values[spec.name] = dst[top].astype(spec.dtype)
                elif spec.name in init:
                    values[spec.name] = init[spec.name].astype(spec.dtype)
                elif spec.name != DEAD.name:
                    values[spec.name] = jnp.full((k,), spec.default, spec.dtype)
            claims.append(
                phases.GrowthClaim(
                    growable=growable,
                    violating=~(level[dst[top]] > level[src[top]]),
                    values=values,
                )
            )
        if ac.max_new_per_step is not None:
            prior = jnp.int32(0)
            for i, claim in enumerate(claims):
                grow32 = claim.growable.astype(jnp.int32)
                rank = jnp.cumsum(grow32) - 1 + prior
                claims[i] = dataclasses.replace(
                    claim, growable=claim.growable & (rank < ac.max_new_per_step)
                )
                prior = prior + grow32.sum()
        new_conns, overflow, resort = [], jnp.bool_(False), jnp.bool_(False)
        for bucket, claim in zip(state.conns, claims, strict=True):
            new_bucket, over_b, resort_b = phases.xla_claim(bucket, claim)
            new_conns.append(new_bucket)
            overflow = overflow | over_b.any()
            resort = resort | resort_b.any()

        def live_count(conns: Any) -> jax.Array:
            return sum(jnp.sum(~c[DEAD.name]) for c in conns)

        return dataclasses.replace(
            state,
            conns=tuple(new_conns),
            needs_resort=state.needs_resort | resort,
            grown=(live_count(new_conns) - live_count(state.conns)).astype(jnp.int32),
            overflow=overflow,
        )

    return phase


def _kill(state: px.NetworkState[Any], frac: float, step: int) -> Any:
    """Tombstone a random slice of the live edges, freeing slots to claim."""
    rng = np.random.default_rng(step)
    conns = []
    for bucket in state.conns:
        dead = np.asarray(bucket[DEAD.name])
        dead = dead | (rng.random(dead.shape) < frac)
        conns.append({**bucket, DEAD.name: jnp.asarray(dead)})
    return dataclasses.replace(state, conns=tuple(conns))


def _same(a: jax.Array, b: jax.Array) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


@functools.cache
def _run(kind: str, selection: str, seed: int) -> dict[str, int]:
    """Run both phases for `_STEPS` steps; return coverage counts."""
    cfg = _config(kind, selection, seed)
    net, static, state = _build(kind, selection, cfg)
    new = jax.jit(phases.build_add_conn_phase(net, static, growth="xla"))
    ref = jax.jit(_reference_phase(net, static))
    inputs = px.StepInputs(
        inputs=jnp.zeros((len(static.input_ids),), jnp.float32), targets=None
    )
    stats = {"overflow": 0, "grown": 0, "buckets_grown": 0, "resort": 0}
    for step in range(_STEPS):
        got, _ = new(state, inputs)
        want = ref(state)
        where = f"{kind}/{selection}/seed{seed} step {step}"
        for b, (gb, wb) in enumerate(zip(got.conns, want.conns, strict=True)):
            assert gb.keys() == wb.keys()
            for name in gb:
                assert _same(gb[name], wb[name]), f"{where}: bucket {b} {name}"
            stats["buckets_grown"] += int(
                np.sum(~np.asarray(gb[DEAD.name]))
                != np.sum(~np.asarray(state.conns[b][DEAD.name]))
            )
        for field in ("overflow", "needs_resort", "grown"):
            assert _same(getattr(got, field), getattr(want, field)), f"{where}: {field}"
        stats["overflow"] += int(got.overflow)
        stats["grown"] += int(got.grown)
        stats["resort"] += int(got.needs_resort)
        state = _kill(
            dataclasses.replace(got, step=got.step + 1, globals_={"t": got.step + 1}),
            cfg["kill"],
            step,
        )
    return stats


_CASES = list(itertools.product(_KINDS, _SELECTIONS, _SEEDS))


@pytest.mark.parametrize(("kind", "selection", "seed"), _CASES)
def test_single_sort_matches_per_bucket_sorts(
    kind: str, selection: str, seed: int
) -> None:
    _run(kind, selection, seed)


def test_the_configs_exercise_overflow_and_several_levels() -> None:
    stats = [_run(*case) for case in _CASES]
    assert sum(s["overflow"] > 0 for s in stats) >= 5
    assert sum(s["buckets_grown"] >= 2 for s in stats) >= 10
    assert sum(s["grown"] for s in stats) > 0
    assert sum(s["resort"] > 0 for s in stats) >= 3
