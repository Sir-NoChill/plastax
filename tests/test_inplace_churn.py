"""In-place structural churn: prune and add rewrite buckets without a resort.

Prune tombstones edges where they sit and add fills whichever dead slots are
free, so after the first churn step a bucket's `to_id` order (the order the
builder and `topo.resort` leave) no longer holds. These tests pin that the
forward stays exact on such scrambled buckets against a numpy reference over
the live edge multiset.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import plastax as px
import plastax.driver

_WIDTH = 24  # units per layer; three layers (input, hidden, output)
_NUM_UNITS = 3 * _WIDTH
_EDGES_PER_LAYER = 160
_KILL_P = 0.2


def _hash01(a: jax.Array, b: jax.Array, c: jax.Array) -> jax.Array:
    h = (a.astype(jnp.uint32) + jnp.uint32(0x9E3779B1)) * jnp.uint32(0x85EBCA77)
    h = (h ^ b.astype(jnp.uint32)) * jnp.uint32(0xC2B2AE3D)
    h = (h ^ c.astype(jnp.uint32)) * jnp.uint32(0x27D4EB2F)
    h = h ^ (h >> 15)
    return (h >> jnp.uint32(8)).astype(jnp.float32) / jnp.float32(1 << 24)


class _SumForward(px.ForwardPass):
    combine = px.monoid.sum_

    def map(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: dict[str, jax.Array],
    ) -> jax.Array:
        del dst, g
        return c[px.WEIGHT, cid] * u[px.ACTIVATION, src]

    def apply(
        self, u: px.UnitView, i: px.UnitIdx, g: dict[str, jax.Array], acc: jax.Array
    ) -> px.UnitWrite:
        del u, i, g
        return px.UnitWrite.of((px.ACTIVATION, acc))


class _HashPrune(px.PruneConn):
    def predicate(
        self, u: px.UnitView, c: px.ConnView, cid: px.ConnIdx, g: dict[str, jax.Array]
    ) -> jax.Array:
        del u
        return _hash01(c[px.FROM_ID, cid], c[px.TO_ID, cid], g["step"]) < _KILL_P


class _HashGrow(px.AddConn):
    max_candidates = 32

    def score(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: dict[str, jax.Array]
    ) -> jax.Array:
        deeper = u[px.LEVEL, dst] == u[px.LEVEL, src] + 1
        return jnp.where(deeper, _hash01(src, dst, g["step"] + 977), -jnp.inf)

    def init(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: dict[str, jax.Array]
    ) -> px.ConnWrite:
        del u
        return px.ConnWrite.of((px.WEIGHT, _hash01(dst, src, g["step"]) - 0.5))


class _Tick(px.ResetGlobal):
    def reset(self, g: dict[str, jax.Array]) -> dict[str, jax.Array]:
        return {"step": g["step"] + 1}


class _ChurnNet(px.Network[dict[str, jax.Array]]):
    forward_pass = _SumForward()
    prune_conn = _HashPrune()
    add_conn = _HashGrow()
    reset_global = _Tick()
    propagation = px.Propagation.TOPOLOGICAL


def _build() -> tuple[px.NetworkStatic, px.NetworkState[dict[str, jax.Array]]]:
    return _build_with(_ChurnNet, headroom=0.5)


def _build_with(
    net: type[px.Network[dict[str, jax.Array]]], *, headroom: float
) -> tuple[px.NetworkStatic, px.NetworkState[dict[str, jax.Array]]]:
    rng = np.random.default_rng(11)
    src, dst = [], []
    for layer in range(2):
        ids = rng.choice(_WIDTH * _WIDTH, _EDGES_PER_LAYER, replace=False)
        src.append(layer * _WIDTH + ids // _WIDTH)
        dst.append((layer + 1) * _WIDTH + ids % _WIDTH)
    frm = np.concatenate(src).astype(np.int32)
    to = np.concatenate(dst).astype(np.int32)
    return px.NetworkBuilder.from_edges(
        net,
        _NUM_UNITS,
        frm,
        to,
        weights=rng.standard_normal(frm.shape[0]).astype(np.float32),
        input_ids=list(range(_WIDTH)),
        output_ids=list(range(2 * _WIDTH, 3 * _WIDTH)),
        globals_={"step": jnp.int32(0)},
        capacity_headroom=headroom,
    )


def _reference_forward(
    state: px.NetworkState[dict[str, jax.Array]], x: np.ndarray
) -> np.ndarray:
    """Layer-by-layer weighted sum over the live edges, in float64."""
    act = np.zeros(_NUM_UNITS)
    act[:_WIDTH] = x
    level = np.asarray(state.units[px.LEVEL.name])
    for src_level, bucket in enumerate(state.conns):
        live = ~np.asarray(bucket[px.DEAD.name])
        frm = np.asarray(bucket[px.FROM_ID.name])[live]
        to = np.asarray(bucket[px.TO_ID.name])[live]
        w = np.asarray(bucket[px.WEIGHT.name])[live].astype(np.float64)
        acc = np.zeros(_NUM_UNITS)
        np.add.at(acc, to, w * act[frm])
        mask = level == src_level + 1
        act[mask] = acc[mask]
    return act


def test_forward_is_exact_on_buckets_scrambled_by_in_place_churn() -> None:
    static, state = _build()
    step = px.make_step(_ChurnNet, static)
    x = np.linspace(-1.0, 1.0, _WIDTH).astype(np.float32)
    inputs = px.StepInputs(inputs=jnp.asarray(x), targets=None)
    for _ in range(6):
        result = step(state, inputs)
        assert not bool(result.overflow)
        state = result.state
    assert not bool(state.needs_resort)

    # The churn really did scramble to_id order: in some bucket the null-slot
    # target sequence (dead -> num_units) descends somewhere.
    def descends(bucket: px.state.Columns) -> bool:
        dead = np.asarray(bucket[px.DEAD.name])
        tgt = np.where(dead, _NUM_UNITS, np.asarray(bucket[px.TO_ID.name]))
        return bool(np.any(np.diff(tgt) < 0))

    assert any(descends(bucket) for bucket in state.conns)

    # One more step: its forward runs on the scrambled buckets, before that
    # step's own prune/add touch them.
    want = _reference_forward(state, x)
    got = np.asarray(step(state, inputs).state.units[px.ACTIVATION.name])
    np.testing.assert_allclose(got[_WIDTH:], want[_WIDTH:], rtol=1e-5, atol=1e-5)


# --- Equivalence under the Driver: in place vs rebuilt from the live edges ---
#
# The plastax counterpart of plastix's `--validate-every` check. Proposal growth
# runs through the Driver long enough to overflow buckets (grow_bucket) and to
# commit same-level edges (needs_resort -> topo.resort); afterwards the forward
# on the churned arena must equal the forward on a net rebuilt from scratch
# from its live edge multiset (parallel edges included).

_FANOUT = 2


class _FanoutGrow(px.ProposeAddConn[dict[str, jax.Array]]):
    """Each unit proposes _FANOUT random partners; only src < dst (a DAG)."""

    max_candidates = 24
    num_proposals = _NUM_UNITS * _FANOUT

    def propose(
        self, u: px.UnitView, j: jax.Array, g: dict[str, jax.Array]
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        del u
        src = j // _FANOUT
        dst = (_hash01(j, g["step"], jnp.int32(5)) * _NUM_UNITS).astype(jnp.int32)
        score = jnp.where(src < dst, _hash01(dst, j, g["step"]), -jnp.inf)
        return src, dst, score

    def init(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: dict[str, jax.Array]
    ) -> px.ConnWrite:
        del u
        return px.ConnWrite.of((px.WEIGHT, _hash01(dst, src, g["step"]) - 0.5))


class _SlowPrune(px.PruneConn):
    def predicate(
        self, u: px.UnitView, c: px.ConnView, cid: px.ConnIdx, g: dict[str, jax.Array]
    ) -> jax.Array:
        del u
        return _hash01(c[px.FROM_ID, cid], c[px.TO_ID, cid], g["step"]) < 0.03


class _GrowingNet(px.Network[dict[str, jax.Array]]):
    forward_pass = _SumForward()
    prune_conn = _SlowPrune()
    add_conn = _FanoutGrow()
    reset_global = _Tick()
    propagation = px.Propagation.TOPOLOGICAL


class _ForwardOnly(px.Network[dict[str, jax.Array]]):
    forward_pass = _SumForward()
    propagation = px.Propagation.TOPOLOGICAL


def _live_edges(
    state: px.NetworkState[dict[str, jax.Array]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frm, to, w = [], [], []
    for bucket in state.conns:
        live = ~np.asarray(bucket[px.DEAD.name])
        frm.append(np.asarray(bucket[px.FROM_ID.name])[live])
        to.append(np.asarray(bucket[px.TO_ID.name])[live])
        w.append(np.asarray(bucket[px.WEIGHT.name])[live])
    return np.concatenate(frm), np.concatenate(to), np.concatenate(w)


def test_in_place_churn_matches_a_rebuild_through_overflow_and_resort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"grow": 0, "resort": 0}
    grow, resort = plastax.driver.grow_bucket, plastax.driver.topo.resort

    def counted_grow(*a: Any) -> Any:
        calls["grow"] += 1
        return grow(*a)

    def counted_resort(*a: Any) -> Any:
        calls["resort"] += 1
        return resort(*a)

    monkeypatch.setattr(plastax.driver, "grow_bucket", counted_grow)
    monkeypatch.setattr(plastax.driver.topo, "resort", counted_resort)

    static, state = _build_with(_GrowingNet, headroom=0.0)
    driver = px.Driver(_GrowingNet, static, state)
    x = np.linspace(-1.0, 1.0, _WIDTH).astype(np.float32)
    inputs = px.StepInputs(inputs=jnp.asarray(x), targets=None)
    # Run until both structural paths have fired: a bucket overflowed
    # (grow_bucket) and a same-level src < dst edge forced a resort.
    for _ in range(40):
        driver.step(inputs)
        if calls["grow"] and calls["resort"]:
            break
    assert calls["grow"] and calls["resort"], calls

    frm, to, w = _live_edges(driver.state)
    assert len(set(zip(frm.tolist(), to.tolist(), strict=True))) < frm.size  # parallel
    fwd = px.make_step(_ForwardOnly, driver.static)
    got = np.asarray(fwd(driver.state, inputs).state.units[px.ACTIVATION.name])

    r_static, r_state = px.NetworkBuilder.from_edges(
        _ForwardOnly,
        _NUM_UNITS,
        frm,
        to,
        weights=w,
        input_ids=list(range(_WIDTH)),
        output_ids=list(range(2 * _WIDTH, 3 * _WIDTH)),
        globals_={"step": jnp.int32(0)},
    )
    rebuilt = px.make_step(_ForwardOnly, r_static)(r_state, inputs)
    want = np.asarray(rebuilt.state.units[px.ACTIVATION.name])
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)
