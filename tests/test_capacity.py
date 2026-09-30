"""Bucket sizing: capacity_policy's power-of-two and aligned rounding, and the
policy carrying through the static config into grow_bucket and resort."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import plastax as px
from plastax import topo
from plastax.state import grow_bucket


def test_power_of_two_policy_is_unchanged() -> None:
    assert topo.capacity_policy(0) == 64
    assert topo.capacity_policy(100) == 128
    assert topo.capacity_policy(128) == 128
    assert topo.capacity_policy(128, headroom=0.01) == 256


def test_aligned_policy_tracks_the_target_closely() -> None:
    assert topo.capacity_policy(1000, align=128) == 1024
    assert topo.capacity_policy(1025, align=128) == 1152
    assert topo.capacity_policy(1000, headroom=0.05, align=128) == 1152  # 1050
    # min_bucket is honoured and itself rounded to the alignment.
    assert topo.capacity_policy(3, align=48) == 96
    # 300M live at 5% headroom: ~315M slots, not the 536,870,912 of pow2.
    cap = topo.capacity_policy(300_000_000, headroom=0.05, align=256)
    assert cap % 256 == 0 and 315_000_000 <= cap < 315_000_256


def test_align_must_be_positive() -> None:
    with pytest.raises(ValueError, match="align"):
        topo.capacity_policy(10, align=0)


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
    ) -> jax.Array:
        del dst, g
        return c[px.WEIGHT, cid] * u[px.ACTIVATION, src]

    def apply(
        self, u: px.UnitView, i: px.UnitIdx, g: None, acc: jax.Array
    ) -> px.UnitWrite:
        del u, i, g
        return px.UnitWrite.of((px.ACTIVATION, acc))


class _Net(px.Network[None]):
    forward_pass = _SumForward()
    propagation = px.Propagation.TOPOLOGICAL


def _build(
    live_per_layer: int, **kw: float | int | None
) -> tuple[px.NetworkStatic, px.NetworkState[None]]:
    width = 40
    rng = np.random.default_rng(0)
    src, dst = [], []
    for layer in range(2):
        ids = rng.choice(width * width, live_per_layer, replace=False)
        src.append(layer * width + ids // width)
        dst.append((layer + 1) * width + ids % width)
    return px.NetworkBuilder.from_edges(
        _Net,
        3 * width,
        np.concatenate(src).astype(np.int32),
        np.concatenate(dst).astype(np.int32),
        input_ids=list(range(width)),
        output_ids=list(range(2 * width, 3 * width)),
        globals_=None,
        **kw,  # type: ignore[arg-type]
    )


def test_from_edges_records_and_applies_the_aligned_policy() -> None:
    static, _ = _build(1000, capacity_headroom=0.1, capacity_align=32)
    assert static.capacity_headroom == 0.1
    assert static.capacity_align == 32
    assert static.level_capacities == (1120, 1120)  # ceil(1100 / 32) * 32


def test_grow_bucket_uses_the_recorded_alignment_and_grows_geometrically() -> None:
    static, state = _build(1000, capacity_align=32)
    assert static.level_capacities[0] == 1024
    grown, grown_state = grow_bucket(static, state, 0)
    # 1.5x of 1024 rounded to 32 -- not the power-of-two 2048.
    assert grown.level_capacities == (1536, 1024)
    assert grown_state.conns[0][px.DEAD.name].shape == (1536,)
    assert grown.capacity_align == 32


def test_resort_keeps_the_build_headroom() -> None:
    static, state = _build(1000, capacity_headroom=0.25, capacity_align=16)
    # Kill 100 edges of bucket 0, then resort: 900 live -> ceil(1125/16)*16.
    b0 = dict(state.conns[0])
    b0[px.DEAD.name] = b0[px.DEAD.name].at[jnp.arange(100)].set(True)
    state = dataclasses.replace(state, conns=(b0, state.conns[1]))
    new_static, _ = topo.resort(static, state)
    assert new_static.level_capacities[0] == 1136
    assert new_static.capacity_headroom == 0.25
