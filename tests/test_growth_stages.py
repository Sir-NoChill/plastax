"""Unit tests for the add_conn stage functions.

`build_add_conn_phase` is assembled from named stages (candidate production,
validity, the two dedupes, selection, claim). Each stage is pure and traced
inside the phase; these tests pin their contracts in isolation. End-to-end
behaviour is pinned separately by `test_growth_claim.py`'s sha digests.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import plastax as px
from plastax import phases, views
from plastax import rng as rng_mod
from plastax._types import DEAD, FROM_ID, LEVEL, TO_ID
from plastax.phases import (
    apply_validity,
    candidates_grid,
    candidates_per_level,
    candidates_shortlist,
    dedupe_live,
    dedupe_step,
    select,
)


def test_candidates_grid_enumerates_every_ordered_pair() -> None:
    src, dst = candidates_grid(3)
    pairs = sorted(zip(np.asarray(src).tolist(), np.asarray(dst).tolist(), strict=True))
    assert pairs == sorted((s, d) for s in range(3) for d in range(3))


def test_candidates_shortlist_crosses_the_top_m_units() -> None:
    importance = jnp.asarray([0.1, 5.0, 0.2, 7.0], dtype=jnp.float32)
    src, dst = candidates_shortlist(importance, 2)
    top = {3, 1}  # the two most important unit ids
    assert set(np.asarray(src).tolist()) == top
    assert set(np.asarray(dst).tolist()) == top
    assert src.shape == dst.shape == (4,)


def test_candidates_per_level_sources_at_level_dests_strictly_deeper() -> None:
    #           unit:    0  1  2  3  4
    levels = jnp.asarray([0, 1, 1, 2, 3], dtype=jnp.int32)
    importance = jnp.asarray([9.0, 8.0, 7.0, 6.0, 5.0], dtype=jnp.float32)
    src, dst = candidates_per_level(importance, levels, 1, 2, 1)
    # sources: the top-2 among units AT level 1 -> {1, 2}
    assert set(np.asarray(src).tolist()) == {1, 2}
    # destinations: strictly deeper within gap 1 -> level 2 only -> unit 3.
    # top_k pads the second slot with the lowest-index -inf unit (0), so the
    # out-of-window unit 4 (level 3) must not appear at all, and neither may
    # the same-level units 1/2 (strictly-deeper window).
    dst_ids = set(np.asarray(dst).tolist())
    assert 3 in dst_ids
    assert dst_ids.isdisjoint({1, 2, 4})


@pytest.mark.parametrize("is_pipeline", [False, True])
def test_apply_validity_window_selfloop_and_source_level(is_pipeline: bool) -> None:
    levels = jnp.asarray([0, 1, 2, 0], dtype=jnp.int32)
    src = jnp.asarray([0, 0, 0, 1, 3], dtype=jnp.int32)
    dst = jnp.asarray([1, 2, 0, 2, 1], dtype=jnp.int32)
    valid = np.asarray(apply_validity(src, dst, levels, 0, 1, is_pipeline))
    # 0->1: gap 1, src level 0 == bucket -> valid in both modes
    assert bool(valid[0])
    # 0->2: gap 2 > max_level_gap 1 -> invalid
    assert not bool(valid[1])
    # 0->0: self-loop -> invalid
    assert not bool(valid[2])
    # 1->2: in window, but src level 1 != bucket 0 -> only PIPELINE admits it
    assert bool(valid[3]) == is_pipeline
    # 3->1: gap 1, src level 0 == bucket -> valid in both modes
    assert bool(valid[4])


def test_dedupe_step_keeps_each_pairs_highest_scored_copy() -> None:
    src = jnp.asarray([0, 0, 1, 0], dtype=jnp.int32)
    dst = jnp.asarray([1, 1, 2, 1], dtype=jnp.int32)
    scores = jnp.asarray([1.0, 3.0, 2.0, -jnp.inf], dtype=jnp.float32)
    out = np.asarray(dedupe_step(scores, src, dst))
    # the (0, 1) pair keeps only its 3.0 copy; the distinct (1, 2) survives
    assert out.tolist()[1] == 3.0
    assert out.tolist()[2] == 2.0
    assert out[0] == -np.inf and out[3] == -np.inf


def test_select_skips_the_sort_when_everything_fits() -> None:
    scores = jnp.asarray([5.0, -jnp.inf, 7.0], dtype=jnp.float32)
    src = jnp.asarray([0, 1, 2], dtype=jnp.int32)
    dst = jnp.asarray([1, 2, 3], dtype=jnp.int32)
    assert np.asarray(select(scores, src, dst, 3)).tolist() == [0, 1, 2]
    # k < n: genuine selection, descending by score
    assert np.asarray(select(scores, src, dst, 2)).tolist() == [2, 0]


def test_select_breaks_score_ties_by_src_then_dst_then_candidate_index() -> None:
    scores = jnp.asarray([4.0, 4.0, 4.0, 4.0, 9.0], dtype=jnp.float32)
    src = jnp.asarray([7, 2, 2, 2, 5], dtype=jnp.int32)
    dst = jnp.asarray([3, 9, 6, 6, 0], dtype=jnp.int32)
    # total order among the 4.0 ties: src 2 before src 7; within src 2,
    # dst 6 before dst 9; within (2, 6), candidate 2 before candidate 3.
    assert np.asarray(select(scores, src, dst, 5)).tolist() == [0, 1, 2, 3, 4]
    assert np.asarray(select(scores, src, dst, 4)).tolist() == [4, 2, 3, 1]


def test_select_is_identical_under_jit() -> None:
    scores = jnp.asarray([1.0, 1.0, 1.0, 2.0, -jnp.inf], dtype=jnp.float32)
    src = jnp.asarray([3, 1, 1, 0, 0], dtype=jnp.int32)
    dst = jnp.asarray([0, 5, 4, 2, 1], dtype=jnp.int32)
    eager = np.asarray(select(scores, src, dst, 3))
    jitted = np.asarray(jax.jit(select, static_argnums=3)(scores, src, dst, 3))
    assert eager.tolist() == jitted.tolist() == [3, 2, 1]


def test_select_sorts_nan_scores_last() -> None:
    scores = jnp.asarray([jnp.nan, 1.0, 0.5], dtype=jnp.float32)
    src = jnp.asarray([0, 1, 2], dtype=jnp.int32)
    dst = jnp.asarray([1, 2, 0], dtype=jnp.int32)
    # a NaN score must not outrank finite candidates (it is vetoed at commit)
    assert np.asarray(select(scores, src, dst, 2)).tolist() == [1, 2]


def test_dedupe_step_keeps_the_total_order_first_copy_on_equal_scores() -> None:
    src = jnp.asarray([4, 4, 4], dtype=jnp.int32)
    dst = jnp.asarray([6, 6, 6], dtype=jnp.int32)
    scores = jnp.asarray([2.0, 2.0, 2.0], dtype=jnp.float32)
    out = np.asarray(dedupe_step(scores, src, dst))
    # equal pair, equal scores: the earliest candidate index survives
    assert out.tolist() == [2.0, -np.inf, -np.inf]


def test_dedupe_live_masks_live_edges_but_not_dead_ones() -> None:
    bucket = {
        FROM_ID.name: jnp.asarray([0, 1, 2, 0], dtype=jnp.int32),
        TO_ID.name: jnp.asarray([1, 2, 3, 0], dtype=jnp.int32),
        DEAD.name: jnp.asarray([False, True, False, True]),
    }
    src = jnp.asarray([0, 1, 2], dtype=jnp.int32)
    dst = jnp.asarray([1, 2, 3], dtype=jnp.int32)
    ok = np.asarray(dedupe_live(bucket, src, dst, 4, None))
    # (0,1) live -> masked; (1,2) only exists dead -> allowed; (2,3) live -> masked
    assert ok.tolist() == [False, True, False]


def test_network_level_neighbourhood_is_rejected_with_guidance() -> None:
    class _Fwd(px.ForwardPass):
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
        ) -> px.UnitWrite:
            return px.UnitWrite.of((px.ACTIVATION, acc))

    with pytest.raises(TypeError, match="max_level_gap"):

        class _Net(px.Network[None]):
            forward_pass = _Fwd()
            neighbourhood = 1


# ---------------------------------------------------------------------------
# Proposer candidate stages (per-unit / per-connection / global)
# ---------------------------------------------------------------------------


class _StageForward(px.ForwardPass):
    combine = px.monoid.sum_

    def map(  # noqa: D102 -- test pass
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

    def apply(  # noqa: D102 -- test pass
        self, u: px.UnitView, i: px.UnitIdx, g: None, acc: jax.Array
    ) -> px.UnitWrite:
        del u, i, g
        return px.UnitWrite.of((px.ACTIVATION, acc))


def _site_uniform_int(seed: int, step: int, key: int, j: int, n: int) -> int:
    """First uniform_int draw of a site, straight from the rng contract."""
    rng = rng_mod.Rng.for_site(seed, jnp.int32(step), 1, jnp.uint32(key), jnp.int32(j))
    return int(rng.uniform_int(n))


def test_per_unit_candidates_lay_out_unit_major() -> None:
    """Position i * P + j holds unit i's j-th proposal (the 6.4-style index)."""
    n_units, p = 5, 3
    u_view = views.UnitView({LEVEL.name: jnp.zeros((n_units,), jnp.int32)})

    def propose(
        u: views.UnitView, i: jax.Array, j: jax.Array, g: None, rng: object
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        del u, g, rng
        return i, (i + j + 1) % n_units, (i * 10 + j).astype(jnp.float32)

    src, dst, score, ok = phases.candidates_propose_per_unit(
        propose, u_view, None, p, n_units, seed=0, step=jnp.int32(0)
    )
    assert src.shape == (n_units * p,)
    for i in range(n_units):
        for j in range(p):
            pos = i * p + j
            assert int(src[pos]) == i
            assert int(dst[pos]) == (i + j + 1) % n_units
            assert float(score[pos]) == i * 10 + j
            assert bool(ok[pos])


def test_per_unit_sites_draw_the_contracted_streams() -> None:
    """The rng handed to (i, j) is the (seed, step, 1, i, j) site stream."""
    n_units, p, seed, step = 4, 2, 11, 7
    u_view = views.UnitView({LEVEL.name: jnp.zeros((n_units,), jnp.int32)})

    def propose(
        u: views.UnitView, i: jax.Array, j: jax.Array, g: None, rng: rng_mod.Rng
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        del u, g
        return i, rng.uniform_int(n_units).astype(jnp.int32), jnp.float32(0.0)

    _, dst, _, _ = phases.candidates_propose_per_unit(
        propose, u_view, None, p, n_units, seed=seed, step=jnp.int32(step)
    )
    for i in range(n_units):
        for j in range(p):
            assert int(dst[i * p + j]) == _site_uniform_int(seed, step, i, j, n_units)


def test_per_conn_candidates_rank_by_src_dst_occurrence() -> None:
    """Ranks follow ascending (src, dst, occurrence); dead proposers veto."""
    # Bucket layout (slot order): (2, 3) dead, (1, 2), (0, 1), (1, 2) -- the
    # second (1, 2) is occurrence 1. Ranks: (0,1)=0, (1,2)occ0=1, (1,2)occ1=2.
    conns = [
        {
            FROM_ID.name: jnp.asarray([2, 1, 0, 1], jnp.int32),
            TO_ID.name: jnp.asarray([3, 2, 1, 2], jnp.int32),
            DEAD.name: jnp.asarray([True, False, False, False]),
        }
    ]
    n_units, p = 4, 1
    u_view = views.UnitView({LEVEL.name: jnp.zeros((n_units,), jnp.int32)})
    seen: list[tuple[int, int]] = []

    def propose(
        u: views.UnitView,
        c: views.ConnView,
        cid: jax.Array,
        j: jax.Array,
        g: None,
        rng: object,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        del u, j, g, rng
        src = c[FROM_ID, cid]
        dst = c[TO_ID, cid]
        return (
            jnp.asarray(src, jnp.int32),
            jnp.asarray(dst, jnp.int32),
            jnp.asarray(src * 10 + dst, jnp.float32),
        )

    src, dst, score, ok = phases.candidates_propose_per_conn(
        propose, u_view, conns, None, p, n_units, seed=0, step=jnp.int32(0)
    )
    del seen
    # Rank-major layout: positions 0..2 are the live edges ascending.
    assert [int(s) for s in src[:3]] == [0, 1, 1]
    assert [int(d) for d in dst[:3]] == [1, 2, 2]
    assert [bool(o) for o in ok[:3]] == [True, True, True]
    assert not bool(ok[3])  # the dead proposer's candidate is vetoed


def test_per_conn_occurrence_distinguishes_parallel_edge_sites() -> None:
    """Parallel edges draw from distinct site streams (occurrence keyed)."""
    conns = [
        {
            FROM_ID.name: jnp.asarray([1, 1], jnp.int32),
            TO_ID.name: jnp.asarray([2, 2], jnp.int32),
            DEAD.name: jnp.asarray([False, False]),
        }
    ]
    n_units, p, seed, step = 65536, 1, 3, 5
    u_view = views.UnitView({LEVEL.name: jnp.zeros((2,), jnp.int32)})

    def propose(
        u: views.UnitView,
        c: views.ConnView,
        cid: jax.Array,
        j: jax.Array,
        g: None,
        rng: rng_mod.Rng,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        del u, c, cid, g
        return (
            jnp.int32(0),
            rng.uniform_int(n_units).astype(jnp.int32),
            jnp.float32(0.0),
        )

    _, dst, _, _ = phases.candidates_propose_per_conn(
        propose, u_view, conns, None, p, n_units, seed=seed, step=jnp.int32(step)
    )
    want = [
        _site_uniform_int(
            seed,
            step,
            int(rng_mod.conn_key(jnp.uint32(1), jnp.uint32(2), jnp.uint32(occ))),
            0,
            n_units,
        )
        for occ in (0, 1)
    ]
    assert [int(d) for d in dst[:2]] == want
    assert want[0] != want[1]  # occurrence actually separates the streams


def test_per_connection_proposer_is_rejected_under_sharding() -> None:
    """Scheme-A sharding cannot rank sharded proposers; the builder says so."""

    class _PerConn(px.ProposeAddConn[None]):
        proposer = "per_connection"
        max_candidates = 1
        proposals_per_proposer = 1

        def propose(  # noqa: D102 -- test rule
            self, *args: object
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            del args
            return jnp.int32(0), jnp.int32(1), jnp.float32(0.0)

        def init(  # noqa: D102 -- test rule
            self, u: object, src: object, dst: object, g: None
        ) -> px.ConnWrite:
            del u, src, dst, g
            return px.ConnWrite.of((px.WEIGHT, jnp.float32(0.0)))

    class _Net(px.Network[None]):
        forward_pass = _StageForward()
        add_conn = _PerConn()
        propagation = px.Propagation.TOPOLOGICAL

    static, state = px.NetworkBuilder.from_edges(
        _Net,
        4,
        np.asarray([0, 1], dtype=np.int32),
        np.asarray([2, 3], dtype=np.int32),
        input_ids=[0, 1],
        output_ids=[2, 3],
        globals_=None,
        sharding=px.ShardSpec(axis_name="conn", num_shards=2),
    )
    with pytest.raises(NotImplementedError, match="per-connection"):
        phases.build_add_conn_phase(_Net, static)
