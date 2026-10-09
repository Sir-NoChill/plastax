"""Unit tests for the add_conn stage functions.

`build_add_conn_phase` is assembled from named stages (candidate production,
validity, the two dedupes, selection, claim). Each stage is pure and traced
inside the phase; these tests pin their contracts in isolation. End-to-end
behaviour is pinned separately by `test_growth_claim.py`'s sha digests.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from plastax._types import DEAD, FROM_ID, TO_ID
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
    assert np.asarray(select(scores, 3)).tolist() == [0, 1, 2]
    # k < n: genuine top-k by score
    assert np.asarray(select(scores, 2)).tolist() == [2, 0]


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
