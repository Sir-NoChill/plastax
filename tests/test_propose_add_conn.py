"""ProposeAddConn: growth from policy-emitted proposals instead of a grid.

A 4-4-4 net (input level 0, hidden level 1, output level 2) starts wired
i -> 4+i -> 8+i, so every unit sits at its layer's level; bucket 0 holds
(i, 4+i) and bucket 1 holds (4+i, 8+i). Each test's policy
emits a fixed proposal table, so every assertion is about the framework:
bucket routing, the level window, vetoes, per-bucket top-k, the live and
within-step duplicate checks, and init.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import plastax as px
from plastax import phases

_DUMMY_INPUTS = px.StepInputs(inputs=jnp.zeros((0,), dtype=jnp.float32), targets=None)
_NEW_WEIGHT = 7.0
_BASE_SRC = [0, 1, 2, 3, 4, 5, 6, 7]
_BASE_DST = [4, 5, 6, 7, 8, 9, 10, 11]
_BASE0 = [(0, 4), (1, 5), (2, 6), (3, 7)]
_BASE1 = [(4, 8), (5, 9), (6, 10), (7, 11)]


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


class _TableProposals(px.ProposeAddConn[None]):
    """Proposal j is row j of a fixed (src, dst, score) table."""

    def __init__(
        self,
        table: list[tuple[int, int, float]],
        max_candidates: int,
        dedupe: bool | None = None,
    ) -> None:
        self.src = jnp.asarray([r[0] for r in table], dtype=jnp.int32)
        self.dst = jnp.asarray([r[1] for r in table], dtype=jnp.int32)
        self.scores = jnp.asarray([r[2] for r in table], dtype=jnp.float32)
        self.num_proposals = len(table)
        self.max_candidates = max_candidates
        if dedupe is not None:
            self.dedupe = dedupe

    def propose(
        self, u: px.UnitView, j: jax.Array, g: None
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        del u, g
        return self.src[j], self.dst[j], self.scores[j]

    def init(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: None
    ) -> px.ConnWrite:
        del u, src, dst, g
        return px.ConnWrite.of((px.WEIGHT, jnp.float32(_NEW_WEIGHT)))


def _net(policy: px.ProposeAddConn[None]) -> type[px.Network[None]]:
    class Net(px.Network[None]):
        forward_pass = _SumForward()
        add_conn = policy
        propagation = px.Propagation.TOPOLOGICAL

    return Net


def _grow(
    table: list[tuple[int, int, float]], max_candidates: int, dedupe: bool | None = None
) -> list[list[tuple[int, int]]]:
    """Build the 4-4-4 net, run one add phase, return each bucket's grown pairs."""
    net = _net(_TableProposals(table, max_candidates, dedupe))
    static, state = px.NetworkBuilder.from_edges(
        net,
        12,
        np.asarray(_BASE_SRC, dtype=np.int32),
        np.asarray(_BASE_DST, dtype=np.int32),
        input_ids=[0, 1, 2, 3],
        output_ids=[8, 9, 10, 11],
        globals_=None,
    )
    new_state, _ = phases.build_add_conn_phase(net, static)(state, _DUMMY_INPUTS)
    out = []
    for bucket in new_state.conns:
        dead = np.asarray(bucket[px.DEAD.name])
        frm = np.asarray(bucket[px.FROM_ID.name])
        to = np.asarray(bucket[px.TO_ID.name])
        w = np.asarray(bucket[px.WEIGHT.name])
        pairs = [
            (int(f), int(t)) for f, t, d in zip(frm, to, dead, strict=True) if not d
        ]
        grown = sorted(pairs)
        for base in (_BASE0, _BASE1):
            for pair in base:
                if pair in grown:
                    grown.remove(pair)
        new_w = [float(x) for x, d in zip(w, dead, strict=True) if not d]
        assert new_w.count(_NEW_WEIGHT) == len(grown)  # every grown edge init'd
        out.append(grown)
    return out


def test_proposals_route_to_their_source_level_and_respect_the_window() -> None:
    table = [
        (1, 4, 1.0),  # level 0 -> 1: bucket 0
        (5, 8, 1.0),  # level 1 -> 2: bucket 1
        (2, 10, 1.0),  # level 0 -> 2: outside neighbourhood 1, dropped
        (6, 6, 1.0),  # self-loop, dropped
        (3, 5, -jnp.inf),  # vetoed
        (99, 5, 1.0),  # out-of-range id, vetoed
    ]
    bucket0, bucket1 = _grow(table, max_candidates=4)
    assert bucket0 == [(1, 4)]
    assert bucket1 == [(5, 8)]


def test_each_bucket_keeps_its_own_top_k_by_score() -> None:
    table = [(1, 4, 1.0), (2, 4, 3.0), (3, 4, 2.0), (5, 8, 0.5)]
    bucket0, bucket1 = _grow(table, max_candidates=2)
    assert bucket0 == [(2, 4), (3, 4)]  # the top 2 of 3 in bucket 0
    assert bucket1 == [(5, 8)]  # bucket 1's lone proposal


def test_default_allows_parallel_edges() -> None:
    table = [(0, 4, 2.0), (1, 4, 1.0), (1, 4, 1.0)]
    (bucket0, _) = _grow(table, max_candidates=3)
    # A repeat of the live (0, 4) and a within-step repeat both grow.
    assert bucket0 == [(0, 4), (1, 4), (1, 4)]


def test_dedupe_excludes_live_and_within_step_duplicates() -> None:
    table = [(0, 4, 2.0), (1, 4, 1.0), (1, 4, 1.0), (2, 4, 0.5)]
    (bucket0, _) = _grow(table, max_candidates=4, dedupe=True)
    assert bucket0 == [(1, 4), (2, 4)]


class _BothSources(_TableProposals):
    def score(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: None
    ) -> jax.Array:
        del u, src, dst, g
        return jnp.float32(0.0)


def test_add_conn_must_be_exactly_one_of_grid_or_propose() -> None:
    with pytest.raises(TypeError, match="exactly one"):
        _net(_BothSources([(1, 5, 1.0)], 1))


def test_within_step_repeats_do_not_consume_top_k_slots_under_dedupe() -> None:
    # Two copies of (1, 4) outscore (2, 4); with k = 2 the repeat must be
    # vetoed before top_k so (2, 4) still grows.
    table = [(1, 4, 1.0), (1, 4, 1.0), (2, 4, 0.5)]
    (bucket0, _) = _grow(table, max_candidates=2, dedupe=True)
    assert bucket0 == [(1, 4), (2, 4)]


def test_num_proposals_must_be_positive() -> None:
    policy = _TableProposals([(1, 4, 1.0)], 1)
    policy.num_proposals = 0
    with pytest.raises(TypeError, match="num_proposals"):
        _net(policy)


class _WideProposals(px.ProposeAddConn[None]):
    max_candidates = 4
    num_proposals = 8

    def propose(
        self, u: px.UnitView, j: jax.Array, g: None
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        del u, g
        return j, j + 1, jnp.float32(1.0)

    def init(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: None
    ) -> px.ConnWrite:
        del u, src, dst, g
        return px.ConnWrite.of((px.WEIGHT, jnp.float32(_NEW_WEIGHT)))


def test_building_a_proposal_phase_allocates_no_candidate_grid() -> None:
    # Regression: the builder eagerly materialised the num_units^2 grid (two
    # int32 columns) for every add_conn policy, including proposal ones.
    num_units = 3000
    net = _net(_WideProposals())
    static, state = px.NetworkBuilder.from_edges(
        net,
        num_units,
        np.asarray([0], dtype=np.int32),
        np.asarray([1], dtype=np.int32),
        input_ids=[0],
        output_ids=[num_units - 1],
        globals_=None,
    )
    before = sum(a.nbytes for a in jax.live_arrays())
    phase = phases.build_add_conn_phase(net, static)
    held = sum(a.nbytes for a in jax.live_arrays()) - before
    assert held < 1 << 20, held  # a grid would be 2 * 3000**2 * 4 = 72 MB
    del phase


class _PipelineNet(px.Network[None]):
    forward_pass = _SumForward()
    add_conn = _TableProposals([(1, 4, 1.0), (5, 8, 1.0), (4, 1, 1.0)], 3)
    propagation = px.Propagation.PIPELINE


def test_pipeline_proposals_land_in_the_single_bucket() -> None:
    static, state = px.NetworkBuilder.from_edges(
        _PipelineNet,
        12,
        np.asarray(_BASE_SRC, dtype=np.int32),
        np.asarray(_BASE_DST, dtype=np.int32),
        input_ids=[0, 1, 2, 3],
        output_ids=[8, 9, 10, 11],
        globals_=None,
    )
    new_state, _ = phases.build_add_conn_phase(_PipelineNet, static)(
        state, _DUMMY_INPUTS
    )
    (bucket,) = new_state.conns
    dead = np.asarray(bucket[px.DEAD.name])
    pairs = {
        (int(f), int(t))
        for f, t, d in zip(
            np.asarray(bucket[px.FROM_ID.name]),
            np.asarray(bucket[px.TO_ID.name]),
            dead,
            strict=True,
        )
        if not d
    }
    # All three are within the window (|level gap| <= 1), including the
    # backward (4, 1): pipeline propagation takes any source level.
    assert {(1, 4), (5, 8), (4, 1)} <= pairs
    assert len(pairs) == len(_BASE_SRC) + 3


def test_new_edges_fill_interleaved_holes_and_leave_live_edges_intact() -> None:
    net = _net(_TableProposals([(1, 4, 3.0), (2, 4, 2.0), (3, 4, 1.0)], 3))
    static, state = px.NetworkBuilder.from_edges(
        net,
        12,
        np.asarray(_BASE_SRC, dtype=np.int32),
        np.asarray(_BASE_DST, dtype=np.int32),
        weights=np.arange(1, 9, dtype=np.float32),
        input_ids=[0, 1, 2, 3],
        output_ids=[8, 9, 10, 11],
        globals_=None,
    )
    # Tombstone slots 1 and 3 of bucket 0, leaving live slots 0 and 2
    # between holes (and the padding from slot 4 on).
    b0 = dict(state.conns[0])
    b0[px.DEAD.name] = b0[px.DEAD.name].at[jnp.asarray([1, 3])].set(True)
    before = {k: np.asarray(v) for k, v in b0.items()}
    state = px.NetworkState(
        units=state.units,
        conns=(b0, state.conns[1]),
        globals_=None,
        needs_resort=state.needs_resort,
    )
    new_state, _ = phases.build_add_conn_phase(net, static)(state, _DUMMY_INPUTS)
    after = {k: np.asarray(v) for k, v in new_state.conns[0].items()}

    for slot in (0, 2):  # surviving live edges are untouched
        for name in (px.FROM_ID.name, px.TO_ID.name, px.WEIGHT.name, px.DEAD.name):
            assert after[name][slot] == before[name][slot], (slot, name)
    # The three new edges take the first three free slots, in top_k order.
    grown = [
        (int(after[px.FROM_ID.name][i]), int(after[px.TO_ID.name][i]))
        for i in (1, 3, 4)
    ]
    assert grown == [(1, 4), (2, 4), (3, 4)]
    assert not after[px.DEAD.name][[1, 3, 4]].any()
    assert after[px.DEAD.name][5:].all()
