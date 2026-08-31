"""AddConn (M4a).

K-bounded candidates, top_k selection, prefix-sum slot claim, overflow flag,
and level-preserving adds do not set needs_resort (rung0 design section 5).

Shared network: unit 0 (ANCHOR) and unit 1 (SRC) both sit at level 0; units
2..6 (DST) sit at level 1, each already wired from ANCHOR with a DISTINCT
weight (5.0, 4.0, 3.0, 2.0, 1.0) so a plain weighted-sum forward pass gives
each DST a distinct, predictable activation to score candidates against.
SRC starts with NO outgoing edges, so every (SRC, dst) pair is a genuinely
fresh add_conn candidate; the (ANCHOR, dst) pairs are already-live edges
that add_conn's own window would also propose (native tolerates such
duplicate/parallel proposals in its sampled GrowFanout path,
dispatch_cpu.hpp:849-856) -- the score policy below gives any SRC-sourced
candidate a large bonus so top_k always prefers the (SRC, dst) set under
test over the (ANCHOR, dst) duplicates, keeping every assertion below fully
deterministic.
"""

from __future__ import annotations

import dataclasses
import warnings

import jax
import jax.numpy as jnp
import numpy as np

import plastax as px
from plastax import phases

_DUMMY_INPUTS = px.StepInputs(inputs=jnp.zeros((0,), dtype=jnp.float32), targets=None)

_ANCHOR = 0
_SRC = 1
_DST = (2, 3, 4, 5, 6)
_ANCHOR_WEIGHTS = (5.0, 4.0, 3.0, 2.0, 1.0)
_NEW_WEIGHT = 42.0
_SRC_BONUS = 100.0


class _SumForward(px.ForwardPass):
    """Weighted-sum forward (test_forward_pipeline.py's _SumForward,
    reused verbatim): map = weight * activation[src], combine = sum,
    apply = identity onto ACTIVATION."""

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
        return c[px.WEIGHT, cid] * u[px.ACTIVATION, src]

    def apply(
        self, u: px.UnitView, i: px.UnitIdx, g: None, acc: jax.Array
    ) -> px.UnitWrite:
        return px.UnitWrite.of((px.ACTIVATION, acc))


class _WindowAddConn(px.AddConn[None]):
    """score favors any (SRC, *) candidate by a large, fixed bonus over
    dst's own activation, so top_k's selection among (SRC, dst) pairs is
    exactly "highest anchor weight first" (dst 2, then 3, then 4) and never
    confuses a (SRC, dst) candidate with an (ANCHOR, dst) duplicate. init
    tags every newly-committed edge with a distinct, recognizable weight
    (_NEW_WEIGHT) so added edges are trivially distinguishable from the
    pre-existing ANCHOR edges (5.0..1.0) and from a dead slot's
    FieldSpec-default weight (0.0)."""

    max_candidates = 3

    def score(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: None
    ) -> jax.Array:
        del g
        bonus = jnp.where(
            src == jnp.int32(_SRC), jnp.float32(_SRC_BONUS), jnp.float32(0.0)
        )
        return u[px.ACTIVATION, dst] + bonus

    def init(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: None
    ) -> px.ConnWrite:
        del u, src, dst, g
        return px.ConnWrite.of((px.WEIGHT, jnp.float32(_NEW_WEIGHT)))


class _AddConnNet(px.Network[None]):
    forward_pass = _SumForward()
    add_conn = _WindowAddConn()
    propagation = px.Propagation.TOPOLOGICAL


class _PipelineAddConnNet(px.Network[None]):
    forward_pass = _SumForward()
    add_conn = _WindowAddConn()
    propagation = px.Propagation.PIPELINE


def _build_net(
    net: type[px.Network[None]],
) -> tuple[px.NetworkStatic, px.NetworkState[None]]:
    builder = px.NetworkBuilder(net, None)
    builder.add_unit()  # 0: ANCHOR
    builder.add_unit()  # 1: SRC
    for _ in _DST:
        builder.add_unit()  # 2..6: DST
    builder.mark_input(_ANCHOR)
    builder.mark_input(_SRC)
    for dst, w in zip(_DST, _ANCHOR_WEIGHTS, strict=True):
        builder.add_conn(_ANCHOR, dst, weight=w)
    return builder.finalize()


def _with_marker_activations(
    state: px.NetworkState[None],
) -> px.NetworkState[None]:
    """A real forward pass would write ANCHOR_WEIGHTS into DST's activation
    (weight * ANCHOR's activation of 1.0); tests that call
    build_add_conn_phase directly (bypassing forward) seed that same
    distinct-per-dst marker by hand so `score` sees what a real forward
    pass would have produced."""
    dst_ids = jnp.asarray(_DST, dtype=jnp.int32)
    markers = jnp.asarray(_ANCHOR_WEIGHTS, dtype=jnp.float32)
    activation = state.units[px.ACTIVATION.name].at[dst_ids].set(markers)
    units = {**state.units, px.ACTIVATION.name: activation}
    return dataclasses.replace(state, units=units)


def test_k_bounded_candidate_count_and_top_k_selects_highest_scored() -> None:
    static, state = _build_net(_AddConnNet)
    state = _with_marker_activations(state)

    phase = phases.build_add_conn_phase(_AddConnNet, static)
    new_state, loss = phase(state, _DUMMY_INPUTS)
    assert float(loss) == 0.0

    bucket = new_state.conns[0]
    dead = np.asarray(bucket[px.DEAD.name])
    from_id = np.asarray(bucket[px.FROM_ID.name])
    to_id = np.asarray(bucket[px.TO_ID.name])
    live = {
        (int(f), int(t)) for f, t, d in zip(from_id, to_id, dead, strict=True) if not d
    }
    anchor_edges = {(_ANCHOR, dst) for dst in _DST}
    new_edges = live - anchor_edges

    # K-bounded: exactly max_candidates (3) new edges land, even though 5
    # distinct (SRC, dst) candidates all pass the level window and would
    # all score above every (ANCHOR, dst) duplicate.
    assert len(new_edges) == 3
    # top_k: the 3 highest-scored (SRC, dst) pairs -- dst 2, 3, 4 (anchor
    # weights 5.0, 4.0, 3.0) -- not dst 5 or 6 (weights 2.0, 1.0).
    assert new_edges == {(_SRC, 2), (_SRC, 3), (_SRC, 4)}


def test_prefix_sum_slot_claim_lands_new_edges_in_the_bucket_dead_slots() -> None:
    static, state = _build_net(_AddConnNet)
    state = _with_marker_activations(state)

    phase = phases.build_add_conn_phase(_AddConnNet, static)
    new_state, _ = phase(state, _DUMMY_INPUTS)

    bucket = new_state.conns[0]
    dead = np.asarray(bucket[px.DEAD.name])
    from_id = np.asarray(bucket[px.FROM_ID.name])
    to_id = np.asarray(bucket[px.TO_ID.name])
    weight = np.asarray(bucket[px.WEIGHT.name])

    # Positions 0-4 held the 5 pre-existing ANCHOR edges (builder-sorted by
    # to_id, IMPLEMENTATION_PLAN.md [D:3]) and must be untouched by the add.
    np.testing.assert_array_equal(from_id[:5], np.full(5, _ANCHOR))
    np.testing.assert_array_equal(to_id[:5], np.array(_DST))
    np.testing.assert_allclose(weight[:5], np.array(_ANCHOR_WEIGHTS))
    assert bool((~dead[:5]).all())

    # The 3 accepted candidates claim the first 3 DEAD slots (positions 5,
    # 6, 7) via the prefix-sum rank over the pre-call dead mask, in
    # lax.top_k's own (highest-score-first) output order.
    assert bool((~dead[5:8]).all())
    np.testing.assert_array_equal(from_id[5:8], np.full(3, _SRC))
    np.testing.assert_array_equal(to_id[5:8], np.array([2, 3, 4]))
    np.testing.assert_allclose(weight[5:8], np.full(3, _NEW_WEIGHT))

    # Everything beyond stays untouched dead padding.
    assert bool(dead[8:].all())


def test_level_preserving_add_does_not_set_needs_resort() -> None:
    static, state = _build_net(_AddConnNet)
    state = _with_marker_activations(state)
    assert bool(state.needs_resort) is False

    phase = phases.build_add_conn_phase(_AddConnNet, static)
    new_state, _ = phase(state, _DUMMY_INPUTS)

    # M4b Deviation (IMPLEMENTATION_PLAN.md): the window is no longer
    # restricted to dst-strictly-ahead-of-src (phases.py's
    # build_add_conn_phase docstring), so a same-level or behind-src
    # candidate CAN now be proposed and scored -- but _SRC_BONUS makes
    # every (SRC, dst in _DST) candidate outscore every such candidate here
    # (see the module docstring), so the top-3 actually COMMITTED are still
    # exactly the level-preserving (SRC, 2/3/4) edges and needs_resort
    # stays False, same outcome as M4a for a different reason.
    assert bool(new_state.needs_resort) is False


def test_overflow_flag_set_and_excess_candidates_dropped_not_miswritten() -> None:
    static, state = _build_net(_AddConnNet)
    state = _with_marker_activations(state)

    # Shrink bucket 0 down to exactly 1 free (dead) slot beyond the 5 live
    # anchor edges, so only the highest-scored of the 3 top-k'd candidates
    # can actually claim a slot -- this needs a hand-truncated (static,
    # state) since NetworkBuilder.finalize's capacity_policy always leaves
    # >= 64 slots per bucket (M1: min_bucket=64).
    truncated = {name: col[:6] for name, col in state.conns[0].items()}
    small_static = dataclasses.replace(static, level_capacities=(6,))
    small_state = dataclasses.replace(state, conns=(truncated,))
    assert bool((~small_state.conns[0][px.DEAD.name][:5]).all())
    assert bool(small_state.conns[0][px.DEAD.name][5])

    overflow_sink: list[jax.Array] = [jnp.bool_(False)]
    phase = phases.build_add_conn_phase(
        _AddConnNet, small_static, overflow_sink=overflow_sink
    )
    new_state, _ = phase(small_state, _DUMMY_INPUTS)

    assert bool(overflow_sink[0]) is True
    bucket = new_state.conns[0]
    dead = np.asarray(bucket[px.DEAD.name])
    from_id = np.asarray(bucket[px.FROM_ID.name])
    to_id = np.asarray(bucket[px.TO_ID.name])
    weight = np.asarray(bucket[px.WEIGHT.name])

    assert dead.shape == (6,)  # capacity unchanged: no corruption/resize
    assert int((~dead).sum()) == 6  # 5 original + exactly 1 new, not 3

    # The sole claimed slot (position 5) is the single HIGHEST-scored
    # candidate; the other 2 (dst 3, dst 4) are dropped, not mis-written
    # into some other (wrong) slot.
    assert not dead[5]
    assert int(from_id[5]) == _SRC
    assert int(to_id[5]) == 2
    np.testing.assert_allclose(float(weight[5]), _NEW_WEIGHT)

    # Positions 0-4 (the pre-existing live anchor edges) are untouched.
    np.testing.assert_array_equal(from_id[:5], np.full(5, _ANCHOR))
    np.testing.assert_array_equal(to_id[:5], np.array(_DST))
    np.testing.assert_allclose(weight[:5], np.array(_ANCHOR_WEIGHTS))


def test_added_edge_participates_in_the_next_forward_sweep() -> None:
    """End to end: step 1's forward computes DST's anchor-only activation,
    which step 1's OWN add_conn phase then scores against (phase order,
    phases.py module docstring: forward runs before add_conn in the SAME
    step) to pick the top-3 (SRC, dst) edges; step 2's forward is the first
    sweep that can observe them, since they did not exist during step 1's
    own forward pass."""
    static, state = _build_net(_AddConnNet)
    step = px.make_step(_AddConnNet, static)
    inputs = px.StepInputs(
        inputs=jnp.asarray([1.0, 1.0], dtype=jnp.float32), targets=None
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error", message=".*Some donated buffers were not usable.*"
        )
        result1 = step(state, inputs)
        assert bool(result1.overflow) is False
        assert bool(result1.state.needs_resort) is False
        result2 = step(result1.state, inputs)

    got = np.asarray(result2.state.units[px.ACTIVATION.name])
    # dst 2, 3, 4 (the top-3 candidates selected during step 1) now ALSO
    # receive the new (SRC, dst) edge's contribution (SRC's activation is
    # 1.0, so the edge adds exactly _NEW_WEIGHT); dst 5, 6 (not selected)
    # are unaffected, still anchor-only.
    for dst, anchor_w in zip(_DST[:3], _ANCHOR_WEIGHTS[:3], strict=True):
        np.testing.assert_allclose(
            float(got[dst]), anchor_w + _NEW_WEIGHT, rtol=1e-5, atol=1e-5
        )
    for dst, anchor_w in zip(_DST[3:], _ANCHOR_WEIGHTS[3:], strict=True):
        np.testing.assert_allclose(float(got[dst]), anchor_w, rtol=1e-5, atol=1e-5)


def test_pipeline_mode_adds_land_in_the_single_bucket_and_never_resort() -> None:
    """PIPELINE's level_capacities is a 1-tuple (rung0 design section 3):
    every accepted candidate, regardless of its source unit's level, must
    land in that one flat bucket. The level WINDOW itself is still
    consulted (native's AddConnections has no Pipeline/Topological
    distinction, dispatch_cpu.hpp:699-762), so the accepted set is
    identical to the TOPOLOGICAL case above.

    needs_resort stays False here for the SAME reason as the TOPOLOGICAL
    test above (_SRC_BONUS keeps the actually-committed top-3 confined to
    the level-preserving (SRC, dst) edges), not because PIPELINE mode
    structurally forbids it (M4b Deviation, IMPLEMENTATION_PLAN.md,
    phases.py's build_add_conn_phase docstring: design section 5's "in
    pipeline mode adds never resort at all" described a consequence of
    M4a's narrower window, not a carve-out in the reassignment formula,
    which never distinguished propagation mode).
    """
    static, state = _build_net(_PipelineAddConnNet)
    assert len(static.level_capacities) == 1
    state = _with_marker_activations(state)

    phase = phases.build_add_conn_phase(_PipelineAddConnNet, static)
    new_state, _ = phase(state, _DUMMY_INPUTS)

    bucket = new_state.conns[0]
    dead = np.asarray(bucket[px.DEAD.name])
    from_id = np.asarray(bucket[px.FROM_ID.name])
    to_id = np.asarray(bucket[px.TO_ID.name])
    live = {
        (int(f), int(t)) for f, t, d in zip(from_id, to_id, dead, strict=True) if not d
    }
    anchor_edges = {(_ANCHOR, dst) for dst in _DST}
    new_edges = live - anchor_edges

    assert new_edges == {(_SRC, 2), (_SRC, 3), (_SRC, 4)}
    assert bool(new_state.needs_resort) is False


class _NoBonusAddConn(px.AddConn[None]):
    """score is dst's activation alone, with NO SRC bonus, so the highest-
    scored candidates are the already-live (ANCHOR, dst) pairs (dst 2/3/4,
    activations 5/4/3). Without dedup those would be selected and grown as
    duplicate parallel edges; the occupancy mask must exclude them, leaving
    growth to the fresh (SRC, dst) pairs of equal dst score."""

    max_candidates = 3

    def score(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: None
    ) -> jax.Array:
        del src, g
        return u[px.ACTIVATION, dst]

    def init(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: None
    ) -> px.ConnWrite:
        del u, src, dst, g
        return px.ConnWrite.of((px.WEIGHT, jnp.float32(_NEW_WEIGHT)))


class _NoBonusAddConnNet(px.Network[None]):
    forward_pass = _SumForward()
    add_conn = _NoBonusAddConn()
    propagation = px.Propagation.TOPOLOGICAL


def test_add_conn_never_grows_a_duplicate_of_a_live_edge() -> None:
    static, state = _build_net(_NoBonusAddConnNet)
    state = _with_marker_activations(state)

    phase = phases.build_add_conn_phase(_NoBonusAddConnNet, static)
    new_state, _ = phase(state, _DUMMY_INPUTS)

    bucket = new_state.conns[0]
    dead = np.asarray(bucket[px.DEAD.name])
    from_id = np.asarray(bucket[px.FROM_ID.name])
    to_id = np.asarray(bucket[px.TO_ID.name])
    live_pairs = [
        (int(f), int(t)) for f, t, d in zip(from_id, to_id, dead, strict=True) if not d
    ]

    # No live pair appears twice: the highest-scored candidates are the
    # already-live (ANCHOR, dst) edges, which the dedup mask excludes rather
    # than regrowing as parallel duplicates.
    assert len(live_pairs) == len(set(live_pairs))
    # Each pre-existing ANCHOR edge survives exactly once.
    for dst in _DST:
        assert live_pairs.count((_ANCHOR, dst)) == 1
    # Growth still happens -- into fresh (SRC, dst) pairs, never duplicates.
    new_edges = set(live_pairs) - {(_ANCHOR, dst) for dst in _DST}
    assert new_edges == {(_SRC, 2), (_SRC, 3), (_SRC, 4)}


class _ShortlistAddConn(px.AddConn[None]):
    """max_candidate_units shortlists growth candidates to the top-M units by
    `importance`. Here importance favors SRC and DST id 6, so the only
    level-increasing candidate the M=2 shortlist can form is (SRC, 6)."""

    max_candidates = 3
    max_candidate_units = 2

    def importance(self, u: px.UnitView, i: px.UnitIdx, g: None) -> jax.Array:
        del u, g
        favored = (i == jnp.int32(_SRC)) | (i == jnp.int32(6))
        return jnp.where(favored, jnp.float32(10.0), jnp.float32(0.0))

    def score(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: None
    ) -> jax.Array:
        del u, src, dst, g
        return jnp.float32(1.0)

    def init(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: None
    ) -> px.ConnWrite:
        del u, src, dst, g
        return px.ConnWrite.of((px.WEIGHT, jnp.float32(_NEW_WEIGHT)))


class _ShortlistAddConnNet(px.Network[None]):
    forward_pass = _SumForward()
    add_conn = _ShortlistAddConn()
    propagation = px.Propagation.TOPOLOGICAL


def test_add_conn_candidate_shortlist_restricts_growth_to_top_m_units() -> None:
    static, state = _build_net(_ShortlistAddConnNet)
    state = _with_marker_activations(state)

    phase = phases.build_add_conn_phase(_ShortlistAddConnNet, static)
    new_state, _ = phase(state, _DUMMY_INPUTS)

    bucket = new_state.conns[0]
    dead = np.asarray(bucket[px.DEAD.name])
    from_id = np.asarray(bucket[px.FROM_ID.name])
    to_id = np.asarray(bucket[px.TO_ID.name])
    live = [
        (int(f), int(t)) for f, t, d in zip(from_id, to_id, dead, strict=True) if not d
    ]

    # Only (SRC, 6) can form from the {SRC, 6} shortlist as a forward edge --
    # growth touches no other pair, and never a duplicate.
    new_edges = set(live) - {(_ANCHOR, dst) for dst in _DST}
    assert new_edges == {(_SRC, 6)}
    assert len(live) == len(set(live))


class _DeeperOnlyAddConn(px.AddConn[None]):
    """Scores deeper candidates finite and vetoes every non-deeper candidate
    with -inf. max_candidates (10) exceeds the number of deeper candidates
    (5: SRC -> each DST), and the bucket has many free dead slots, so the veto
    is the only thing keeping growth to deeper edges: without it, top_k would
    back-fill the surplus slots with the -inf-scored same-level candidates it
    surfaces once the finite ones run out (setting needs_resort)."""

    max_candidates = 10

    def score(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: None
    ) -> jax.Array:
        del g
        deeper = u[px.LEVEL, dst] > u[px.LEVEL, src]
        return jnp.where(deeper, jnp.float32(1.0), -jnp.inf)

    def init(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: None
    ) -> px.ConnWrite:
        del u, src, dst, g
        return px.ConnWrite.of((px.WEIGHT, jnp.float32(_NEW_WEIGHT)))


class _DeeperOnlyNet(px.Network[None]):
    forward_pass = _SumForward()
    add_conn = _DeeperOnlyAddConn()
    propagation = px.Propagation.TOPOLOGICAL


def test_veto_score_is_never_committed_even_with_free_slots() -> None:
    static, state = _build_net(_DeeperOnlyNet)
    state = _with_marker_activations(state)

    phase = phases.build_add_conn_phase(_DeeperOnlyNet, static)
    new_state, _ = phase(state, _DUMMY_INPUTS)

    bucket = new_state.conns[0]
    dead = np.asarray(bucket[px.DEAD.name])
    from_id = np.asarray(bucket[px.FROM_ID.name])
    to_id = np.asarray(bucket[px.TO_ID.name])
    live = {
        (int(f), int(t)) for f, t, d in zip(from_id, to_id, dead, strict=True) if not d
    }
    new_edges = live - {(_ANCHOR, dst) for dst in _DST}

    # Only the deeper (SRC, dst) edges grow; the vetoed (-inf) non-deeper
    # candidates are never committed, even though free dead slots remain and
    # max_candidates leaves room for them.
    assert new_edges == {(_SRC, dst) for dst in _DST}
    # No non-deeper edge was back-filled, so leveling is preserved.
    assert bool(new_state.needs_resort) is False


# ids for a 3-level net: two inputs, two hidden, two outputs.
_IN, _HID, _OUT = (0, 1), (2, 3), (4, 5)


def _build_3level(
    net: type[px.Network[None]],
) -> tuple[px.NetworkStatic, px.NetworkState[None]]:
    """A→H0, B→H1, H0→O0, H1→O1: three levels, two source-level buckets."""
    builder = px.NetworkBuilder(net, None)
    for _ in range(6):
        builder.add_unit()
    for i in _IN:
        builder.mark_input(i)
    for o in _OUT:
        builder.mark_output(o)
    builder.add_conn(_IN[0], _HID[0], weight=1.0)
    builder.add_conn(_IN[1], _HID[1], weight=1.0)
    builder.add_conn(_HID[0], _OUT[0], weight=1.0)
    builder.add_conn(_HID[1], _OUT[1], weight=1.0)
    return builder.finalize()


class _DeepestImportanceGrow(px.AddConn[None]):
    """Importance favours the deepest level, so a GLOBAL top-M (M=2) shortlist
    fills with the two output units -- among which no deeper edge exists -- and
    the shallow input->hidden transition grows nothing. score grows any deeper
    candidate; init tags it. Subclasses only toggle `shortlist_per_level`."""

    max_candidates = 4
    max_candidate_units = 2

    def importance(self, u: px.UnitView, i: px.UnitIdx, g: None) -> jax.Array:
        del g
        return jnp.where(
            u[px.LEVEL, i] == jnp.int32(2), jnp.float32(10.0), jnp.float32(1.0)
        )

    def score(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: None
    ) -> jax.Array:
        del g
        deeper = u[px.LEVEL, dst] > u[px.LEVEL, src]
        return jnp.where(deeper, jnp.float32(1.0), -jnp.inf)

    def init(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: None
    ) -> px.ConnWrite:
        del u, src, dst, g
        return px.ConnWrite.of((px.WEIGHT, jnp.float32(_NEW_WEIGHT)))


class _GlobalShortlistNet(px.Network[None]):
    forward_pass = _SumForward()
    add_conn = _DeepestImportanceGrow()
    propagation = px.Propagation.TOPOLOGICAL


class _PerLevelGrow(_DeepestImportanceGrow):
    shortlist_per_level = True


class _PerLevelShortlistNet(px.Network[None]):
    forward_pass = _SumForward()
    add_conn = _PerLevelGrow()
    propagation = px.Propagation.TOPOLOGICAL


def _bucket0_new_edges(
    net: type[px.Network[None]],
) -> set[tuple[int, int]]:
    static, state = _build_3level(net)
    phase = phases.build_add_conn_phase(net, static)
    new_state, _ = phase(state, _DUMMY_INPUTS)
    bucket = new_state.conns[0]  # source-level-0 edges: the input->hidden transition
    dead = np.asarray(bucket[px.DEAD.name])
    frm = np.asarray(bucket[px.FROM_ID.name])
    to = np.asarray(bucket[px.TO_ID.name])
    live = {(int(f), int(t)) for f, t, d in zip(frm, to, dead, strict=True) if not d}
    return live - {(_IN[0], _HID[0]), (_IN[1], _HID[1])}  # minus the two seeds


def test_per_level_shortlist_serves_a_transition_the_global_starves() -> None:
    # Importance concentrates the global top-M on the two output units, so the
    # input->hidden bucket sees no shortlisted source and grows nothing.
    assert _bucket0_new_edges(_GlobalShortlistNet) == set()
    # Per-level draws that bucket its own (top-M level-0 sources x top-M level-1
    # destinations) grid, so the shallow transition grows the two cross edges.
    assert _bucket0_new_edges(_PerLevelShortlistNet) == {
        (_IN[0], _HID[1]),
        (_IN[1], _HID[0]),
    }
