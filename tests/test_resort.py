"""Resort + retrace contract (M4b).

recompute_levels vs host Kahn (initial_levels); resort produces sorted,
compacted, correctly-capacitied buckets; the widened AddConn window
(phases.py's build_add_conn_phase) can genuinely set needs_resort; the
retrace-count contract (rung0 design section 4): a level-preserving
add/prune workload compiles exactly once, one resort recompiles exactly
once more; a Driver end-to-end run exercises both escalation paths
(overflow -> grow_bucket, needs_resort -> topo.resort).

Retrace-count measurement note: `jax._src.test_util.count_jit_and_pmap_
lowerings`/`assert_num_jit_and_pmap_compilations` count EVERY
`lower_jaxpr_to_module` call, including ones from ordinary eager (non-jit)
jax ops -- confirmed empirically (a handful of eager `jnp`/`lax` calls
costs over a dozen). `make_step`'s own eager prelude (`build_phases`
precomputing masks like `unit_id_mask` before the traced `step` closure
even exists) and `topo.resort`'s eager device work (segment reductions,
`lax.sort_key_val`, `lax.fori_loop`) both cost a real, nonzero, and
UNSPECIFIED number of such events -- neither is the thing the retrace
contract describes (design section 4: "a new NetworkStatic ... is a new
jit PyTreeDef, so it [the STEP FUNCTION] recompiles"). Every retrace-count
assertion below therefore keeps `Driver`/`make_step` CONSTRUCTION (and, for
the resort test, the resort-triggering call itself) outside the counted
`with` block, so only the jitted step closure's own first-ever invocation
is ever counted -- confirmed to be exactly 1 lowering event per fresh
(net, static) pair, 0 for every repeat call (see the module's own
empirical check this file's implementation was validated against).
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
from jax._src import test_util as jtu

import plastax as px
from plastax import phases, topo
from plastax.state import live_conn_count

_DUMMY_INPUTS = px.StepInputs(inputs=jnp.zeros((0,), dtype=jnp.float32), targets=None)


class _SumForward(px.ForwardPass):
    """Weighted-sum forward (test_forward_pipeline.py's _SumForward,
    reused verbatim across this file's networks): map = weight *
    activation[src], combine = sum, apply = identity onto ACTIVATION."""

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


def _live_edges(state: px.NetworkState[object]) -> set[tuple[int, int]]:
    """(from, to) pairs of every live conn, across every bucket."""
    edges: set[tuple[int, int]] = set()
    for bucket in state.conns:
        dead = np.asarray(bucket[px.DEAD.name])
        from_id = np.asarray(bucket[px.FROM_ID.name])
        to_id = np.asarray(bucket[px.TO_ID.name])
        edges |= {
            (int(f), int(t))
            for f, t, d in zip(from_id, to_id, dead, strict=True)
            if not d
        }
    return edges


# ---------------------------------------------------------------------------
# recompute_levels vs host initial_levels
# ---------------------------------------------------------------------------


class _RecomputeNet(px.Network[None]):
    forward_pass = _SumForward()
    propagation = px.Propagation.TOPOLOGICAL


def _dag_with_skip_edge() -> tuple[int, np.ndarray, tuple[int, ...]]:
    """0, 1 input (level 0); 2 <- 0, 1 (level 1); 3 <- 2 (level 2);
    4 <- 3, 0(skip) (level 3) -- same shape as test_forward_topo.py's
    graph, the case that discriminates a real level walk from a buggy one
    (a skip edge whose source is two levels back from its destination)."""
    num_units = 5
    edges = np.array([[0, 2], [1, 2], [2, 3], [3, 4], [0, 4]], dtype=np.int32)
    return num_units, edges, (0, 1)


def _build_dag_with_skip_edge() -> tuple[px.NetworkStatic, px.NetworkState[None]]:
    num_units, edges, input_ids = _dag_with_skip_edge()
    builder = px.NetworkBuilder(_RecomputeNet, None)
    for _ in range(num_units):
        builder.add_unit()
    for unit_id in input_ids:
        builder.mark_input(unit_id)
    for src, dst in edges.tolist():
        builder.add_conn(int(src), int(dst), weight=1.0)
    return builder.finalize()


def test_recompute_levels_matches_host_initial_levels_on_a_dag_with_a_skip_edge() -> (
    None
):
    num_units, edges, _ = _dag_with_skip_edge()
    expected = topo.initial_levels(num_units, edges)
    static, state = _build_dag_with_skip_edge()

    # Sanity: the builder's OWN construction-time levels (M1's host
    # initial_levels, the same function) already match -- the real
    # assertion below is the ON-DEVICE recompute agreeing with the SAME
    # reference from a cold start, not merely reproducing a value it
    # inherited from the arena.
    np.testing.assert_array_equal(np.asarray(state.units[px.LEVEL.name]), expected)

    got = topo.recompute_levels(static, state)
    np.testing.assert_array_equal(np.asarray(got), expected)


def test_recompute_levels_ignores_a_stale_level_column_and_recomputes_from_conns() -> (
    None
):
    """recompute_levels reads LIVE CONNS (topo.py docstring), never the
    existing LEVEL column -- corrupting it first proves that, rather than
    coincidentally reproducing an already-correct value."""
    num_units, edges, _ = _dag_with_skip_edge()
    expected = topo.initial_levels(num_units, edges)
    static, state = _build_dag_with_skip_edge()

    corrupted = dataclasses.replace(
        state,
        units={
            **state.units,
            px.LEVEL.name: jnp.zeros((num_units,), dtype=jnp.int32),
        },
    )
    got = topo.recompute_levels(static, corrupted)
    np.testing.assert_array_equal(np.asarray(got), expected)


# ---------------------------------------------------------------------------
# resort(): redistribution, (dead, to_id) compaction, capacity_policy
# ---------------------------------------------------------------------------


class _ResortNet(px.Network[None]):
    forward_pass = _SumForward()
    propagation = px.Propagation.TOPOLOGICAL


def _build_prune_relevel_graph() -> tuple[px.NetworkStatic, px.NetworkState[None]]:
    """0, 1 input; 2 <- 0 (lvl1); 3 <- 1 (lvl1); 4 <- 2, 3, 0(skip) (lvl2).
    Distinct weights per edge so the resort's data (not just ids) is
    checkable through the compaction + sort."""
    builder = px.NetworkBuilder(_ResortNet, None)
    for _ in range(5):
        builder.add_unit()
    builder.mark_input(0)
    builder.mark_input(1)
    builder.add_conn(0, 2, weight=0.1)
    builder.add_conn(1, 3, weight=0.2)
    builder.add_conn(2, 4, weight=0.3)
    builder.add_conn(3, 4, weight=0.4)
    builder.add_conn(0, 4, weight=0.5)
    return builder.finalize()


def test_resort_redistributes_sorts_and_compacts_after_a_pruning_relevel() -> None:
    """Pruning 1->3 (3's only incoming edge) orphans unit 3 from level 1
    back to level 0 -- recompute_levels agrees with initial_levels only on
    the ORIGINAL graph, so this is a genuine post-construction level
    change: conn 3->4 must move from bucket 1 (3's old source level) into
    bucket 0 (3's new one), landing alongside 0->2 and the 0->4 skip in an
    order NEITHER the original arena position NOR pure to_id-of-arrival
    would produce by accident -- the real exercise of "gather per level,
    then stable-sort by (dead, to_id)" (topo.py's resort docstring), not
    just of compaction alone.
    """
    static, state = _build_prune_relevel_graph()
    assert len(static.level_capacities) == 2
    got_level = np.asarray(state.units[px.LEVEL.name])
    np.testing.assert_array_equal(got_level, [0, 0, 1, 1, 2])

    # Pre-resort bucket 0 layout the rest of this test's hand-derivation
    # assumes (builder-sorted by to_id: 0->2, 1->3, 0->4).
    from_id0 = np.asarray(state.conns[0][px.FROM_ID.name])
    to_id0 = np.asarray(state.conns[0][px.TO_ID.name])
    np.testing.assert_array_equal(from_id0[:3], [0, 1, 0])
    np.testing.assert_array_equal(to_id0[:3], [2, 3, 4])

    dead0 = np.asarray(state.conns[0][px.DEAD.name]).copy()
    dead0[1] = True  # tombstone 1->3
    pruned_conns0 = {**state.conns[0], px.DEAD.name: jnp.asarray(dead0)}
    pruned_state = dataclasses.replace(state, conns=(pruned_conns0, state.conns[1]))

    new_static, new_state = topo.resort(static, pruned_state)

    new_level = np.asarray(new_state.units[px.LEVEL.name])
    np.testing.assert_array_equal(new_level, [0, 0, 1, 0, 2])  # unit 3 orphaned

    assert len(new_static.level_capacities) == 2
    assert new_static.level_capacities == (
        topo.capacity_policy(3),  # bucket 0: 0->2, 0->4, 3->4
        topo.capacity_policy(1),  # bucket 1: 2->4 only
    )

    bucket0 = new_state.conns[0]
    dead_b0 = np.asarray(bucket0[px.DEAD.name])
    from_b0 = np.asarray(bucket0[px.FROM_ID.name])
    to_b0 = np.asarray(bucket0[px.TO_ID.name])
    weight_b0 = np.asarray(bucket0[px.WEIGHT.name])
    assert dead_b0.shape == (topo.capacity_policy(3),)
    np.testing.assert_array_equal(from_b0[:3], [0, 0, 3])
    np.testing.assert_array_equal(to_b0[:3], [2, 4, 4])  # sorted by to_id
    np.testing.assert_allclose(weight_b0[:3], [0.1, 0.5, 0.4])
    assert bool((~dead_b0[:3]).all())
    assert bool(dead_b0[3:].all())  # padding: real compaction, not a full copy

    bucket1 = new_state.conns[1]
    dead_b1 = np.asarray(bucket1[px.DEAD.name])
    from_b1 = np.asarray(bucket1[px.FROM_ID.name])
    to_b1 = np.asarray(bucket1[px.TO_ID.name])
    weight_b1 = np.asarray(bucket1[px.WEIGHT.name])
    assert not dead_b1[0]
    assert int(from_b1[0]) == 2
    assert int(to_b1[0]) == 4
    np.testing.assert_allclose(float(weight_b1[0]), 0.3)
    assert bool(dead_b1[1:].all())

    assert bool(new_state.needs_resort) is False
    assert _live_edges(new_state) == {(0, 2), (0, 4), (3, 4), (2, 4)}


# ---------------------------------------------------------------------------
# The widened AddConn window can genuinely set needs_resort
# ---------------------------------------------------------------------------


class _SidewaysPipelineAddConn(px.AddConn[None]):
    """Only ever proposes the fixed same-source-level pair (1, 2) -- a
    candidate the M4a window could never even score (module docstring,
    phases.py's build_add_conn_phase), now reachable under M4b's widened
    one."""

    max_candidates = 1

    def score(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: None
    ) -> jnp.ndarray:
        del u, g
        hit = (src == jnp.int32(1)) & (dst == jnp.int32(2))
        return jnp.where(hit, jnp.float32(1.0), jnp.float32(-1e9))

    def init(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: None
    ) -> px.ConnWrite:
        del u, src, dst, g
        return px.ConnWrite.of((px.WEIGHT, jnp.float32(0.5)))


class _SidewaysPipelineNet(px.Network[None]):
    forward_pass = _SumForward()
    add_conn = _SidewaysPipelineAddConn()
    propagation = px.Propagation.PIPELINE


def test_widened_window_lets_a_same_level_add_actually_set_needs_resort() -> None:
    """Direct (Driver-bypassing) evidence that the M4b window can commit a
    same-source-level candidate: PIPELINE's single bucket accepts a source
    at any level (phases.py's build_add_conn_phase docstring), so this
    needs no pre-existing per-level bucket structure the way a
    TOPOLOGICAL demonstration would (see test_one_resort_triggers_
    exactly_one_additional_compile's docstring below for why the
    retrace-COUNT test manufactures needs_resort directly instead)."""
    builder = px.NetworkBuilder(_SidewaysPipelineNet, None)
    builder.add_unit()  # 0: ANCHOR
    builder.add_unit()  # 1: A
    builder.add_unit()  # 2: B
    builder.mark_input(0)
    builder.add_conn(0, 1, weight=1.0)
    builder.add_conn(0, 2, weight=1.0)
    static, state = builder.finalize()
    got_level = np.asarray(state.units[px.LEVEL.name])
    assert int(got_level[1]) == int(got_level[2]) == 1  # genuinely same-level

    phase = phases.build_add_conn_phase(_SidewaysPipelineNet, static)
    new_state, _ = phase(state, _DUMMY_INPUTS)

    assert bool(new_state.needs_resort) is True
    assert (1, 2) in _live_edges(new_state)


# ---------------------------------------------------------------------------
# Retrace count: pure add/prune (level-preserving) workload
# ---------------------------------------------------------------------------

_R_ANCHOR = 0
_R_SRC = 1
_R_DST = (2, 3, 4)
_R_ANCHOR_W = (5.0, 4.0, 3.0)
_R_MARK = 999.0


class _SafeAddConn(px.AddConn[None]):
    """Only ever proposes (SRC, dst) pairs where dst is genuinely ahead of
    src, read via the view rather than hardcoded: level-preserving by this
    POLICY's own choice, not because the (now-widened) window forbids
    anything else -- a real user policy is free to stay this conservative."""

    max_candidates = 2

    def score(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: None
    ) -> jnp.ndarray:
        del g
        ahead = u[px.LEVEL, dst] > u[px.LEVEL, src]
        from_src = src == jnp.int32(_R_SRC)
        return jnp.where(ahead & from_src, jnp.float32(1.0), jnp.float32(-1e9))

    def init(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: None
    ) -> px.ConnWrite:
        del u, src, dst, g
        return px.ConnWrite.of((px.WEIGHT, jnp.float32(_R_MARK)))


class _ChurnPrune(px.PruneConn[None]):
    """Removes exactly what the PRIOR step's add_conn committed (tagged
    _R_MARK, phase order forward/.../prune_conn/add_conn means THIS step's
    own fresh additions survive to be pruned only on the NEXT step), so
    the live-conn count oscillates in a small, bounded range across
    arbitrarily many steps -- never overflowing, never idle."""

    def predicate(
        self, u: px.UnitView, c: px.ConnView, cid: px.ConnIdx, g: None
    ) -> jnp.ndarray:
        del u, g
        return c[px.WEIGHT, cid] == jnp.float32(_R_MARK)


class _RetraceNet(px.Network[None]):
    forward_pass = _SumForward()
    add_conn = _SafeAddConn()
    prune_conn = _ChurnPrune()
    propagation = px.Propagation.TOPOLOGICAL


def _build_retrace_net() -> tuple[px.NetworkStatic, px.NetworkState[None]]:
    builder = px.NetworkBuilder(_RetraceNet, None)
    builder.add_unit()  # 0: ANCHOR
    builder.add_unit()  # 1: SRC
    for _ in _R_DST:
        builder.add_unit()
    builder.mark_input(_R_ANCHOR)
    builder.mark_input(_R_SRC)
    for dst, w in zip(_R_DST, _R_ANCHOR_W, strict=True):
        builder.add_conn(_R_ANCHOR, dst, weight=w)
    return builder.finalize()


def test_pure_add_prune_level_preserving_workload_compiles_exactly_once() -> None:
    static, state = _build_retrace_net()
    driver = px.Driver(_RetraceNet, static, state)  # eager prelude, uncounted
    inputs = px.StepInputs(
        inputs=jnp.asarray([1.0, 1.0], dtype=jnp.float32), targets=None
    )

    # The whole workload -- the step closure's one real trace on its very
    # first call, plus 19 more steps of genuine add/prune churn -- costs
    # exactly the ONE jit lowering the retrace contract promises (rung0
    # design section 4): the static never changes, so nothing ever
    # recompiles (module docstring: construction's eager prelude is
    # deliberately outside this window).
    with jtu.assert_num_jit_and_pmap_compilations(1):
        for _ in range(20):
            driver.step(inputs)

    assert driver.static == static
    assert bool(driver.state.needs_resort) is False
    # Genuine churn happened, not a no-op: ANCHOR's 3 original edges plus
    # whatever SRC-sourced pairs survived this run's last add-then-prune.
    assert int(live_conn_count(driver.state)) >= len(_R_DST)


# ---------------------------------------------------------------------------
# Retrace count: one resort recompiles exactly once more
# ---------------------------------------------------------------------------


class _ChainNet(px.Network[None]):
    forward_pass = _SumForward()
    propagation = px.Propagation.TOPOLOGICAL


def _build_shrinking_chain() -> tuple[px.NetworkStatic, px.NetworkState[None]]:
    """ANCHOR(0, input, lvl0) -> A(1, lvl1) -> B(2, lvl2) -> C(3, lvl3): a
    plain 3-bucket chain (source-levels 0, 1, 2 each already have a
    bucket)."""
    builder = px.NetworkBuilder(_ChainNet, None)
    for _ in range(4):
        builder.add_unit()
    builder.mark_input(0)
    builder.add_conn(0, 1, weight=1.0)
    builder.add_conn(1, 2, weight=1.0)
    builder.add_conn(2, 3, weight=1.0)
    return builder.finalize()


def test_one_resort_triggers_exactly_one_additional_compile() -> None:
    """Manufactures needs_resort directly rather than through a real
    AddConn commit (M4a's deviation note's sanctioned fallback, recorded
    as a Deviation for M4b too): build_add_conn_phase's window only ever
    proposes a candidate sourced at a level that ALREADY has a bucket
    (`src_ok = src_level == bucket_idx`, only true for bucket_idx <
    num_buckets), and Kahn's `level(dst) = max(incoming src levels) + 1`
    means any such commit's resulting level is bounded by the CURRENT max
    level -- so a real widened-window commit can shrink or hold the
    bucket count but, on a small from-scratch test graph, cannot provably
    GROW it without either a much larger seed graph or an actual cycle.
    test_widened_window_lets_a_same_level_add_actually_set_needs_resort
    above already gives direct evidence the window itself works; this
    test's job is make_step's OWN cache behaviour, which does not care
    what set needs_resort. Pruning C's only incoming edge (bucket 2's sole
    conn) orphans C from level 3 back to level 0, so the new max level (2)
    needs one FEWER bucket than the old static's 3 -- a different tuple
    LENGTH, unambiguously a new jit PyTreeDef.
    """
    static, state = _build_shrinking_chain()
    assert len(static.level_capacities) == 3

    bucket2 = state.conns[2]
    dead2 = np.asarray(bucket2[px.DEAD.name]).copy()
    assert not dead2[0]
    dead2[0] = True
    pruned_conns = (*state.conns[:2], {**bucket2, px.DEAD.name: jnp.asarray(dead2)})
    pruned_state = dataclasses.replace(
        state, conns=pruned_conns, needs_resort=jnp.bool_(True)
    )

    driver = px.Driver(_ChainNet, static, pruned_state)  # eager prelude, uncounted
    inputs = px.StepInputs(inputs=jnp.asarray([1.0], dtype=jnp.float32), targets=None)

    # Uncounted: the original static's real first trace, PLUS
    # topo.resort's own eager device work, PLUS the new static's step
    # assembly -- none of that is the thing under test (module docstring).
    driver.step(inputs)
    assert len(driver.static.level_capacities) == 2  # genuinely shrunk
    assert bool(driver.state.needs_resort) is False
    assert _live_edges(driver.state) == {(0, 1), (1, 2)}

    # Driver.step's needs_resort branch reassigns self._step to a freshly
    # assembled but NOT YET CALLED closure and returns -- so this next
    # call is that closure's first-ever invocation: exactly the one
    # additional jit retrace the resort should have caused.
    with jtu.assert_num_jit_and_pmap_compilations(1):
        driver.step(inputs)

    # Steady state thereafter: the post-resort static is stable (nothing
    # further mutates it), so more steps add zero further compiles.
    with jtu.assert_num_jit_and_pmap_compilations(0):
        for _ in range(8):
            driver.step(inputs)

    assert len(driver.static.level_capacities) == 2


# ---------------------------------------------------------------------------
# Driver end to end: overflow -> grow_bucket, needs_resort -> topo.resort
# ---------------------------------------------------------------------------

_E_ANCHOR = 0
_E_DST_A = 1
_E_DST_B = 2
_E_SINK = 3


class _E2EAddConn(px.AddConn[None]):
    """Only ever proposes the fixed same-source-level pair (DST_A, DST_B)
    -- deliberately the one candidate this small graph's single truncated
    bucket has no room for on the first attempt."""

    max_candidates = 1

    def score(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: None
    ) -> jnp.ndarray:
        del u, g
        hit = (src == jnp.int32(_E_DST_A)) & (dst == jnp.int32(_E_DST_B))
        return jnp.where(hit, jnp.float32(1.0), jnp.float32(-1e9))

    def init(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: None
    ) -> px.ConnWrite:
        del u, src, dst, g
        return px.ConnWrite.of((px.WEIGHT, jnp.float32(0.5)))


class _E2ENet(px.Network[None]):
    forward_pass = _SumForward()
    add_conn = _E2EAddConn()
    propagation = px.Propagation.TOPOLOGICAL
    # 0 (not the class default of 1): restricts the window to STRICTLY
    # same-level pairs, so bucket 0 (ANCHOR, the only level-0 unit, with no
    # other level-0 unit to pair against) has NO valid candidates at all --
    # without this, ANCHOR's own bucket would ALSO have valid (if
    # low-scored) in-window candidates every step (any dst within
    # neighbourhood 1, including plain duplicates of ANCHOR's existing
    # edges), and since top_k + prefix-sum commits WHATEVER it selects
    # once that candidate is valid and a slot is free -- regardless of how
    # low build_add_conn_phase's caller-supplied `score` rated it -- bucket
    # 0 would commit an unplanned parallel edge every step, muddying this
    # test's edge-set assertions for no reason relevant to what it checks.
    neighbourhood = 0


def _build_e2e_net() -> tuple[px.NetworkStatic, px.NetworkState[None]]:
    """ANCHOR(0, input, lvl0) -> DST_A(1, lvl1) -> SINK(3, lvl2); ANCHOR ->
    DST_B(2, lvl1): bucket 1 (source level 1) already exists (DST_A ->
    SINK), so DST_A is a valid add_conn SOURCE even before any resort --
    the window only ever proposes a candidate sourced at a level that
    already has a bucket (`src_ok = src_level == bucket_idx`), so the
    same-source-level (DST_A, DST_B) pair needs this pre-existing sibling
    edge to be reachable at all (see test_one_resort_triggers_exactly_
    one_additional_compile's docstring for the general version of this
    constraint). Bucket 1's capacity is then hand-truncated to its 1 live
    conn (zero headroom, mirroring test_add_conn.py's own overflow test)
    so the very first step's sideways-candidate proposal has nowhere to
    land.
    """
    builder = px.NetworkBuilder(_E2ENet, None)
    builder.add_unit()  # 0: ANCHOR
    builder.add_unit()  # 1: DST_A
    builder.add_unit()  # 2: DST_B
    builder.add_unit()  # 3: SINK
    builder.mark_input(_E_ANCHOR)
    builder.add_conn(_E_ANCHOR, _E_DST_A, weight=1.0)
    builder.add_conn(_E_ANCHOR, _E_DST_B, weight=1.0)
    builder.add_conn(_E_DST_A, _E_SINK, weight=1.0)
    static, state = builder.finalize()
    assert len(static.level_capacities) == 2
    truncated = {name: col[:1] for name, col in state.conns[1].items()}
    small_static = dataclasses.replace(
        static, level_capacities=(static.level_capacities[0], 1)
    )
    small_state = dataclasses.replace(state, conns=(state.conns[0], truncated))
    return small_static, small_state


def test_driver_end_to_end_exercises_overflow_grow_and_needs_resort_resort() -> None:
    """One Driver.step call exercising BOTH escalation paths in sequence
    (driver.py's docstring): attempt 1 overflows (bucket 1 has zero free
    slots), so the driver grows it and retries the same inputs; attempt 2
    has room, commits the (DST_A, DST_B) same-level candidate, and sets
    needs_resort, so the driver resorts before returning. capacity_policy's
    min_bucket=64 floor means neither bucket's CAPACITY VALUE is
    guaranteed to visibly change once grown/resorted past a handful of
    live conns (both round up to the same 64) -- unlike
    test_one_resort_triggers_exactly_one_additional_compile, this test
    is about the escalation PATHS actually firing and leaving a correct
    final state, not about forcing a jit retrace, so it checks the
    genuinely-guaranteed signals instead (capacity growth off its
    hand-truncated floor of 1, the recomputed LEVEL column, needs_resort
    clearing, and the live edge set).
    """
    static, state = _build_e2e_net()
    assert static.level_capacities[1] == 1
    driver = px.Driver(_E2ENet, static, state)
    inputs = px.StepInputs(inputs=jnp.asarray([1.0], dtype=jnp.float32), targets=None)

    driver.step(inputs)

    # overflow -> grow: bucket 1's capacity grew past the truncated 1.
    assert driver.static.level_capacities[1] > 1
    assert bool(driver.state.needs_resort) is False

    assert _live_edges(driver.state) == {
        (_E_ANCHOR, _E_DST_A),
        (_E_ANCHOR, _E_DST_B),
        (_E_DST_A, _E_SINK),
        (_E_DST_A, _E_DST_B),
    }
    got_level = np.asarray(driver.state.units[px.LEVEL.name])
    assert int(got_level[_E_ANCHOR]) == 0
    assert int(got_level[_E_DST_A]) == 1
    assert int(got_level[_E_DST_B]) == 2  # bumped: now also fed by DST_A
    assert int(got_level[_E_SINK]) == 2

    # Stable afterward: further steps neither overflow nor resort again.
    for _ in range(3):
        driver.step(inputs)
    assert len(driver.static.level_capacities) == 2
    assert bool(driver.state.needs_resort) is False
