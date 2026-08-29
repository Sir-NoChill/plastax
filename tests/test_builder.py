"""Builder -> finalize invariants (M1).

Levels correct; buckets sorted by (dead, to_id); capacities obey
capacity_policy. Implemented when M1 lands.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import plastax as px
from plastax import topo, topology
from plastax._types import FieldSpec
from plastax.state import Columns
from plastax.views import UnitWrite


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
        return c[px.WEIGHT, cid] * u[px.ACTIVATION, src]

    def apply(
        self, u: px.UnitView, i: px.UnitIdx, g: None, acc: jax.Array
    ) -> UnitWrite:
        return UnitWrite.of((px.ACTIVATION, acc))


class _TopoNet(px.Network[None]):
    forward_pass = _SumForward()
    propagation = px.Propagation.TOPOLOGICAL


class _PipelineNet(px.Network[None]):
    forward_pass = _SumForward()
    propagation = px.Propagation.PIPELINE


Bias = FieldSpec.float32("bias")
Tag = FieldSpec.float32("tag")


class _ExtraFieldsNet(px.Network[None]):
    forward_pass = _SumForward()
    propagation = px.Propagation.TOPOLOGICAL
    extra_unit_fields = (Bias,)
    extra_conn_fields = (Tag,)


def _assert_bucket_sorted_by_dead_then_to_id(cols: Columns) -> None:
    dead = np.asarray(cols["dead"])
    to_id = np.asarray(cols["to_id"])
    # dead is False (0) before True (1): non-decreasing.
    assert np.all(dead[:-1] <= dead[1:])
    live = int((~dead).sum())
    # ascending to_id within the live prefix.
    assert np.all(to_id[:live][:-1] <= to_id[:live][1:])
    assert np.all(dead[live:])  # every slot past the live prefix is dead


def _build_diamond(
    net: type[px.Network[None]],
) -> tuple[px.NetworkStatic, px.NetworkState[None]]:
    """0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3: levels [0, 1, 1, 2]."""
    builder = px.NetworkBuilder(net, None)
    n0, n1, n2, n3 = (builder.add_unit() for _ in range(4))
    builder.mark_input(n0)
    builder.mark_output(n3)
    builder.add_conn(n0, n1, weight=1.0)
    builder.add_conn(n0, n2, weight=2.0)
    builder.add_conn(n1, n3, weight=3.0)
    builder.add_conn(n2, n3, weight=4.0)
    return builder.finalize()


def test_topological_levels_and_bucket_count() -> None:
    static, state = _build_diamond(_TopoNet)
    assert list(np.asarray(state.units["level"])) == [0, 1, 1, 2]
    # source levels used are {0, 1} -> two buckets.
    assert len(static.level_capacities) == 2
    assert static.input_ids == (0,)
    assert static.output_ids == (3,)


def test_buckets_sorted_and_live_counts() -> None:
    static, state = _build_diamond(_TopoNet)
    for bucket in state.conns:
        _assert_bucket_sorted_by_dead_then_to_id(bucket)

    assert int(px.state.live_conn_count(state, 0)) == 2
    assert int(px.state.live_conn_count(state, 1)) == 2
    assert int(px.state.live_conn_count(state)) == 4

    # bucket 0 carries unit 0's edges to 1 and 2, sorted ascending by to_id.
    assert np.asarray(state.conns[0]["to_id"])[:2].tolist() == [1, 2]
    assert np.asarray(state.conns[0]["from_id"])[:2].tolist() == [0, 0]
    # bucket 1 carries 1->3 and 2->3.
    assert sorted(np.asarray(state.conns[1]["from_id"])[:2].tolist()) == [1, 2]
    assert np.asarray(state.conns[1]["to_id"])[:2].tolist() == [3, 3]


def test_bucket_capacities_obey_capacity_policy() -> None:
    static, state = _build_diamond(_TopoNet)
    for level in range(len(state.conns)):
        live = int(px.state.live_conn_count(state, level))
        assert static.level_capacities[level] == topo.capacity_policy(live)


def test_capacity_policy_grows_past_min_bucket_for_a_large_bucket() -> None:
    """10x10 bipartite: 100 live edges in one bucket forces next_pow2(100)."""
    builder = px.NetworkBuilder(_TopoNet, None)
    layer0 = [builder.add_unit() for _ in range(10)]
    layer1 = [builder.add_unit() for _ in range(10)]
    for src in layer0:
        for dst in layer1:
            builder.add_conn(src, dst, weight=0.1)
    static, state = builder.finalize()

    assert len(static.level_capacities) == 1  # only source level 0 has edges
    assert int(px.state.live_conn_count(state, 0)) == 100
    assert static.level_capacities[0] == 128
    assert static.level_capacities[0] == topo.capacity_policy(100)
    _assert_bucket_sorted_by_dead_then_to_id(state.conns[0])


def test_pipeline_mode_uses_a_single_bucket_but_still_computes_levels() -> None:
    static, state = _build_diamond(_PipelineNet)
    assert len(static.level_capacities) == 1
    assert int(px.state.live_conn_count(state)) == 4
    _assert_bucket_sorted_by_dead_then_to_id(state.conns[0])
    # LEVEL is still the topological depth, independent of bucketing mode.
    assert list(np.asarray(state.units["level"])) == [0, 1, 1, 2]


def test_add_unit_dense_zero_based_ids() -> None:
    builder = px.NetworkBuilder(_TopoNet, None)
    ids = [builder.add_unit() for _ in range(5)]
    assert ids == [0, 1, 2, 3, 4]


def test_add_unit_field_defaults_and_overrides() -> None:
    builder = px.NetworkBuilder(_ExtraFieldsNet, None)
    default_unit = builder.add_unit()
    biased_unit = builder.add_unit(bias=2.5, activation=1.0)
    builder.add_conn(default_unit, biased_unit, weight=1.0, tag=9.0)
    static, state = builder.finalize()

    assert np.asarray(state.units["bias"]).tolist() == [0.0, 2.5]
    assert np.asarray(state.units["activation"]).tolist() == [0.0, 1.0]
    assert "tag" in {spec.name for spec in static.conn_fields}
    tag_values = np.asarray(state.conns[0]["tag"])
    live = int(px.state.live_conn_count(state, 0))
    assert tag_values[:live].tolist() == [9.0]


def test_add_unit_rejects_unknown_field() -> None:
    builder = px.NetworkBuilder(_TopoNet, None)
    with pytest.raises(ValueError):
        builder.add_unit(not_a_field=1.0)


def test_add_unit_rejects_level_as_a_settable_field() -> None:
    builder = px.NetworkBuilder(_TopoNet, None)
    with pytest.raises(ValueError):
        builder.add_unit(level=3)


def test_add_conn_rejects_unknown_field() -> None:
    builder = px.NetworkBuilder(_TopoNet, None)
    a, b = builder.add_unit(), builder.add_unit()
    with pytest.raises(ValueError):
        builder.add_conn(a, b, not_a_field=1.0)


def test_add_conn_rejects_from_id_to_id_dead_as_settable_fields() -> None:
    builder = px.NetworkBuilder(_TopoNet, None)
    a, b = builder.add_unit(), builder.add_unit()
    for name in ("from_id", "to_id", "dead"):
        with pytest.raises(ValueError):
            builder.add_conn(a, b, **{name: 0})


def test_finalize_rejects_out_of_range_conn_endpoints() -> None:
    builder = px.NetworkBuilder(_TopoNet, None)
    a = builder.add_unit()
    builder.add_conn(a, 99, weight=1.0)
    with pytest.raises(ValueError):
        builder.finalize()


def test_finalize_rejects_out_of_range_marked_ids() -> None:
    builder = px.NetworkBuilder(_TopoNet, None)
    builder.add_unit()
    builder.mark_output(41)
    with pytest.raises(ValueError):
        builder.finalize()


def test_from_topology_equals_manual_builder_construction() -> None:
    key = jax.random.PRNGKey(7)
    topo_fn = topology.sequential(
        topology.input_units(3),
        topology.dense(3, 5),
        topology.dense(5, 2),
    )
    spec = topo_fn(key)

    static_auto, state_auto = px.NetworkBuilder.from_topology(
        _TopoNet, topo_fn, key, globals_=None
    )

    manual = px.NetworkBuilder(_TopoNet, None)
    for _ in range(spec.num_units):
        manual.add_unit()
    for unit_id in spec.input_ids:
        manual.mark_input(unit_id)
    for unit_id in spec.output_ids:
        manual.mark_output(unit_id)
    for from_id, to_id, weight in zip(
        spec.edges.from_ids.tolist(),
        spec.edges.to_ids.tolist(),
        spec.edges.weights.tolist(),
        strict=True,
    ):
        manual.add_conn(from_id, to_id, weight=weight)
    static_manual, state_manual = manual.finalize()

    assert static_auto == static_manual
    assert state_auto.units.keys() == state_manual.units.keys()
    for name in state_auto.units:
        assert jnp.array_equal(state_auto.units[name], state_manual.units[name])
    assert len(state_auto.conns) == len(state_manual.conns)
    for auto_bucket, manual_bucket in zip(
        state_auto.conns, state_manual.conns, strict=True
    ):
        for name in auto_bucket:
            assert jnp.array_equal(auto_bucket[name], manual_bucket[name])


# --- from_edges (vectorized construction) -------------------------------


def _assert_arenas_equal(
    a: tuple[px.NetworkStatic, px.NetworkState[None]],
    b: tuple[px.NetworkStatic, px.NetworkState[None]],
) -> None:
    """Byte-identical (static, state) arenas: same config and every column."""
    static_a, state_a = a
    static_b, state_b = b
    assert static_a == static_b
    assert state_a.units.keys() == state_b.units.keys()
    for name in state_a.units:
        assert jnp.array_equal(state_a.units[name], state_b.units[name])
    assert len(state_a.conns) == len(state_b.conns)
    for bucket_a, bucket_b in zip(state_a.conns, state_b.conns, strict=True):
        assert bucket_a.keys() == bucket_b.keys()
        for name in bucket_a:
            assert jnp.array_equal(bucket_a[name], bucket_b[name])


def test_from_edges_equals_manual_builder_with_weights_and_extra_columns() -> None:
    """The vectorized path reproduces the per-edge path byte-for-byte.

    Diamond 0->1, 0->2, 1->3, 2->3 with per-edge weights and a settable extra
    conn field (tag): from_edges must bucket, stable-sort by dst, and pad
    exactly as add_conn + finalize do, including permuting the tag column with
    its edges and default-filling the untouched Bias unit field.
    """
    from_ids = np.array([0, 0, 1, 2], dtype=np.int32)
    to_ids = np.array([1, 2, 3, 3], dtype=np.int32)
    weights = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    tags = np.array([9.0, 8.0, 7.0, 6.0], dtype=np.float32)

    vectorized = px.NetworkBuilder.from_edges(
        _ExtraFieldsNet,
        4,
        from_ids,
        to_ids,
        weights=weights,
        input_ids=(0,),
        output_ids=(3,),
        globals_=None,
        extra_conn_columns={"tag": tags},
    )

    manual = px.NetworkBuilder(_ExtraFieldsNet, None)
    for _ in range(4):
        manual.add_unit()
    manual.mark_input(0)
    manual.mark_output(3)
    for s, d, w, t in zip(
        from_ids.tolist(), to_ids.tolist(), weights.tolist(), tags.tolist(), strict=True
    ):
        manual.add_conn(s, d, weight=w, tag=t)

    _assert_arenas_equal(vectorized, manual.finalize())


def test_from_edges_multi_bucket_matches_manual_at_scale() -> None:
    """A three-level net wired with hundreds of edges matches the per-edge path.

    Exercises multiple source-level buckets and next_pow2 capacity growth in a
    single vectorized pass -- the regime the per-edge loop is slow in.
    """
    rng = np.random.default_rng(0)
    # layers [0,32) -> [32,96) -> [96,112); wire each pair densely-ish random.
    froms: list[np.ndarray] = []
    tos: list[np.ndarray] = []
    for lo_in, hi_in, lo_out, hi_out in [(0, 32, 32, 96), (32, 96, 96, 112)]:
        src = rng.integers(lo_in, hi_in, size=400)
        dst = rng.integers(lo_out, hi_out, size=400)
        # one seed edge per output unit so no unit orphans to level 0.
        seed_dst = np.arange(lo_out, hi_out)
        seed_src = rng.integers(lo_in, hi_in, size=seed_dst.size)
        froms.append(np.concatenate([seed_src, src]))
        tos.append(np.concatenate([seed_dst, dst]))
    from_ids = np.concatenate(froms).astype(np.int32)
    to_ids = np.concatenate(tos).astype(np.int32)
    weights = rng.standard_normal(from_ids.size).astype(np.float32)

    vectorized = px.NetworkBuilder.from_edges(
        _TopoNet,
        112,
        from_ids,
        to_ids,
        weights=weights,
        input_ids=tuple(range(32)),
        output_ids=tuple(range(96, 112)),
        globals_=None,
    )

    manual = px.NetworkBuilder(_TopoNet, None)
    for _ in range(112):
        manual.add_unit()
    for i in range(32):
        manual.mark_input(i)
    for i in range(96, 112):
        manual.mark_output(i)
    for s, d, w in zip(
        from_ids.tolist(), to_ids.tolist(), weights.tolist(), strict=True
    ):
        manual.add_conn(s, d, weight=w)

    _assert_arenas_equal(vectorized, manual.finalize())


def test_from_edges_empty_edge_set_matches_manual() -> None:
    """No edges: one empty tombstoned bucket, exactly as finalize builds."""
    empty = np.zeros((0,), dtype=np.int32)
    vectorized = px.NetworkBuilder.from_edges(
        _TopoNet,
        3,
        empty,
        empty,
        input_ids=(0, 1, 2),
        output_ids=(0, 1, 2),
        globals_=None,
    )

    manual = px.NetworkBuilder(_TopoNet, None)
    for _ in range(3):
        manual.add_unit()
    for i in range(3):
        manual.mark_input(i)
        manual.mark_output(i)

    _assert_arenas_equal(vectorized, manual.finalize())


def test_from_edges_pipeline_mode_single_bucket() -> None:
    """PIPELINE construction lands every edge in one flat bucket."""
    from_ids = np.array([0, 1, 2], dtype=np.int32)
    to_ids = np.array([1, 2, 0], dtype=np.int32)  # a cycle: legal in pipeline
    static, state = px.NetworkBuilder.from_edges(
        _PipelineNet,
        3,
        from_ids,
        to_ids,
        weights=np.ones(3, dtype=np.float32),
        input_ids=(0,),
        output_ids=(2,),
        globals_=None,
    )
    assert len(static.level_capacities) == 1
    assert int(px.state.live_conn_count(state)) == 3


def test_from_edges_defaults_weight_when_omitted() -> None:
    """Omitting weights fills the WEIGHT column with its spec default (0)."""
    static, state = px.NetworkBuilder.from_edges(
        _TopoNet,
        2,
        np.array([0], dtype=np.int32),
        np.array([1], dtype=np.int32),
        input_ids=(0,),
        output_ids=(1,),
        globals_=None,
    )
    live = int(px.state.live_conn_count(state, 0))
    assert np.asarray(state.conns[0]["weight"])[:live].tolist() == [0.0]


def test_from_edges_rejects_mismatched_endpoint_lengths() -> None:
    with pytest.raises(ValueError):
        px.NetworkBuilder.from_edges(
            _TopoNet,
            3,
            np.array([0, 1], dtype=np.int32),
            np.array([1], dtype=np.int32),
            input_ids=(0,),
            output_ids=(2,),
            globals_=None,
        )


def test_from_edges_rejects_mismatched_weight_length() -> None:
    with pytest.raises(ValueError):
        px.NetworkBuilder.from_edges(
            _TopoNet,
            3,
            np.array([0, 1], dtype=np.int32),
            np.array([1, 2], dtype=np.int32),
            weights=np.ones(3, dtype=np.float32),
            input_ids=(0,),
            output_ids=(2,),
            globals_=None,
        )


def test_from_edges_rejects_out_of_range_endpoint() -> None:
    with pytest.raises(ValueError):
        px.NetworkBuilder.from_edges(
            _TopoNet,
            3,
            np.array([0], dtype=np.int32),
            np.array([99], dtype=np.int32),
            input_ids=(0,),
            output_ids=(2,),
            globals_=None,
        )


def test_from_edges_rejects_unknown_extra_conn_column() -> None:
    with pytest.raises(ValueError):
        px.NetworkBuilder.from_edges(
            _TopoNet,
            2,
            np.array([0], dtype=np.int32),
            np.array([1], dtype=np.int32),
            input_ids=(0,),
            output_ids=(1,),
            globals_=None,
            extra_conn_columns={"not_a_field": np.zeros(1, dtype=np.float32)},
        )
