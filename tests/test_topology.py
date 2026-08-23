"""Topology generators (M1).

dense edge count = n_in*n_out; conv2d edge enumeration matches
lax.conv_general_dilated shape semantics (positions, receptive fields,
stride); initializer statistics sane; sequential id offsetting; from_topology
equals the equivalent manual builder calls. Implemented when M1 lands.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import plastax as px
from plastax import topology
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


# --- dense ------------------------------------------------------------------


def test_dense_edge_count_is_n_in_times_n_out() -> None:
    block = topology.dense(4, 7)
    assert block.num_units == 7
    edges = block.edges(jax.random.PRNGKey(0), 0, 4)
    assert edges.from_ids.shape == (28,)
    assert edges.to_ids.shape == (28,)
    assert edges.weights.shape == (28,)
    assert edges.from_ids.dtype == np.int32
    assert edges.to_ids.dtype == np.int32
    assert edges.weights.dtype == np.float32


def test_dense_is_fully_connected_bipartite_with_offsets_applied() -> None:
    offset_in, offset_out = 10, 100
    block = topology.dense(3, 2)
    edges = block.edges(jax.random.PRNGKey(0), offset_in, offset_out)

    seen = {(int(f), int(t)) for f, t in zip(edges.from_ids, edges.to_ids, strict=True)}
    expected = {(offset_in + i, offset_out + o) for i in range(3) for o in range(2)}
    assert seen == expected


def test_dense_initializer_statistics_are_sane() -> None:
    """Glorot uniform: Var = 2 / (fan_in + fan_out); loose bound on a large
    sample keeps this a shape/scale sanity check, not a statistical test."""
    n_in, n_out = 200, 200
    block = topology.dense(n_in, n_out)
    edges = block.edges(jax.random.PRNGKey(0), 0, n_in)
    weights = np.asarray(edges.weights)

    expected_var = 2.0 / (n_in + n_out)
    assert abs(float(weights.mean())) < 0.01
    assert 0.5 * expected_var < float(weights.var()) < 1.5 * expected_var


# --- conv2d -------------------------------------------------------------


def _decode_flat(index: int, minor_size: int) -> tuple[int, int]:
    """Inverse of (major * minor_size + minor); returns (major, minor)."""
    return index // minor_size, index % minor_size


@pytest.mark.parametrize(
    ("h", "w", "c_in", "kh", "kw", "c_out", "stride"),
    [
        (5, 5, 2, 2, 2, 3, 1),
        (6, 6, 2, 3, 3, 4, 2),
        (7, 5, 1, 2, 3, 2, 1),
    ],
)
def test_conv2d_output_shape_matches_lax_conv_general_dilated(
    h: int, w: int, c_in: int, kh: int, kw: int, c_out: int, stride: int
) -> None:
    block = topology.conv2d((h, w, c_in), (kh, kw, c_out), stride=stride)

    x = jnp.zeros((1, h, w, c_in))
    kernel = jnp.zeros((kh, kw, c_in, c_out))
    lax_out = jax.lax.conv_general_dilated(
        x,
        kernel,
        window_strides=(stride, stride),
        padding="VALID",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    _, out_h, out_w, out_c = lax_out.shape

    assert block.num_units == out_h * out_w * out_c
    edges = block.edges(jax.random.PRNGKey(0), 0, h * w * c_in)
    assert edges.from_ids.shape == (out_h * out_w * out_c * kh * kw * c_in,)


def test_conv2d_receptive_field_membership_matches_conv_semantics() -> None:
    h, w, c_in = 6, 6, 2
    kh, kw, c_out = 3, 3, 4
    stride = 2
    out_w = (w - kw) // stride + 1
    offset_in, offset_out = 0, 10_000

    block = topology.conv2d((h, w, c_in), (kh, kw, c_out), stride=stride)
    edges = block.edges(jax.random.PRNGKey(3), offset_in, offset_out)

    target_oh, target_ow = 1, 1
    expected_patch = {
        (target_oh * stride + dh, target_ow * stride + dw, ci)
        for dh in range(kh)
        for dw in range(kw)
        for ci in range(c_in)
    }

    found_patch: set[tuple[int, int, int]] = set()
    matched_edges = 0
    for from_id, to_id in zip(
        edges.from_ids.tolist(), edges.to_ids.tolist(), strict=True
    ):
        out_local = to_id - offset_out
        oh, ow = _decode_flat(out_local // c_out, out_w)
        if (oh, ow) != (target_oh, target_ow):
            continue
        matched_edges += 1
        in_local = from_id - offset_in
        ci = in_local % c_in
        ih, iw = _decode_flat(in_local // c_in, w)
        found_patch.add((ih, iw, ci))

    assert matched_edges == kh * kw * c_in * c_out
    assert found_patch == expected_patch


@pytest.mark.parametrize(
    ("h", "w", "c_in", "kh", "kw", "c_out", "stride"),
    [
        (5, 5, 2, 2, 2, 3, 1),
        (6, 6, 2, 3, 3, 4, 2),
        (7, 7, 3, 3, 3, 5, 3),
    ],
)
def test_conv2d_forward_computes_a_convolution(
    h: int, w: int, c_in: int, kh: int, kw: int, c_out: int, stride: int
) -> None:
    """Running a conv2d layer through the sweep computes an actual convolution.

    The edge-enumeration tests above check structure; this checks the numbers.
    At init every unrolled edge carries the shared kernel value, so the per-unit
    weighted sum the forward sweep produces must equal
    jax.lax.conv_general_dilated on the same image and kernel.
    """
    kernel = jax.random.normal(jax.random.PRNGKey(0), (kh, kw, c_in, c_out))
    build = topology.sequential(
        topology.input_units(h * w * c_in),
        topology.conv2d(
            (h, w, c_in), (kh, kw, c_out), stride=stride, init=lambda _k, _s: kernel
        ),
    )
    static, state = px.NetworkBuilder.from_topology(
        _TopoNet, build, jax.random.PRNGKey(1), globals_=None
    )
    image = jax.random.normal(jax.random.PRNGKey(2), (h, w, c_in))
    result = px.make_step(_TopoNet, static)(
        state, px.StepInputs(inputs=image.reshape(-1), targets=None)
    )
    got = np.asarray(result.state.units[px.ACTIVATION.name])[
        np.asarray(static.output_ids)
    ]
    reference = jax.lax.conv_general_dilated(
        image[None],
        kernel,
        window_strides=(stride, stride),
        padding="VALID",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )[0].reshape(-1)
    np.testing.assert_allclose(got, np.asarray(reference), atol=1e-4)


def test_conv2d_initializer_statistics_are_sane() -> None:
    """He normal: Var = 2 / fan_in, fan_in = c_in * kh * kw."""
    h, w, c_in = 10, 10, 4
    kh, kw, c_out = 3, 3, 6
    block = topology.conv2d((h, w, c_in), (kh, kw, c_out))
    edges = block.edges(jax.random.PRNGKey(1), 0, h * w * c_in)
    weights = np.asarray(edges.weights)

    fan_in = c_in * kh * kw
    expected_var = 2.0 / fan_in
    assert abs(float(weights.mean())) < 0.05
    assert 0.5 * expected_var < float(weights.var()) < 1.5 * expected_var


# --- input_units --------------------------------------------------------


def test_input_units_has_no_edges() -> None:
    block = topology.input_units(5)
    assert block.num_units == 5
    edges = block.edges(jax.random.PRNGKey(0), 0, 0)
    assert edges.from_ids.shape == (0,)
    assert edges.to_ids.shape == (0,)
    assert edges.weights.shape == (0,)


# --- sequential -----------------------------------------------------------


def test_sequential_offsets_ids_with_no_overlap_and_wires_consecutive_blocks() -> None:
    topo_fn = topology.sequential(
        topology.input_units(3),
        topology.dense(3, 4),
        topology.dense(4, 2),
    )
    spec = topo_fn(jax.random.PRNGKey(0))

    assert spec.num_units == 3 + 4 + 2
    assert spec.input_ids == (0, 1, 2)
    assert spec.output_ids == (7, 8)
    assert spec.edges.from_ids.shape == (3 * 4 + 4 * 2,)

    # block boundaries: [0,3) input, [3,7) hidden, [7,9) output.
    first_layer = [
        (int(f), int(t))
        for f, t in zip(spec.edges.from_ids, spec.edges.to_ids, strict=True)
        if t < 7
    ]
    second_layer = [
        (int(f), int(t))
        for f, t in zip(spec.edges.from_ids, spec.edges.to_ids, strict=True)
        if t >= 7
    ]
    assert len(first_layer) == 12
    assert all(0 <= f < 3 and 3 <= t < 7 for f, t in first_layer)
    assert len(second_layer) == 8
    assert all(3 <= f < 7 and 7 <= t < 9 for f, t in second_layer)


def test_sequential_is_deterministic_for_a_fixed_key() -> None:
    topo_fn = topology.sequential(topology.input_units(2), topology.dense(2, 3))
    key = jax.random.PRNGKey(123)
    spec_a = topo_fn(key)
    spec_b = topo_fn(key)
    assert np.array_equal(spec_a.edges.from_ids, spec_b.edges.from_ids)
    assert np.array_equal(spec_a.edges.to_ids, spec_b.edges.to_ids)
    assert np.array_equal(spec_a.edges.weights, spec_b.edges.weights)


def test_sequential_requires_at_least_one_block() -> None:
    with pytest.raises(ValueError):
        topology.sequential()


def test_sequential_single_block_is_both_input_and_output() -> None:
    topo_fn = topology.sequential(topology.input_units(4))
    spec = topo_fn(jax.random.PRNGKey(0))
    assert spec.num_units == 4
    assert spec.input_ids == (0, 1, 2, 3)
    assert spec.output_ids == (0, 1, 2, 3)
    assert spec.edges.from_ids.shape == (0,)


# --- from_topology equivalence ------------------------------------------


def test_from_topology_equals_manual_builder_for_a_conv_then_dense_net() -> None:
    key = jax.random.PRNGKey(9)
    topo_fn = topology.sequential(
        topology.input_units(3 * 3 * 1),
        topology.conv2d((3, 3, 1), (2, 2, 2)),
        topology.dense(4 * 2, 3),
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
    for name in state_auto.units:
        assert jnp.array_equal(state_auto.units[name], state_manual.units[name])
    for auto_bucket, manual_bucket in zip(
        state_auto.conns, state_manual.conns, strict=True
    ):
        for name in auto_bucket:
            assert jnp.array_equal(auto_bucket[name], manual_bucket[name])
