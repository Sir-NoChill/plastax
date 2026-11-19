"""Topology: the shapes handles carry and what each layer creates.

Design document sections 2.1 to 2.3. Nothing here runs a step; every test builds a
network and looks at what was built.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import plastax as px

from . import support


def built(call: Any, input_shape: tuple[int, ...], **layers: Any) -> support.Runner:
    return support.Runner(
        support.network(support.topology(call, **layers), input_shape=input_shape)
    )


# --------------------------------------------------------------------------- #
# Shapes
# --------------------------------------------------------------------------- #
def test_layer_output_shapes() -> None:
    shapes: dict[str, tuple[int, ...]] = {}

    def call(m: Any, x: Any) -> Any:
        shapes["input"] = x.shape
        h = m.conv(x)
        shapes["conv"] = h.shape
        p = m.pool(h)
        shapes["pool"] = p.shape
        flat = p.flatten()
        shapes["flat"] = flat.shape
        out = m.head(flat)
        shapes["head"] = out.shape
        return out

    built(
        call,
        (8, 8, 1),
        conv=px.layers.Conv(4, 3, padding="same", shared=True),
        pool=px.layers.MaxPool(2),
        head=px.layers.Dense(10),
    )

    assert shapes == {
        "input": (8, 8, 1),
        "conv": (8, 8, 4),
        "pool": (4, 4, 4),
        "flat": (64,),
        "head": (10,),
    }


def test_dense_acts_on_the_last_axis() -> None:
    shapes: dict[str, tuple[int, ...]] = {}

    def call(m: Any, x: Any) -> Any:
        out = m.fc(x)
        shapes["out"] = out.shape
        return out

    built(call, (5, 4), fc=px.layers.Dense(3))

    assert shapes["out"] == (5, 3)


def test_shape_relabelling_and_broadcasting() -> None:
    shapes: dict[str, tuple[int, ...]] = {}

    def call(m: Any, x: Any) -> Any:  # x: (2, 3)
        shapes["none"] = x[:, None, :].shape
        shapes["reshape"] = x.reshape(3, 2).shape
        shapes["flatten"] = x.flatten().shape
        shapes["product"] = (x[:, None, :] * x[None, :, :]).shape
        shapes["sum"] = x.sum(axis=-1).shape
        shapes["sum_keepdims"] = x.sum(axis=-1, keepdims=True).shape
        shapes["mean"] = x.mean(axis=0).shape
        shapes["difference"] = (x - x.mean(axis=-1, keepdims=True)).shape
        return x

    built(call, (2, 3))

    assert shapes == {
        "none": (2, 1, 3),
        "reshape": (3, 2),
        "flatten": (6,),
        "product": (2, 2, 3),
        "sum": (2,),
        "sum_keepdims": (2, 1),
        "mean": (3,),
        "difference": (2, 3),
    }


# --------------------------------------------------------------------------- #
# What each layer creates
# --------------------------------------------------------------------------- #
def test_dense_layer_creates_all_to_all_connections() -> None:
    fc = px.layers.Dense(3, bias=False)
    r = built(lambda m, x: m.fc(x), (4,), fc=fc)

    assert r.num_units() == 4 + 3
    assert len(r.pairs(fc)) == 3 * 4


def test_bias_is_one_constant_unit_and_one_connection_per_unit() -> None:
    fc = px.layers.Dense(3, bias=True)
    r = built(lambda m, x: m.fc(x), (4,), fc=fc)

    assert r.num_units() == 4 + 3 + 1
    assert r.num_connections() == 3 * 4 + 3


def test_shared_dense_has_one_weight_per_entry_and_one_connection_per_use() -> None:
    positions, d_in, d_out = 5, 4, 3
    fc = px.layers.Dense(d_out, shared=True, bias=False, init=px.init.lecun_uniform)
    r = built(lambda m, x: m.fc(x), (positions, d_in), fc=fc)

    weights = r.conn_field(fc, px.WEIGHT)
    assert len(weights) == positions * d_out * d_in
    assert len(np.unique(weights)) == d_out * d_in


def test_unshared_dense_on_leading_axes_has_a_weight_per_position() -> None:
    positions, d_in, d_out = 5, 4, 3
    fc = px.layers.Dense(d_out, shared=False, bias=False, init=px.init.lecun_uniform)
    r = built(lambda m, x: m.fc(x), (positions, d_in), fc=fc)

    weights = r.conn_field(fc, px.WEIGHT)
    assert len(weights) == positions * d_out * d_in
    assert len(np.unique(weights)) == positions * d_out * d_in


def test_shared_convolution_counts() -> None:
    conv = px.layers.Conv(
        2, 3, padding="valid", shared=True, bias=False, init=px.init.lecun_uniform
    )
    r = built(lambda m, x: m.conv(x), (5, 5, 1), conv=conv)

    weights = r.conn_field(conv, px.WEIGHT)
    # 3x3 output positions, 2 output channels, a 3x3x1 window each.
    assert len(weights) == (3 * 3 * 2) * (3 * 3 * 1)
    assert len(np.unique(weights)) == 3 * 3 * 1 * 2
    assert r.num_units() == 5 * 5 * 1 + 3 * 3 * 2


def test_same_padding_drops_the_connections_that_fall_outside() -> None:
    conv = px.layers.Conv(1, 3, padding="same", shared=True, bias=False)
    r = built(lambda m, x: m.conv(x), (4, 4, 1), conv=conv)

    # A 3x3 window over a 4x4 grid: 4 corners see 4 inputs, 8 edges see 6, 4 centres see
    # 9.
    assert len(r.pairs(conv)) == 4 * 4 + 8 * 6 + 4 * 9


def test_max_pool_creates_one_unit_per_window() -> None:
    pool = px.layers.MaxPool(2)
    r = built(lambda m, x: m.pool(x), (4, 4, 3), pool=pool)

    assert r.num_units() == 4 * 4 * 3 + 2 * 2 * 3
    assert len(r.pairs(pool)) == 4 * 4 * 3  # every input feeds exactly one window


def test_sparse_fan_in_gives_every_unit_that_many_distinct_inputs() -> None:
    fc = px.layers.Dense(6, connectivity=px.layers.sparse(fan_in=3), bias=False)
    r = built(lambda m, x: m.fc(x), (10,), fc=fc)

    pairs = r.pairs(fc)
    assert len(pairs) == 6 * 3
    assert len(set(pairs)) == len(pairs)
    for unit in {dst for _, dst in pairs}:
        assert sum(dst == unit for _, dst in pairs) == 3


def test_sparse_density_gives_about_that_fraction_of_connections() -> None:
    fc = px.layers.Dense(40, connectivity=px.layers.sparse(density=0.25), bias=False)
    r = built(lambda m, x: m.fc(x), (40,), fc=fc)

    assert 0.15 * 1600 < len(r.pairs(fc)) < 0.35 * 1600


def test_explicit_pairs_are_the_only_connections() -> None:
    wiring = px.layers.pairs(src=np.array([0, 1, 2, 3]), dst=np.array([0, 0, 1, 1]))
    fc = px.layers.Dense(2, connectivity=wiring, bias=False)
    r = built(lambda m, x: m.fc(x), (4,), fc=fc)

    sources = np.asarray(px.inspect.sources(r.net, fc)).reshape(-1)
    destinations = np.asarray(px.inspect.destinations(r.net, fc)).reshape(-1)
    expected = {
        (int(sources[s]), int(destinations[d]))
        for s, d in [(0, 0), (1, 0), (2, 1), (3, 1)]
    }
    assert set(r.pairs(fc)) == expected


# --------------------------------------------------------------------------- #
# Handle operations: what creates units and what does not
# --------------------------------------------------------------------------- #
def test_relabelling_a_handle_creates_nothing() -> None:
    def plain(m: Any, x: Any) -> Any:
        return x

    def relabelled(m: Any, x: Any) -> Any:
        return x[:, None, :].reshape(3, 2).flatten()

    a, b = built(plain, (2, 3)), built(relabelled, (2, 3))

    assert b.num_units() == a.num_units() == 6
    assert b.num_connections() == a.num_connections() == 0


def test_reshape_relabels_in_row_major_order() -> None:
    grid = px.UnitsRef("grid")
    flat = px.UnitsRef("flat")

    def call(m: Any, x: Any) -> Any:  # x: (2, 3)
        x.bind(grid)
        return x.reshape(3, 2).flatten().bind(flat)

    r = built(call, (2, 3))

    grid_ids, flat_ids = r.unit_ids(grid), r.unit_ids(flat)
    assert grid_ids.shape == (2, 3)
    np.testing.assert_array_equal(flat_ids, grid_ids.reshape(-1))


def test_product_creates_one_unit_per_broadcast_element() -> None:
    r = built(lambda m, x: x[:, None, :] * x[None, :, :], (2, 3))

    assert r.num_units() == 6 + 2 * 2 * 3
    assert r.num_connections() == 2 * (2 * 2 * 3)  # a left and a right factor each


def test_sum_over_an_axis_creates_one_unit_per_remaining_element() -> None:
    r = built(lambda m, x: x.sum(axis=-1), (2, 3))

    assert r.num_units() == 6 + 2
    assert r.num_connections() == 6


# --------------------------------------------------------------------------- #
# The merge of `a + b`
# --------------------------------------------------------------------------- #
def test_sum_is_merged_into_an_operand_with_no_other_reader() -> None:
    """Neither `fa(x)` nor `fb(x)` is read elsewhere, so `+` adds no units."""
    r = built(
        lambda m, x: m.fa(x) + m.fb(x),
        (4,),
        fa=px.layers.Dense(3, bias=False),
        fb=px.layers.Dense(3, bias=False),
    )

    assert r.num_units() == 4 + 3 + 3


def test_sum_is_its_own_units_when_no_operand_can_absorb_it() -> None:
    """`a` is also returned and `b` has an activation, so neither can absorb the sum."""
    relu = px.activations.relu

    def call(m: Any, x: Any) -> Any:
        a = m.fa(x)
        b = relu(m.fb(x))
        return a + b, a

    r = built(
        call,
        (4,),
        fa=px.layers.Dense(3, bias=False),
        fb=px.layers.Dense(3, bias=False),
    )

    assert r.num_units() == 4 + 3 + 3 + 3


@pytest.mark.parametrize("op", ["add", "subtract"])
def test_sum_never_changes_its_operands(op: str) -> None:
    """A unit read by a sum and by something else keeps its own value."""

    def call(m: Any, x: Any) -> Any:
        a, b = m.fa(x), m.fb(x)
        return (a + b if op == "add" else a - b), a

    wa = np.arange(12, dtype=np.float32).reshape(3, 4) / 10
    wb = np.ones((3, 4), np.float32)
    r = built(
        call,
        (4,),
        fa=px.layers.Dense(3, bias=False, init=px.init.array(wa)),
        fb=px.layers.Dense(3, bias=False, init=px.init.array(wb)),
    )
    x = np.array([1.0, -2.0, 3.0, 0.5], np.float32)
    combined, a = r.predict(x).outputs

    support.close(a, wa @ x)
    support.close(combined, wa @ x + (wb @ x if op == "add" else -(wb @ x)))
