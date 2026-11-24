"""Where a network built from a topology stores its blocks: runs, capacity, spare room.

Plan `plans/20261123T215119Z-layer-policies-stage3.md` (workspace root), part 2. No
test here asserts a literal unit id or slot: runs are read from `static.layout` and
compared with the blocks of the build record.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import plastax as px
import plastax.layers as pxl
from plastax.builder import BlockSpan
from plastax.layers import BuildError, CreateConnections, CreateUnits, Dense, Units
from plastax.state import Layout

KEY = jax.random.key(0)
ALIGN = 64  # small buckets, so a test reads a whole one


class WeightedSum(px.policies.ForwardConnPolicy):
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


class Summed(px.policies.ForwardUnitPolicy):
    combine = px.monoid.sum_

    def apply(
        self, u: px.UnitView, i: px.UnitIdx, g: None, acc: jax.Array
    ) -> px.UnitWrite:
        return px.UnitWrite.of((px.ACTIVATION, acc))


class Anywhere(px.policies.ScoreAddConnPolicy[None]):
    """The network's own growth: any two units, deeper only."""

    max_new_per_level = 1

    def score(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: None
    ) -> jax.Array:
        return jnp.where(u[px.LEVEL, dst] > u[px.LEVEL, src], 1.0, -jnp.inf)

    def init(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: None
    ) -> px.ConnWrite:
        return px.ConnWrite.of((px.WEIGHT, 0.0))


def module(call: Callable[..., Any], **layers: Any) -> pxl.Module:
    """A topology with the given layers whose `__call__` is `call(module, x)`."""

    class Topology(pxl.Module):
        def setup(self) -> None:
            for name, layer in layers.items():
                setattr(self, name, layer)

    Topology.__call__ = call  # type: ignore[method-assign]
    return Topology()


def network(
    topology: pxl.Module, shape: tuple[int, ...] = (5,), **attributes: Any
) -> type[px.Network[None]]:
    """A linear network over `topology`, with small buckets unless told otherwise."""
    fields = {
        "topology": topology,
        "input_shape": shape,
        "forward": px.policies.Forward(conn=WeightedSum(), unit=Summed()),
        "capacity_align": ALIGN,
        **attributes,
    }
    return type("Net", (px.Network,), fields)


def three_layers(bias: bool = True, **middle: Any) -> pxl.Module:
    return module(
        lambda m, x: m.head(m.middle(m.first(x))),
        first=Dense(4, bias=bias),
        middle=Dense(3, bias=bias, **middle),
        head=Dense(2, bias=bias),
    )


def built(
    net: type[px.Network[None]],
) -> tuple[pxl.Record, Layout, px.NetworkStatic, px.NetworkState[None]]:
    static, state = net.build(KEY)
    assert static.layout is not None
    return net._trace(), static.layout, static, state


def pairs(bucket: Any, start: int, stop: int) -> list[tuple[int, int]]:
    """The live connections in slots `[start, stop)` of a bucket, in slot order."""
    live = ~np.asarray(bucket["dead"])[start:stop]
    return list(
        zip(
            np.asarray(bucket["from_id"])[start:stop][live].tolist(),
            np.asarray(bucket["to_id"])[start:stop][live].tolist(),
            strict=True,
        )
    )


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #
def test_every_connection_of_a_block_is_inside_its_runs() -> None:
    record, layout, _, state = built(network(three_layers()))

    assert len(layout.conn_runs) == len(record.blocks)
    for block, runs in zip(record.blocks, layout.conn_runs, strict=True):
        stored = [
            pair
            for run in runs
            for pair in pairs(state.conns[run.bucket], run.start, run.stop)
        ]
        written = zip(block.src.tolist(), block.dst.tolist(), strict=True)
        assert sorted(stored) == sorted(written)


def test_a_run_is_exactly_its_block_when_no_capacity_is_given() -> None:
    record, layout, _, state = built(network(three_layers()))

    for block, runs in zip(record.blocks, layout.conn_runs, strict=True):
        assert sum(run.stop - run.start for run in runs) == block.src.size
        for run in runs:
            assert not np.asarray(state.conns[run.bucket]["dead"])[
                run.start : run.stop
            ].any()


def test_the_runs_of_a_bucket_follow_one_another_and_the_rest_is_spare() -> None:
    _, layout, static, state = built(network(three_layers()))

    for bucket, (spare_start, spare_stop) in enumerate(layout.spare):
        runs = sorted(
            (run for of_block in layout.conn_runs for run in of_block),
            key=lambda run: run.start,
        )
        at = 0
        for run in (run for run in runs if run.bucket == bucket):
            assert run.start == at
            at = run.stop
        assert spare_start == at
        assert spare_stop == static.level_capacities[bucket]
        assert np.asarray(state.conns[bucket]["dead"])[spare_start:].all()


def test_inside_a_run_connections_are_sorted_by_source_then_destination() -> None:
    _, layout, _, state = built(network(three_layers()))

    for runs in layout.conn_runs:
        for run in runs:
            stored = pairs(state.conns[run.bucket], run.start, run.stop)
            assert stored == sorted(stored)


def test_unit_runs_are_the_ids_of_each_block_of_units() -> None:
    record, layout, _, _ = built(network(three_layers()))

    assert len(layout.unit_runs) == len(record.unit_blocks)
    for units, (start, stop) in zip(record.unit_blocks, layout.unit_runs, strict=True):
        assert start == units.start
        assert stop - start == int(np.prod(units.shape))
    in_blocks = {i for start, stop in layout.unit_runs for i in range(start, stop)}
    assert in_blocks.isdisjoint(record.inputs["x"].reshape(-1).tolist())


def test_a_sequential_network_is_stored_in_the_order_one_sort_would_give() -> None:
    """Block after block, and sorted by source over the whole bucket, are one order."""
    net = network(three_layers())
    record, _, _, state = built(net)
    _, unblocked = px.NetworkBuilder.from_edges(
        net,
        record.num_units,
        np.concatenate([block.src for block in record.blocks]),
        np.concatenate([block.dst for block in record.blocks]),
        input_ids=tuple(record.inputs["x"].reshape(-1).tolist()),
        output_ids=tuple(record.outputs[0].reshape(-1).tolist()),
        globals_=None,
        capacity_align=ALIGN,
    )

    for laid_out, sorted_whole in zip(state.conns, unblocked.conns, strict=True):
        for column in ("from_id", "to_id", "dead"):
            np.testing.assert_array_equal(laid_out[column], sorted_whole[column])


# --------------------------------------------------------------------------- #
# Capacity
# --------------------------------------------------------------------------- #
def test_a_capacity_leaves_dead_slots_at_the_end_of_the_run() -> None:
    record, layout, _, state = built(network(three_layers(capacity=20)))
    (index,) = [
        k for k, block in enumerate(record.blocks) if block.path == ("middle", "weight")
    ]
    (run,) = layout.conn_runs[index]
    size = record.blocks[index].src.size

    assert size == 12
    assert run.stop - run.start == 20
    dead = np.asarray(state.conns[run.bucket]["dead"])[run.start : run.stop]
    assert not dead[:size].any()
    assert dead[size:].all()


def test_a_capacity_moves_no_other_block_out_of_order() -> None:
    plain = built(network(three_layers()))[1]
    roomy = built(network(three_layers(capacity=20)))[1]

    order = lambda layout: [  # noqa: E731
        [k for k, runs in enumerate(layout.conn_runs) for r in runs if r.bucket == b]
        for b in range(len(layout.spare))
    ]
    assert order(plain) == order(roomy)


def test_a_network_with_room_in_a_layer_computes_what_it_did_without() -> None:
    x = jnp.asarray(np.random.default_rng(0).normal(size=5), jnp.float32)
    inputs = px.StepInputs(inputs=x, targets=None)

    plain = network(three_layers()).init(KEY).step(inputs, outputs=True)
    roomy = network(three_layers(capacity=20)).init(KEY).step(inputs, outputs=True)

    assert plain.outputs is not None and roomy.outputs is not None
    np.testing.assert_allclose(roomy.outputs, plain.outputs, rtol=1e-6)


# --------------------------------------------------------------------------- #
# Blocks that share sources, and blocks on several levels
# --------------------------------------------------------------------------- #
def branches() -> pxl.Module:
    """Two blocks that read the same units."""

    def call(m: Any, x: Units) -> Units:
        y = m.units(shape=(3,))
        m.even(x[None, :], y[:, None])
        m.again(x[None, :], y[:, None])
        return y

    return module(
        call, units=CreateUnits(), even=CreateConnections(), again=CreateConnections()
    )


def test_blocks_that_read_the_same_units_are_stored_one_after_the_other() -> None:
    record, layout, _, state = built(network(branches()))
    (first,), (second,) = layout.conn_runs

    assert first.bucket == second.bucket
    assert (first.start, first.stop, second.stop) == (0, 15, 30)
    assert second.start == first.stop
    # Sorted by source over the whole bucket, the two blocks would alternate.
    stored = pairs(state.conns[first.bucket], 0, 30)
    assert stored != sorted(stored)
    assert stored[:15] == sorted(stored[:15])
    assert sorted(stored[:15]) == sorted(
        zip(record.blocks[0].src.tolist(), record.blocks[0].dst.tolist(), strict=True)
    )


def two_levels(**out: Any) -> pxl.Module:
    """A block `out` whose sources are on two levels: `h[:2]` is fed, `h[2:]` is not."""

    def call(m: Any, x: Units) -> Units:
        h = m.hidden(shape=(4,))
        m.feed(x[None, :], h[:2, None])
        z = m.z(shape=(2,))
        m.out(h[None, :], z[:, None])
        return z

    return module(
        call,
        hidden=CreateUnits(),
        feed=CreateConnections(),
        z=CreateUnits(),
        out=CreateConnections(**out),
    )


def test_a_block_with_sources_on_two_levels_has_a_run_in_each_bucket() -> None:
    record, layout, _, state = built(network(two_levels()))
    runs = layout.conn_runs[1]

    assert record.blocks[1].path == ("out",)
    assert [run.bucket for run in runs] == [0, 1]
    assert [run.stop - run.start for run in runs] == [4, 4]
    for run in runs:
        assert len(pairs(state.conns[run.bucket], run.start, run.stop)) == 4


def test_a_capacity_on_a_block_with_sources_on_two_levels_is_refused() -> None:
    net = network(two_levels(capacity=16))

    with pytest.raises(BuildError, match=r"out is given capacity=16.*2 levels"):
        net.build(KEY)


# --------------------------------------------------------------------------- #
# Room that belongs to no layer
# --------------------------------------------------------------------------- #
def test_a_bucket_is_rounded_up_to_a_multiple_of_the_alignment() -> None:
    _, layout, static, _ = built(network(three_layers()))

    assert all(capacity % ALIGN == 0 for capacity in static.level_capacities)
    assert all(stop - start < ALIGN for start, stop in layout.spare)


def test_the_alignment_is_1024_unless_the_network_says_otherwise() -> None:
    net = network(three_layers())
    del net.capacity_align

    assert net.capacity_align == 1024
    assert all(capacity % 1024 == 0 for capacity in net.build(KEY)[0].level_capacities)


def test_headroom_is_added_to_the_spare_room_of_each_bucket() -> None:
    topology = module(lambda m, x: m.fc(x), fc=Dense(40, bias=False))
    tight = built(network(topology, capacity_align=1))[1]
    roomy = built(network(topology, capacity_align=1, capacity_headroom=0.5))[1]
    (run,) = roomy.conn_runs[0]

    # 200 connections; the bucket is never smaller than 64 slots.
    assert tight.spare == ((200, 200),)
    assert (run.start, run.stop) == (0, 200)
    assert roomy.spare == ((200, 300),)


@pytest.mark.parametrize("align", [0, 1000, 1.5, True])
def test_an_alignment_that_is_not_a_power_of_two_is_refused(align: Any) -> None:
    with pytest.raises(BuildError, match="capacity_align.*power of two"):
        network(three_layers(), capacity_align=align).build(KEY)


@pytest.mark.parametrize("headroom", [-0.5, "half"])
def test_headroom_that_is_not_a_fraction_is_refused(headroom: Any) -> None:
    with pytest.raises(BuildError, match="capacity_headroom"):
        network(three_layers(), capacity_headroom=headroom).build(KEY)


# --------------------------------------------------------------------------- #
# Networks with no layout
# --------------------------------------------------------------------------- #
# The layers below have no bias: a bias is never regrown, which differs from a network
# that grows, and a layer that differs from its network is refused until stage 3.
def test_a_network_built_from_edge_arrays_has_no_layout() -> None:
    net = network(three_layers())

    static, _ = px.NetworkBuilder.from_edges(
        net,
        3,
        np.array([0, 1]),
        np.array([2, 2]),
        input_ids=(0, 1),
        output_ids=(2,),
        globals_=None,
    )

    assert static.layout is None


def test_a_network_that_grows_its_own_connections_has_no_layout() -> None:
    net = network(three_layers(bias=False), grow=px.policies.Grow(conn=Anywhere()))

    static, state = net.build(KEY)

    assert static.layout is None
    # Stored as it was before blocks: each bucket sorted by source as a whole.
    for bucket in state.conns:
        stored = pairs(bucket, 0, np.asarray(bucket["dead"]).size)
        assert stored == sorted(stored)


def test_a_capacity_beside_the_networks_own_growth_is_refused() -> None:
    net = network(
        three_layers(bias=False, capacity=20), grow=px.policies.Grow(conn=Anywhere())
    )

    with pytest.raises(BuildError, match=r"middle\.weight is given capacity=20"):
        net.build(KEY)


# --------------------------------------------------------------------------- #
# The builder
# --------------------------------------------------------------------------- #
def edges(blocks: list[BlockSpan] | None) -> px.NetworkStatic:
    net = network(three_layers())
    static, _ = px.NetworkBuilder.from_edges(
        net,
        4,
        np.array([0, 1, 0, 1]),
        np.array([2, 2, 3, 3]),
        input_ids=(0, 1),
        output_ids=(2, 3),
        globals_=None,
        blocks=blocks,
    )
    return static


def test_from_edges_lays_out_the_blocks_it_is_given() -> None:
    layout = edges([BlockSpan(2), BlockSpan(2, capacity=6)]).layout

    assert layout is not None
    assert [(r.start, r.stop) for runs in layout.conn_runs for r in runs] == [
        (0, 2),
        (2, 8),
    ]
    assert layout.unit_runs == ()  # the builder knows of no blocks of units


@pytest.mark.parametrize(
    ("blocks", "message"),
    [
        ([BlockSpan(3)], "hold 3 edges and 4 were given"),
        ([BlockSpan(2), BlockSpan(2, capacity=1)], "2 edges and a capacity of 1"),
        ([BlockSpan(4), BlockSpan(0, capacity=3)], "in 0 buckets"),
    ],
)
def test_from_edges_refuses_blocks_it_cannot_lay_out(
    blocks: list[BlockSpan], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        edges(blocks)
