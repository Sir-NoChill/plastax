"""`CreateUnits` and `CreateConnections`, checked on the record they write.

Plan `plans/20261119T171340Z-steps-4-5-primitives.md` (workspace root). No test here
asserts a literal unit id: the order in which calls number their units is not
specified. Ids are compared with the ids of the handles involved.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import plastax.layers as pxl
from plastax.layers import (
    BuildError,
    ConstInit,
    CreateConnections,
    CreateUnits,
    Record,
    ShapeError,
    Units,
)


def traced(
    call: Callable[[Any, Units], Any], shape: tuple[int, ...] = (4,), **layers: Any
) -> tuple[Record, dict[str, np.ndarray]]:
    """Trace a topology with the given layers whose `__call__` is `call(module, x)`.

    Returns the record, and the ids of every handle `call` stored on the module as
    an attribute whose name starts with `h_`.
    """

    class Net(pxl.Module):
        def setup(self) -> None:
            for name, layer in layers.items():
                setattr(self, name, layer)

        def __call__(self, x: Units) -> Any:
            return call(self, x)

    net = Net()
    record = pxl.trace(net, {"x": shape})
    kept = {k: v.ids for k, v in vars(net).items() if k.startswith("h_")}
    return record, kept


def pairs(record: Record, block: int = 0) -> set[tuple[int, int]]:
    b = record.blocks[block]
    return set(zip(b.src.tolist(), b.dst.tolist(), strict=True))


# --------------------------------------------------------------------------- #
# CreateUnits
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape", [(), (3,), (2, 5), (0, 4)])
def test_units_from_a_shape(shape: tuple[int, ...]) -> None:
    record, _ = traced(lambda m, x: m.units(shape=shape), units=CreateUnits())
    out = record.outputs[0]

    assert out.shape == shape
    assert record.num_units == 4 + int(np.prod(shape))
    assert np.unique(out).size == out.size
    assert np.intersect1d(out, record.inputs["x"]).size == 0
    assert record.blocks == []


def test_units_like_a_handle_take_its_shape_and_connect_nothing() -> None:
    record, _ = traced(lambda m, x: m.units(x), (2, 3), units=CreateUnits())
    out = record.outputs[0]

    assert out.shape == (2, 3)
    assert record.num_units == 6 + 6
    assert np.intersect1d(out, record.inputs["x"]).size == 0
    assert record.blocks == []


def test_a_call_writes_one_unit_block() -> None:
    record, _ = traced(lambda m, x: m.units(shape=(2, 3)), units=CreateUnits())
    (block,) = record.unit_blocks
    out = record.outputs[0]

    assert block.path == ("units",)
    assert block.shape == (2, 3)
    assert block.constant is None
    np.testing.assert_array_equal(
        out, np.arange(block.start, block.start + 6).reshape(2, 3)
    )


def test_input_units_are_not_in_a_unit_block() -> None:
    record, _ = traced(lambda m, x: x)

    assert record.unit_blocks == []
    assert record.num_units == 4


@pytest.mark.parametrize("value", [1.0, 0.5, -2, 0])
def test_constant_units_hold_the_value_given(value: float) -> None:
    record, _ = traced(lambda m, x: m.one(shape=(2,)), one=CreateUnits(constant=value))
    (block,) = record.unit_blocks

    assert block.constant == value
    assert isinstance(block.constant, float)
    assert record.constants.dtype == np.int32
    np.testing.assert_array_equal(np.sort(record.constants), np.sort(record.outputs[0]))


def test_constants_lists_only_the_constant_blocks() -> None:
    def call(m: Any, x: Units) -> Units:
        m.h_plain = m.plain(shape=(3,))
        m.h_one = m.one(shape=())
        m.h_half = m.half(shape=(2,))
        return x

    record, kept = traced(
        call,
        plain=CreateUnits(),
        one=CreateUnits(constant=1.0),
        half=CreateUnits(constant=0.5),
    )

    expected = np.concatenate([kept["h_one"].reshape(-1), kept["h_half"]])
    np.testing.assert_array_equal(np.sort(record.constants), np.sort(expected))
    assert [b.constant for b in record.unit_blocks] == [None, 1.0, 0.5]


# --------------------------------------------------------------------------- #
# CreateConnections
# --------------------------------------------------------------------------- #
def connected(
    arrange: Callable[[Units, Units], tuple[Units, Units]],
    x_shape: tuple[int, ...],
    y_shape: tuple[int, ...],
) -> tuple[Record, np.ndarray, np.ndarray, list[Any]]:
    """Connect input `x` to new units `y`, arranged by `arrange(x, y)`."""
    returned: list[Any] = []

    def call(m: Any, x: Units) -> Units:
        y = m.units(shape=y_shape)
        returned.append(m.conn(*arrange(x, y)))
        return y

    record, _ = traced(call, x_shape, units=CreateUnits(), conn=CreateConnections())
    return record, record.inputs["x"], record.outputs[0], returned


def test_connections_create_no_units_and_return_nothing() -> None:
    record, _, _, returned = connected(lambda x, y: (x, y), (4,), (4,))

    assert record.num_units == 4 + 4
    assert len(record.unit_blocks) == 1
    assert returned == [None]


def test_a_block_carries_its_path_and_initializer() -> None:
    init = ConstInit(0.25)

    def call(m: Any, x: Units) -> Units:
        m.skip(x, x[::-1])
        return x

    record, _ = traced(call, skip=CreateConnections(init=init))
    (block,) = record.blocks

    assert block.path == ("skip",)
    assert block.init is init
    assert block.src.dtype == block.dst.dtype == np.int32


def test_a_block_is_in_row_major_order_over_the_broadcast_shape() -> None:
    record, x, y, _ = connected(lambda x, y: (x[None, :], y[:, None]), (4,), (3,))
    (block,) = record.blocks

    assert block.shape == (3, 4)  # of neither handle: what the two broadcast to
    np.testing.assert_array_equal(
        block.src.reshape(block.shape), np.broadcast_to(x[None, :], (3, 4))
    )
    np.testing.assert_array_equal(
        block.dst.reshape(block.shape), np.broadcast_to(y[:, None], (3, 4))
    )


def test_all_to_all_along_the_last_axis_does_not_cross_positions() -> None:
    record, x, y, _ = connected(
        lambda x, y: (x[..., None, :], y[..., :, None]), (5, 4), (5, 3)
    )
    (block,) = record.blocks

    position = np.full(record.num_units, -1)
    position[x] = np.arange(5)[:, None]
    position[y] = np.arange(5)[:, None]

    assert block.src.size == 5 * 3 * 4
    np.testing.assert_array_equal(position[block.src], position[block.dst])


def test_several_calls_into_the_same_units_are_separate_blocks() -> None:
    def call(m: Any, x: Units) -> Units:
        y = m.units(x)
        m.first(x, y)
        m.second(x, y)
        return y

    record, _ = traced(
        call, units=CreateUnits(), first=CreateConnections(), second=CreateConnections()
    )

    assert [b.path for b in record.blocks] == [("first",), ("second",)]
    assert pairs(record, 0) == pairs(record, 1)  # the same pairs, twice


def test_a_unit_may_be_connected_to_itself() -> None:
    def call(m: Any, x: Units) -> Units:
        m.loop(x, x)
        return x

    record, _ = traced(call, loop=CreateConnections())

    np.testing.assert_array_equal(record.blocks[0].src, record.blocks[0].dst)


# --------------------------------------------------------------------------- #
# Explicit wiring: handles indexed with integer arrays
# --------------------------------------------------------------------------- #
def test_listed_pairs_are_the_only_connections_in_the_order_given() -> None:
    src, dst = np.array([0, 1, 2, 3, 3]), np.array([0, 0, 1, 1, 0])
    record, x, y, _ = connected(lambda x, y: (x[src], y[dst]), (4,), (2,))
    (block,) = record.blocks

    assert block.shape == (5,)
    np.testing.assert_array_equal(block.src, x[src])
    np.testing.assert_array_equal(block.dst, y[dst])


# --------------------------------------------------------------------------- #
# Errors, raised at the line that wrote them
# --------------------------------------------------------------------------- #
def test_units_need_a_handle_or_a_shape() -> None:
    with pytest.raises(TypeError, match="handle or with `shape=`"):
        traced(lambda m, x: m.units(), units=CreateUnits())


def test_units_do_not_take_both_a_handle_and_a_shape() -> None:
    with pytest.raises(TypeError, match="not both"):
        traced(lambda m, x: m.units(x, shape=(2,)), units=CreateUnits())


def test_a_shape_that_is_not_a_shape() -> None:
    with pytest.raises(ShapeError, match="non-negative"):
        traced(lambda m, x: m.units(shape=(2, -1)), units=CreateUnits())


# No message is matched below: the test run type-checks plastax's arguments
# (jaxtyping), and that check raises its own TypeError before the layer's.
@pytest.mark.parametrize("constant", ["1", True])
def test_a_constant_must_be_a_number(constant: Any) -> None:
    with pytest.raises(TypeError):
        CreateUnits(constant=constant)


def test_connections_need_two_handles() -> None:
    with pytest.raises(TypeError):
        traced(lambda m, x: m.conn(x, x.ids), conn=CreateConnections())


def test_handles_that_do_not_broadcast() -> None:
    with pytest.raises(ShapeError, match="do not broadcast"):
        connected(lambda x, y: (x, y), (4,), (3,))


def test_the_same_pair_twice_in_one_call() -> None:
    twice = np.array([0, 1, 0])
    with pytest.raises(ValueError, match="more than once"):
        connected(lambda x, y: (x[twice], y[twice]), (4,), (2,))


def test_a_refused_call_writes_no_block() -> None:
    seen: list[Record] = []

    def call(m: Any, x: Units) -> Units:
        seen.append(x._record)
        m.conn(x, x[:3])
        return x

    with pytest.raises(ShapeError):
        traced(call, conn=CreateConnections())

    assert seen[0].blocks == []


@pytest.mark.parametrize(
    "call",
    [
        lambda m, x: m.units(shape=(2,)) and m.units(shape=(2,)),
        lambda m, x: (m.conn(x, x), m.conn(x, x)),
    ],
    ids=["units", "connections"],
)
def test_a_primitive_called_twice_is_not_implemented(call: Any) -> None:
    with pytest.raises(NotImplementedError, match="second time"):
        traced(call, units=CreateUnits(), conn=CreateConnections())


def test_units_created_outside_any_trace() -> None:
    with pytest.raises(BuildError, match="not part of a topology being traced"):
        CreateUnits()(shape=(3,))


def test_a_primitive_that_is_not_in_the_topology() -> None:
    stray_units, stray_conn = CreateUnits(), CreateConnections()

    with pytest.raises(BuildError, match="not part of a topology being traced"):
        traced(lambda m, x: stray_units(shape=(3,)))
    with pytest.raises(BuildError, match="not part of the topology"):
        traced(lambda m, x: stray_conn(x, x))


def test_primitives_called_after_their_trace_returned() -> None:
    kept: dict[str, Any] = {}

    def call(m: Any, x: Units) -> Units:
        kept.update(module=m, x=x)
        return x

    traced(call, units=CreateUnits(), conn=CreateConnections())

    with pytest.raises(BuildError, match="not part of a topology being traced"):
        kept["module"].units(shape=(2,))
    with pytest.raises(BuildError, match="finished"):
        kept["module"].conn(kept["x"], kept["x"])


def test_handles_from_two_traces() -> None:
    other = pxl.trace(pxl.Dense(2, bias=False), {"x": (4,)})
    stale = Units(other.outputs[0], other)

    with pytest.raises(BuildError, match="different traces"):
        traced(lambda m, x: m.conn(x, stale[:1]), conn=CreateConnections())
    with pytest.raises(BuildError, match="another trace"):
        traced(lambda m, x: m.units(stale), units=CreateUnits())


def test_a_module_cannot_be_traced_while_it_is_being_traced() -> None:
    class Nested(pxl.Module):
        def __call__(self, x: Units) -> Units:
            pxl.trace(self, {"x": (2,)})
            return x

    with pytest.raises(BuildError, match="already being traced"):
        pxl.trace(Nested(), {"x": (2,)})


def test_a_module_can_be_traced_again_after_an_error() -> None:
    units = CreateUnits()
    with pytest.raises(TypeError):
        traced(lambda m, x: m.units(), units=units)

    class Net(pxl.Module):
        def setup(self) -> None:
            self.units = CreateUnits()

        def __call__(self, x: Units) -> Units:
            return self.units(x)

    net = Net()
    with pytest.raises(BuildError):
        pxl.trace(net, {"x": (2,), "extra": ()})
    record = pxl.trace(net, {"x": (2,)})

    assert record.num_units == 4
    assert net._record is None
