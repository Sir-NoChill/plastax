"""`Dense` with private weights, checked on the record it writes.

Plan `plans/20261119T152006Z-step4-dense-draft.md` (workspace root). No test here
asserts a literal unit id: the order in which a call numbers its units is not
specified. Ids are compared with the ids of the handles involved.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import plastax.layers as pxl
from plastax.layers import BuildError, ConstInit, Dense, Record, ShapeError, Units


def net(fc: Dense, call: Callable[[Any, Units], Any] | None = None) -> pxl.Module:
    """A topology with one layer, `fc`, whose `__call__` is `call(module, x)`."""

    class Net(pxl.Module):
        def setup(self) -> None:
            self.fc = fc

        def __call__(self, x: Units) -> Any:
            return self.fc(x) if call is None else call(self, x)

    return Net()


def traced(fc: Dense, shape: tuple[int, ...]) -> tuple[Record, np.ndarray, np.ndarray]:
    """Trace `fc` over one input; return the record, the input ids, the output ids."""
    record = pxl.trace(net(fc), {"x": shape})
    return record, record.inputs["x"], record.outputs[0]


def raised_in_this_file(excinfo: pytest.ExceptionInfo[BaseException]) -> bool:
    return any(str(entry.path) == __file__ for entry in excinfo.traceback)


# --------------------------------------------------------------------------- #
# Shapes and counts
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("shape", "features", "expected"),
    [((4,), 3, (3,)), ((5, 4), 3, (5, 3)), ((2, 3, 4), 5, (2, 3, 5))],
)
def test_dense_acts_on_last_axis(
    shape: tuple[int, ...], features: int, expected: tuple[int, ...]
) -> None:
    _, _, out = traced(Dense(features), shape)

    assert out.shape == expected


def test_without_bias_there_is_one_block_and_no_constant() -> None:
    record, x, out = traced(Dense(3, bias=False), (4,))

    assert record.num_units == 4 + 3
    assert [block.name for block in record.blocks] == ["weight"]
    assert record.blocks[0].src.shape == (12,)
    assert record.blocks[0].dst.shape == (12,)
    assert record.constants.size == 0
    assert np.intersect1d(x, out).size == 0  # the layer's units are new


def test_bias_is_one_constant_unit_and_one_connection_per_unit() -> None:
    record, x, out = traced(Dense(3, bias=True), (4,))
    weight, bias = record.blocks

    assert record.num_units == 4 + 3 + 1
    assert (weight.name, bias.name) == ("weight", "bias")
    assert weight.src.size + bias.src.size == 3 * 4 + 3
    assert record.constants.shape == (1,)
    one = record.constants[0]
    assert one not in x and one not in out
    np.testing.assert_array_equal(bias.src, np.full(3, one))
    np.testing.assert_array_equal(bias.dst, out)


def test_bias_on_leading_axes_is_one_constant_and_a_connection_per_unit() -> None:
    record, _, out = traced(Dense(3), (5, 4))
    bias = record.blocks[1]

    assert record.constants.shape == (1,)
    np.testing.assert_array_equal(bias.src, np.full(15, record.constants[0]))
    np.testing.assert_array_equal(bias.dst, out.reshape(-1))


# --------------------------------------------------------------------------- #
# Which units are connected, and in what order
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape", [(4,), (5, 4), (2, 3, 4)])
def test_weight_block_is_row_major_over_position_destination_source(
    shape: tuple[int, ...],
) -> None:
    """A weight array of shape `lead + (features, n_in)`, flattened, is in block
    order.
    """
    features = 3
    record, x, out = traced(Dense(features, bias=False), shape)
    block = record.blocks[0]
    weights = (*shape[:-1], features, shape[-1])

    np.testing.assert_array_equal(
        block.src.reshape(weights), np.broadcast_to(x[..., None, :], weights)
    )
    np.testing.assert_array_equal(
        block.dst.reshape(weights), np.broadcast_to(out[..., :, None], weights)
    )


def test_every_source_reaches_every_new_unit_exactly_once() -> None:
    record, x, out = traced(Dense(3, bias=False), (4,))
    block = record.blocks[0]

    pairs = set(zip(block.src.tolist(), block.dst.tolist(), strict=True))
    assert len(pairs) == block.src.size == 12
    assert pairs == {(s, d) for s in x.tolist() for d in out.tolist()}


def test_no_connection_crosses_positions() -> None:
    record, x, out = traced(Dense(3, bias=False), (5, 4))
    block = record.blocks[0]

    position = np.full(record.num_units, -1)
    position[x] = np.arange(5)[:, None]
    position[out] = np.arange(5)[:, None]

    np.testing.assert_array_equal(position[block.src], position[block.dst])


def test_a_leading_axis_of_length_zero_creates_nothing() -> None:
    record, _, out = traced(Dense(3), (0, 4))

    assert out.shape == (0, 3)
    assert record.blocks[0].src.size == 0
    assert record.blocks[1].src.size == 0


# --------------------------------------------------------------------------- #
# What a block carries
# --------------------------------------------------------------------------- #
def test_blocks_carry_the_layer_path_and_the_initializers_given() -> None:
    init, bias_init = ConstInit(0.25), ConstInit(-1.0)
    record, _, _ = traced(Dense(3, init=init, bias_init=bias_init), (4,))
    weight, bias = record.blocks

    assert weight.path == bias.path == ("fc",)
    assert weight.init is init
    assert bias.init is bias_init
    assert init.value == 0.25


def test_initializers_default_to_none() -> None:
    record, _, _ = traced(Dense(3), (4,))

    assert [block.init for block in record.blocks] == [None, None]


def test_block_ids_are_int32() -> None:
    record, _, _ = traced(Dense(3), (4,))

    for block in record.blocks:
        assert block.src.dtype == np.int32
        assert block.dst.dtype == np.int32
    assert record.constants.dtype == np.int32


def test_a_dense_layer_can_be_the_whole_topology() -> None:
    record = pxl.trace(Dense(3, bias=False), {"x": (4,)})

    assert record.blocks[0].path == ()
    assert record.outputs[0].shape == (3,)


# --------------------------------------------------------------------------- #
# More than one layer
# --------------------------------------------------------------------------- #
class TwoLayers(pxl.Module):
    def setup(self) -> None:
        self.hidden = Dense(8)
        self.out = Dense(2)

    def __call__(self, x: Units) -> Units:
        self.h = self.hidden(x)
        return self.out(self.h)


def test_the_second_layer_reads_the_first_layers_units() -> None:
    module = TwoLayers()
    record = pxl.trace(module, {"x": (4,)})
    hidden_ids = module.h.ids
    by_key = {(block.path, block.name): block for block in record.blocks}

    assert set(by_key) == {
        (("hidden",), "weight"),
        (("hidden",), "bias"),
        (("out",), "weight"),
        (("out",), "bias"),
    }
    assert set(by_key[("hidden",), "weight"].dst.tolist()) == set(hidden_ids.tolist())
    assert set(by_key[("out",), "weight"].src.tolist()) == set(hidden_ids.tolist())
    assert record.num_units == 4 + 8 + 1 + 2 + 1


def test_each_biased_layer_has_its_own_constant() -> None:
    record = pxl.trace(TwoLayers(), {"x": (4,)})
    by_key = {(block.path, block.name): block for block in record.blocks}

    hidden_one = by_key[("hidden",), "bias"].src[0]
    out_one = by_key[("out",), "bias"].src[0]
    assert hidden_one != out_one
    assert set(record.constants.tolist()) == {hidden_one, out_one}


def test_tracing_twice_gives_the_same_blocks() -> None:
    module = TwoLayers()
    first = pxl.trace(module, {"x": (3, 4)})
    second = pxl.trace(module, {"x": (3, 4)})

    assert len(first.blocks) == len(second.blocks) == 4
    for a, b in zip(first.blocks, second.blocks, strict=True):
        assert (a.path, a.name) == (b.path, b.name)
        np.testing.assert_array_equal(a.src, b.src)
        np.testing.assert_array_equal(a.dst, b.dst)
    np.testing.assert_array_equal(first.constants, second.constants)


# --------------------------------------------------------------------------- #
# Errors, raised at the line that wrote them
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("features", [0, -1])
def test_features_must_be_at_least_one(features: int) -> None:
    with pytest.raises(ValueError, match="at least 1") as excinfo:
        Dense(features)
    assert raised_in_this_file(excinfo)


# No message is matched in the two tests below: the test run type-checks plastax's
# arguments (jaxtyping), and that check raises its own TypeError before the layer's.
@pytest.mark.parametrize("features", [2.5, True, "3"])
def test_features_must_be_an_integer(features: Any) -> None:
    with pytest.raises(TypeError):
        Dense(features)


def test_a_handle_with_no_axes() -> None:
    with pytest.raises(ShapeError, match="last axis") as excinfo:
        pxl.trace(net(Dense(3)), {"x": ()})
    assert raised_in_this_file(excinfo)


def test_a_last_axis_of_length_zero() -> None:
    with pytest.raises(ShapeError, match="at least one source"):
        pxl.trace(net(Dense(3)), {"x": (4, 0)})


def test_called_with_something_that_is_not_a_handle() -> None:
    with pytest.raises(TypeError):
        pxl.trace(net(Dense(3), lambda m, x: m.fc(x.ids)), {"x": (4,)})


def test_a_layer_called_twice_is_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="Dense at fc .* second time"):
        pxl.trace(net(Dense(4), lambda m, x: m.fc(m.fc(x))), {"x": (4,)})


def test_a_layer_called_twice_writes_nothing_the_second_time() -> None:
    seen: list[Record] = []

    def call(m: Any, x: Units) -> Units:
        seen.append(x._record)
        return m.fc(m.fc(x))

    with pytest.raises(NotImplementedError):
        pxl.trace(net(Dense(4, bias=False), call), {"x": (4,)})

    assert seen[0].num_units == 4 + 4
    assert len(seen[0].blocks) == 1


def test_a_layer_that_is_not_in_the_topology() -> None:
    stray = Dense(2)

    with pytest.raises(BuildError, match="not part of the topology"):
        pxl.trace(net(Dense(3), lambda m, x: stray(m.fc(x))), {"x": (4,)})


def test_a_layer_called_after_its_trace_returned() -> None:
    module = TwoLayers()
    pxl.trace(module, {"x": (4,)})

    with pytest.raises(BuildError, match="finished"):
        module.out(module.h)


def test_a_module_can_be_traced_again_after_an_error() -> None:
    fc = Dense(4, bias=False)
    with pytest.raises(NotImplementedError):
        pxl.trace(net(fc, lambda m, x: m.fc(m.fc(x))), {"x": (4,)})

    record = pxl.trace(net(fc), {"x": (4,)})

    assert len(record.blocks) == 1
