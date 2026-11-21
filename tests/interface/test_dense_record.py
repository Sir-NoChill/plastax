"""`Dense` with private weights, checked on the record it writes.

Plan `plans/20261119T171340Z-steps-4-5-primitives.md` (workspace root). `Dense` is a
helper written with `CreateUnits` and `CreateConnections`, which have their own tests
in test_primitives_record.py. No test here asserts a literal unit id: the order in
which a call numbers its units is not specified. Ids are compared with the ids of the
handles involved.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import plastax.layers as pxl
from plastax.layers import (
    BuildError,
    Dense,
    Record,
    ShapeError,
    Units,
)


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


# --------------------------------------------------------------------------- #
# Shapes and counts
# --------------------------------------------------------------------------- #


def test_without_bias_there_is_one_block_and_no_constant() -> None:
    record, x, out = traced(Dense(3, bias=False), (4,))

    assert record.num_units == 4 + 3
    assert [block.path for block in record.blocks] == [("fc", "weight")]
    assert record.blocks[0].src.shape == (12,)
    assert record.blocks[0].dst.shape == (12,)
    assert record.constants.size == 0
    assert np.intersect1d(x, out).size == 0  # the layer's units are new


def test_bias_on_leading_axes_is_one_constant_and_a_connection_per_unit() -> None:
    record, _, out = traced(Dense(3), (5, 4))
    bias = record.blocks[1]

    assert record.constants.shape == (1,)
    assert bias.shape == (5, 3)
    np.testing.assert_array_equal(bias.src, np.full(15, record.constants[0]))
    np.testing.assert_array_equal(bias.dst, out.reshape(-1))


# --------------------------------------------------------------------------- #
# Which units are connected, and in what order
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape", [(4,), (2, 3, 4)])
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

    assert block.shape == weights
    np.testing.assert_array_equal(
        block.src.reshape(block.shape), np.broadcast_to(x[..., None, :], weights)
    )
    np.testing.assert_array_equal(
        block.dst.reshape(block.shape), np.broadcast_to(out[..., :, None], weights)
    )


def test_a_leading_axis_of_length_zero_creates_nothing() -> None:
    record, _, out = traced(Dense(3), (0, 4))

    assert out.shape == (0, 3)
    assert record.blocks[0].src.size == 0
    assert record.blocks[1].src.size == 0


# --------------------------------------------------------------------------- #
# What a block carries
# --------------------------------------------------------------------------- #


def test_a_dense_layer_can_be_the_whole_topology() -> None:
    record = pxl.trace(Dense(3, bias=False), {"x": (4,)})

    assert record.blocks[0].path == ("weight",)
    assert record.outputs[0].shape == (3,)


def test_dense_writes_its_units_as_unit_blocks() -> None:
    record, _, out = traced(Dense(3), (5, 4))
    units, one = record.unit_blocks

    assert (units.path, units.shape, units.constant) == (("fc", "units"), (5, 3), None)
    assert (one.path, one.shape, one.constant) == (("fc", "one"), (), 1.0)


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
    by_path = {block.path: block for block in record.blocks}

    assert set(by_path) == {
        ("hidden", "weight"),
        ("hidden", "bias_weight"),
        ("out", "weight"),
        ("out", "bias_weight"),
    }
    assert set(by_path["hidden", "weight"].dst.tolist()) == set(hidden_ids.tolist())
    assert set(by_path["out", "weight"].src.tolist()) == set(hidden_ids.tolist())
    assert record.num_units == 4 + 8 + 1 + 2 + 1


def test_each_biased_layer_has_its_own_constant() -> None:
    record = pxl.trace(TwoLayers(), {"x": (4,)})
    by_path = {block.path: block for block in record.blocks}

    hidden_one = by_path["hidden", "bias_weight"].src[0]
    out_one = by_path["out", "bias_weight"].src[0]
    assert hidden_one != out_one
    assert set(record.constants.tolist()) == {hidden_one, out_one}


def test_tracing_twice_gives_the_same_blocks() -> None:
    module = TwoLayers()
    first = pxl.trace(module, {"x": (3, 4)})
    second = pxl.trace(module, {"x": (3, 4)})

    assert len(first.blocks) == len(second.blocks) == 4
    for a, b in zip(first.blocks, second.blocks, strict=True):
        assert a.path == b.path
        np.testing.assert_array_equal(a.src, b.src)
        np.testing.assert_array_equal(a.dst, b.dst)
    np.testing.assert_array_equal(first.constants, second.constants)


# --------------------------------------------------------------------------- #
# Errors, raised at the line that wrote them
# --------------------------------------------------------------------------- #
def test_features_must_be_at_least_one() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        Dense(0)


# No message is matched in the two tests below: the test run type-checks plastax's
# arguments (jaxtyping), and that check raises its own TypeError before the layer's.
def test_features_must_be_an_integer() -> None:
    with pytest.raises(TypeError):
        Dense(True)


def test_a_handle_with_no_axes() -> None:
    with pytest.raises(ShapeError, match="last axis"):
        pxl.trace(net(Dense(3)), {"x": ()})


def test_a_last_axis_of_length_zero() -> None:
    with pytest.raises(ShapeError, match="at least one source"):
        pxl.trace(net(Dense(3)), {"x": (4, 0)})


def test_a_layer_called_twice_is_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="Dense at fc .* second time"):
        pxl.trace(net(Dense(4), lambda m, x: m.fc(m.fc(x))), {"x": (4,)})


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
