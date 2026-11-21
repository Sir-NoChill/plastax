"""The notation a topology is written in, checked without building a network.

Plan `plans/20261116T132654Z-layers-module.md` (workspace root). So far: handles
(step 2), modules and
tracing (step 3). Layers are in test_primitives_record.py and test_dense_record.py.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import plastax.layers as pxl
from plastax.layers import BuildError, Record, ShapeError, Units


def handle(*shape: int) -> Units:
    """A handle to `prod(shape)` units numbered from 0, in row-major order."""
    return Units(
        np.arange(int(np.prod(shape)), dtype=np.int32).reshape(shape), Record()
    )


# --------------------------------------------------------------------------- #
# What a handle reports
# --------------------------------------------------------------------------- #


def test_ids_are_int32_and_cannot_be_written() -> None:
    h = handle(2, 3)

    assert h.ids.dtype == np.int32
    with pytest.raises(ValueError, match="read-only"):
        h.ids[0, 0] = 9


def test_a_handle_does_not_follow_the_array_it_was_made_from() -> None:
    ids = np.arange(4, dtype=np.int32)
    h = Units(ids, Record())

    ids[0] = 99

    assert h.ids[0] == 0


def test_ids_must_be_integers() -> None:
    with pytest.raises(TypeError, match="integers"):
        Units(np.zeros(3, np.float32), Record())


# --------------------------------------------------------------------------- #
# Relabelling: the same units, rearranged
# --------------------------------------------------------------------------- #
def test_reshape_takes_a_shape_or_its_axes() -> None:
    h = handle(2, 3)

    np.testing.assert_array_equal(h.reshape((3, 2)).ids, h.reshape(3, 2).ids)


@pytest.mark.parametrize(
    ("shape", "index", "unit"),
    [((2, 3), (1, 2), 5)],
)
def test_integers_for_every_axis_select_a_single_unit(
    shape: tuple[int, ...], index: Any, unit: int
) -> None:
    """The result is a handle with no axes, not a bare id."""
    one = handle(*shape)[index]

    assert isinstance(one, Units)
    assert one.shape == ()
    assert one.ids == unit


def test_iterating_a_handle_with_one_axis_gives_its_units_one_by_one() -> None:
    units = list(handle(3))

    assert [u.shape for u in units] == [(), (), ()]
    assert [int(u.ids) for u in units] == [0, 1, 2]


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
def test_reshape_to_a_different_number_of_units() -> None:
    with pytest.raises(ShapeError, match="reshape"):
        handle(2, 3).reshape(4, 2)


def test_an_index_that_does_not_fit() -> None:
    with pytest.raises(ShapeError):
        handle(2, 3)[5]


@pytest.mark.parametrize(
    "index",
    [True, np.array([True, False]), "0"],
    ids=["bool", "mask", "string"],
)
def test_an_index_that_is_not_integers(index: Any) -> None:
    with pytest.raises(TypeError):
        handle(2, 3)[index]


def test_a_handle_is_not_an_array() -> None:
    with pytest.raises(TypeError, match="holds no values"):
        np.asarray(handle(2, 3))


def test_a_handle_with_no_axes_cannot_be_iterated() -> None:
    with pytest.raises(TypeError):
        list(handle())


# --------------------------------------------------------------------------- #
# Tracing a module with no layers
# --------------------------------------------------------------------------- #
class Passthrough(pxl.Module):
    def __call__(self, x: Units) -> Units:
        return x.flatten()


class TwoInputs(pxl.Module):
    def __call__(self, obs: Units, gamma: Units) -> tuple[Units, Units]:
        return gamma, obs


def test_a_module_that_returns_its_input() -> None:
    record = pxl.trace(Passthrough(), {"x": (2, 3)})

    assert record.num_units == 6
    np.testing.assert_array_equal(record.inputs["x"], [[0, 1, 2], [3, 4, 5]])
    assert record.blocks == []
    assert record.constants.size == 0
    assert len(record.outputs) == 1
    np.testing.assert_array_equal(record.outputs[0], np.arange(6))


def test_inputs_are_numbered_in_the_order_given() -> None:
    record = pxl.trace(TwoInputs(), {"gamma": (), "obs": (3,)})

    np.testing.assert_array_equal(record.inputs["gamma"], 0)
    np.testing.assert_array_equal(record.inputs["obs"], [1, 2, 3])


def test_inputs_reach_arguments_by_name_and_outputs_keep_their_order() -> None:
    record = pxl.trace(TwoInputs(), {"gamma": (), "obs": (3,)})

    assert [out.shape for out in record.outputs] == [(), (3,)]
    np.testing.assert_array_equal(record.outputs[1], record.inputs["obs"])


# --------------------------------------------------------------------------- #
# setup, arguments and paths
# --------------------------------------------------------------------------- #
class Inner(pxl.Module):
    def __call__(self, x: Units) -> Units:
        return x


class Middle(pxl.Module):
    def setup(self) -> None:
        self.inner = Inner()

    def __call__(self, x: Units) -> Units:
        return self.inner(x)


class Outer(pxl.Module):
    def __init__(self, depth: int) -> None:
        self.depth = depth
        self.setups = 0

    def setup(self) -> None:
        self.setups += 1
        self.first = Middle()
        self.rest = [Middle() for _ in range(self.depth)]

    def __call__(self, x: Units) -> Units:
        x = self.first(x)
        for module in self.rest:
            x = module(x)
        return x


def test_setup_runs_once_however_often_the_module_is_traced() -> None:
    module = Outer(depth=2)
    assert module.setups == 0

    pxl.trace(module, {"x": (3,)})
    pxl.trace(module, {"x": (3,)})

    assert module.setups == 1


def test_paths_follow_attribute_names_and_list_indices() -> None:
    module = Outer(depth=2)
    pxl.trace(module, {"x": (3,)})

    assert module.path == ()
    assert module.first.path == ("first",)
    assert module.first.inner.path == ("first", "inner")
    assert module.rest[1].inner.path == ("rest", "1", "inner")


def test_a_module_has_no_path_before_it_is_traced() -> None:
    with pytest.raises(BuildError, match="no path"):
        _ = Passthrough().path


def test_one_module_declared_in_two_places() -> None:
    class Twice(pxl.Module):
        def setup(self) -> None:
            self.a = Inner()
            self.b = self.a

        def __call__(self, x: Units) -> Units:
            return self.b(self.a(x))

    with pytest.raises(BuildError, match="declared at"):
        pxl.trace(Twice(), {"x": (3,)})


# --------------------------------------------------------------------------- #
# Errors raised by a trace
# --------------------------------------------------------------------------- #
def test_an_argument_with_no_input() -> None:
    with pytest.raises(BuildError, match=r"arguments \['gamma'\] have no input"):
        pxl.trace(TwoInputs(), {"obs": (3,)})


def test_an_input_with_no_argument() -> None:
    with pytest.raises(BuildError, match=r"inputs \['extra'\] have no argument"):
        pxl.trace(Passthrough(), {"x": (3,), "extra": ()})


def test_call_must_name_its_inputs() -> None:
    class Starred(pxl.Module):
        def __call__(self, *handles: Units) -> Units:
            return handles[0]

    with pytest.raises(BuildError, match="name each input"):
        pxl.trace(Starred(), {"x": (3,)})


def test_a_module_without_call() -> None:
    class Empty(pxl.Module):
        pass

    with pytest.raises(BuildError, match="no __call__"):
        pxl.trace(Empty(), {})


def test_returning_something_that_is_not_a_handle() -> None:
    class ReturnsIds(pxl.Module):
        def __call__(self, x: Units) -> Any:
            return x.ids

    with pytest.raises(BuildError, match="must return a handle"):
        pxl.trace(ReturnsIds(), {"x": (3,)})


def test_returning_a_handle_from_another_trace() -> None:
    class Leaks(pxl.Module):
        def __call__(self, x: Units) -> Units:
            return handle(3)

    with pytest.raises(BuildError, match="another trace"):
        pxl.trace(Leaks(), {"x": (3,)})


def test_an_input_shape_that_is_not_a_shape() -> None:
    with pytest.raises(ShapeError, match="non-negative"):
        pxl.trace(Passthrough(), {"x": (2, -1)})
