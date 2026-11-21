"""A network's behaviour declared by phase, each phase a pair of halves.

Plan `plans/20261121T123437Z-policy-sets.md` (workspace root). A pass is a
`Forward` or `Backward` pair: the `conn` half supplies `map`, the `unit` half
`combine` and `apply`, and the two are chosen independently. One object with
all three may stand for the pair. `update`, `prune` and `grow` are pairs whose
halves are optional, and an optimizer brings its own update and fields.
"""

from __future__ import annotations

import warnings
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import plastax as px
import plastax.init as init
import plastax.layers as pxl
from plastax.layers import Dense, Units

f32 = np.float32
KEY = jax.random.key(0)
W = np.array([[1.0, -2.0, 3.0], [0.5, 4.0, -1.0]], f32)


class WeightedSum(px.policies.ForwardConnPolicy):
    linear_input = px.ACTIVATION

    def map(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: Any,
    ) -> jax.Array:
        return c[px.WEIGHT, cid] * u[px.ACTIVATION, src]


class Reduced(px.policies.ForwardUnitPolicy):
    """A unit's activation is whatever `combine` leaves."""

    def __init__(self, combine: Any) -> None:
        self.combine = combine

    def apply(
        self, u: px.UnitView, i: px.UnitIdx, g: Any, acc: jax.Array
    ) -> px.UnitWrite:
        return px.UnitWrite.of((px.ACTIVATION, acc))


class OneLayer(pxl.Module):
    def setup(self) -> None:
        self.fc = Dense(2, bias=False, init=init.array(W))

    def __call__(self, x: Units) -> Units:
        return self.fc(x)


def outputs_of(net: type[px.Network[None]], x: np.ndarray) -> np.ndarray:
    inputs = px.StepInputs(inputs=jnp.asarray(x), targets=None)
    return np.asarray(net.init(KEY).step(inputs, outputs=True).outputs)


def forward_with(combine: Any) -> px.policies.Forward[None]:
    return px.policies.Forward(conn=WeightedSum(), unit=Reduced(combine))


def test_one_map_serves_two_unit_policies_that_reduce_differently() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        class SumNet(px.Network[None]):
            topology = OneLayer()
            input_shape = (3,)
            forward = forward_with(px.monoid.sum_)

        class MaxNet(px.Network[None]):
            topology = OneLayer()
            input_shape = (3,)
            forward = forward_with(px.monoid.max_)

    x = np.array([1.0, 2.0, 3.0], f32)

    np.testing.assert_allclose(outputs_of(SumNet, x), (W * x).sum(axis=1), rtol=1e-6)
    np.testing.assert_allclose(outputs_of(MaxNet, x), (W * x).max(axis=1), rtol=1e-6)


class Both(WeightedSum, px.policies.ForwardUnitPolicy):
    """One class with map, combine and apply."""

    combine = px.monoid.sum_

    def apply(
        self, u: px.UnitView, i: px.UnitIdx, g: Any, acc: jax.Array
    ) -> px.UnitWrite:
        return px.UnitWrite.of((px.ACTIVATION, acc))


def test_one_object_with_both_halves_is_read_as_the_pair() -> None:
    both = Both()

    class OneObject(px.Network[None]):
        topology = OneLayer()
        input_shape = (3,)
        forward = both

    x = np.array([1.0, 2.0, 3.0], f32)

    assert OneObject.forward == px.policies.Forward(conn=both, unit=both)
    np.testing.assert_allclose(outputs_of(OneObject, x), W @ x, rtol=1e-6)


def test_a_pass_that_is_only_one_half_is_refused() -> None:
    with pytest.raises(TypeError, match="forward must be a plastax.policies.Forward"):

        class OnlyAMap(px.Network[None]):
            forward = WeightedSum()


def test_a_unit_policy_in_the_conn_half_is_refused() -> None:
    with pytest.raises(TypeError, match=r"forward\.conn must satisfy"):

        class Swapped(px.Network[None]):
            forward = px.policies.Forward(
                conn=Reduced(px.monoid.sum_),  # type: ignore[arg-type]
                unit=Reduced(px.monoid.sum_),
            )


def test_linear_input_without_a_semiring_the_engine_has_warns() -> None:
    with pytest.warns(UserWarning, match=r"linear_input.*\(sum, prod\)"):

        class MaxNet(px.Network[None]):
            forward = forward_with(px.monoid.max_)


def test_linear_input_with_a_sum_does_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        class SumNet(px.Network[None]):
            forward = forward_with(px.monoid.sum_)


def test_a_pair_with_neither_half_is_refused() -> None:
    with pytest.raises(ValueError, match="neither a conn nor a unit"):
        px.policies.Prune()


# --------------------------------------------------------------------------- #
# The optimizer brings its update and its fields
# --------------------------------------------------------------------------- #
GRAD = px.FieldSpec("grad", np.float32, 0.0)
COUNT = px.FieldSpec("opt/count", np.float32, 0.0)
SEEN = px.FieldSpec("opt/seen", np.float32, 0.0)


class CountConn(px.policies.UpdateConnPolicy):
    """Adds one to the optimizer's connection column every step."""

    def incoming(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: Any,
    ) -> px.ConnWrite:
        return px.ConnWrite.of((COUNT, c[COUNT, cid] + 1.0))

    def outgoing(
        self,
        u: px.UnitView,
        src: px.UnitIdx,
        dst: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: Any,
    ) -> px.ConnWrite:
        return px.ConnWrite.of()


class CountUnit(px.policies.UpdateUnitPolicy):
    """Adds one to the optimizer's unit column every step."""

    def update(self, u: px.UnitView, i: px.UnitIdx, g: Any) -> px.UnitWrite:
        return px.UnitWrite.of((SEEN, u[SEEN, i] + 1.0))


class Counting:
    """An optimizer with state on connections and on units."""

    conn_fields = (COUNT,)
    unit_fields = (SEEN,)
    needs_step_counter = False

    def update(self) -> px.policies.Update[Any]:
        return px.policies.Update(conn=CountConn(), unit=CountUnit())


def test_an_optimizer_brings_its_fields_and_both_halves_of_its_update() -> None:
    class Counted(px.Network[None]):
        topology = OneLayer()
        input_shape = (3,)
        forward = forward_with(px.monoid.sum_)
        optimizer = Counting()
        extra_unit_fields = (GRAD,)

    driver = Counted.init(KEY)
    step = px.StepInputs(inputs=jnp.ones(3, jnp.float32), targets=None)
    driver.step(step)
    driver.step(step)

    assert set(driver.state.units) >= {"grad", "opt/seen"}
    np.testing.assert_array_equal(np.asarray(driver.state.units["opt/seen"]), 2.0)
    counts = np.asarray(driver.state.conns[0]["opt/count"])
    dead = np.asarray(driver.state.conns[0]["dead"])
    np.testing.assert_array_equal(counts[~dead], 2.0)


def test_update_fills_the_half_the_optimizer_leaves() -> None:
    class Mixed(px.Network[None]):
        topology = OneLayer()
        input_shape = (3,)
        forward = forward_with(px.monoid.sum_)
        optimizer = px.optim.sgd(0.1, GRAD)
        update = px.policies.Update(unit=CountUnit())
        extra_unit_fields = (GRAD, SEEN)

    resolved = Mixed.resolved()

    assert isinstance(resolved.update_unit, CountUnit)
    assert resolved.update_conn is not None


def test_a_half_given_by_both_the_optimizer_and_update_is_refused() -> None:
    with pytest.raises(TypeError, match="unit update is given by both"):

        class Twice(px.Network[None]):
            forward = forward_with(px.monoid.sum_)
            optimizer = Counting()
            update = px.policies.Update(unit=CountUnit())


def test_a_field_named_by_the_network_and_the_optimizer_is_refused() -> None:
    with pytest.raises(ValueError, match="duplicate"):

        class Clash(px.Network[None]):
            forward = forward_with(px.monoid.sum_)
            optimizer = Counting()
            extra_unit_fields = (SEEN,)
