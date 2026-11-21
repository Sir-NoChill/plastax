"""The two policy sets on a network: `conn_policy` and `unit_policy`.

Plan `plans/20261121T123437Z-policy-sets.md` (workspace root). A pass is cut in
two: the connection set supplies `map`, the unit set `combine` and `apply`, and
the two are chosen independently.
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


class GradBackwardConn(px.policies.BackwardConnPolicy):
    def map(
        self,
        u: px.UnitView,
        src: px.UnitIdx,
        dst: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: Any,
    ) -> jax.Array:
        return c[px.WEIGHT, cid] * u[px.ACTIVATION, dst]


class OneLayer(pxl.Module):
    def setup(self) -> None:
        self.fc = Dense(2, bias=False, init=init.array(W))

    def __call__(self, x: Units) -> Units:
        return self.fc(x)


def outputs_of(net: type[px.Network[None]], x: np.ndarray) -> np.ndarray:
    inputs = px.StepInputs(inputs=jnp.asarray(x), targets=None)
    return np.asarray(net.init(KEY).step(inputs, outputs=True).outputs)


def test_one_map_serves_two_unit_policies_that_reduce_differently() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        class SumNet(px.Network[None]):
            topology = OneLayer()
            input_shape = (3,)
            conn_policy = px.policies.Conn(forward=WeightedSum())
            unit_policy = px.policies.Unit(forward=Reduced(px.monoid.sum_))

        class MaxNet(px.Network[None]):
            topology = OneLayer()
            input_shape = (3,)
            conn_policy = px.policies.Conn(forward=WeightedSum())
            unit_policy = px.policies.Unit(forward=Reduced(px.monoid.max_))

    x = np.array([1.0, 2.0, 3.0], f32)

    np.testing.assert_allclose(outputs_of(SumNet, x), (W * x).sum(axis=1), rtol=1e-6)
    np.testing.assert_allclose(outputs_of(MaxNet, x), (W * x).max(axis=1), rtol=1e-6)


def test_linear_input_without_a_semiring_the_engine_has_warns() -> None:
    with pytest.warns(UserWarning, match=r"linear_input.*\(\+, \*\)"):

        class MaxNet(px.Network[None]):
            conn_policy = px.policies.Conn(forward=WeightedSum())
            unit_policy = px.policies.Unit(forward=Reduced(px.monoid.max_))


def test_linear_input_with_a_sum_does_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        class SumNet(px.Network[None]):
            conn_policy = px.policies.Conn(forward=WeightedSum())
            unit_policy = px.policies.Unit(forward=Reduced(px.monoid.sum_))


def test_a_network_without_both_sets_is_refused() -> None:
    with pytest.raises(TypeError, match="unit_policy is required"):

        class NoUnits(px.Network[None]):
            conn_policy = px.policies.Conn(forward=WeightedSum())


def test_a_backward_half_on_one_set_only_is_refused() -> None:
    with pytest.raises(TypeError, match=r"conn_policy\.backward is set"):

        class HalfBackward(px.Network[None]):
            conn_policy = px.policies.Conn(
                forward=WeightedSum(), backward=GradBackwardConn()
            )
            unit_policy = px.policies.Unit(forward=Reduced(px.monoid.sum_))


def test_a_unit_policy_in_a_connection_slot_is_refused() -> None:
    with pytest.raises(TypeError, match=r"conn_policy\.forward must satisfy"):

        class Swapped(px.Network[None]):
            conn_policy = px.policies.Conn(forward=Reduced(px.monoid.sum_))  # type: ignore[arg-type]
            unit_policy = px.policies.Unit(forward=Reduced(px.monoid.sum_))
