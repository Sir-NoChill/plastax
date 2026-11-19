"""Initial values of weights and fields.

Design document section 2.8. The topology sets a connection's weight through the layer's
initializer. A connection policy set's `init` fills in the other fields and may be a
function of the globals and the drawn weight. Values of connections created at run time
by a growth policy are in test_connection_growth.py.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, Float

import plastax as px

from . import support

Scalar = Float[Array, ""]
f32 = np.float32

W = (np.arange(12, dtype=f32).reshape(3, 4) - 5.0) / 10.0

# Fields with no policy behind them, set only by `init`.
SIGN = px.ConnField[f32]("sign", default=0.0)
LOG_STEP = px.ConnField[f32]("log_step", default=0.0)
TAG = px.ConnField[np.int32]("tag", default=0)
BIAS_TOWARD = px.UnitField[f32]("bias_toward", default=0.0)


@dataclasses.dataclass(frozen=True)
class Params:
    alpha_init: float = 0.01


@px.globals
class Globals:
    hp: Params


def conn_set(**more: Any) -> Any:
    p = px.policies
    return p.Conn(
        forward=p.WeightedSum(),
        backward=p.WeightedSumBackward(),
        update=None,
        prune=None,
        add=None,
        **more,
    )


def built(fc: Any, input_shape: tuple[int, ...] = (4,), **attrs: Any) -> support.Runner:
    model = support.topology(lambda m, x: m.fc(x), fc=fc)
    globals_ = attrs.pop("globals", None)
    return support.Runner(
        support.network(model, input_shape=input_shape, **attrs), globals=globals_
    )


# --------------------------------------------------------------------------- #
# The layer's initializer sets the weight
# --------------------------------------------------------------------------- #
def test_constant_initializer() -> None:
    fc = px.layers.Dense(3, bias=False, init=px.init.constant(0.25))

    support.close(built(fc).weights(fc), np.full((3, 4), 0.25, f32))


def test_array_initializer() -> None:
    fc = px.layers.Dense(3, bias=False, init=px.init.array(W))

    support.close(built(fc).weights(fc), W)


def test_zeros_initializer() -> None:
    fc = px.layers.Dense(3, bias=False, init=px.init.zeros)

    support.close(built(fc).weights(fc), np.zeros((3, 4), f32))


def test_lecun_uniform_is_bounded_by_the_fan_in() -> None:
    fan_in = 100
    fc = px.layers.Dense(50, bias=False, init=px.init.lecun_uniform)
    w = built(fc, input_shape=(fan_in,)).weights(fc)

    bound = np.sqrt(3.0 / fan_in)
    assert np.abs(w).max() <= bound
    assert np.abs(w).max() > 0.9 * bound
    assert abs(w.mean()) < 0.1 * bound
    support.close(w.var(), 1.0 / fan_in, rtol=0.1)


def test_initializer_sees_the_fan_in_of_a_sparse_layer() -> None:
    """With 4 inputs per unit out of 400, the bound follows 4, not 400."""
    fc = px.layers.Dense(
        50,
        connectivity=px.layers.sparse(fan_in=4),
        bias=False,
        init=px.init.lecun_uniform,
    )
    r = built(fc, input_shape=(400,))
    w = r.conn_field(fc, px.WEIGHT)

    assert np.abs(w).max() <= np.sqrt(3.0 / 4)
    assert np.abs(w).max() > np.sqrt(3.0 / 400)


def test_initial_weights_replay_for_the_same_seed() -> None:
    def weights(seed: int) -> np.ndarray:
        fc = px.layers.Dense(3, bias=False, init=px.init.lecun_uniform)
        model = support.topology(lambda m, x: m.fc(x), fc=fc)
        r = support.Runner(support.network(model, input_shape=(4,)), seed=seed)
        return r.weights(fc)

    np.testing.assert_array_equal(weights(1), weights(1))
    assert not np.array_equal(weights(1), weights(2))


@pytest.mark.open_question(
    "Plastax ships a default initializer per layer, in the manner of PyTorch and JAX; "
    "which one is not chosen. This only assumes it scales down with the fan-in."
)
def test_default_initializer_scales_with_the_fan_in() -> None:
    def spread(fan_in: int) -> float:
        fc = px.layers.Dense(64, bias=False)
        return float(built(fc, input_shape=(fan_in,)).weights(fc).std())

    narrow, wide = spread(4), spread(400)

    assert 0.0 < wide < narrow
    assert np.isfinite(narrow)


# --------------------------------------------------------------------------- #
# A policy set's `init` fills in the other fields
# --------------------------------------------------------------------------- #
def test_policy_set_init_gives_fields_fixed_values() -> None:
    fc = px.layers.Dense(
        3,
        bias=False,
        init=px.init.array(W),
        conn_policy=conn_set(init={TAG: 7, LOG_STEP: -2.0}),
    )
    r = built(fc)

    np.testing.assert_array_equal(r.conn_field(fc, TAG), 7)
    support.close(r.conn_field(fc, LOG_STEP), np.full(12, -2.0, f32))
    support.close(r.weights(fc), W)  # the weight still comes from the layer


def test_policy_set_init_may_depend_on_the_drawn_weight() -> None:
    def init(g: px.Globals, w: Scalar) -> dict[Any, Scalar]:
        return {SIGN: jnp.sign(w)}

    fc = px.layers.Dense(
        3, bias=False, init=px.init.array(W), conn_policy=conn_set(init=init)
    )
    r = built(fc)

    np.testing.assert_array_equal(
        np.sort(r.conn_field(fc, SIGN)), np.sort(np.sign(W).reshape(-1))
    )
    weights, signs = r.conn_field(fc, px.WEIGHT), r.conn_field(fc, SIGN)
    np.testing.assert_array_equal(signs, np.sign(weights))


def test_policy_set_init_may_depend_on_a_hyperparameter() -> None:
    def init(g: Globals, w: Scalar) -> dict[Any, Scalar]:
        return {LOG_STEP: jnp.log(g.hp.alpha_init)}

    fc = px.layers.Dense(
        3, bias=False, init=px.init.zeros, conn_policy=conn_set(init=init)
    )
    r = built(fc, globals=Globals(hp=Params(alpha_init=0.05)))

    support.close(r.conn_field(fc, LOG_STEP), np.full(12, np.log(0.05), f32))


def test_field_without_an_init_starts_at_its_default() -> None:
    fc = px.layers.Dense(
        3, bias=False, init=px.init.array(W), conn_policy=conn_set(init={TAG: 7})
    )
    r = built(fc)

    support.close(r.conn_field(fc, LOG_STEP), np.zeros(12, f32))


@pytest.mark.open_question(
    "The design gives connection policy sets an `init`. The sketches' feature units "
    "need "
    "initial unit fields too (a threshold per unit); `px.policies.Unit(init=...)` is a "
    "guess at the same form for units."
)
def test_unit_policy_set_init_gives_unit_fields_fixed_values() -> None:
    out = px.UnitsRef("out")
    p = px.policies
    unit = p.Unit(
        forward=p.SumThenActivate(),
        backward=p.PassThroughBackward(),
        init={BIAS_TOWARD: 0.75},
    )
    fc = px.layers.Dense(3, bias=False, unit_policy=unit)
    model = support.topology(lambda m, x: m.fc(x).bind(out), fc=fc)
    r = support.Runner(support.network(model, input_shape=(4,)))

    support.close(r.field(BIAS_TOWARD, out), np.full(3, 0.75, f32))
