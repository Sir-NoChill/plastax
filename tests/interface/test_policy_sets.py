"""Policy sets: network defaults, overrides on layers and handles, frozen sets.

Design document sections 2.6, 3.1 and 3.9.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from jaxtyping import Array, Float

import plastax as px

from . import support

Scalar = Float[Array, ""]

X = np.array([1.0, -2.0, 3.0, 0.5], np.float32)
T = np.array([0.5, -1.0], np.float32)
W1 = (np.arange(12, dtype=np.float32).reshape(3, 4) - 5.0) / 10.0
W2 = np.array([[0.5, -1.0, 2.0], [1.5, 0.25, -0.75]], np.float32)


@px.policy
class Doubling(px.ForwardUnitPolicy[Scalar, px.Globals]):
    """A unit that outputs twice the sum of its inputs."""

    combine = px.monoid.sum_

    def apply(self, u: px.UnitView, i: px.UnitIdx, g: px.Globals, acc: Scalar) -> Any:
        return px.UnitWrite.of((px.ACTIVATION, 2.0 * acc))


def two_layers(**overrides: Any) -> tuple[Any, Any, Any]:
    """(fc1, fc2, model) for x -> fc1 -> fc2, with keyword overrides per layer."""
    fc1 = px.layers.Dense(
        3, bias=False, init=px.init.array(W1), **overrides.get("fc1", {})
    )
    fc2 = px.layers.Dense(
        2, bias=False, init=px.init.array(W2), **overrides.get("fc2", {})
    )
    model = support.topology(lambda m, x: m.fc2(m.fc1(x)), fc1=fc1, fc2=fc2)
    return fc1, fc2, model


def changed(r: support.Runner, layer: Any, before: np.ndarray) -> bool:
    return not np.allclose(r.weights(layer), before)


# --------------------------------------------------------------------------- #
# Connection policy sets
# --------------------------------------------------------------------------- #
def test_network_policy_is_the_default_for_every_layer() -> None:
    fc1, fc2, model = two_layers()
    r = support.Runner(support.network(model, input_shape=(4,)))
    r.step(X, T)

    assert changed(r, fc1, W1)
    assert changed(r, fc2, W2)


def test_layer_policy_overrides_the_default() -> None:
    fc1, fc2, model = two_layers(fc1={"conn_policy": support.fixed()})
    r = support.Runner(support.network(model, input_shape=(4,)))
    r.step(X, T)

    support.close(r.weights(fc1), W1)
    assert changed(r, fc2, W2)


def test_frozen_set_still_carries_the_forward_and_backward_pass() -> None:
    """A frozen layer between two trained ones passes values down and gradient up."""
    fc1, fc2, model = two_layers(fc2={"conn_policy": support.fixed()})
    r = support.Runner(
        support.network(model, input_shape=(4,), conn_policy=support.learned(0.1))
    )
    result = r.step(X, T)

    seed = W2 @ W1 @ X - T
    support.close(result.outputs, W2 @ W1 @ X)
    support.close(r.weights(fc2), W2)
    support.close(r.weights(fc1), W1 - 0.1 * np.outer(W2.T @ seed, X))


def test_none_as_the_default_is_satisfied_when_every_layer_supplies_a_set() -> None:
    fc1, fc2, model = two_layers(
        fc1={"conn_policy": support.learned()}, fc2={"conn_policy": support.fixed()}
    )
    r = support.Runner(support.network(model, input_shape=(4,), conn_policy=None))
    r.step(X, T)

    assert changed(r, fc1, W1)
    support.close(r.weights(fc2), W2)


# --------------------------------------------------------------------------- #
# Unit policy sets
# --------------------------------------------------------------------------- #
def test_units_without_an_activation_use_the_identity() -> None:
    _, _, model = two_layers()
    r = support.Runner(support.network(model, input_shape=(4,)))

    support.close(r.predict(X).outputs, W2 @ W1 @ X)


def test_activation_on_a_handle_sets_the_units_policy() -> None:
    relu = px.activations.relu
    fc1 = px.layers.Dense(3, bias=False, init=px.init.array(W1))
    fc2 = px.layers.Dense(2, bias=False, init=px.init.array(W2))
    model = support.topology(lambda m, x: m.fc2(relu(m.fc1(x))), fc1=fc1, fc2=fc2)
    r = support.Runner(support.network(model, input_shape=(4,)))

    support.close(r.predict(X).outputs, W2 @ np.maximum(W1 @ X, 0.0))


def test_with_policy_gives_a_handle_its_own_unit_policy() -> None:
    doubling = px.policies.Unit(
        forward=Doubling(), backward=px.policies.PassThroughBackward()
    )
    fc1 = px.layers.Dense(3, bias=False, init=px.init.array(W1))
    fc2 = px.layers.Dense(2, bias=False, init=px.init.array(W2))
    model = support.topology(
        lambda m, x: m.fc2(m.fc1(x).with_policy(doubling)), fc1=fc1, fc2=fc2
    )
    r = support.Runner(support.network(model, input_shape=(4,)))

    support.close(r.predict(X).outputs, W2 @ (2.0 * (W1 @ X)))


def test_layer_unit_policy_overrides_the_default() -> None:
    doubling = px.policies.Unit(
        forward=Doubling(), backward=px.policies.PassThroughBackward()
    )
    fc = px.layers.Dense(3, bias=False, init=px.init.array(W1), unit_policy=doubling)
    model = support.topology(lambda m, x: m.fc(x), fc=fc)
    r = support.Runner(support.network(model, input_shape=(4,)))

    support.close(r.predict(X).outputs, 2.0 * (W1 @ X))


def test_network_unit_policy_may_be_a_set_or_a_function_of_the_activation() -> None:
    relu = px.activations.relu
    fc = px.layers.Dense(3, bias=False, init=px.init.array(W1))

    as_function = support.network(
        support.topology(lambda m, x: relu(m.fc(x)), fc=fc),
        input_shape=(4,),
        unit_policy=px.policies.summing,
    )
    support.close(support.Runner(as_function).predict(X).outputs, np.maximum(W1 @ X, 0))

    fc = px.layers.Dense(3, bias=False, init=px.init.array(W1))
    as_set = support.network(
        support.topology(lambda m, x: m.fc(x), fc=fc),
        input_shape=(4,),
        unit_policy=px.policies.summing(px.activations.relu),
    )
    support.close(support.Runner(as_set).predict(X).outputs, np.maximum(W1 @ X, 0))


# --------------------------------------------------------------------------- #
# Loss override
# --------------------------------------------------------------------------- #
def test_loss_on_a_handle_overrides_the_network_loss() -> None:
    identity = px.activations.identity
    cross_entropy = px.SoftmaxCrossEntropyLossPolicy(seed=px.policies.SEED)
    fc = px.layers.Dense(3, bias=False, init=px.init.array(W1))
    model = support.topology(lambda m, x: identity(m.fc(x), loss=cross_entropy), fc=fc)
    r = support.Runner(support.network(model, input_shape=(4,)))
    target = np.array([0.0, 1.0, 0.0], np.float32)
    result = r.step(X, target)

    logits = W1 @ X
    log_softmax = logits - np.log(np.sum(np.exp(logits)))
    support.close(result.loss, -np.sum(target * log_softmax))
