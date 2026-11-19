"""Losses and outputs.

Design document section 3.11. A loss sees a group of units. For a regular grid of groups
the handle's shape gives the groups and the loss runs over its last axis; otherwise a
loss is applied to each handle and targets are matched by position.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, Float, Int32

import plastax as px

from . import support

f32 = np.float32
LR = 0.1

X = np.array([1.0, -2.0, 3.0, 0.5], f32)
W = (np.arange(24, dtype=f32).reshape(6, 4) - 11.0) / 20.0


def cross_entropy(logits: jax.Array, target: jax.Array) -> jax.Array:
    return -jnp.sum(target * jax.nn.log_softmax(logits, axis=-1))


@px.policy
class ErrorTimesTwo(px.LossPolicy[px.Globals]):
    """loss = sum((y - t)^2); the value it returns as the seed is 2 * (y - t)."""

    seed: px.UnitField[f32]

    def calculate_loss(
        self,
        u: px.UnitView,
        outputs: Int32[Array, " n"],
        targets: Float[Array, " n"],
        g: px.Globals,
    ) -> tuple[Float[Array, ""], Float[Array, " n"]]:
        error = u.gather(px.ACTIVATION, outputs) - targets
        return jnp.sum(error * error), 2.0 * error


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #
def test_outputs_have_the_shape_the_topology_returned() -> None:
    fc = px.layers.Dense(6, bias=False, init=px.init.array(W))
    model = support.topology(lambda m, x: m.fc(x).reshape(2, 3), fc=fc)
    out = support.Runner(support.network(model, input_shape=(4,))).predict(X).outputs

    support.close(out, (W @ X).reshape(2, 3))


def test_outputs_are_a_tuple_when_the_topology_returns_several_handles() -> None:
    head_a = px.layers.Dense(3, bias=False, init=px.init.array(W[:3]))
    head_b = px.layers.Dense(2, bias=False, init=px.init.array(W[3:5]))
    model = support.topology(lambda m, x: (m.a(x), m.b(x)), a=head_a, b=head_b)
    a, b = support.Runner(support.network(model, input_shape=(4,))).predict(X).outputs

    support.close(a, W[:3] @ X)
    support.close(b, W[3:5] @ X)


def test_outputs_of_a_step_come_from_before_its_update() -> None:
    fc = px.layers.Dense(6, bias=False, init=px.init.array(W))
    model = support.topology(lambda m, x: m.fc(x), fc=fc)
    r = support.Runner(support.network(model, input_shape=(4,)))
    result = r.step(X, np.ones(6, f32))

    support.close(result.outputs, W @ X)
    assert not np.allclose(r.weights(fc), W)


# --------------------------------------------------------------------------- #
# Loss groups
# --------------------------------------------------------------------------- #
def test_loss_over_the_last_axis_is_one_softmax_per_group() -> None:
    target = np.array([[0, 1, 0], [1, 0, 0]], f32)
    fc = px.layers.Dense(6, bias=False, init=px.init.array(W))
    model = support.topology(lambda m, x: m.fc(x).reshape(2, 3), fc=fc)
    net = support.network(
        model,
        input_shape=(4,),
        conn_policy=support.learned(LR),
        loss=px.SoftmaxCrossEntropyLossPolicy(seed=px.policies.SEED, axis=-1),
    )
    r = support.Runner(net)
    result = r.step(X, target)

    def loss(w: jax.Array) -> jax.Array:
        return cross_entropy((w @ X).reshape(2, 3), target)

    support.close(result.loss, loss(W))
    support.close(r.weights(fc), W - LR * np.asarray(jax.grad(loss)(jnp.asarray(W))))


def test_loss_applied_to_each_handle_takes_targets_by_position() -> None:
    identity = px.activations.identity
    ce = px.SoftmaxCrossEntropyLossPolicy(seed=px.policies.SEED)
    head_a = px.layers.Dense(3, bias=False, init=px.init.array(W[:3]))
    head_b = px.layers.Dense(2, bias=False, init=px.init.array(W[3:5]))

    def call(m: Any, x: Any) -> Any:
        return identity(m.a(x), loss=ce), identity(m.b(x), loss=ce)

    model = support.topology(call, a=head_a, b=head_b)
    net = support.network(
        model, input_shape=(4,), conn_policy=support.learned(LR), loss=None
    )
    r = support.Runner(net)
    targets = (np.array([0, 0, 1], f32), np.array([1, 0], f32))
    result = r.step(X, targets)

    def loss_a(w: jax.Array) -> jax.Array:
        return cross_entropy(w @ X, targets[0])

    def loss_b(w: jax.Array) -> jax.Array:
        return cross_entropy(w @ X, targets[1])

    support.close(result.loss, loss_a(W[:3]) + loss_b(W[3:5]))
    support.close(r.weights(head_a), W[:3] - LR * np.asarray(jax.grad(loss_a)(W[:3])))
    support.close(r.weights(head_b), W[3:5] - LR * np.asarray(jax.grad(loss_b)(W[3:5])))


# --------------------------------------------------------------------------- #
# A custom loss
# --------------------------------------------------------------------------- #
def test_custom_loss_sees_the_output_units_and_writes_its_seed_there() -> None:
    seed = px.policies.SEED
    hidden, out = px.UnitsRef("hidden"), px.UnitsRef("out")
    fc1 = px.layers.Dense(3, bias=False, init=px.init.array(W[:3]))
    w2 = np.array([[0.5, -1.0, 2.0], [1.5, 0.25, -0.75]], f32)
    fc2 = px.layers.Dense(2, bias=False, init=px.init.array(w2))
    model = support.topology(
        lambda m, x: m.fc2(m.fc1(x).bind(hidden)).bind(out), fc1=fc1, fc2=fc2
    )
    net = support.network(
        model,
        input_shape=(4,),
        conn_policy=support.learned(LR),
        loss=ErrorTimesTwo(seed=seed),
    )
    r = support.Runner(net)
    target = np.array([0.5, -1.0], f32)
    result = r.step(X, target)

    error = w2 @ W[:3] @ X - target
    support.close(result.loss, np.sum(error**2))
    support.close(r.field(seed, out), 2.0 * error)
    support.close(
        r.field(seed, hidden), np.zeros(3, f32)
    )  # written at the outputs only
    support.close(r.weights(fc2), w2 - LR * np.outer(2.0 * error, W[:3] @ X))


@pytest.mark.open_question(
    "How a network says it has no loss at all (a reservoir). `loss = None` already "
    "means "
    "'every output handle supplies its own'. `px.NO_LOSS` is a guess."
)
def test_network_with_no_loss_runs_forward_and_has_no_loss_value() -> None:
    fc = px.layers.Dense(6, bias=False, init=px.init.array(W))
    model = support.topology(lambda m, x: m.fc(x), fc=fc)
    net = support.network(
        model, input_shape=(4,), conn_policy=support.fixed(), loss=px.NO_LOSS
    )
    result = support.Runner(net).step(X)

    support.close(result.outputs, W @ X)
    assert result.loss is None


def test_targets_of_the_wrong_shape_are_an_error_at_the_step() -> None:
    fc = px.layers.Dense(6, bias=False, init=px.init.array(W))
    model = support.topology(lambda m, x: m.fc(x).reshape(2, 3), fc=fc)
    r = support.Runner(support.network(model, input_shape=(4,)))

    with pytest.raises(px.ShapeError, match="targets"):
        r.step(X, np.zeros(6, f32))
