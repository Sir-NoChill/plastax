"""The backward pass and one optimizer step, against `jax.grad` of the same function.

Design document sections 3.3 and 3.7. Plastax does not differentiate policies; these
tests check that the hand-written backward policies and the SGD update produce the
gradient.

The loss is squared error, 0.5 * sum((y - t)^2), so the seed at the outputs is y - t.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import plastax as px

from . import support

LR = 0.1
X = np.array([1.0, -2.0, 3.0, 0.5], np.float32)
T = np.array([0.5, -1.0], np.float32)
W1 = (np.arange(12, dtype=np.float32).reshape(3, 4) - 5.0) / 10.0
W2 = np.array([[0.5, -1.0, 2.0], [1.5, 0.25, -0.75]], np.float32)


def squared_error(y: jax.Array, t: jax.Array) -> jax.Array:
    return 0.5 * jnp.sum((y - t) ** 2)


def after_one_step(
    f: Callable[..., jax.Array], params: tuple[np.ndarray, ...], x: np.ndarray
) -> list[np.ndarray]:
    """The parameters after one SGD step on squared_error(f(*params, x), T)."""
    grads = jax.grad(lambda p: squared_error(f(*p, x), T))(params)
    return [np.asarray(p - LR * g) for p, g in zip(params, grads, strict=True)]


def trained(call: Any, x: np.ndarray, **layers: Any) -> support.Runner:
    net = support.network(
        support.topology(call, **layers),
        input_shape=x.shape,
        conn_policy=support.learned(LR),
    )
    r = support.Runner(net)
    r.step(x, T)
    return r


def test_two_layer_network_takes_the_gradient_step() -> None:
    tanh = px.activations.tanh
    fc1 = px.layers.Dense(3, bias=False, init=px.init.array(W1))
    fc2 = px.layers.Dense(2, bias=False, init=px.init.array(W2))
    r = trained(lambda m, x: m.fc2(tanh(m.fc1(x))), X, fc1=fc1, fc2=fc2)

    w1, w2 = after_one_step(lambda a, b, x: b @ jnp.tanh(a @ x), (W1, W2), X)
    support.close(r.weights(fc1), w1)
    support.close(r.weights(fc2), w2)


def test_grad_field_holds_the_gradient_at_the_pre_activation() -> None:
    tanh = px.activations.tanh
    hidden, out = px.UnitsRef("hidden"), px.UnitsRef("out")

    def call(m: Any, x: Any) -> Any:
        return m.fc2(tanh(m.fc1(x)).bind(hidden)).bind(out)

    r = trained(
        call,
        X,
        fc1=px.layers.Dense(3, bias=False, init=px.init.array(W1)),
        fc2=px.layers.Dense(2, bias=False, init=px.init.array(W2)),
    )

    z1 = W1 @ X
    seed = W2 @ np.tanh(z1) - T
    support.close(r.field(px.policies.GRAD, out), seed)
    support.close(
        r.field(px.policies.GRAD, hidden), (W2.T @ seed) * (1 - np.tanh(z1) ** 2)
    )


def test_bias_takes_the_gradient_step() -> None:
    fc = px.layers.Dense(
        2, bias=True, init=px.init.array(W2), bias_init=px.init.constant(0.5)
    )
    x = X[:3]
    r = trained(lambda m, x: m.fc(x), x, fc=fc)

    (w,) = after_one_step(lambda a, x: a @ x + 0.5, (W2,), x)
    support.close(r.weights(fc), w)
    # d/db of the loss is the seed itself.
    out = r.predict(x).outputs
    bias = out - r.weights(fc) @ x
    support.close(bias, 0.5 - LR * (W2 @ x + 0.5 - T))


def test_shared_weight_gradient_is_the_sum_over_its_uses() -> None:
    x = np.stack([X, 2 * X, -X])  # three positions
    w = W1[:2]  # (2, 4), shared by the three positions
    fc = px.layers.Dense(2, shared=True, bias=False, init=px.init.array(w))
    r = trained(lambda m, x: m.fc(x).sum(axis=0), x, fc=fc)

    (expected,) = after_one_step(lambda a, x: (x @ a.T).sum(axis=0), (w,), x)
    weights = r.conn_field(fc, px.WEIGHT)
    assert len(np.unique(weights)) == expected.size
    support.close(np.sort(np.unique(weights)), np.sort(expected.reshape(-1)))


def test_residual_connection_carries_gradient_and_is_not_trained() -> None:
    tanh = px.activations.tanh
    w_a = np.eye(4, dtype=np.float32) * 0.5 + 0.1
    w_b = (np.arange(16, dtype=np.float32).reshape(4, 4) - 8.0) / 20.0
    w_out = np.array([[1.0, -1.0, 0.5, 2.0], [0.0, 0.5, -0.5, 1.0]], np.float32)
    fa = px.layers.Dense(4, bias=False, init=px.init.array(w_a))
    fb = px.layers.Dense(4, bias=False, init=px.init.array(w_b))
    out = px.layers.Dense(2, bias=False, init=px.init.array(w_out))

    def call(m: Any, x: Any) -> Any:
        h = tanh(m.fa(x))
        return m.out(tanh(m.fb(h) + h))

    r = trained(call, X, fa=fa, fb=fb, out=out)

    def f(a: jax.Array, b: jax.Array, o: jax.Array, x: jax.Array) -> jax.Array:
        h = jnp.tanh(a @ x)
        return o @ jnp.tanh(b @ h + h)

    for layer, expected in zip(
        (fa, fb, out), after_one_step(f, (w_a, w_b, w_out), X), strict=True
    ):
        support.close(r.weights(layer), expected)
    # The identity connections of `+` still carry weight 1: a second step agrees with
    # the same function at the new parameters.
    y = r.predict(X).outputs
    support.close(y, f(r.weights(fa), r.weights(fb), r.weights(out), X))


def test_activation_derivative_need_not_be_the_true_one() -> None:
    """LTU: a 0/1 step forward, the sigmoid's derivative backward."""
    ltu = px.activations.ltu
    fc1 = px.layers.Dense(3, bias=False, init=px.init.array(W1))
    fc2 = px.layers.Dense(2, bias=False, init=px.init.array(W2))
    r = trained(lambda m, x: m.fc2(ltu(m.fc1(x))), X, fc1=fc1, fc2=fc2)

    z1 = W1 @ X
    h = (z1 > 0).astype(np.float32)
    seed = W2 @ h - T
    sigmoid = 1.0 / (1.0 + np.exp(-z1))
    grad_z1 = (W2.T @ seed) * sigmoid * (1.0 - sigmoid)
    support.close(r.weights(fc2), W2 - LR * np.outer(seed, h))
    support.close(r.weights(fc1), W1 - LR * np.outer(grad_z1, X))


def test_softmax_cross_entropy_takes_the_gradient_step() -> None:
    target = np.array([0.0, 1.0, 0.0], np.float32)
    fc = px.layers.Dense(3, bias=False, init=px.init.array(W1))
    net = support.network(
        support.topology(lambda m, x: m.fc(x), fc=fc),
        input_shape=(4,),
        conn_policy=support.learned(LR),
        loss=px.SoftmaxCrossEntropyLossPolicy(seed=px.policies.SEED),
    )
    r = support.Runner(net)
    result = r.step(X, target)

    def loss(w: jax.Array) -> jax.Array:
        return -jnp.sum(target * jax.nn.log_softmax(w @ X))

    support.close(result.loss, loss(W1))
    support.close(r.weights(fc), W1 - LR * np.asarray(jax.grad(loss)(W1)))
