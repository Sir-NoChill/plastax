"""Array notation on handles: the values it computes and the gradients through it.

Design document sections 2.3 and 3.7. Two operands `a` and `b` are made from the single
input by two dense layers with fixed weights, so their values are known:
a = WA @ x and b = WB @ x.

The compositions at the end (softmax, layer norm, attention) come from
`docs/design/sketches/toy_transformer.py`.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import plastax as px

from . import support

LR = 0.1
X = np.array([1.0, -2.0, 3.0, 0.5], np.float32)
WA = (np.arange(12, dtype=np.float32).reshape(3, 4) - 5.0) / 10.0
WB = np.array(
    [[0.5, -1.0, 2.0, 0.0], [1.5, 0.25, -0.75, 1.0], [0.0, 1.0, 0.5, -2.0]], np.float32
)
A, B = WA @ X, WB @ X


def operands() -> dict[str, Any]:
    return {
        "fa": px.layers.Dense(3, bias=False, init=px.init.array(WA)),
        "fb": px.layers.Dense(3, bias=False, init=px.init.array(WB)),
    }


def value(call: Any) -> Any:
    """The output of `call(a, b)` on the two operand handles."""
    model = support.topology(lambda m, x: call(m.fa(x), m.fb(x)), **operands())
    net = support.network(model, input_shape=(4,))
    return support.Runner(net).predict(X).outputs


# --------------------------------------------------------------------------- #
# Values
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("call", "expected"),
    [
        (lambda a, b: a + b, A + B),
        (lambda a, b: a - b, A - B),
        (lambda a, b: a * b, A * B),
        (lambda a, b: a[:, None] * b[None, :], np.outer(A, B)),
        (lambda a, b: a[:, None] + b[None, :], A[:, None] + B[None, :]),
        (lambda a, b: a.sum(axis=0), np.sum(A)),
        (lambda a, b: a.mean(axis=0), np.mean(A)),
        (lambda a, b: a.sum(axis=0, scale=0.25), 0.25 * np.sum(A)),
        (lambda a, b: a.sum(axis=0, keepdims=True), np.sum(A, keepdims=True)),
        (lambda a, b: (a[:, None] * b[None, :]).sum(axis=-1), np.outer(A, B).sum(-1)),
        (lambda a, b: (a[:, None] * b[None, :]).mean(axis=0), np.outer(A, B).mean(0)),
        (lambda a, b: a - a.mean(axis=0, keepdims=True), A - A.mean()),
        (lambda a, b: a * a, A * A),
    ],
    ids=[
        "add",
        "subtract",
        "multiply",
        "multiply_broadcast",
        "add_broadcast",
        "sum",
        "mean",
        "sum_scale",
        "sum_keepdims",
        "sum_of_product",
        "mean_of_product",
        "centre",
        "square",
    ],
)
def test_value(call: Any, expected: np.ndarray) -> None:
    support.close(value(call), np.asarray(expected, np.float32))


def test_activation_on_the_result_of_an_operation() -> None:
    exp = px.activations.exp

    support.close(value(lambda a, b: exp(a - b)), np.exp(A - B))


@pytest.mark.parametrize(
    ("name", "fn"),
    [
        ("exp", np.exp),
        ("reciprocal", lambda v: 1.0 / v),
        ("tanh", np.tanh),
        ("relu", lambda v: np.maximum(v, 0.0)),
    ],
)
def test_shipped_activation_value(name: str, fn: Any) -> None:
    activation = getattr(px.activations, name)

    support.close(value(lambda a, b: activation(a)), fn(A))


def test_rsqrt_adds_its_epsilon_first() -> None:
    rsqrt = px.activations.rsqrt(eps=1e-2)

    support.close(value(lambda a, b: rsqrt(a * a)), 1.0 / np.sqrt(A * A + 1e-2))


# --------------------------------------------------------------------------- #
# Gradients through the operations
# --------------------------------------------------------------------------- #
def gradient_step(call: Any, f: Any, target: np.ndarray) -> None:
    """One SGD step through `call(a, b)` matches jax.grad of `f(a, b)`."""
    layers = operands()
    model = support.topology(lambda m, x: call(m.fa(x), m.fb(x)), **layers)
    net = support.network(model, input_shape=(4,), conn_policy=support.learned(LR))
    r = support.Runner(net)
    r.step(X, target)

    def loss(params: tuple[jax.Array, jax.Array]) -> jax.Array:
        wa, wb = params
        return 0.5 * jnp.sum((f(wa @ X, wb @ X) - target) ** 2)

    ga, gb = jax.grad(loss)((jnp.asarray(WA), jnp.asarray(WB)))
    support.close(r.weights(layers["fa"]), WA - LR * np.asarray(ga), rtol=1e-4)
    support.close(r.weights(layers["fb"]), WB - LR * np.asarray(gb), rtol=1e-4)


@pytest.mark.parametrize(
    ("call", "f", "target"),
    [
        (lambda a, b: a + b, lambda a, b: a + b, np.zeros(3, np.float32)),
        (lambda a, b: a - b, lambda a, b: a - b, np.zeros(3, np.float32)),
        (lambda a, b: a * b, lambda a, b: a * b, np.ones(3, np.float32)),
        (lambda a, b: a * a, lambda a, b: a * a, np.ones(3, np.float32)),
        (
            lambda a, b: (a[:, None] * b[None, :]).sum(axis=-1, scale=0.5),
            lambda a, b: 0.5 * jnp.outer(a, b).sum(-1),
            np.ones(3, np.float32),
        ),
        (
            lambda a, b: a - a.mean(axis=0, keepdims=True),
            lambda a, b: a - a.mean(),
            np.ones(3, np.float32),
        ),
    ],
    ids=["add", "subtract", "multiply", "square", "sum_of_product", "centre"],
)
def test_gradient(call: Any, f: Any, target: np.ndarray) -> None:
    gradient_step(call, f, target)


def test_connections_made_by_operations_are_not_trained() -> None:
    """The +1, -1 and 1/n weights stay what the topology set them to."""

    def call(a: Any, b: Any) -> Any:
        return (a - b).mean(axis=0, keepdims=True)

    layers = operands()
    model = support.topology(lambda m, x: call(m.fa(x), m.fb(x)), **layers)
    r = support.Runner(support.network(model, input_shape=(4,)))
    r.step(X, np.ones(1, np.float32))

    wa, wb = r.weights(layers["fa"]), r.weights(layers["fb"])
    support.close(r.predict(X).outputs, [np.mean(wa @ X - wb @ X)])


# --------------------------------------------------------------------------- #
# Compositions from the transformer sketch
# --------------------------------------------------------------------------- #
def softmax(s: Any, axis: int) -> Any:
    e = px.activations.exp(s)
    return e * px.activations.reciprocal(e.sum(axis=axis, keepdims=True))


def layer_norm(x: Any) -> Any:
    centred = x - x.mean(axis=-1, keepdims=True)
    variance = (centred * centred).mean(axis=-1, keepdims=True)
    return centred * px.activations.rsqrt(eps=1e-5)(variance)


def test_softmax_from_sums_products_and_activations() -> None:
    out = value(lambda a, b: softmax(a[:, None] + b[None, :], axis=-1))

    expected = jax.nn.softmax(A[:, None] + B[None, :], axis=-1)
    support.close(out, expected)
    support.close(np.sum(out, axis=-1), np.ones(3, np.float32))


def test_softmax_gradient() -> None:
    gradient_step(
        lambda a, b: softmax(a + b, axis=0),
        lambda a, b: jax.nn.softmax(a + b),
        np.array([1.0, 0.0, 0.0], np.float32),
    )


def test_layer_norm_from_sums_products_and_activations() -> None:
    out = value(lambda a, b: layer_norm(a[:, None] * b[None, :]))

    grid = np.outer(A, B)
    centred = grid - grid.mean(-1, keepdims=True)
    expected = centred / np.sqrt((centred**2).mean(-1, keepdims=True) + 1e-5)
    support.close(out, expected, rtol=1e-4)


def test_layer_norm_gradient() -> None:
    def f(a: jax.Array, b: jax.Array) -> jax.Array:
        centred = a - a.mean()
        return centred * jax.lax.rsqrt((centred**2).mean() + 1e-5)

    gradient_step(lambda a, b: layer_norm(a), f, np.array([1.0, 0.0, -1.0], np.float32))


def test_single_head_attention() -> None:
    """scores = q k^T / sqrt(d), weights = softmax(scores), out = weights v."""
    length, d = 3, 4
    rng = np.random.default_rng(0)
    x = rng.normal(size=(length, d)).astype(np.float32)
    wq, wk, wv = (rng.normal(size=(d, d)).astype(np.float32) * 0.5 for _ in range(3))

    def call(m: Any, x: Any) -> Any:
        q, k, v = m.query(x), m.key(x), m.value(x)
        scores = (q[:, None, :] * k[None, :, :]).sum(axis=-1, scale=d**-0.5)
        weights = softmax(scores, axis=-1)
        return (weights[:, :, None] * v[None, :, :]).sum(axis=1)

    model = support.topology(
        call,
        query=px.layers.Dense(d, shared=True, bias=False, init=px.init.array(wq)),
        key=px.layers.Dense(d, shared=True, bias=False, init=px.init.array(wk)),
        value=px.layers.Dense(d, shared=True, bias=False, init=px.init.array(wv)),
    )
    out = support.Runner(support.network(model, input_shape=x.shape)).predict(x).outputs

    q, k, v = x @ wq.T, x @ wk.T, x @ wv.T
    expected = jax.nn.softmax(q @ k.T / np.sqrt(d), axis=-1) @ v
    support.close(out, expected, rtol=1e-4, atol=1e-5)
