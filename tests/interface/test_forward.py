"""Forward arithmetic of the layers, against numpy.

Design document sections 2.2 and 3.3. Weights are fixed through the layer initializer,
so every expected value is written out in the test.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import plastax as px

from . import support

X = np.array([1.0, -2.0, 3.0, 0.5], np.float32)
W1 = (np.arange(12, dtype=np.float32).reshape(3, 4) - 5.0) / 10.0
W2 = np.array([[0.5, -1.0, 2.0], [1.5, 0.25, -0.75]], np.float32)


def outputs(call: Any, x: np.ndarray, **layers: Any) -> Any:
    net = support.network(support.topology(call, **layers), input_shape=x.shape)
    return support.Runner(net).predict(x).outputs


def test_dense_layer_is_a_matrix_product() -> None:
    out = outputs(
        lambda m, x: m.fc(x),
        X,
        fc=px.layers.Dense(3, bias=False, init=px.init.array(W1)),
    )

    support.close(out, W1 @ X)


def test_activation_is_applied_to_each_unit() -> None:
    relu = px.activations.relu
    out = outputs(
        lambda m, x: relu(m.fc(x)),
        X,
        fc=px.layers.Dense(3, bias=False, init=px.init.array(W1)),
    )

    support.close(out, np.maximum(W1 @ X, 0.0))


def test_two_layers_in_one_step() -> None:
    """In the topologically sorted mode a signal crosses every layer in one step."""
    tanh = px.activations.tanh
    out = outputs(
        lambda m, x: m.fc2(tanh(m.fc1(x))),
        X,
        fc1=px.layers.Dense(3, bias=False, init=px.init.array(W1)),
        fc2=px.layers.Dense(2, bias=False, init=px.init.array(W2)),
    )

    support.close(out, W2 @ np.tanh(W1 @ X))


def test_bias_adds_a_learned_constant() -> None:
    fc = px.layers.Dense(
        3, bias=True, init=px.init.array(W1), bias_init=px.init.constant(0.5)
    )
    out = outputs(lambda m, x: m.fc(x), X, fc=fc)

    support.close(out, W1 @ X + 0.5)


def test_shared_dense_applies_the_same_weights_at_every_position() -> None:
    x = np.stack([X, 2 * X, -X])  # (3, 4)
    fc = px.layers.Dense(3, shared=True, bias=False, init=px.init.array(W1))
    out = outputs(lambda m, x: m.fc(x), x, fc=fc)

    support.close(out, x @ W1.T)


def test_grid_is_flattened_before_a_dense_layer_reads_all_of_it() -> None:
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    w = np.linspace(-1.0, 1.0, 12, dtype=np.float32).reshape(2, 6)
    fc = px.layers.Dense(2, bias=False, init=px.init.array(w))
    out = outputs(lambda m, x: m.fc(x.flatten()), x, fc=fc)

    support.close(out, w @ x.reshape(-1))


def test_shared_convolution_with_same_padding() -> None:
    x = np.arange(16, dtype=np.float32).reshape(4, 4, 1) / 4.0
    kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], np.float32)
    conv = px.layers.Conv(
        1,
        3,
        padding="same",
        shared=True,
        bias=False,
        init=px.init.array(kernel.reshape(3, 3, 1, 1)),
    )
    out = outputs(lambda m, x: m.conv(x), x, conv=conv)

    padded = np.pad(x[..., 0], 1)
    expected = np.zeros((4, 4), np.float32)
    for i in range(4):
        for j in range(4):
            expected[i, j] = np.sum(padded[i : i + 3, j : j + 3] * kernel)
    support.close(out, expected[..., None])


def test_strided_valid_convolution_with_two_channels() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 5, 2)).astype(np.float32)
    kernel = rng.normal(size=(3, 3, 2, 3)).astype(np.float32)  # (kh, kw, c_in, c_out)
    conv = px.layers.Conv(
        3,
        3,
        stride=2,
        padding="valid",
        shared=True,
        bias=False,
        init=px.init.array(kernel),
    )
    out = outputs(lambda m, x: m.conv(x), x, conv=conv)

    expected = np.zeros((2, 2, 3), np.float32)
    for i in range(2):
        for j in range(2):
            window = x[2 * i : 2 * i + 3, 2 * j : 2 * j + 3, :]
            expected[i, j] = np.einsum("hwc,hwco->o", window, kernel)
    support.close(out, expected, rtol=1e-4, atol=1e-5)


def test_explicit_pairs_sum_only_what_they_connect() -> None:
    wiring = px.layers.pairs(src=np.array([0, 1, 2, 3]), dst=np.array([0, 0, 1, 1]))
    fc = px.layers.Dense(2, connectivity=wiring, bias=False, init=px.init.constant(2.0))
    out = outputs(lambda m, x: m.fc(x), X, fc=fc)

    support.close(out, [2.0 * (X[0] + X[1]), 2.0 * (X[2] + X[3])])


def test_sparse_layer_sums_only_its_connections() -> None:
    fc = px.layers.Dense(
        5,
        connectivity=px.layers.sparse(fan_in=2),
        bias=False,
        init=px.init.constant(1.0),
    )
    net = support.network(
        support.topology(lambda m, x: m.fc(x), fc=fc), input_shape=(4,)
    )
    r = support.Runner(net)

    support.close(r.predict(X).outputs, r.weights(fc) @ X)
    assert np.all((r.weights(fc) != 0).sum(axis=1) == 2)
