"""Max pooling: the forward maximum and where the gradient goes.

Design document section 3.7. No winner is stored. A pooling connection passes the
gradient back when its source's activation equals its destination's, so tied inputs each
receive the full gradient.

The network is: input (16,) -> an identity dense layer -> reshape (4, 4, 1) -> pool ->
flatten. The identity layer's units are the ones whose gradient is read.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import plastax as px

from . import support

IDENTITY = np.eye(16, dtype=np.float32)


def pooled(grid: np.ndarray, targets: np.ndarray | None = None) -> tuple[Any, Any, Any]:
    """Run one step on a 4x4 grid. Returns (runner, result, ref to the grid's units)."""
    cells = px.UnitsRef("cells")
    pre = px.layers.Dense(16, bias=False, init=px.init.array(IDENTITY))

    def call(m: Any, x: Any) -> Any:
        return m.pool(m.pre(x).bind(cells).reshape(4, 4, 1)).flatten()

    model = support.topology(call, pre=pre, pool=px.layers.MaxPool(2))
    r = support.Runner(support.network(model, input_shape=(16,)))
    targets = np.zeros(4, np.float32) if targets is None else targets
    result = r.step(grid.reshape(-1).astype(np.float32), targets)
    return r, result, cells


def test_forward_is_the_maximum_of_each_window() -> None:
    grid = np.array([[1, 5, 2, 0], [3, 4, 9, 1], [-1, -2, 6, 6], [-3, -4, 7, 8]])
    _, result, _ = pooled(grid)

    support.close(result.outputs, [5.0, 9.0, -1.0, 8.0])


def test_gradient_goes_to_the_input_that_won() -> None:
    grid = np.array([[1, 5, 2, 0], [3, 4, 9, 1], [-1, -2, 6, 6], [-3, -4, 7, 8]])
    targets = np.array([1.0, 2.0, 3.0, 4.0], np.float32)
    r, result, cells = pooled(grid, targets)

    seed = np.asarray(result.outputs) - targets  # squared error
    expected = np.zeros((4, 4), np.float32)
    expected[0, 1] = seed[0]  # the 5
    expected[1, 2] = seed[1]  # the 9
    expected[2, 0] = seed[2]  # the -1
    expected[3, 3] = seed[3]  # the 8
    support.close(r.field(px.policies.GRAD, cells), expected.reshape(-1))


def test_tied_inputs_each_receive_the_full_gradient() -> None:
    grid = np.zeros((4, 4))
    grid[0, 0] = grid[1, 1] = 3.0  # two winners in the top-left window
    targets = np.array([1.0, 0.0, 0.0, 0.0], np.float32)
    r, result, cells = pooled(grid, targets)

    grad = r.field(px.policies.GRAD, cells).reshape(4, 4)
    seed = float(result.outputs[0]) - 1.0
    assert seed == 2.0
    support.close(grad[0, 0], seed)
    support.close(grad[1, 1], seed)
    support.close(grad[0, 1], 0.0)


def test_pooling_connections_are_not_trained() -> None:
    grid = np.array([[1, 5, 2, 0], [3, 4, 9, 1], [-1, -2, 6, 6], [-3, -4, 7, 8]])
    r, _, _ = pooled(grid, np.array([1.0, 2.0, 3.0, 4.0], np.float32))

    # After a training step the pool still takes the plain maximum of what feeds it.
    pre = r.net.topology.pre
    fed = (r.weights(pre) @ grid.reshape(-1)).reshape(2, 2, 2, 2)
    expected = fed.transpose(0, 2, 1, 3).reshape(4, 4).max(axis=1)
    support.close(r.predict(grid.reshape(-1).astype(np.float32)).outputs, expected)


def test_pooling_over_channels_keeps_them_apart() -> None:
    x = np.zeros((2, 2, 2), np.float32)
    x[..., 0] = [[1, 2], [3, 4]]
    x[..., 1] = [[-1, -5], [-2, -3]]
    model = support.topology(lambda m, x: m.pool(x), pool=px.layers.MaxPool(2))
    out = support.Runner(support.network(model, input_shape=x.shape)).predict(x).outputs

    support.close(out, np.array([[[4.0, -1.0]]], np.float32))
