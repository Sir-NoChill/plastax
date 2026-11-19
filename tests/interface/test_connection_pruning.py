"""Pruning connections.

Design document sections 3.3 and 3.10. A prune policy's predicate removes a connection.
Pruning runs after the update within a step. In a dense layer a pruned connection is
masked out.

The rule used throughout: a connection with a negative weight is pruned.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, Bool

import plastax as px

from . import support

Flag = Bool[Array, ""]
f32 = np.float32
LR = 0.1

X = np.array([1.0, -2.0, 3.0, 0.5], f32)
T = np.array([0.5, -1.0], f32)
W1 = (np.arange(12, dtype=f32).reshape(3, 4) - 5.0) / 10.0  # five negative entries
W2 = np.array([[0.5, -1.0, 2.0], [1.5, 0.25, -0.75]], f32)  # two negative entries
KEPT1 = np.where(W1 >= 0, W1, 0.0)
KEPT2 = np.where(W2 >= 0, W2, 0.0)


@px.policy
class NegativeWeight(px.PruneConnPolicy[px.Globals]):
    def predicate(
        self, u: px.UnitView, c: px.ConnView, cid: px.ConnIdx, g: px.Globals
    ) -> Flag:
        return c[px.WEIGHT, cid] < 0.0


def pruning(update: Any = None) -> Any:
    p = px.policies
    return p.Conn(
        forward=p.WeightedSum(),
        backward=p.WeightedSumBackward(),
        update=update,
        prune=NegativeWeight(),
        add=None,
    )


def one_layer(policy: Any, **layer_args: Any) -> tuple[support.Runner, Any]:
    fc = px.layers.Dense(3, bias=False, init=px.init.array(W1), **layer_args)
    model = support.topology(lambda m, x: m.fc(x), fc=fc)
    return support.Runner(
        support.network(model, input_shape=(4,), conn_policy=policy)
    ), fc


def test_predicate_removes_exactly_the_connections_it_selects() -> None:
    r, fc = one_layer(pruning())
    r.step(X, np.zeros(3, f32))

    assert len(r.pairs(fc)) == int((W1 >= 0).sum())
    support.close(r.weights(fc), KEPT1)


def test_pruning_runs_after_the_forward_pass_of_its_step() -> None:
    r, _ = one_layer(pruning())
    result = r.step(X, np.zeros(3, f32))

    support.close(result.outputs, W1 @ X)


def test_pruned_connection_no_longer_contributes() -> None:
    r, _ = one_layer(pruning())
    r.step(X, np.zeros(3, f32))

    support.close(r.predict(X).outputs, KEPT1 @ X)


def test_unit_that_loses_every_input_outputs_zero() -> None:
    all_negative = -np.abs(W1) - 0.1
    fc = px.layers.Dense(3, bias=False, init=px.init.array(all_negative))
    model = support.topology(lambda m, x: m.fc(x), fc=fc)
    r = support.Runner(support.network(model, input_shape=(4,), conn_policy=pruning()))
    r.step(X, np.zeros(3, f32))

    assert r.pairs(fc) == []
    support.close(r.predict(X).outputs, np.zeros(3, f32))


def test_pruning_runs_after_the_update_of_its_step() -> None:
    """A weight the update carries below zero is removed in that same step."""
    non_negative = np.abs(W1)
    fc = px.layers.Dense(3, bias=False, init=px.init.array(non_negative))
    model = support.topology(lambda m, x: m.fc(x), fc=fc)
    net = support.network(
        model,
        input_shape=(4,),
        conn_policy=pruning(update=px.optim.sgd(lr=LR, grad=px.policies.GRAD)),
    )
    r = support.Runner(net)
    target = np.array([5.0, -5.0, 0.0], f32)
    r.step(X, target)

    updated = non_negative - LR * np.outer(non_negative @ X - target, X)
    assert (updated < 0).any()  # the case is not vacuous
    support.close(r.weights(fc), np.where(updated >= 0, updated, 0.0))
    assert len(r.pairs(fc)) == int((updated >= 0).sum())


def test_no_gradient_passes_through_a_pruned_connection() -> None:
    """Layer 2 prunes its negative weights in step 1; step 2's gradient for layer 1 is
    the gradient of the network without them.
    """
    fc1 = px.layers.Dense(3, bias=False, init=px.init.array(W1))
    fc2 = px.layers.Dense(2, bias=False, init=px.init.array(W2), conn_policy=pruning())
    model = support.topology(lambda m, x: m.fc2(m.fc1(x)), fc1=fc1, fc2=fc2)
    r = support.Runner(
        support.network(model, input_shape=(4,), conn_policy=support.learned(LR))
    )
    r.step(X, T)
    r.step(X, T)

    def step(w1: jax.Array, w2: np.ndarray) -> jax.Array:
        grad = jax.grad(lambda w: 0.5 * jnp.sum((w2 @ (w @ X) - T) ** 2))(w1)
        return w1 - LR * grad

    expected = step(step(jnp.asarray(W1), W2), KEPT2)
    support.close(r.weights(fc1), expected, rtol=1e-4)
    support.close(r.weights(fc2), KEPT2)


def test_set_with_no_prune_policy_is_never_pruned() -> None:
    """The network default prunes; this layer's set says `prune=None`."""
    fc1 = px.layers.Dense(3, bias=False, init=px.init.array(W1))
    fc2 = px.layers.Dense(
        2, bias=False, init=px.init.array(W2), conn_policy=support.fixed()
    )
    model = support.topology(lambda m, x: m.fc2(m.fc1(x)), fc1=fc1, fc2=fc2)
    r = support.Runner(support.network(model, input_shape=(4,), conn_policy=pruning()))
    r.step(X, T)

    support.close(r.weights(fc1), KEPT1)
    support.close(r.weights(fc2), W2)


@pytest.mark.parametrize("fmt", ["COO", "CSR", "DENSE"])
def test_pruning_gives_the_same_result_in_every_format(fmt: str) -> None:
    """In a dense layer the pruned connections are masked out."""
    r, fc = one_layer(pruning(), hint=px.Hint(format=getattr(px.Format, fmt)))
    r.step(X, np.zeros(3, f32))

    support.close(r.weights(fc), KEPT1)
    support.close(r.predict(X).outputs, KEPT1 @ X)
    assert len(r.pairs(fc)) == int((W1 >= 0).sum())
