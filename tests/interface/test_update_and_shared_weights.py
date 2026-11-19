"""The connection update (contribution, combine, step) and shared weights.

Design document sections 3.3 and 3.8.

The update used here is deliberately not a gradient rule, so the three parts are
visible: each use of a weight contributes its source's activation, the contributions are
combined, and one step adds the total to the weight.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from jaxtyping import Array, Bool, Float

import plastax as px

from . import support

Scalar = Float[Array, ""]
Flag = Bool[Array, ""]
f32 = np.float32

POSITIONS, D_IN, D_OUT = 3, 4, 2
W = (np.arange(8, dtype=f32).reshape(D_OUT, D_IN) - 3.0) / 10.0
X1 = np.array([1.0, -2.0, 3.0, 0.5], f32)
X = np.stack([X1, 2 * X1, -0.5 * X1])  # (POSITIONS, D_IN)


@px.policy
class Accumulate(px.UpdateConnPolicy[Scalar, px.Globals]):
    """w <- w + combine(source activations of the weight's uses)."""

    combine = px.monoid.sum_
    steps: px.ConnField[np.int32] = 0  # how many times `step` ran for this weight
    last_total: px.ConnField[f32] = 0.0

    def contribution(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: px.Globals,
    ) -> Scalar:
        return u[px.ACTIVATION, src]

    def step(
        self, c: px.ConnView, cid: px.ConnIdx, g: px.Globals, total: Scalar
    ) -> Any:
        return px.ConnWrite.of(
            (px.WEIGHT, c[px.WEIGHT, cid] + total),
            (self.steps, c[self.steps, cid] + 1),
            (self.last_total, total),
        )


@px.policy
class AccumulateMax(Accumulate):
    """The same, combining the uses by maximum."""

    combine = px.monoid.max_


@px.policy
class CountUses(Accumulate):
    """Every use contributes 1, so the total is the number of uses."""

    def contribution(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: px.Globals,
    ) -> Scalar:
        return u[px.ACTIVATION, src] * 0.0 + 1.0


@px.policy
class FromLoudSource(px.PruneConnPolicy[px.Globals]):
    """True for a connection whose source's activation is above 100."""

    def predicate(
        self, u: px.UnitView, c: px.ConnView, cid: px.ConnIdx, g: px.Globals
    ) -> Flag:
        return u[px.ACTIVATION, c[px.FROM_ID, cid]] > 100.0


def conn_set(update: Any, **more: Any) -> Any:
    p = px.policies
    return p.Conn(
        forward=p.WeightedSum(), backward=p.WeightedSumBackward(), update=update, **more
    )


def one_layer(policy: Any, *, shared: bool, x: np.ndarray) -> tuple[Any, Any, Any]:
    """Build x -> fc, run one step. Returns (runner, layer, result)."""
    fc = px.layers.Dense(D_OUT, shared=shared, bias=False, init=px.init.array(W))
    model = support.topology(lambda m, x: m.fc(x), fc=fc)
    r = support.Runner(support.network(model, input_shape=x.shape, conn_policy=policy))
    result = r.step(x, np.zeros(x.shape[:-1] + (D_OUT,), f32))
    return r, fc, result


def per_position(r: support.Runner, fc: Any) -> np.ndarray:
    """The weights each position uses, as (POSITIONS, D_OUT, D_IN)."""
    full = r.weights(fc).reshape(POSITIONS, D_OUT, POSITIONS, D_IN)
    return np.stack([full[p, :, p, :] for p in range(POSITIONS)])


# --------------------------------------------------------------------------- #
# The three parts of an update
# --------------------------------------------------------------------------- #
def test_private_weight_is_the_case_of_one_use() -> None:
    r, fc, _ = one_layer(conn_set(Accumulate()), shared=False, x=X1)

    support.close(r.weights(fc), W + X1[None, :])


def test_shared_weight_receives_the_sum_over_its_uses() -> None:
    r, fc, _ = one_layer(conn_set(Accumulate()), shared=True, x=X)

    expected = W + X.sum(axis=0)[None, :]
    for position in per_position(r, fc):
        support.close(position, expected)


def test_combine_can_be_another_monoid() -> None:
    r, fc, _ = one_layer(conn_set(AccumulateMax()), shared=True, x=X)

    expected = W + X.max(axis=0)[None, :]
    for position in per_position(r, fc):
        support.close(position, expected)


def test_step_may_write_several_fields() -> None:
    update = Accumulate()
    r, fc, _ = one_layer(conn_set(update), shared=False, x=X1)

    np.testing.assert_array_equal(r.conn_field(fc, update.steps), 1)
    totals = r.conn_field(fc, update.last_total)
    assert sorted(set(totals.tolist())) == sorted(set(X1.tolist()))


# --------------------------------------------------------------------------- #
# Shared weights
# --------------------------------------------------------------------------- #
def test_every_use_reads_the_same_weight() -> None:
    fc = px.layers.Dense(D_OUT, shared=True, bias=False, init=px.init.array(W))
    model = support.topology(lambda m, x: m.fc(x), fc=fc)
    r = support.Runner(support.network(model, input_shape=X.shape))

    for position in per_position(r, fc):
        support.close(position, W)


def test_step_runs_once_per_weight_not_once_per_use() -> None:
    update = CountUses()
    r, fc, _ = one_layer(conn_set(update), shared=True, x=X)

    for position in per_position(r, fc):
        support.close(position, W + POSITIONS)
    np.testing.assert_array_equal(r.conn_field(fc, update.steps), 1)


def test_fields_of_an_update_policy_belong_to_the_weight() -> None:
    """Read through any use, the update's fields are the weight's one value."""
    update = CountUses()
    r, fc, _ = one_layer(conn_set(update), shared=True, x=X)
    r.step(X, np.zeros((POSITIONS, D_OUT), f32))

    np.testing.assert_array_equal(r.conn_field(fc, update.steps), 2)
    support.close(
        r.conn_field(fc, update.last_total),
        np.full(POSITIONS * D_OUT * D_IN, POSITIONS, f32),
    )


def test_optimizer_state_on_a_shared_weight_matches_one_weight_seeing_every_use() -> (
    None
):
    """Adam on a shared layer equals Adam on the summed gradient of one weight."""
    import jax
    import jax.numpy as jnp

    lr, steps = 0.05, 3
    target = np.ones((POSITIONS, D_OUT), f32)
    adam = px.policies.Conn(
        forward=px.policies.WeightedSum(),
        backward=px.policies.WeightedSumBackward(),
        update=px.optim.adam(lr=lr, grad=px.policies.GRAD),
    )
    fc = px.layers.Dense(D_OUT, shared=True, bias=False, init=px.init.array(W))
    model = support.topology(lambda m, x: m.fc(x), fc=fc)
    r = support.Runner(support.network(model, input_shape=X.shape, conn_policy=adam))
    for _ in range(steps):
        r.step(X, target)

    # Adam written out, with its usual defaults.
    b1, b2, eps = 0.9, 0.999, 1e-8
    w, m, v = jnp.asarray(W), jnp.zeros_like(W), jnp.zeros_like(W)
    grad = jax.grad(lambda w: 0.5 * jnp.sum((X @ w.T - target) ** 2))
    for t in range(1, steps + 1):
        g = grad(w)
        m, v = b1 * m + (1 - b1) * g, b2 * v + (1 - b2) * g * g
        w = w - lr * (m / (1 - b1**t)) / (jnp.sqrt(v / (1 - b2**t)) + eps)
    for position in per_position(r, fc):
        support.close(position, w, rtol=1e-4, atol=1e-5)


def test_pruning_one_use_leaves_the_other_uses_and_the_weight() -> None:
    """Position 0's first input is loud, so the uses reading it are pruned."""
    x = X.copy()
    x[0, 0] = 1000.0
    policy = conn_set(None, prune=FromLoudSource())
    r, fc, _ = one_layer(policy, shared=True, x=x)

    assert len(r.pairs(fc)) == POSITIONS * D_OUT * D_IN - D_OUT
    out = r.predict(x).outputs
    support.close(out[0], W[:, 1:] @ x[0, 1:])  # the pruned uses no longer contribute
    support.close(out[1:], x[1:] @ W.T)  # the others still read the weight


@pytest.mark.open_question(
    "The untie trait is not designed: its name, its place in the step and its place in "
    "a policy set are guesses. Untying is one-way."
)
def test_untied_use_becomes_a_private_weight_and_stays_one() -> None:
    x = X.copy()
    x[0, 0] = 1000.0

    @px.policy
    class UntieFromLoudSource(px.UntieConnPolicy[px.Globals]):
        def predicate(
            self, u: px.UnitView, c: px.ConnView, cid: px.ConnIdx, g: px.Globals
        ) -> Flag:
            return u[px.ACTIVATION, c[px.FROM_ID, cid]] > 100.0

    policy = conn_set(Accumulate(), untie=UntieFromLoudSource())
    r, fc, _ = one_layer(policy, shared=True, x=x)
    quiet = X  # nothing is loud any more: nothing further is untied
    r.step(quiet, np.zeros((POSITIONS, D_OUT), f32))
    r.step(quiet, np.zeros((POSITIONS, D_OUT), f32))

    weights = per_position(r, fc)
    # Untying copied the value, so no connection was lost.
    assert len(r.pairs(fc)) == POSITIONS * D_OUT * D_IN
    # Positions 1 and 2 still share every weight.
    support.close(weights[1], weights[2])
    # Position 0's uses of input 0 are private now: they only see their own source, so
    # they have moved apart from the shared weight and stay apart.
    assert not np.allclose(weights[0][:, 0], weights[1][:, 0])
    # Its other uses are still shared.
    support.close(weights[0][:, 1:], weights[1][:, 1:])
