"""The DeepR rule from `docs/design/sketches/deepr_multimnist.py`.

Every weight takes a gradient step, an L1 pull toward zero and a little noise:

    w <- w - lr * (dL/dw + l1 * sign(w)) + sqrt(2 * lr * T) * noise

In the rewired layer a connection whose weight changes sign is removed and the layer is
refilled at random with connections that start at zero. The other layer takes the same
update and keeps its structure.

The policies are the sketch's, copied here so the test stands alone.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float

import plastax as px

from . import support

Scalar = Float[Array, ""]
Flag = Bool[Array, ""]
f32 = np.float32

X = np.array([1.0, -2.0, 3.0, 0.5], f32)
T = np.array([0.5, -1.0], f32)
W1 = (np.arange(12, dtype=f32).reshape(3, 4) - 5.0) / 10.0
W2 = np.array([[0.5, -1.0, 2.0], [1.5, 0.25, -0.75]], f32)


@dataclasses.dataclass(frozen=True)
class Params:
    lr: float = 1e-3
    l1: float = 1e-2
    temperature: float = 0.0


@px.globals
class Globals:
    hp: Params


@px.policy
class DeepRUpdate(px.UpdateConnPolicy[Scalar, Globals]):
    combine = px.monoid.sum_
    crossed: px.ConnField[np.bool_] = False

    def contribution(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: Globals,
    ) -> Scalar:
        return u[px.policies.GRAD, dst] * u[px.ACTIVATION, src]

    def step(self, c: px.ConnView, cid: px.ConnIdx, g: Globals, total: Scalar) -> Any:
        hp = g.hp
        w = c[px.WEIGHT, cid]
        noise = jnp.sqrt(2.0 * hp.lr * hp.temperature) * self.rng(g, cid).normal()
        w_new = w - hp.lr * (total + hp.l1 * jnp.sign(w)) + noise
        return px.ConnWrite.of((px.WEIGHT, w_new), (self.crossed, w * w_new < 0.0))


@px.policy
class SignCrossPrune(px.PruneConnPolicy[Globals]):
    crossed: px.ConnField[np.bool_]

    def predicate(
        self, u: px.UnitView, c: px.ConnView, cid: px.ConnIdx, g: Globals
    ) -> Flag:
        return c[self.crossed, cid]


@px.policy
class RandomRegrow(px.ProposeAddConnPolicy[Globals]):
    proposer = px.Proposer.PER_LAYER
    proposals_per_proposer = 256
    dedupe_live = True
    dedupe_step = True
    on_overflow = px.Overflow.DROP

    def propose(
        self, u: px.UnitView, layer: px.LayerView, j: px.ProposalIdx, g: Globals
    ) -> Any:
        rng = self.rng(g, j)
        return px.Proposal(
            src=layer.random_source(rng),
            dst=layer.random_destination(rng),
            score=rng.uniform(),
        )

    def init(self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: Globals) -> Any:
        return px.ConnWrite.of((px.WEIGHT, 0.0))


def policy_sets(*, regrow: bool) -> tuple[Any, Any]:
    """(rewired, fixed): the sketch's two connection sets, sharing one update policy."""
    p = px.policies
    update = DeepRUpdate()
    rewired = p.Conn(
        forward=p.WeightedSum(),
        backward=p.WeightedSumBackward(),
        update=update,
        prune=SignCrossPrune(crossed=update.crossed),
        add=RandomRegrow() if regrow else None,
    )
    fixed = p.Conn(
        forward=p.WeightedSum(),
        backward=p.WeightedSumBackward(),
        update=update,
        prune=None,
        add=None,
    )
    return rewired, fixed


def two_layers(
    hp: Params, w1: np.ndarray = W1, seed: int = 0
) -> tuple[support.Runner, Any, Any]:
    """x -> hidden (tanh, rewired without regrowth) -> out (fixed structure)."""
    tanh = px.activations.tanh
    rewired, fixed = policy_sets(regrow=False)
    hidden = px.layers.Dense(3, bias=False, init=px.init.array(w1), conn_policy=rewired)
    out = px.layers.Dense(2, bias=False, init=px.init.array(W2), conn_policy=fixed)
    model = support.topology(
        lambda m, x: m.out(tanh(m.hidden(x))), hidden=hidden, out=out
    )
    net = support.network(
        model,
        input_shape=(4,),
        conn_policy=None,
        add_conn=None,
        config=px.Config(propagation=px.Propagation.TOPOLOGICAL, seed=seed),
    )
    return support.Runner(net, globals=Globals(hp=hp)), hidden, out


def expected_update(hp: Params, w1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Both layers after one noiseless step, before any pruning."""

    def loss(params: tuple[jax.Array, jax.Array]) -> jax.Array:
        a, b = params
        return 0.5 * jnp.sum((b @ jnp.tanh(a @ X) - T) ** 2)

    g1, g2 = jax.grad(loss)((jnp.asarray(w1), jnp.asarray(W2)))
    new1 = w1 - hp.lr * (np.asarray(g1) + hp.l1 * np.sign(w1))
    new2 = W2 - hp.lr * (np.asarray(g2) + hp.l1 * np.sign(W2))
    return new1, new2


# --------------------------------------------------------------------------- #
# The update
# --------------------------------------------------------------------------- #
def test_noiseless_update_is_a_gradient_step_with_an_l1_pull() -> None:
    hp = Params(lr=1e-3, l1=1e-2, temperature=0.0)
    r, hidden, out = two_layers(hp)
    r.step(X, T)

    new1, new2 = expected_update(hp, W1)
    assert not (W1 * new1 < 0).any()  # nothing crosses zero at this step size
    support.close(r.weights(hidden), new1)
    support.close(r.weights(out), new2)


def test_noise_has_the_scale_the_temperature_sets() -> None:
    """With zero input the gradient is zero, so a weight moves by the noise alone."""
    hp = Params(lr=0.5, l1=0.0, temperature=1e-4)
    fan_in, units = 32, 64
    rewired, _ = policy_sets(regrow=False)
    fc = px.layers.Dense(
        units, bias=False, init=px.init.constant(1.0), conn_policy=rewired
    )
    model = support.topology(lambda m, x: m.fc(x), fc=fc)
    net = support.network(model, input_shape=(fan_in,), conn_policy=None, add_conn=None)
    r = support.Runner(net, globals=Globals(hp=hp))
    r.step(np.zeros(fan_in, f32), np.zeros(units, f32))

    moved = r.conn_field(fc, px.WEIGHT) - 1.0
    scale = np.sqrt(2.0 * hp.lr * hp.temperature)  # 0.01
    assert len(moved) == units * fan_in  # nothing came near zero
    assert abs(moved.mean()) < 0.1 * scale
    support.close(moved.std(), scale, rtol=0.1)


def test_noise_replays_for_the_same_seed() -> None:
    hp = Params(lr=0.5, l1=0.0, temperature=1e-4)

    def weights(seed: int) -> np.ndarray:
        r, hidden, _ = two_layers(hp, w1=np.abs(W1) + 1.0, seed=seed)
        r.step(X, T)
        return r.weights(hidden)

    np.testing.assert_array_equal(weights(1), weights(1))
    assert not np.array_equal(weights(1), weights(2))


# --------------------------------------------------------------------------- #
# Sign changes
# --------------------------------------------------------------------------- #
def test_connection_whose_weight_changes_sign_is_removed_in_the_rewired_layer() -> None:
    hp = Params(lr=1.0, l1=0.0, temperature=0.0)
    r, hidden, _ = two_layers(hp)
    r.step(X, T)

    new1, _ = expected_update(hp, W1)
    crossed = W1 * new1 < 0
    assert crossed.any()  # the case is not vacuous
    assert len(r.pairs(hidden)) == int((~crossed).sum())
    support.close(r.weights(hidden), np.where(crossed, 0.0, new1), rtol=1e-4)


def test_layer_with_no_prune_policy_keeps_a_weight_that_changes_sign() -> None:
    hp = Params(lr=3.0, l1=0.0, temperature=0.0)
    r, _, out = two_layers(hp)
    r.step(X, T)

    _, new2 = expected_update(hp, W1)
    assert (W2 * new2 < 0).any()  # the case is not vacuous
    assert len(r.pairs(out)) == W2.size
    support.close(r.weights(out), new2, rtol=1e-4)


def test_weight_starting_at_zero_may_take_either_sign() -> None:
    """A regrown connection starts at exactly zero; its first move is not a crossing."""
    hp = Params(lr=0.05, l1=0.0, temperature=0.0)
    zeros = np.zeros_like(W1)
    r, hidden, _ = two_layers(hp, w1=zeros)
    r.step(X, T)

    new1, _ = expected_update(hp, zeros)
    assert (new1 > 0).any() and (new1 < 0).any()
    assert len(r.pairs(hidden)) == zeros.size
    support.close(r.weights(hidden), new1, rtol=1e-4)


# --------------------------------------------------------------------------- #
# The whole rule
# --------------------------------------------------------------------------- #
def test_rewired_layer_keeps_its_connection_count_while_it_rewires() -> None:
    n_in, n_hidden, fan_in = 16, 8, 4
    hp = Params(lr=0.05, l1=0.01, temperature=1e-4)
    rewired, fixed = policy_sets(regrow=True)
    hidden = px.layers.Dense(
        n_hidden,
        connectivity=px.layers.sparse(fan_in=fan_in, capacity=px.Capacity.INITIAL),
        bias=False,
        init=px.init.lecun_uniform(),
        conn_policy=rewired,
        hint=px.Hint(format=px.Format.COO),
    )
    out = px.layers.Dense(2, bias=False, conn_policy=fixed)
    model = support.topology(
        lambda m, x: m.out(px.activations.tanh(m.hidden(x))), hidden=hidden, out=out
    )
    net = support.network(model, input_shape=(n_in,), conn_policy=None, add_conn=None)
    r = support.Runner(net, globals=Globals(hp=hp))
    initial = set(r.pairs(hidden))
    assert len(initial) == n_hidden * fan_in

    rng = np.random.default_rng(0)
    for _ in range(50):
        r.step(rng.normal(size=n_in).astype(f32), rng.normal(size=2).astype(f32))
        pairs = r.pairs(hidden)
        assert len(pairs) == n_hidden * fan_in
        assert len(set(pairs)) == len(pairs)

    assert set(r.pairs(hidden)) != initial  # it did rewire
    assert len(r.pairs(out)) == n_hidden * 2
