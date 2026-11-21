"""SKETCH -- does not run. A CNN with a residual layer on MNIST, written against the
interface we are designing, to see how it reads and where it has gaps.

Nothing imported from `px.layers`, `px.activations` or `px.policies` exists yet, and the
`Policy`-suffixed traits, typed fields and `px.Config` are proposals. Lines marked `GAP:` are
places where writing this exposed something the interface spec has not settled.

The three things a user specifies are laid out in order:
    1. topology       -- `MnistCNN`
    2. behaviour      -- the policies
    3. hints + config -- `hint=` on a layer, `px.Config` on the network
and `Mnist(px.Network)` is their union.

The base policies are written out here so the whole interface is visible in one file. In
practice they would be imported from `plastax.policies`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

import plastax as px

relu = px.activations.relu

# Every policy method is written for ONE unit or ONE connection and vmapped by the framework,
# so every index and value in the signatures below is a scalar. `u` and `c` are read-only
# views of all unit and all connection state; a policy returns a write for its one element.
# This network has no hyperparameters in global state, so `g` is the framework's own
# `px.Globals` (the step counter and the seed).
Scalar = Float[Array, ""]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_mnist(root: str = "/tmp/data") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Images (N, 28, 28, 1) float32 in [0, 1]; labels int64. torchvision is a reader only."""
    from torchvision.datasets import MNIST

    train = MNIST(root, train=True, download=True)
    test = MNIST(root, train=False, download=True)

    def images(d: MNIST) -> np.ndarray:
        return (d.data.numpy().astype(np.float32) / 255.0)[..., None]

    return (
        images(train),
        train.targets.numpy().astype(np.int64),
        images(test),
        test.targets.numpy().astype(np.int64),
    )


# --------------------------------------------------------------------------- #
# 1. Topology: the initial structure
# --------------------------------------------------------------------------- #
class MnistCNN(px.layers.Module):
    """conv -> relu -> [conv + skip] -> relu -> max pool -> dense."""

    def setup(self) -> None:
        # A layer is a set of connections. `shared=True` makes every kernel entry one shared
        # weight, used once per spatial position.
        self.conv1 = px.layers.Conv(16, 3, padding="same", shared=True)
        self.conv2 = px.layers.Conv(16, 3, padding="same", shared=True)
        # Pooling ships with its own policies, applied as a layer-wide override of the
        # network defaults: frozen identity-weight connections, and units that reduce by
        # max and send the gradient back to the input that won (`max_pool` below).
        self.pool = px.layers.MaxPool(2)
        # A hint: predict that the head is best run as a dense matmul.
        self.head = px.layers.Dense(10, hint=px.Hint(format=px.Format.DENSE))

    def __call__(self, x: px.layers.Units) -> px.layers.Units:
        # `x` is a handle to the input units, shape (28, 28, 1). Handles carry shapes, not data.
        h1 = relu(self.conv1(x))  # (28, 28, 16)
        # `a + b` means new summing units fed by identity connections: weight 1 (set here,
        # by the topology), created with `update=None`, so no update touches them. Here conv2's
        # units have no other reader and no activation, so the builder merges the sum into
        # their reduce: no new units in the built network, one relu over the sum.
        h2 = relu(self.conv2(h1) + h1)  # (28, 28, 16)
        p = self.pool(h2)  # (14, 14, 16)
        # A dense layer acts on the last axis, so the pooled grid is flattened first.
        # `flatten` only relabels the handle's shape; it creates nothing.
        # No activation on the head: its units carry the logits.
        return self.head(p.flatten())  # (14 * 14 * 16,) -> (10,)


# --------------------------------------------------------------------------- #
# 2. Behaviour: per-connection and per-unit policies
# --------------------------------------------------------------------------- #
# A field shared between policies is one module-level object passed to each of them.
GRAD = px.UnitField[np.float32]("grad_pre_act", default=0.0)  # dL/d(pre-activation)
SEED = px.UnitField[np.float32]("loss_seed", default=0.0)  # dL/d(activation), from the loss


@px.policy
class WeightedSum(px.policies.ForwardConnPolicy[Scalar, px.Globals]):
    """One connection's contribution to its destination."""

    def map(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: px.Globals,
    ) -> Scalar:
        # c[px.WEIGHT, cid] is the weight whether it is private or shared.
        return c[px.WEIGHT, cid] * u[px.ACTIVATION, src]


@px.policy
class SumThenActivate(px.policies.ForwardUnitPolicy[Scalar, px.Globals]):
    """A unit sums its incoming contributions and applies its activation."""

    combine = px.monoid.sum_
    # Set per unit by calling an activation on a handle: `relu(h)` is this policy with
    # `activation=relu`, and the backward half below with the same.
    activation: px.Activation = px.activations.identity
    # Declared here, so namespaced to this policy: no clash with another policy's `pre`.
    pre: px.UnitField[np.float32] = 0.0

    def apply(self, u: px.UnitView, i: px.UnitIdx, g: px.Globals, acc: Scalar) -> px.UnitWrite:
        return px.UnitWrite.of((self.pre, acc), (px.ACTIVATION, self.activation.fn(acc)))


@px.policy
class WeightedSumBackward(px.policies.BackwardConnPolicy[Scalar, px.Globals]):
    """One connection's contribution to its source's gradient."""

    def map(
        self,
        u: px.UnitView,
        src: px.UnitIdx,
        dst: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: px.Globals,
    ) -> Scalar:
        return c[px.WEIGHT, cid] * u[GRAD, dst]


@px.policy
class ActivationBackward(px.policies.BackwardUnitPolicy[Scalar, px.Globals]):
    """dL/da -> dL/dz through the unit's own activation derivative."""

    combine = px.monoid.sum_
    # The forward half's field, passed in, as any field one policy reads from another is.
    pre: px.UnitField[np.float32]
    # Set by the same call that sets the forward half's: `relu(h)` replaces the field named
    # `activation` in both unit halves of h's units, so the two always agree.
    activation: px.Activation = px.activations.identity

    def apply(self, u: px.UnitView, i: px.UnitIdx, g: px.Globals, acc: Scalar) -> px.UnitWrite:
        d = self.activation.derivative(u[self.pre, i], u[px.ACTIVATION, i])
        return px.UnitWrite.of((GRAD, (acc + u[SEED, i]) * d))


# An activation is applied to a handle: `relu(h)`. It replaces the field named `activation` in
# the forward and the backward unit half of h's units, and is refused if either half has no
# such field. Units with no activation applied keep the identity.
#
# PROPOSED (plans/20261121T191023Z-layer-policies.md, Q3): the field name is the contract.


# Max pooling, as `px.layers.MaxPool` ships it. Written out here to show the standard: no
# winner is stored. The forward reduce is a plain max, and the backward map of each pooling
# connection recomputes whether its source won by comparing activations. Tied inputs each
# receive the full gradient.
@px.policy
class MaxReduce(px.policies.ForwardUnitPolicy[Scalar, px.Globals]):
    combine = px.monoid.max_

    def apply(self, u: px.UnitView, i: px.UnitIdx, g: px.Globals, acc: Scalar) -> px.UnitWrite:
        return px.UnitWrite.of((px.ACTIVATION, acc))


@px.policy
class MaxPoolBackward(px.policies.BackwardConnPolicy[Scalar, px.Globals]):
    def map(
        self,
        u: px.UnitView,
        src: px.UnitIdx,
        dst: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: px.Globals,
    ) -> Scalar:
        won = u[px.ACTIVATION, src] == u[px.ACTIVATION, dst]
        return jnp.where(won, u[GRAD, dst], 0.0)


@px.policy
class PassThroughBackward(px.policies.BackwardUnitPolicy[Scalar, px.Globals]):
    """No activation, so dL/dz is dL/da."""

    combine = px.monoid.sum_

    def apply(self, u: px.UnitView, i: px.UnitIdx, g: px.Globals, acc: Scalar) -> px.UnitWrite:
        return px.UnitWrite.of((GRAD, acc + u[SEED, i]))


# What `px.layers.MaxPool` passes for its own connections and units, overriding the network.
# A layer takes the network's keywords. Its `forward` pair is self-contained: the `conn` half
# maps the pooling connections and the `unit` half reduces them at the pooled units.
#
# Its `backward` pair is not. The `conn` half is the map of the pooling connections, and what
# it returns is reduced at their sources, the convolution's units, by the convolution's
# backward unit half. The `unit` half reduces what the layer after the pool sends back. Both
# pairings are checked when the network is built (Q5).
#
# Pooling connections are frozen. There is no flag for that: `None` says this layer has no
# update, prune or growth, where leaving the keyword out would inherit the network's. Their
# weight of 1 is set by the topology (the `MaxPool` layer creates them with it), so
# `WeightedSum` passes the source's activation through unchanged.
max_pool = dict(
    forward=px.policies.Forward(conn=WeightedSum(), unit=MaxReduce()),
    backward=px.policies.Backward(conn=MaxPoolBackward(), unit=PassThroughBackward()),
    update=None,  # frozen: nothing may change, create or remove these connections
    prune=None,
    grow=None,
)
# A group of keywords reused across layers is a dict: `SomeLayer(..., **max_pool)`.


_sum = SumThenActivate()


# --------------------------------------------------------------------------- #
# 3. The network: topology + behaviour + hints and configuration
# --------------------------------------------------------------------------- #
class Mnist(px.Network):
    topology = MnistCNN()
    input_shape = (28, 28, 1)

    # The network's policies, by phase: the default for every connection and unit unless a
    # layer or a handle is given its own under the same keyword.
    forward = px.policies.Forward(conn=WeightedSum(), unit=_sum)
    backward = px.policies.Backward(
        conn=WeightedSumBackward(), unit=ActivationBackward(pre=_sum.pre)
    )
    # The optimizer supplies the connection update and the state only it reads: a per-use
    # contribution (GRAD[dst] * ACTIVATION[src]), summed over the uses of a weight, then one
    # Adam step.
    optimizer = px.optim.adam(lr=1e-3, grad=GRAD)
    loss = px.policies.SoftmaxCrossEntropyLossPolicy(seed=SEED)  # over the output units

    config = px.Config(
        format=px.Format.AUTO,  # per-layer hints override this
        propagation=px.Propagation.TOPOLOGICAL,
        representation_optimization=False,
        seed=0,
    )


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def main(steps: int = 20_000, eval_every: int = 2_000) -> None:
    xtr, ytr, xte, yte = load_mnist()

    # `init` materializes the network with its policies attached. The step is the policies
    # composed in phase order and compiled by the jit.
    state = Mnist.init(jax.random.PRNGKey(0))
    step = px.make_step(Mnist)
    # Evaluation is the forward policies alone. Forward policies can write state (here the
    # stored pre-activation), so there are two modes, both to be designed and documented:
    #   DISCARD  run forward, return the outputs, throw the writes away (read-only)
    #   KEEP     run forward and keep what it wrote, for networks whose state advances on
    #            every input, like the imprinting learner
    predict = px.make_step(Mnist, phases=px.Phases.FORWARD, writes=px.Writes.DISCARD)

    rng = np.random.default_rng(0)
    for t in range(steps):
        i = int(rng.integers(len(xtr)))
        # Batching is deferred, so this is online: one sample per step.
        inputs = px.StepInputs(inputs=xtr[i], targets=jax.nn.one_hot(ytr[i], 10))
        result = step(state, inputs)
        state = result.state

        if (t + 1) % eval_every == 0:
            correct = 0
            for x, y in zip(xte[:1000], yte[:1000], strict=True):
                # `outputs` has the shape the topology returned: the (10,) logits.
                out = predict(state, px.StepInputs(inputs=x)).outputs
                correct += int(jnp.argmax(out) == y)
            print(f"step {t + 1}: loss {float(result.loss):.4f}, test acc {correct / 1000:.3f}")


if __name__ == "__main__":
    main()
