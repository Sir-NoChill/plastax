"""SKETCH -- does not run. A two-layer transformer on a toy task, written against the
interface we are designing. This is the sketch expected to stress the interface most.

Task. A sequence of L random tokens goes in; the same sequence reversed comes out. Position i
must attend to position L-1-i, so the task cannot be done without attention.

The question this sketch answers is what attention is, unit by unit. The answer used here
needs one new kind of unit and nothing else:

    a product unit    has exactly two incoming connections and outputs their product

Everything in a transformer is then sums, products and activations:

    scores    s[i, j] = sum_c q[i, c] * k[j, c]       a product unit per (i, j, c), then a sum
    softmax   a = exp(s) / sum_j exp(s)               exp activation, a sum, a reciprocal
                                                      activation, a product
    mixing    o[i, c] = sum_j a[i, j] * v[j, c]       a product unit per (i, j, c), then a sum
    layer norm                                        means, a subtraction, products, an
                                                      rsqrt activation, a learned scale

The topology is written with ordinary array notation on handles (`*`, `+`, `-`, `.sum`,
`.mean`, indexing with `None`). Each of those is a build-time operation that creates units
and frozen connections; none of them touches data.

The cost, stated plainly: one attention layer is about 2 * L * L * d product units (1024
each for scores and mixing at L = 8, d = 16). Unit by unit this is what attention is. Running
it as two matrix products is the job of a future attention layer builder, noted below.

The transformer is a distant use case for plastax. This sketch exists to see what it brings
to the surface.

The weighted-sum policies are imported from `plastax.policies`; they are the ones written out
in `mnist_cnn_residual.py`. Lines marked `GAP:` are places where writing this exposed
something the interface spec has not settled.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

import plastax as px
from plastax.layers import Units
from plastax.policies import (
    GRAD,
    SEED,
    ActivationBackward,
    PassThroughBackward,
    SumThenActivate,
    WeightedSum,
    WeightedSumBackward,
)

relu = px.activations.relu
exp = px.activations.exp  # derivative: y
reciprocal = px.activations.reciprocal  # derivative: -y^2
rsqrt = px.activations.rsqrt(eps=1e-5)  # 1 / sqrt(x + eps); derivative: -y^3 / 2

Scalar = Float[Array, ""]
f32 = np.float32

# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #
SEQ_LEN = 8
VOCAB = 16
D_MODEL = 16
D_HEAD = 16  # one head, to keep the shapes readable; more heads is one more axis
D_MLP = 4 * D_MODEL
N_LAYERS = 2


# --------------------------------------------------------------------------- #
# Behaviour: the product unit
# --------------------------------------------------------------------------- #
# What a connection hands a product unit: (left factor, right factor). A connection fills
# its own side and puts 1 in the other, and the unit multiplies each side separately.
Factors = tuple[Scalar, Scalar]


@px.policy
class ProductUnit(px.policies.ForwardUnitPolicy[Factors, px.Globals]):
    """Outputs left * right, and keeps both for the backward pass."""

    combine = (px.monoid.prod, px.monoid.prod)
    left: px.UnitField[f32] = 1.0
    right: px.UnitField[f32] = 1.0

    def apply(
        self, u: px.UnitView, i: px.UnitIdx, g: px.Globals, acc: Factors
    ) -> px.UnitWrite:
        left, right = acc
        return px.UnitWrite.of(
            (self.left, left), (self.right, right), (px.ACTIVATION, left * right)
        )


@px.policy
class Factor(px.policies.ForwardConnPolicy[Factors, px.Globals]):
    """One of the two inputs of a product unit. The connection's weight is not used."""

    is_left: bool  # static: which side this connection fills

    def map(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: px.Globals,
    ) -> Factors:
        a = u[px.ACTIVATION, src]
        one = jnp.float32(1.0)
        return (a, one) if self.is_left else (one, a)  # a Python branch: resolved at trace


@px.policy
class FactorBackward(px.policies.BackwardConnPolicy[Scalar, px.Globals]):
    """d(left * right) / d(this side) is the other side, which the product unit kept."""

    other: px.UnitField[f32]  # the product unit's field for the opposite side, passed in

    def map(
        self,
        u: px.UnitView,
        src: px.UnitIdx,
        dst: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: px.Globals,
    ) -> Scalar:
        return u[GRAD, dst] * u[self.other, dst]


_product = ProductUnit()
# Frozen: nothing may change, create or remove these. `None` says so; leaving a keyword out
# would inherit the network's.
frozen = dict(update=None, prune=None, grow=None)


def multiply(a: Units, b: Units) -> Units:
    """What `a * b` on two handles does, written with the two primitives.

    A primitive creates one kind of element, so it takes that half of each phase directly:
    unit halves for `CreateUnits`, connection halves for `CreateConnections`.
    """
    a, b = px.layers.broadcast(a, b)
    out = px.layers.CreateUnits(forward=_product, backward=PassThroughBackward(), **frozen)(a)
    px.layers.CreateConnections(
        forward=Factor(is_left=True), backward=FactorBackward(other=_product.right), **frozen
    )(a, out)
    px.layers.CreateConnections(
        forward=Factor(is_left=False), backward=FactorBackward(other=_product.left), **frozen
    )(b, out)
    return out


# `a * b` on two handles broadcasts the two shapes, creates one product unit per element,
# and connects each to its element of `a` as its left factor and of `b` as its right.
# `multiply` and its policies would ship with plastax, as pooling's do.
#
# The product unit is in the standard monoid form: its accumulator is a pair, each side
# combined by the `prod` monoid plastax already has, and a struct of monoids is a monoid.
#
# Product units have a different forward accumulator (`Factors`) from every other unit here
# (`Scalar`). The framework handles that with one more monoid: the network's accumulator is
# the product of its policies' accumulators, here (sum, (prod, prod)). A connection fills
# its own part and leaves the identity in the rest; a unit reads its own part.
#
# GAP: the alternative to the product unit is a connection whose weight is a reference to a
# unit's activation, in the shape of a shared weight (a connection row referencing a weight
# stored elsewhere). It needs half the units. See "Weights as references" in the design
# document; not adopted here.


# --------------------------------------------------------------------------- #
# Topology
# --------------------------------------------------------------------------- #
def softmax(s: Units, axis: int) -> Units:
    """exp(s) / sum(exp(s)) along `axis`, from an activation, a sum and a product.

    A shipped version would first subtract the maximum along `axis` for numerical safety.
    That needs a max-reducing unit (as in pooling) whose connections pass no gradient back.
    """
    e = exp(s)  # sets the activation of s's units
    return e * reciprocal(e.sum(axis=axis, keepdims=True))


class LayerNorm(px.layers.Module):
    """Normalise each position's vector to zero mean and unit variance, then rescale."""

    def setup(self) -> None:
        # One learned weight per channel and a bias, the same at every position.
        self.rescale = px.layers.Scale(shared=True, bias=True)

    def __call__(self, x: Units) -> Units:  # (L, D)
        # `x - m` is a sum with frozen weights +1 and -1; `.mean` is a sum with frozen
        # weights 1/D. The topology sets those weights.
        centred = x - x.mean(axis=-1, keepdims=True)
        variance = (centred * centred).mean(axis=-1, keepdims=True)
        return self.rescale(centred * rsqrt(variance))


class Attention(px.layers.Module):
    """Single-head self-attention."""

    def setup(self) -> None:
        # A `Dense` acts on the last axis of its handle, once per position along the leading
        # axes, and `shared=True` makes the positions use the same weights (as a
        # convolution's positions do).
        self.query = px.layers.Dense(D_HEAD, shared=True, bias=False)
        self.key = px.layers.Dense(D_HEAD, shared=True, bias=False)
        self.value = px.layers.Dense(D_HEAD, shared=True, bias=False)
        self.out = px.layers.Dense(D_MODEL, shared=True, bias=False)

    def __call__(self, x: Units) -> Units:  # (L, D)
        q, k, v = self.query(x), self.key(x), self.value(x)  # each (L, H)

        # s[i, j] = sum_c q[i, c] * k[j, c] / sqrt(H). Indexing with `None` adds an axis to a
        # handle; it creates nothing. `scale` is the frozen weight of the sum's connections.
        products = q[:, None, :] * k[None, :, :]  # (L, L, H) product units
        scores = products.sum(axis=-1, scale=D_HEAD**-0.5)  # (L, L)

        attention = softmax(scores, axis=-1)  # (L, L): how much i reads from j

        # o[i, c] = sum_j a[i, j] * v[j, c]
        mixed = (attention[:, :, None] * v[None, :, :]).sum(axis=1)  # (L, H)
        return self.out(mixed)

    # Unit by unit, the two lines above are 2 * L * L * H product units. In a dense format
    # they are two matrix products. The framework will not try to recognise that; the route
    # is an attention layer builder (`px.layers.Attention`) that creates these units and
    # knows it can run them as tensor operations. If an attention layer started to change
    # structure it would have to be split back into units. This is far down the road.


class Block(px.layers.Module):
    def setup(self) -> None:
        self.norm1 = LayerNorm()
        self.attention = Attention()
        self.norm2 = LayerNorm()
        self.fc1 = px.layers.Dense(D_MLP, shared=True)
        self.fc2 = px.layers.Dense(D_MODEL, shared=True)

    def __call__(self, x: Units) -> Units:  # (L, D)
        # `a + b` always means new summing units, and the result is a new handle. `x` is
        # read by the norm, by the attention and by the next block, so the sum must not
        # change `x` itself. When the network is built, where one operand has no other
        # reader and no activation, the builder merges the sum into that operand's reduce.
        # Here the attention's and the MLP's outputs are such operands, so neither line
        # below adds a level of units.
        x = x + self.attention(self.norm1(x))
        x = x + self.fc2(relu(self.fc1(self.norm2(x))))
        return x


class Transformer(px.layers.Module):
    def setup(self) -> None:
        # The input is one-hot, so a dense layer shared across positions is the token
        # embedding. The position embedding is a private bias on each of the (L, D) units: a
        # bias is a learned connection from a unit that is always 1.
        self.embed = px.layers.Dense(D_MODEL, shared=True, bias=False)
        self.position = px.layers.Bias(shared=False)
        self.blocks = [Block() for _ in range(N_LAYERS)]
        self.norm = LayerNorm()
        self.unembed = px.layers.Dense(VOCAB, shared=True)

    def __call__(self, tokens: Units) -> Units:  # (L, VOCAB) one-hot
        x = self.position(self.embed(tokens))  # (L, D)
        for block in self.blocks:
            x = block(x)
        return self.unembed(self.norm(x))  # (L, VOCAB) logits


# --------------------------------------------------------------------------- #
# The network
# --------------------------------------------------------------------------- #
_sum = SumThenActivate()


class Reverser(px.Network):
    topology = Transformer()
    input_shape = (SEQ_LEN, VOCAB)

    # The defaults: every layer's connections are learned, every unit sums and applies its
    # activation. The handle operations (`*`, `+`, `-`, `.sum`, `.mean`) create their
    # elements frozen, with their own policies.
    forward = px.policies.Forward(conn=WeightedSum(), unit=_sum)
    backward = px.policies.Backward(
        conn=WeightedSumBackward(), unit=ActivationBackward(pre=_sum.pre)
    )
    # A shared weight is used once per position. The optimizer's update sums the per-use
    # contributions (GRAD[dst] * ACTIVATION[src]) and takes one Adam step.
    optimizer = px.optim.adam(lr=1e-3, grad=GRAD)
    # The logits are (position, token): one softmax per position, summed over positions.
    loss = px.policies.SoftmaxCrossEntropyLossPolicy(seed=SEED, axis=-1)

    config = px.Config(
        format=px.Format.AUTO,  # nothing here changes structure: a candidate for dense
        propagation=px.Propagation.TOPOLOGICAL,
        representation_optimization=False,
        seed=0,
    )


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def main(steps: int = 50_000, window: int = 1_000) -> None:
    state = Reverser.init(jax.random.PRNGKey(0))
    step = px.make_step(Reverser)

    rng = np.random.default_rng(0)
    loss_sum, correct = 0.0, 0.0
    for t in range(steps):
        tokens = rng.integers(VOCAB, size=SEQ_LEN)
        inputs = px.StepInputs(
            inputs=jax.nn.one_hot(tokens, VOCAB),
            targets=jax.nn.one_hot(tokens[::-1].copy(), VOCAB),
        )
        result = step(state, inputs)
        state = result.state

        loss_sum += float(result.loss)
        correct += float(jnp.mean(jnp.argmax(result.outputs, axis=-1) == tokens[::-1]))
        if (t + 1) % window == 0:
            print(f"step {t + 1}: loss {loss_sum / window:.4f}, accuracy {correct / window:.3f}")
            loss_sum, correct = 0.0, 0.0


if __name__ == "__main__":
    main()
