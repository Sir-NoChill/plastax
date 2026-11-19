"""SKETCH -- does not run. Deep Rewiring (DeepR; Bellec et al. 2018) on parallel MNIST,
written against the interface we are designing. This is the sketch for connections being
destroyed and created at run time while the units stay fixed.

The problem and the network follow `phd/structure_search` in ejmejm/phd_research:
`experiments/set_parallel_mnist.ipynb` for the problem, and
`experiments/deep_r_progression/01_deep_r_2layer.ipynb` for the DeepR network.

Problem. A sample is T MNIST images drawn independently and concatenated (each pixel
standardized over the training set); task t's target is the label of image t. Every
`PERMUTE_PERIOD` steps one task, chosen at random, gets a new random mapping from digits to
labels. The ideal network is block-diagonal: each task's logits depend only on that task's
pixels. The loss is the sum of each task's softmax cross-entropy, one sample per step.

Network. 784*T inputs -> hidden LTU units -> 10*T logits, no biases. Each hidden unit starts
with `INPUT_FANIN` random inputs and is wired to exactly one logit.

DeepR, as in that notebook. Every weight takes a gradient step, an L1 pull toward zero and a
little noise. In the input layer, a connection whose weight changes sign is removed. This
sketch then also refills the layer at random (the notebook's first stage only prunes; DeepR
proper regrows, and this sketch is for both). The output layer takes the same update and
its structure never changes.

What this sketch exercises that the other two did not:
    - sparse initial wiring, random (`fan_in=`) and explicit (`pairs`)
    - a connection update that needs random noise
    - pruning and growth confined to a layer, with the layer's count conserved
    - a loss over groups of output units

The plain weighted-sum policies are imported from `plastax.policies`; they are the ones
written out in `mnist_cnn_residual.py`. Lines marked `GAP:` are places where writing this
exposed something the interface spec has not settled.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float

import plastax as px
from plastax.policies import GRAD, SEED, WeightedSum, WeightedSumBackward, summing

Scalar = Float[Array, ""]
Flag = Bool[Array, ""]

# --------------------------------------------------------------------------- #
# Structure: these fix shapes and counts, so they are static
# --------------------------------------------------------------------------- #
N_TASKS = 2
N_PIXELS = 28 * 28
N_CLASSES = 10
N_INPUTS = N_TASKS * N_PIXELS
N_OUTPUTS = N_TASKS * N_CLASSES

N_HIDDEN = 100  # split equally between the logits
INPUT_FANIN = 256  # inputs each hidden unit starts with
PROPOSALS_PER_STEP = 512  # growth candidates drawn per step in the input layer

PERMUTE_PERIOD = 2000 // N_TASKS  # steps between label changes


# --------------------------------------------------------------------------- #
# Data: the stream is ordinary host-side code
# --------------------------------------------------------------------------- #
def parallel_mnist(seed: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (inputs, labels) forever.

    Args:
        seed: generator seed for the images drawn and the label changes.

    Yields:
        inputs: (N_INPUTS,) float32, T standardized images concatenated.
        labels: (N_TASKS,) int64, each task's current label for its image.
    """
    from torchvision.datasets import MNIST

    mnist = MNIST("/tmp/data", train=True, download=True)
    images = mnist.data.numpy().astype(np.float32).reshape(-1, N_PIXELS) / 255.0
    std = images.std(0)
    images = (images - images.mean(0)) / np.where(std < 1e-6, 1.0, std)
    digits = mnist.targets.numpy()

    rng = np.random.default_rng(seed)
    label_maps = np.tile(np.arange(N_CLASSES), (N_TASKS, 1))  # (task, digit) -> label
    step = 0
    while True:
        step += 1
        if step % PERMUTE_PERIOD == 0:
            label_maps[rng.integers(N_TASKS)] = rng.permutation(N_CLASSES)
        index = rng.integers(len(images), size=N_TASKS)
        yield images[index].reshape(-1), label_maps[np.arange(N_TASKS), digits[index]]


# --------------------------------------------------------------------------- #
# Hyperparameters
# --------------------------------------------------------------------------- #
@dataclasses.dataclass(frozen=True)
class Params:
    """Numeric hyperparameters, read by the policies as `g.hp`."""

    lr: float = 2.0**-7
    l1: float = 1e-4  # pull of each weight toward zero
    temperature: float = 1e-7  # scale of the noise


@px.globals
class Globals:
    hp: Params


# --------------------------------------------------------------------------- #
# Behaviour: the DeepR connection
# --------------------------------------------------------------------------- #
@px.policy
class DeepRUpdate(px.UpdateConnPolicy[Scalar, Globals]):
    """w <- w - lr * (dL/dw + l1 * sign(w)) + sqrt(2 * lr * T) * noise.

    Records whether the step carried the weight across zero. Whether that removes the
    connection is the prune rule's business, so the same update serves both layers.
    """

    combine = px.monoid.sum_
    crossed: px.ConnField[np.bool_] = False  # this step moved the weight to the other sign

    def contribution(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: Globals,
    ) -> Scalar:
        return u[GRAD, dst] * u[px.ACTIVATION, src]  # dL/dw for this use of the weight

    def step(self, c: px.ConnView, cid: px.ConnIdx, g: Globals, total: Scalar) -> px.ConnWrite:
        hp = g.hp
        w = c[px.WEIGHT, cid]
        # A policy that needs randomness asks for it: this policy's own stream, for this
        # connection and this step.
        rng = self.rng(g, cid)
        noise = jnp.sqrt(2.0 * hp.lr * hp.temperature) * rng.normal()
        w_new = w - hp.lr * (total + hp.l1 * jnp.sign(w)) + noise
        # A new connection starts at exactly zero and is free to take either sign.
        return px.ConnWrite.of((px.WEIGHT, w_new), (self.crossed, w * w_new < 0.0))


@px.policy
class SignCrossPrune(px.PruneConnPolicy[Globals]):
    """Remove a connection whose weight has just changed sign."""

    crossed: px.ConnField[np.bool_]  # the update's field, passed in

    def predicate(self, u: px.UnitView, c: px.ConnView, cid: px.ConnIdx, g: Globals) -> Flag:
        return c[self.crossed, cid]


@px.policy
class RandomRegrow(px.ProposeAddConnPolicy[Globals]):
    """Refill the layer with connections at random places, starting at zero."""

    proposer = px.Proposer.PER_LAYER
    proposals_per_proposer = PROPOSALS_PER_STEP
    dedupe_live = True  # never a second connection between the same two units
    dedupe_step = True
    on_overflow = px.Overflow.DROP  # the layer is full: surplus proposals are discarded

    # Growth has two forms (owner). The network's `add_conn` is the default and may connect
    # any two units, as in the imprinting learner. A connection policy set may carry its own
    # (`add=` below), which overrides the default for the layer the set is attached to:
    # a new connection then goes from that layer's sources to that layer's destinations,
    # belongs to that set, and counts against that layer's slots.
    #
    # `layer` is a view of the layer's population: its sources and its destinations, the
    # unit indices the layer connected when it was built.
    def propose(
        self,
        u: px.UnitView,
        layer: px.LayerView,
        j: px.ProposalIdx,
        g: Globals,
    ) -> px.Proposal:
        rng = self.rng(g, j)  # this policy's stream for proposal j of this step
        return px.Proposal(
            src=layer.random_source(rng),
            dst=layer.random_destination(rng),
            score=rng.uniform(),  # every candidate is as good as any other
        )

    def init(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: Globals
    ) -> px.ConnWrite:
        # A connection created at run time has no layer initializer: its values come from
        # here.
        return px.ConnWrite.of((px.WEIGHT, 0.0))


_update = DeepRUpdate()

# The input layer: updated, pruned on a sign change, refilled.
rewired = px.policies.Conn(
    forward=WeightedSum(),
    backward=WeightedSumBackward(),
    update=_update,
    prune=SignCrossPrune(crossed=_update.crossed),
    add=RandomRegrow(),
)

# The output layer: the same update, structure fixed.
fixed = px.policies.Conn(
    forward=WeightedSum(),
    backward=WeightedSumBackward(),
    update=_update,
    prune=None,
    add=None,
)


# --------------------------------------------------------------------------- #
# Topology
# --------------------------------------------------------------------------- #
class ParallelMLP(px.layers.Module):
    """inputs -> sparse rewired layer -> LTU hidden units -> one logit each."""

    def setup(self) -> None:
        # Sparse initial wiring is structure. `fan_in` gives every hidden unit that many
        # random inputs. `capacity=INITIAL` gives the layer exactly as many connection slots
        # as it starts with connections: pruning frees a slot and growth can only fill free
        # slots, so the count is conserved without a counter in global state.
        self.hidden = px.layers.Dense(
            N_HIDDEN,
            connectivity=px.layers.sparse(fan_in=INPUT_FANIN, capacity=px.Capacity.INITIAL),
            init=px.init.lecun_uniform,  # the topology sets the initial weights
            bias=False,
            conn_policy=rewired,
            hint=px.Hint(format=px.Format.COO),
        )
        # Explicit initial wiring: hidden unit i feeds logit i // (N_HIDDEN // N_OUTPUTS).
        # Plain numpy, since this is build-time code.
        hidden = np.arange(N_HIDDEN)
        self.logits = px.layers.Dense(
            N_OUTPUTS,
            connectivity=px.layers.pairs(src=hidden, dst=hidden // (N_HIDDEN // N_OUTPUTS)),
            init=px.init.lecun_uniform,
            bias=False,
            conn_policy=fixed,
            hint=px.Hint(format=px.Format.COO),
        )

    def __call__(self, x: px.layers.Units) -> px.layers.Units:
        # LTU: 0/1 step forward, sigmoid derivative backward. An activation is a function
        # and its derivative, so one whose derivative is not the true one fits as it is.
        h = px.activations.ltu(self.hidden(x))
        # The groups a loss sees are given by the shape of the handle. The logits are
        # reshaped to (task, class) and the network's one softmax loss runs over the last
        # axis, once per task. `reshape` relabels the handle in row-major order, so logit
        # t * N_CLASSES + c becomes (t, c), and creates nothing. It is checked here, where
        # it is written: the new shape must hold exactly the handle's units.
        return self.logits(h).reshape(N_TASKS, N_CLASSES)


# --------------------------------------------------------------------------- #
# The network
# --------------------------------------------------------------------------- #
class DeepR(px.Network):
    topology = ParallelMLP()
    input_shape = (N_INPUTS,)

    conn_policy = None  # asserts that every layer names its own set
    add_conn = None  # no network-wide growth: only the input layer's own
    unit_policy = summing
    # One softmax cross-entropy per task, summed over the tasks.
    loss = px.SoftmaxCrossEntropyLossPolicy(seed=SEED, axis=-1)

    config = px.Config(
        # Growth stays inside a layer, so the levels never change.
        propagation=px.Propagation.TOPOLOGICAL,
        representation_optimization=False,
        seed=0,
    )


# --------------------------------------------------------------------------- #
# Training: online, each sample scored before it is learned from
# --------------------------------------------------------------------------- #
def main(steps: int = 500_000, window: int = 2_000) -> None:
    state = DeepR.init(jax.random.PRNGKey(0), globals=Globals(hp=Params()))
    # One step is: forward, loss, backward, update (the weights move), prune (the ones that
    # changed sign), add (refill). A connection removed in a step is replaced in that step.
    step = px.make_step(DeepR)

    loss_sum, correct = 0.0, 0.0
    for t, (x, labels) in zip(range(steps), parallel_mnist(seed=0), strict=False):
        targets = jax.nn.one_hot(labels, N_CLASSES)  # (task, class), the shape of the outputs
        result = step(state, px.StepInputs(inputs=x, targets=targets))
        state = result.state

        # `result.outputs` has the shape the topology returned: (task, class) logits, from
        # this step's forward pass, before the update.
        loss_sum += float(result.loss)
        correct += float(jnp.mean(jnp.argmax(result.outputs, axis=-1) == labels))
        if (t + 1) % window == 0:
            print(f"step {t + 1}: loss {loss_sum / window:.4f}, accuracy {correct / window:.3f}")
            loss_sum, correct = 0.0, 0.0


if __name__ == "__main__":
    main()
