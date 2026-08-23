"""Dense MLP on MNIST, two ways: plastax optim.sgd vs jax+optax, online SGD.

Both nets share architecture and initial weights, so plain SGD makes them track
to float32 precision -- the reference-oracle parity demo for plastax.optim.sgd
(Track A.1, ECOSYSTEM_ROADMAP.md). plastax runs online (one sample per step,
host-driven) and is much slower than the batched jax reference; the point is
that the weight updates agree and both learn, not throughput.

Data: ``torchvision.datasets.MNIST`` (a dev dependency) reads/downloads MNIST
under ``$root`` (default /tmp/data) on first use -- only its dataset reader is
used, no torch compute.

Run:  uv run python examples/mnist_sgd.py --steps 4000
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
from mlp_xor import GradPreAct, LossGrad, MSELoss, SigmoidBackward, SigmoidForward

import plastax as px

Weights = list[np.ndarray]


# --------------------------------------------------------------------------- #
# MNIST loading (torchvision dataset reader; a dev/example dependency)
# --------------------------------------------------------------------------- #
def load_mnist(
    root: str = "/tmp/data",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST via torchvision (downloads on first use).

    Returns (train_x, train_y, test_x, test_y): images (N, 28, 28) float32 in
    [0, 1], labels int64. Only torchvision's dataset reader is used, no torch
    compute.
    """
    from torchvision.datasets import MNIST

    train = MNIST(root, train=True, download=True)
    test = MNIST(root, train=False, download=True)
    xtr = train.data.numpy().astype(np.float32) / 255.0
    ytr = train.targets.numpy().astype(np.int64)
    xte = test.data.numpy().astype(np.float32) / 255.0
    yte = test.targets.numpy().astype(np.int64)
    return xtr, ytr, xte, yte


def preprocess(images: np.ndarray, pool: int) -> np.ndarray:
    """Average-pool 28x28 by ``pool`` and flatten, shrinking the input layer."""
    if pool > 1:
        n, h, w = images.shape
        hh, ww = h // pool, w // pool
        images = images[:, : hh * pool, : ww * pool]
        images = images.reshape(n, hh, pool, ww, pool).mean(axis=(2, 4))
    return images.reshape(images.shape[0], -1).astype(np.float32)


# --------------------------------------------------------------------------- #
# The two implementations, from shared init weights
# --------------------------------------------------------------------------- #
def glorot_weights(sizes: list[int], rng: np.random.Generator) -> Weights:
    """One Glorot-uniform matrix per dense layer."""
    out: Weights = []
    for a, b in zip(sizes[:-1], sizes[1:], strict=True):
        limit = float(np.sqrt(6.0 / (a + b)))
        out.append(rng.uniform(-limit, limit, size=(a, b)).astype(np.float32))
    return out


def _const(mat: np.ndarray) -> Callable[[jax.Array, tuple[int, ...]], jax.Array]:
    def init(key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
        del key, shape
        return jnp.asarray(mat)

    return init


def _mlp_class(lr: float, *, train: bool) -> type[px.Network[None]]:
    """A plastax sigmoid-MLP Network class; forward-only when ``train`` is False."""
    if train:

        class _Train(px.Network[None]):
            forward_pass = SigmoidForward()
            backward_pass = SigmoidBackward()
            loss = MSELoss()
            update_conn = px.optim.sgd(lr, GradPreAct).update_conn()
            extra_unit_fields = (GradPreAct, LossGrad)
            propagation = px.Propagation.TOPOLOGICAL

        return _Train

    class _Eval(px.Network[None]):
        forward_pass = SigmoidForward()
        extra_unit_fields = (GradPreAct, LossGrad)
        propagation = px.Propagation.TOPOLOGICAL

    return _Eval


def build_plastax(
    sizes: list[int], weights: Weights, lr: float
) -> tuple[type[px.Network[None]], px.NetworkStatic, px.NetworkState[None]]:
    cls = _mlp_class(lr, train=True)
    blocks = [px.topology.input_units(sizes[0])]
    for i, w in enumerate(weights):
        blocks.append(px.topology.dense(sizes[i], sizes[i + 1], init=_const(w)))
    topo = px.topology.sequential(*blocks)
    # Every dense layer inits via _const(w), so the shared numpy weights are the
    # sole source of truth and from_topology's key seeds nothing observable. The
    # API still requires a valid key (sequential splits it per block), so pass a
    # fixed dummy rather than thread args.seed through inertly.
    static, state = px.NetworkBuilder.from_topology(
        cls, topo, jax.random.PRNGKey(0), globals_=None
    )
    return cls, static, state


def plastax_weights(state: px.NetworkState[None], sizes: list[int]) -> Weights:
    """Reassemble live edge weights into per-layer matrices via unit-id ranges."""
    offsets = np.cumsum([0, *sizes])
    mats = [
        np.zeros((sizes[i], sizes[i + 1]), np.float32) for i in range(len(sizes) - 1)
    ]
    for bucket in state.conns:
        live = ~np.asarray(bucket[px.DEAD.name])
        froms = np.asarray(bucket[px.FROM_ID.name])[live]
        tos = np.asarray(bucket[px.TO_ID.name])[live]
        vals = np.asarray(bucket[px.WEIGHT.name])[live]
        for f, t, v in zip(froms, tos, vals, strict=True):
            layer = int(np.searchsorted(offsets, t, side="right") - 2)
            mats[layer][f - offsets[layer], t - offsets[layer + 1]] = v
    return mats


def jax_forward(weights: Weights, x: jax.Array) -> jax.Array:
    a = x
    for w in weights:
        a = jax.nn.sigmoid(a @ jnp.asarray(w))
    return a


def accuracy(weights: Weights, x: np.ndarray, y: np.ndarray) -> float:
    pred = np.asarray(jnp.argmax(jax_forward(weights, jnp.asarray(x)), axis=1))
    return float(np.mean(pred == y))


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=4000, help="online SGD steps")
    parser.add_argument(
        "--pool", type=int, default=2, help="avg-pool factor (28//pool)"
    )
    parser.add_argument("--hidden", type=int, default=32, help="hidden units")
    parser.add_argument("--lr", type=float, default=0.5, help="learning rate")
    parser.add_argument("--eval-n", type=int, default=2000, help="test images scored")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    xtr, ytr, xte, yte = load_mnist()
    xtr = preprocess(xtr, args.pool)
    xte = preprocess(xte, args.pool)[: args.eval_n]
    yte = yte[: args.eval_n]
    n_in = xtr.shape[1]
    sizes = [n_in, args.hidden, 10]
    print(f"MLP {sizes}  lr={args.lr}  online steps={args.steps}  (pool={args.pool})")

    weights = glorot_weights(sizes, rng)

    # plastax
    cls, static, state = build_plastax(sizes, weights, args.lr)
    step = px.make_step(cls, static)

    # jax + optax reference (identical init)
    params = {f"W{i}": jnp.asarray(w) for i, w in enumerate(weights)}
    opt = optax.sgd(args.lr)
    opt_state = opt.init(params)

    def loss_fn(p: dict[str, jax.Array], x: jax.Array, y: jax.Array) -> jax.Array:
        diff = jax_forward([p[f"W{i}"] for i in range(len(weights))], x) - y
        return 0.5 * jnp.sum(diff * diff)

    @jax.jit
    def jax_step(
        p: dict[str, jax.Array], os: optax.OptState, x: jax.Array, y: jax.Array
    ) -> tuple[dict[str, jax.Array], optax.OptState, jax.Array]:
        loss, grad = jax.value_and_grad(loss_fn)(p, x, y)
        updates, os = opt.update(grad, os, p)
        return optax.apply_updates(p, updates), os, loss

    order = rng.integers(0, xtr.shape[0], size=args.steps)  # fixed online sample stream
    max_dloss = 0.0
    print(f"{'step':>6}  {'plastax_loss':>12}  {'optax_loss':>12}  {'|Δ|':>9}")
    for t, idx in enumerate(order):
        x = jnp.asarray(xtr[idx])
        y = jax.nn.one_hot(ytr[idx], 10, dtype=jnp.float32)
        result = step(state, px.StepInputs(inputs=x, targets=y))
        state = result.state
        p_loss = float(result.loss)
        params, opt_state, j_loss = jax_step(params, opt_state, x, y)
        max_dloss = max(max_dloss, abs(p_loss - float(j_loss)))
        if t % max(1, args.steps // 10) == 0 or t == args.steps - 1:
            print(f"{t:>6}  {p_loss:>12.6f}  {float(j_loss):>12.6f}  {max_dloss:>9.2e}")

    pw = plastax_weights(state, sizes)
    jw = [np.asarray(params[f"W{i}"]) for i in range(len(weights))]
    max_dw = max(float(np.max(np.abs(a - b))) for a, b in zip(pw, jw, strict=True))
    print(f"\nplastax test accuracy: {accuracy(pw, xte, yte):.4f}")
    print(f"optax   test accuracy: {accuracy(jw, xte, yte):.4f}")
    print(f"max |Δloss| (per-step parity): {max_dloss:.2e}")
    print(f"max |ΔW|    (final parity)    : {max_dw:.2e}")


if __name__ == "__main__":
    main()
