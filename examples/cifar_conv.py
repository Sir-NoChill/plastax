"""A simple plastax convnet on CIFAR-10.

The MLP failed on CIFAR not only on the adam lr (see cifar_dst) but on inductive
bias: each hidden unit saw a random ~4% of pixels. A convnet fixes that with
*local receptive fields* -- and in plastax a convnet is just a different
topology (`topology.conv2d`, an untied/locally-connected layer) trained with the
exact same traits as the MLP (sigmoid forward/backward, MSE loss, adam). No new
policies. conv2d's small fan-in (kernel*channels) also sidesteps the large-fan-in
adam blow-up, though a uniform lr=1e-3 stays safe for the dense head.

Online, batch-1. Loads CIFAR-10 (32x32x3, HWC) via torchvision through
cifar_dst.load_cifar10.

Run:  .venv-plastax-gpu/bin/python examples/cifar_conv.py
"""

from __future__ import annotations

import time

import cifar_dst as CD
import jax
import jax.numpy as jnp
import numpy as np
from mlp_xor import GradPreAct, LossGrad, MSELoss, SigmoidBackward, SigmoidForward

import plastax as px

_IMG = (32, 32, 3)
_CLASSES = 10


def _conv_topology(channels: tuple[int, int], kernel: int, stride: int):  # noqa: ANN202
    """Two conv2d layers + a dense head over the 32x32x3 image grid."""
    h, w, c = _IMG
    h1, w1 = (h - kernel) // stride + 1, (w - kernel) // stride + 1
    h2, w2 = (h1 - kernel) // stride + 1, (w1 - kernel) // stride + 1
    return px.topology.sequential(
        px.topology.input_units(h * w * c),
        px.topology.conv2d((h, w, c), (kernel, kernel, channels[0]), stride=stride),
        px.topology.conv2d(
            (h1, w1, channels[0]), (kernel, kernel, channels[1]), stride=stride
        ),
        px.topology.dense(h2 * w2 * channels[1], _CLASSES),
    )


def make_convnet(
    optimizer: px.optim.Optimizer, *, train: bool
) -> type[px.Network[None]]:
    """Build the convnet trait class; train adds backward/loss/update, eval is
    forward-only over the same field layout (so both share one static/state)."""
    if train:

        class _Train(px.Network[None]):
            forward_pass = SigmoidForward()
            backward_pass = SigmoidBackward()
            loss = MSELoss()
            update_conn = optimizer.update_conn()
            extra_unit_fields = (GradPreAct, LossGrad)
            extra_conn_fields = optimizer.state_fields
            propagation = px.Propagation.TOPOLOGICAL

        return _Train

    class _Eval(px.Network[None]):
        forward_pass = SigmoidForward()
        extra_unit_fields = (GradPreAct, LossGrad)
        extra_conn_fields = optimizer.state_fields
        propagation = px.Propagation.TOPOLOGICAL

    return _Eval


def run(
    steps: int = 120000,
    lr: float = 1e-3,
    channels: tuple[int, int] = (16, 32),
    kernel: int = 3,
    stride: int = 2,
    seed: int = 0,
    verbose: bool = True,
) -> float:
    """Train the convnet online on CIFAR-10 and return test accuracy."""
    x_tr, y_tr, x_te, y_te = CD.load_cifar10()
    optimizer = px.optim.adam(lr, GradPreAct)
    train_net = make_convnet(optimizer, train=True)
    static, state = px.NetworkBuilder.from_topology(
        train_net,
        _conv_topology(channels, kernel, stride),
        jax.random.PRNGKey(0),
        globals_=None,
    )
    if verbose:
        print(
            f"convnet: units {static.num_units}  live edges "
            f"{int(px.state.live_conn_count(state))}  ({steps} online steps)",
            flush=True,
        )
    train_step = px.make_step(train_net, static)
    eval_step = px.make_step(make_convnet(optimizer, train=False), static)

    eye = np.eye(_CLASSES, dtype=np.float32)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for i in range(steps):
        j = int(rng.integers(len(x_tr)))
        state = train_step(
            state,
            px.StepInputs(
                inputs=jnp.asarray(x_tr[j]), targets=jnp.asarray(eye[int(y_tr[j])])
            ),
        ).state
        if verbose and (i + 1) % 20000 == 0:
            print(f"  step {i + 1}  ({time.time() - t0:.0f}s)", flush=True)

    output_ids = np.asarray(static.output_ids)
    correct = 0
    for j in range(len(x_te)):
        state = eval_step(
            state, px.StepInputs(inputs=jnp.asarray(x_te[j]), targets=None)
        ).state
        preds = np.asarray(state.units[px.ACTIVATION.name])[output_ids]
        correct += int(np.argmax(preds) == int(y_te[j]))
    acc = correct / len(x_te)
    if verbose:
        print(f"test accuracy: {acc:.3f}  (chance {1 / _CLASSES:.2f})", flush=True)
    return acc


def main() -> None:
    """Train the convnet and assert it clears the MLP's ceiling."""
    acc = run()
    assert acc > 0.4, f"convnet underperformed ({acc:.3f})"


if __name__ == "__main__":
    main()
