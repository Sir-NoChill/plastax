"""CIFAR-10: plastax native-sparse SET/RigL (O(E) arena) vs a dense mask-based
reference running the SAME algorithm (O(N^2) weights + a binary mask).

Both train an online (batch-1) sigmoid MLP with the identical dynamic-sparse
rule -- per-destination-unit half-normal magnitude prune (tau = sqrt(pi) *
erfinv(zeta) * mean|w|), count-conserving regrowth (SET: random; RigL: largest
|dL/dw| = |grad_pre_act[dst] * activation[src]|, the delta-rule factorization) --
at the same architecture and sparsity. The only difference is the representation:
plastax stores E live edges + O(E) optimizer state; the dense reference stores the
full N x N weight matrix, its mask, and O(N^2) optimizer state, and does dense
matmuls. So matched accuracy at far fewer stored parameters is the result to
look for.

Data: the pure-numpy CIFAR-10 python-batch reader (the same cifar-10-batches-py
torchvision would download); fetch once with
`curl -L -o /tmp/data/cifar-10-python.tar.gz \
    https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz && \
    tar xzf /tmp/data/cifar-10-python.tar.gz -C /tmp/data`.

Run (GPU strongly preferred):
    .venv-plastax-gpu/bin/python examples/cifar_dst.py
"""

from __future__ import annotations

import time

import dst_sparse as D
import jax
import jax.numpy as jnp
import numpy as np

import plastax as px

_ALPHA = float(
    jnp.sqrt(jnp.pi) * jax.scipy.special.erfinv(jnp.float32(0.3))
)  # zeta=0.3


# --------------------------------------------------------------------------- #
# CIFAR-10 (pure numpy over the python batches)
# --------------------------------------------------------------------------- #
def load_cifar10(
    root: str = "/tmp/data",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load CIFAR-10 via torchvision (downloads on first use).

    Returns flattened, per-pixel-standardized (N, 3072) float32 inputs (HWC row
    order, so a convnet can reshape back to 32x32x3) and int64 labels.
    """
    from torchvision.datasets import CIFAR10  # dev dependency; no torch compute

    train = CIFAR10(root, train=True, download=True)
    test = CIFAR10(root, train=False, download=True)
    x_train = train.data.reshape(len(train.data), -1).astype(np.float32)
    x_test = test.data.reshape(len(test.data), -1).astype(np.float32)
    y_train = np.asarray(train.targets, np.int64)
    y_test = np.asarray(test.targets, np.int64)
    mean, std = x_train.mean(0), x_train.std(0) + 1e-6
    return (x_train - mean) / std, y_train, (x_test - mean) / std, y_test


def _with_bias(x_row: np.ndarray) -> jax.Array:
    """Append the constant bias input, as a float32 jax vector."""
    return jnp.asarray(np.append(x_row, 1.0), dtype=jnp.float32)


# --------------------------------------------------------------------------- #
# plastax native-sparse side (reuses dst_sparse's policies + arena)
# --------------------------------------------------------------------------- #
def run_plastax(
    method: str,
    data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    layers: tuple[int, int, int],
    budgets: tuple[int, int],
    *,
    shortlist: int | None,
    lr: float,
    steps: int,
    churn_every: int,
    seed: int,
) -> dict[str, float]:
    """Train the plastax sparse MLP online with SET/RigL, eval on the test set."""
    x_tr, y_tr, x_te, y_te = data
    classes = layers[-1]
    opt = px.optim.adam(lr, D.GradPreAct)
    mc = shortlist * shortlist if shortlist else max(budgets)
    train_net = D.make_net(opt, method=method, mode="train")
    churn_net = D.make_net(
        opt, method=method, mode="churn", max_candidates=mc, shortlist=shortlist
    )
    eval_net = D.make_net(opt, method=method, mode="eval")
    static, state = D.build_sparse_mlp(train_net, layers, budgets, seed)
    train_step = px.make_step(train_net, static)
    churn_step = px.make_step(churn_net, static)
    eval_step = px.make_step(eval_net, static)

    rng = np.random.default_rng(seed)
    last = jnp.zeros((layers[0],), dtype=jnp.float32)
    t0 = time.time()
    for step in range(steps):
        i = int(rng.integers(len(x_tr)))
        last = _with_bias(x_tr[i])
        state = train_step(
            state, px.StepInputs(inputs=last, targets=D._one_hot(int(y_tr[i]), classes))
        ).state
        if (step + 1) % churn_every == 0:
            state = churn_step(state, px.StepInputs(inputs=last, targets=None)).state
    jax.block_until_ready(state.units[px.ACTIVATION.name])
    train_s = time.time() - t0

    output_ids = np.asarray(static.output_ids)
    correct = 0
    for i in range(len(x_te)):
        state = eval_step(
            state, px.StepInputs(inputs=_with_bias(x_te[i]), targets=None)
        ).state
        preds = np.asarray(state.units[px.ACTIVATION.name])[output_ids]
        correct += int(np.argmax(preds) == int(y_te[i]))
    live = int(px.state.live_conn_count(state))
    return {"acc": correct / len(x_te), "live": live, "train_s": train_s}


# --------------------------------------------------------------------------- #
# dense mask-based reference (the SAME algorithm, O(N^2) weights + mask)
# --------------------------------------------------------------------------- #
def _dense_init(
    layers: tuple[int, int, int], budgets: tuple[int, int], seed: int
) -> dict[str, jax.Array]:
    """Dense weights + a binary mask at the target per-layer sparsity."""
    rng = np.random.default_rng(seed)
    i_dim, h_dim, c_dim = layers

    def layer(n_in: int, n_out: int, budget: int) -> tuple[np.ndarray, np.ndarray]:
        src, dst = D._choose_pairs(n_in, n_out, budget, rng)
        mask = np.zeros((n_in, n_out), np.float32)
        mask[src, dst] = 1.0
        scale = 1.0 / np.sqrt(max(1.0, budget / n_out))
        w = (rng.standard_normal((n_in, n_out)) * scale).astype(np.float32) * mask
        return w, mask

    w1, m1 = layer(i_dim, h_dim, budgets[0])
    w2, m2 = layer(h_dim, c_dim, budgets[1])
    zeros = [np.zeros_like(w1), np.zeros_like(w1), np.zeros_like(w2), np.zeros_like(w2)]
    return {
        "W1": jnp.asarray(w1),
        "M1": jnp.asarray(m1),
        "W2": jnp.asarray(w2),
        "M2": jnp.asarray(m2),
        "m1_": jnp.asarray(zeros[0]),
        "v1_": jnp.asarray(zeros[1]),
        "m2_": jnp.asarray(zeros[2]),
        "v2_": jnp.asarray(zeros[3]),
        "t": jnp.zeros((), jnp.float32),
    }


def _dense_step(s: dict[str, jax.Array], x: jax.Array, y: jax.Array, lr: float) -> dict:
    """One online adam step; returns state + the full delta-rule weight grads."""
    p1 = x @ (s["W1"] * s["M1"])
    a1 = jax.nn.sigmoid(p1)
    out = jax.nn.sigmoid(a1 @ (s["W2"] * s["M2"]))
    gp2 = (out - y) * out * (1.0 - out)  # dL/dpreact2
    gp1 = (gp2 @ (s["W2"] * s["M2"]).T) * a1 * (1.0 - a1)  # dL/dpreact1
    gW1 = jnp.outer(x, gp1)  # full (I,H) delta-rule grad (dead entries included)
    gW2 = jnp.outer(a1, gp2)  # full (H,C)
    t = s["t"] + 1.0
    new = dict(s, t=t)
    for w, mask, gw, mk, vk in (
        ("W1", "M1", gW1, "m1_", "v1_"),
        ("W2", "M2", gW2, "m2_", "v2_"),
    ):
        g = gw * s[mask]  # live-edge gradient
        m = 0.9 * s[mk] + 0.1 * g
        v = 0.999 * s[vk] + 0.001 * g * g
        upd = (m / (1 - 0.9**t)) / (jnp.sqrt(v / (1 - 0.999**t)) + 1e-8)
        new[w] = jnp.where(s[mask] > 0, s[w] - lr * upd, s[w])
        new[mk], new[vk] = m, v
    return new, gW1, gW2


def _dense_rewire(
    s: dict[str, jax.Array],
    gW1: jax.Array,
    gW2: jax.Array,
    method: str,
    rng: np.random.Generator,
) -> dict[str, jax.Array]:
    """Host-side per-column magnitude prune + count-conserving regrow (SET/RigL)."""
    new = dict(s)
    for wk, mk, gw, m_, v_ in (
        ("W1", "M1", gW1, "m1_", "v1_"),
        ("W2", "M2", gW2, "m2_", "v2_"),
    ):
        w = np.asarray(s[wk])
        m = np.asarray(s[mk]) > 0
        g = np.asarray(gw)
        live = m.sum(0)
        mean_abs = np.abs(w * m).sum(0) / np.maximum(live, 1)
        prune = m & (np.abs(w) < (_ALPHA * mean_abs)[None, :])
        m2 = m & ~prune
        pruned = prune.sum(0)
        dead = ~m2
        if method == "set":
            score = np.where(dead, rng.random(w.shape), -np.inf)
        else:
            score = np.where(dead, np.abs(g), -np.inf)
        rank = np.argsort(np.argsort(-score, axis=0), axis=0)
        regrow = dead & (rank < pruned[None, :])
        m_new = m2 | regrow
        new[mk] = jnp.asarray(m_new.astype(np.float32))
        new[wk] = jnp.asarray(np.where(regrow, 0.0, w).astype(np.float32))
        new[m_] = jnp.asarray(np.where(regrow, 0.0, np.asarray(s[m_])))
        new[v_] = jnp.asarray(np.where(regrow, 0.0, np.asarray(s[v_])))
    return new


def run_dense(
    method: str,
    data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    layers: tuple[int, int, int],
    budgets: tuple[int, int],
    *,
    lr: float,
    steps: int,
    churn_every: int,
    seed: int,
) -> dict[str, float]:
    """Train the dense mask-based reference online with the same SET/RigL rule."""
    x_tr, y_tr, x_te, y_te = data
    classes = layers[-1]
    s = _dense_init(layers, budgets, seed)
    step_fn = jax.jit(lambda st, x, y: _dense_step(st, x, y, lr))
    rng = np.random.default_rng(seed)
    eye = np.eye(classes, dtype=np.float32)
    gW1 = gW2 = None
    t0 = time.time()
    for step in range(steps):
        i = int(rng.integers(len(x_tr)))
        x = _with_bias(x_tr[i])
        y = jnp.asarray(eye[int(y_tr[i])])
        s, gW1, gW2 = step_fn(s, x, y)
        if (step + 1) % churn_every == 0:
            s = _dense_rewire(s, gW1, gW2, method, rng)
    jax.block_until_ready(s["W1"])
    train_s = time.time() - t0

    @jax.jit
    def predict(st: dict[str, jax.Array], x: jax.Array) -> jax.Array:
        a1 = jax.nn.sigmoid(x @ (st["W1"] * st["M1"]))
        return jax.nn.sigmoid(a1 @ (st["W2"] * st["M2"]))

    correct = 0
    for i in range(len(x_te)):
        out = np.asarray(predict(s, _with_bias(x_te[i])))
        correct += int(np.argmax(out) == int(y_te[i]))
    live = int((np.asarray(s["M1"]) > 0).sum() + (np.asarray(s["M2"]) > 0).sum())
    dense_params = layers[0] * layers[1] + layers[1] * layers[2]
    return {
        "acc": correct / len(x_te),
        "live": live,
        "dense_params": dense_params,
        "train_s": train_s,
    }


# --------------------------------------------------------------------------- #
_LAYERS = (3073, 1024, 10)  # 32x32x3 + bias -> hidden -> 10 classes
_BUDGETS = (131072, 4096)  # packed powers of two; ~135k live edges
_SHORTLIST = 512
_STEPS = 80000
_CHURN_EVERY = 100
# adam lr: a large fan-in (3073 inputs) makes adam's near-unit first step blow
# up the pre-activations at the ~0.05 that suited the tiny synthetic task, so
# the sigmoid layer saturates and nothing learns; 1e-3 is stable here.
_LR = 1e-3


def main() -> None:
    """Run SET and RigL on both backends and print the comparison."""
    print(f"backend={jax.default_backend()}  loading CIFAR-10...")
    data = load_cifar10()
    dense_params = _LAYERS[0] * _LAYERS[1] + _LAYERS[1] * _LAYERS[2]
    live = sum(_BUDGETS)
    print(
        f"MLP {_LAYERS}  live edges {live}  dense weights {dense_params}  "
        f"sparsity {live / dense_params:.1%}  ({_STEPS} online steps)"
    )
    print(
        f"{'method':6s} {'backend':8s} {'test acc':>9s} {'live':>8s} "
        f"{'stored params':>14s} {'train s':>8s}"
    )
    for method in ("set", "rigl"):
        p = run_plastax(
            method,
            data,
            _LAYERS,
            _BUDGETS,
            shortlist=_SHORTLIST,
            lr=_LR,
            steps=_STEPS,
            churn_every=_CHURN_EVERY,
            seed=0,
        )
        print(
            f"{method:6s} {'plastax':8s} {p['acc']:9.3f} {p['live']:8d} "
            f"{p['live']:14d} {p['train_s']:8.1f}"
        )
        d = run_dense(
            method,
            data,
            _LAYERS,
            _BUDGETS,
            lr=_LR,
            steps=_STEPS,
            churn_every=_CHURN_EVERY,
            seed=0,
        )
        print(
            f"{method:6s} {'dense':8s} {d['acc']:9.3f} {d['live']:8d} "
            f"{d['dense_params']:14d} {d['train_s']:8.1f}"
        )


if __name__ == "__main__":
    main()
