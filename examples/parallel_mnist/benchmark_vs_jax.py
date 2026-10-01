"""Side-by-side benchmark: a colleague's JAX networks vs plastax networks.

Both sides train the SAME parallel-MNIST task (K=5 independent sub-tasks,
batch_size=1 online, Adam, cross-entropy) at the SAME input resolution and the
SAME live-parameter budget, and we measure throughput (steps/sec) and
asymptotic accuracy for three network types:

    dense   -- colleague MLP            vs  plastax dense
    block   -- colleague BlockSparseMLP vs  plastax block-oracle
    dynamic -- colleague DynamicNetwork vs  plastax dynamic-sparse

--------------------------------------------------------------------------
PATH / ENVIRONMENT ASSUMPTIONS (read before running)
--------------------------------------------------------------------------
* Run with the GPU venv interpreter that has jax(gpu), plastax, optax, equinox:
      /home/stormblessed/Code/Research/Plastix/.venv-plastax-gpu/bin/python
* Run from the plastax repo root (this file lives in examples/parallel_mnist/).
  We do sys.path.insert(0, "examples") to import `parallel_mnist`.
* Colleague repo is expected at:
      /home/stormblessed/Code/Research/Plastix/edan-phd-research
  (override with env var EDAN_REPO). It is importable as `phd...`. We stub
  `phd.feature_search` before importing `phd` so that `phd/__init__.py` does
  NOT pull in torch (which is not installed in the GPU venv) -- we only need
  the pure jax/equinox model + optimizer + connectivity modules.
* MNIST is read directly from the torchvision idx cache at
      /tmp/data/MNIST/raw
  (no torchvision import). Both sides receive the identical pooled arrays.

Neither repo's source files are modified; this script only reads them.

--------------------------------------------------------------------------
TIMING METHODOLOGY (identical for both sides -> fair)
--------------------------------------------------------------------------
Two-point subtraction. For each (type, impl) we time a short run of
`N_SMALL` online steps and a long run of `N_BIG` online steps, each run
executed to full completion (device work forced via block_until_ready / a
host pull of the final state). Steady-state throughput is then

    steps/sec = (N_BIG - N_SMALL) / (t_big - t_small)

which cancels one-time compilation and fixed build/setup overhead for BOTH
implementations without needing to instrument plastax's internal loop.
Asymptotic accuracy is the mean over the final 10% of sampled windows of the
N_BIG run.
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import os
import struct
import sys
import time
import types
import warnings
from typing import Any

import numpy as np

# --- resolution / task constants (held equal across all six configs) -------
K_TASKS = 5
N_CLASSES = 10
POOL = 2  # 28 -> 14  =>  196 px / task
PIX = (28 // POOL) ** 2  # 196
LR = 2e-3
WIDTH = 12  # plastax per-task hidden width
N_SMALL = 500
N_BIG = 3000
SAMPLE_EVERY = 200
REWIRE_PERIOD = 20  # dynamic: restructure cadence (both sides)
SEED = 0

EDAN_REPO = os.environ.get(
    "EDAN_REPO", "/home/stormblessed/Code/Research/Plastix/edan-phd-research"
)
MNIST_RAW = os.environ.get("MNIST_RAW", "/tmp/data/MNIST/raw")


# ---------------------------------------------------------------------------
# MNIST from the raw idx cache (no torchvision), pooled exactly as plastax does
# ---------------------------------------------------------------------------
def _load_idx(path: str) -> np.ndarray:
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rb") as f:
        (magic,) = struct.unpack(">I", f.read(4))
        if magic == 2051:
            n, r, c = struct.unpack(">III", f.read(12))
            return np.frombuffer(f.read(), np.uint8).reshape(n, r, c)
        if magic == 2049:
            (n,) = struct.unpack(">I", f.read(4))
            return np.frombuffer(f.read(), np.uint8)
        raise ValueError(f"bad idx magic {magic} in {path}")


def load_mnist_pooled(
    pool: int = POOL, split: str = "train"
) -> tuple[np.ndarray, np.ndarray]:
    pre = "train" if split == "train" else "t10k"
    imgs = _load_idx(f"{MNIST_RAW}/{pre}-images-idx3-ubyte").astype(np.float32) / 255.0
    lbls = _load_idx(f"{MNIST_RAW}/{pre}-labels-idx1-ubyte").astype(np.int64)
    if pool > 1:
        side = 28 // pool
        trimmed = imgs[:, : side * pool, : side * pool]
        imgs = trimmed.reshape(imgs.shape[0], side, pool, side, pool).mean(axis=(2, 4))
    return imgs.reshape(imgs.shape[0], -1), lbls


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Row:
    type: str
    impl: str
    params: int
    steps_per_sec: float
    accuracy: float
    note: str = ""


def _two_point(t_small: float, t_big: float) -> float:
    dt = t_big - t_small
    if dt <= 0:
        return float("nan")
    return (N_BIG - N_SMALL) / dt


# ===========================================================================
# COLLEAGUE SIDE  (phd... models, minimal optax online step)
# ===========================================================================
def _import_phd() -> None:
    """Import the colleague package, stubbing feature_search to dodge torch."""
    if EDAN_REPO not in sys.path:
        sys.path.insert(0, EDAN_REPO)
    sys.modules.setdefault("phd.feature_search", types.ModuleType("phd.feature_search"))


def _count_params(model: Any) -> int:
    import equinox as eqx
    import jax

    return int(sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array))))


def _colleague_stream(images: np.ndarray, labels: np.ndarray, seed: int) -> Any:
    from phd.structure_search.data import ParallelMNISTStream

    return ParallelMNISTStream(
        images=images,
        labels=labels,
        n_tasks=K_TASKS,
        batch_size=1,
        seed=seed + 100,
        permute_period=0,
    )


def _run_colleague_static(
    kind: str, images: np.ndarray, labels: np.ndarray, n_steps: int
) -> Any:
    """Online train a colleague dense/block model for n_steps; return
    (elapsed_sec, asymptotic_accuracy, n_params). Fully blocked at the end."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import optax
    from phd.jax_core.models import MLP
    from phd.structure_search.block_sparse_mlp import BlockSparseMLP

    key = jax.random.key(SEED)
    if kind == "dense":
        model = MLP(
            input_dim=K_TASKS * PIX,
            output_dim=K_TASKS * N_CLASSES,
            n_layers=2,
            hidden_dim=K_TASKS * WIDTH,
            weight_init_method="lecun_uniform",
            activation="relu",
            key=key,
        )
    else:  # block
        model = BlockSparseMLP(
            n_tasks=K_TASKS,
            input_dim_per_task=PIX,
            output_dim_per_task=N_CLASSES,
            n_layers=2,
            hidden_dim=WIDTH,
            weight_init_method="lecun_uniform",
            activation="relu",
            key=key,
        )
    n_params = _count_params(model)
    opt = optax.adam(LR)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(model: Any, opt_state: Any, x: Any, y: Any) -> Any:
        def loss_fn(m: Any) -> Any:
            raw, _ = m(x)
            logits = raw.reshape(K_TASKS, N_CLASSES)
            oh = jax.nn.one_hot(y, N_CLASSES)
            loss = -jnp.mean(jnp.sum(oh * jax.nn.log_softmax(logits, -1), axis=-1))
            return loss, logits

        (loss, logits), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
        updates, opt_state = opt.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        acc = (jnp.argmax(logits, -1) == y).astype(jnp.float32).mean()
        return model, opt_state, loss, acc

    stream = _colleague_stream(images, labels, SEED)
    imgs_np, lbls_np = stream.sample_batch(n_steps)
    imgs_np = imgs_np[:, 0, :]
    lbls_np = lbls_np[:, 0, :]

    acc_samples: list[float] = []
    csum = jnp.float32(0.0)
    cnt = 0
    t0 = time.perf_counter()
    for t in range(n_steps):
        x = jnp.asarray(imgs_np[t])
        y = jnp.asarray(lbls_np[t])
        model, opt_state, loss, acc = step(model, opt_state, x, y)
        csum = csum + acc
        cnt += 1
        if t > 0 and t % SAMPLE_EVERY == 0:
            acc_samples.append(float(csum) / cnt)
            csum = jnp.float32(0.0)
            cnt = 0
    jax.block_until_ready((model, opt_state))
    elapsed = time.perf_counter() - t0
    if cnt:
        acc_samples.append(float(csum) / cnt)
    n_tail = max(1, len(acc_samples) // 10)
    asy_acc = float(np.mean(acc_samples[-n_tail:])) if acc_samples else float("nan")
    return elapsed, asy_acc, n_params


def _run_colleague_dynamic(
    images: np.ndarray, labels: np.ndarray, n_steps: int, restructure: bool = True
) -> Any:
    """Online train the colleague DynamicNetwork (padded/masked dense arrays)
    for n_steps, restructuring every REWIRE_PERIOD steps via ConnectivityManager.
    Returns (elapsed_sec, asymptotic_accuracy, n_params_active)."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import optax
    from phd.jax_core.optimizers.optimizer import EqxOptimizer
    from phd.structure_search.connectivity_manager import (
        ConnectivityManager,
        contribution_utility,
    )
    from phd.structure_search.dynamic_network import (
        count_active_connections,
        init_random_dynamic_network,
        sync_outgoing_weights,
    )

    key = jax.random.key(SEED)
    k1, k2 = jax.random.split(key)
    # Sized so the initial active-connection count (~4.9k) is in the same
    # ballpark as plastax's live-edge budget (~3.9k) for the same task.
    net = init_random_dynamic_network(
        input_dim=K_TASKS * PIX,
        output_dim=K_TASKS * N_CLASSES,
        n_layers=1,
        units_per_layer=K_TASKS * WIDTH,
        max_units_per_layer=128,
        max_connections_per_unit=32,
        activations=("relu",),
        max_fan_out=64,
        key=k1,
    )
    active0 = int(count_active_connections(net))

    spec = jax.tree.map(lambda _: False, net)
    spec = eqx.tree_at(lambda n: (n.weights, n.output_weights), spec, (True, True))
    opt = EqxOptimizer(optax.adam(LR), net, spec, name="adam")
    tracker = ConnectivityManager(
        model=net,
        prune_rate=1e-4,
        connection_budget=float(active0),
        decay_rate=0.99,
        maturity_threshold=100,
        max_new_units_per_step=8,
        output_connect_strategy="all",
        utility_fn=contribution_utility,
        generate_fn=None,
        rng=k2,
    )

    @eqx.filter_jit
    def step(
        net: Any,
        opt: Any,
        tracker: Any,
        images_b: Any,
        labels_b: Any,
        do_restructure: Any,
    ) -> Any:
        def loss_fn(m: Any) -> Any:
            raw, buf = jax.vmap(m)(images_b)  # (1, K*nc), (1, buf)
            logits = raw.reshape(-1, K_TASKS, N_CLASSES)
            oh = jax.nn.one_hot(labels_b, N_CLASSES)  # (1, K, nc)
            loss = -jnp.mean(jnp.sum(oh * jax.nn.log_softmax(logits, -1), axis=-1))
            return loss, (raw, buf)

        (loss, (raw, buf)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            net
        )
        updates, opt = opt.with_update(grads, net)
        net = eqx.apply_updates(net, updates)
        net = sync_outgoing_weights(net)
        oh = jax.nn.one_hot(labels_b, N_CLASSES).reshape(raw.shape)
        tracker = tracker.update_stats(
            net, buf, grads=grads, updates=updates, targets=oh, predictions=raw
        )
        if do_restructure:
            rng, rr = jax.random.split(tracker.rng)
            res = tracker.modify_structure(net, opt, rng=rr)
            tracker, net, opt = res[0], res[1], res[2]
        pred = jnp.argmax(raw.reshape(-1, K_TASKS, N_CLASSES), axis=-1)
        acc = (pred == labels_b).astype(jnp.float32).mean()
        return net, opt, tracker, loss, acc

    stream = _colleague_stream(images, labels, SEED)
    imgs_np, lbls_np = stream.sample_batch(n_steps)  # (n,1,980),(n,1,K)
    imgs_np = imgs_np[:, 0, :]  # (n,980)
    lbls_np = lbls_np[:, 0, :]  # (n,K)

    acc_samples: list[float] = []
    csum = jnp.float32(0.0)
    cnt = 0
    t0 = time.perf_counter()
    for t in range(n_steps):
        xb = jnp.asarray(imgs_np[t])[None, :]  # (1,980)
        yb = jnp.asarray(lbls_np[t])[None, :]  # (1,K)
        do_r = restructure and (t > 0 and t % REWIRE_PERIOD == 0)
        net, opt, tracker, loss, acc = step(net, opt, tracker, xb, yb, do_r)
        csum = csum + acc
        cnt += 1
        if t > 0 and t % SAMPLE_EVERY == 0:
            acc_samples.append(float(csum) / cnt)
            csum = jnp.float32(0.0)
            cnt = 0
    jax.block_until_ready((net, opt, tracker))
    elapsed = time.perf_counter() - t0
    if cnt:
        acc_samples.append(float(csum) / cnt)
    n_tail = max(1, len(acc_samples) // 10)
    asy_acc = float(np.mean(acc_samples[-n_tail:])) if acc_samples else float("nan")
    n_params = int(count_active_connections(net))
    return elapsed, asy_acc, n_params


# ===========================================================================
# PLASTAX SIDE  (examples/parallel_mnist/run.py black-boxed)
# ===========================================================================
def _plastax_run(
    kind: str, images: np.ndarray, labels: np.ndarray, n_steps: int
) -> Any:
    """Time run.run_<kind> end-to-end (fully synchronous) and read
    (elapsed_sec, asymptotic_accuracy, n_params) back from its outputs."""
    from parallel_mnist import run

    cfg_kw = dict(
        n_tasks=K_TASKS,
        pool=POOL,
        width=WIDTH,
        n_steps=n_steps,
        optimizer="adam",
        learning_rate=LR,
    )
    if kind == "dynamic":
        cfg_kw.update(prune_threshold=1e-4, density=0.2)
    cfg = run.Config(**cfg_kw)
    fn = {"dense": run.run_dense, "block": run.run_block, "dynamic": run.run_dynamic}[
        kind
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        m, params = fn(cfg, images, labels)
        elapsed = time.perf_counter() - t0  # run.* pulls final state -> blocked
    asy_acc = m.summary()["asymptotic_accuracy"]
    return elapsed, asy_acc, int(params)


# ===========================================================================
# Driver
# ===========================================================================
def bench_colleague(kind: str, images: np.ndarray, labels: np.ndarray) -> Row:
    runner = (
        _run_colleague_dynamic
        if kind == "dynamic"
        else (lambda im, lb, n: _run_colleague_static(kind, im, lb, n))
    )
    t_small, _, _ = runner(images, labels, N_SMALL)
    t_big, acc, params = runner(images, labels, N_BIG)
    note = f"restructure every {REWIRE_PERIOD}" if kind == "dynamic" else ""
    return Row(kind, "colleague", params, _two_point(t_small, t_big), acc, note)


def bench_plastax(kind: str, images: np.ndarray, labels: np.ndarray) -> Row:
    t_small, _, _ = _plastax_run(kind, images, labels, N_SMALL)
    t_big, acc, params = _plastax_run(kind, images, labels, N_BIG)
    return Row(kind, "plastax", params, _two_point(t_small, t_big), acc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--types",
        default="dense,block,dynamic",
        help="comma list subset of dense,block,dynamic",
    )
    ap.add_argument("--md-out", default=None, help="optional path to write a table")
    args = ap.parse_args()

    sys.path.insert(0, "examples")
    _import_phd()

    import jax

    print(f"jax backend: {jax.default_backend()}  devices: {jax.devices()}")

    images, labels = load_mnist_pooled()
    print(
        f"parallel MNIST: K={K_TASKS} tasks, {PIX} px/task, width={WIDTH}/task, "
        f"N_SMALL={N_SMALL} N_BIG={N_BIG}\n"
    )

    types_ = [t.strip() for t in args.types.split(",") if t.strip()]
    rows: list[Row] = []
    for kind in types_:
        for side in ("colleague", "plastax"):
            print(f"running {kind:8s} / {side} ...", flush=True)
            try:
                if side == "colleague":
                    rows.append(bench_colleague(kind, images, labels))
                else:
                    rows.append(bench_plastax(kind, images, labels))
                r = rows[-1]
                print(
                    f"    params={r.params:>7d}  steps/s={r.steps_per_sec:8.1f}  "
                    f"acc={r.accuracy:.3f}  {r.note}",
                    flush=True,
                )
            except Exception as e:  # noqa: BLE001
                print(f"    FAILED: {type(e).__name__}: {e}", flush=True)

    # --- table ---
    hdr = f"\n{'type':<9}{'impl':<11}{'params':>9}{'steps/sec':>12}{'accuracy':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r.type:<9}{r.impl:<11}{r.params:>9d}{r.steps_per_sec:>12.1f}"
            f"{r.accuracy:>10.3f}"
        )

    if args.md_out:
        with open(args.md_out, "w") as f:
            f.write("| type | impl | params | steps/sec | accuracy |\n")
            f.write("|------|------|-------:|----------:|---------:|\n")
            for r in rows:
                f.write(
                    f"| {r.type} | {r.impl} | {r.params} | "
                    f"{r.steps_per_sec:.1f} | {r.accuracy:.3f} |\n"
                )
        print(f"\nwrote {args.md_out}")


if __name__ == "__main__":
    main()
