"""Extreme multi-label classification traits (examples/xmc.py).

The XMC net departs from the sigmoid MLP in three ways that are easy to get
silently wrong -- a linear output unit, dL/dz taken straight from the loss at
the output level, and a ReLU derivative everywhere else -- so the load-bearing
test here is a gradient check: one SGD step must move every weight by exactly
`-lr * dL/dw`, with `dL/dw` from `jax.grad` on an explicit dense model of the
same computation. That check is what caught the backward pass reading
`u[GradPreAct, dst]` instead of `src`: `build_backward_accumulate` passes the
accumulator target (FROM_ID, the shallower unit) as the callback's `dst`
parameter and the already-finalized deeper unit as `src`, inverted from the
forward pass, and reading the wrong one leaves every hidden gradient at zero
while the output layer still trains correctly.

examples/xmc.py is loaded by file path after the modules it imports, mirroring
the other example-backed tests.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import plastax as px

_EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def _load_example(name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, _EXAMPLES_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_load_example("mlp_xor")
_load_example("dst_sparse")
_load_example("xmc_data")
xmc = _load_example("xmc")
GradPreAct = xmc.GradPreAct

_D, _H, _L = 3, 4, 5
_LR = 0.1
_POS_WEIGHT = 3.0


def _dense_edges() -> tuple[np.ndarray, np.ndarray]:
    """Every (input -> hidden) and (hidden -> label) edge of the tiny net."""
    src1, dst1 = np.meshgrid(np.arange(_D), np.arange(_H), indexing="ij")
    src2, dst2 = np.meshgrid(np.arange(_H), np.arange(_L), indexing="ij")
    from_ids = np.concatenate([src1.ravel(), src2.ravel() + _D]).astype(np.int32)
    to_ids = np.concatenate([dst1.ravel() + _D, dst2.ravel() + _D + _H]).astype(
        np.int32
    )
    return from_ids, to_ids


def _edge_weights(state: px.NetworkState[None]) -> dict[tuple[int, int], float]:
    """Map every live edge's (from, to) to its weight."""
    out: dict[tuple[int, int], float] = {}
    for bucket in state.conns:
        from_ids = np.asarray(bucket[px.FROM_ID.name])
        to_ids = np.asarray(bucket[px.TO_ID.name])
        weights = np.asarray(bucket[px.WEIGHT.name])
        dead = np.asarray(bucket[px.DEAD.name])
        for k in range(len(from_ids)):
            if not dead[k]:
                out[(int(from_ids[k]), int(to_ids[k]))] = float(weights[k])
    return out


def test_gradients_match_autodiff() -> None:
    """One SGD step moves every weight by -lr * the autodiff gradient."""
    optimizer = px.optim.sgd(_LR, GradPreAct)
    net = xmc.make_net(optimizer, method="rigl", mode="train", pos_weight=_POS_WEIGHT)
    from_ids, to_ids = _dense_edges()
    rng = np.random.default_rng(0)
    static, state = px.NetworkBuilder.from_edges(
        net,
        _D + _H + _L,
        from_ids,
        to_ids,
        weights=rng.standard_normal(len(from_ids)).astype(np.float32),
        input_ids=tuple(range(_D)),
        output_ids=tuple(range(_D + _H, _D + _H + _L)),
        globals_=None,
    )
    state = xmc.mark_outputs(static, state)

    inputs = jnp.asarray(rng.standard_normal(_D), dtype=jnp.float32)
    targets = jnp.asarray([1.0, 0.0, 1.0, 0.0, 0.0], dtype=jnp.float32)
    before = _edge_weights(state)
    w_hidden = jnp.asarray(
        [[before[(i, _D + j)] for j in range(_H)] for i in range(_D)]
    )
    w_label = jnp.asarray(
        [[before[(_D + j, _D + _H + k)] for k in range(_L)] for j in range(_H)]
    )

    def dense_loss(w1: jax.Array, w2: jax.Array) -> jax.Array:
        logits = jax.nn.relu(inputs @ w1) @ w2
        scale = 1.0 + targets * (_POS_WEIGHT - 1.0)
        stable = jnp.maximum(logits, 0.0) + jnp.log1p(jnp.exp(-jnp.abs(logits)))
        return jnp.sum(scale * stable - _POS_WEIGHT * targets * logits)

    expected_loss = dense_loss(w_hidden, w_label)
    grad_hidden, grad_label = jax.grad(dense_loss, argnums=(0, 1))(w_hidden, w_label)

    result = px.make_step(net, static)(
        state, px.StepInputs(inputs=inputs, targets=targets)
    )
    after = _edge_weights(result.state)

    np.testing.assert_allclose(float(result.loss), float(expected_loss), rtol=1e-5)
    for i in range(_D):
        for j in range(_H):
            want = float(w_hidden[i, j]) - _LR * float(grad_hidden[i, j])
            np.testing.assert_allclose(after[(i, _D + j)], want, atol=1e-6)
    for j in range(_H):
        for k in range(_L):
            want = float(w_label[j, k]) - _LR * float(grad_label[j, k])
            np.testing.assert_allclose(after[(_D + j, _D + _H + k)], want, atol=1e-6)


def test_churn_conserves_live_edges() -> None:
    """SET and RigL rewiring hold the live-edge count exactly."""
    split = xmc.synthetic_split(64, 128, 256, nnz=8, rank=8, seed=0)
    for method in ("set", "rigl"):
        _, live_history, _ = xmc.run(
            split,
            method=method,
            hidden=32,
            hidden_fan_in=16,
            label_fan_in=4,
            shortlist=32,
            num_cycles=4,
            steps_per_cycle=16,
            eval_points=8,
            verbose=False,
        )
        assert len(set(live_history)) == 1, (
            f"{method}: live-edge count drifted: {sorted(set(live_history))}"
        )


def test_zeta_column_controls_pruning() -> None:
    """zeta is a runtime column: 0 makes churn a no-op, >0 rewires.

    Both cases run the SAME compiled churn step, which is what lets the host
    anneal between cycles without retracing.
    """
    split = xmc.synthetic_split(32, 128, 256, nnz=8, rank=8, seed=0)
    optimizer = px.optim.adam(0.01, GradPreAct)
    churn_net = xmc.make_net(
        optimizer, method="rigl", mode="churn", max_candidates=4096, shortlist=32
    )
    static, base = xmc.build_xmc_net(
        churn_net,
        split.num_features,
        split.num_labels,
        32,
        hidden_fan_in=16,
        label_fan_in=4,
        seed=0,
    )
    churn_step = px.make_step(churn_net, static)
    inputs = px.StepInputs(inputs=jnp.asarray(split.dense_input(0)), targets=None)

    def live_after(zeta: float) -> set[tuple[int, int]]:
        state = jax.tree.map(jnp.copy, base)
        state = churn_step(xmc.set_zeta(state, zeta), inputs).state
        return set(_edge_weights(state))

    before = set(_edge_weights(base))
    assert live_after(0.0) == before, "zeta=0 must leave the wiring untouched"
    rewired = live_after(0.3)
    assert len(rewired) == len(before), "churn must conserve the live-edge count"
    assert rewired != before, "zeta>0 must actually rewire"
