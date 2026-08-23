"""RigL (SPARSE_PLAN.md, growth variant): gradient-informed regrowth holds
sparsity, grows *high-gradient* edges (unlike SET's random growth), and learns.

examples/rigl_sparse.py supplies RigLRegrow and reuses set_sparse's magnitude
prune + task; both are loaded by file path after mlp_xor (which they import),
mirroring test_set_sparse.py.

The RigL-specific property under test is that a churn grows the absent edges with
the largest |dL/dw| = |grad_pre_act[dst] * activation[src]| -- so the grown
edges' mean gradient magnitude far exceeds the candidate pool's, which random
(SET) growth would merely match.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import plastax as px

_EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def _load_example(name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, _EXAMPLES_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


mlp_xor = _load_example("mlp_xor")
set_sparse = _load_example("set_sparse")  # rigl_sparse imports it
rigl_sparse = _load_example("rigl_sparse")


def _live_pairs(state: px.NetworkState[None]) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for bucket in state.conns:
        dead = np.asarray(bucket[px.DEAD.name])
        frm = np.asarray(bucket[px.FROM_ID.name])
        to = np.asarray(bucket[px.TO_ID.name])
        for i in np.flatnonzero(~dead):
            pairs.add((int(frm[i]), int(to[i])))
    return pairs


@pytest.fixture(scope="module")
def rigl() -> dict[str, object]:
    """Train the RigL sparse MLP, then instrument one churn's growth gradients."""
    optimizer = px.optim.adam(0.05, set_sparse.GradPreAct)
    train_net = rigl_sparse.make_net(optimizer, mode="train")
    churn_net = rigl_sparse.make_net(
        optimizer, mode="churn", zeta=0.3, max_candidates=max(set_sparse._BUDGETS)
    )
    eval_net = rigl_sparse.make_net(optimizer, mode="eval")

    static, state = set_sparse.build_sparse_mlp(
        train_net, set_sparse._LAYER_SIZES, set_sparse._BUDGETS, 0
    )
    train_step = px.make_step(train_net, static)
    churn_step = px.make_step(churn_net, static)
    eval_step = px.make_step(eval_net, static)

    teacher = set_sparse._teacher(0)
    rng = np.random.default_rng(1)
    dummy = jnp.zeros((set_sparse._LAYER_SIZES[0],), dtype=jnp.float32)

    live: list[int] = []
    acc_before = 0.0
    last_inputs = dummy
    for cycle in range(8):
        for _ in range(40):
            inputs, label = set_sparse._sample(teacher, rng)
            state = train_step(
                state, px.StepInputs(inputs=inputs, targets=set_sparse._one_hot(label))
            ).state
            last_inputs = inputs
        state = churn_step(state, px.StepInputs(inputs=last_inputs, targets=None)).state
        live.append(int(px.state.live_conn_count(state)))
        if cycle == 0:
            acc_before, state = set_sparse.evaluate(
                eval_step, static, state, teacher, rng, 256
            )

    # One instrumented churn: snapshot the exact columns growth reads (the last
    # train step already scattered `last_inputs`, so input activations match the
    # persisted hidden activations / grad_pre_act), score every deeper-absent
    # candidate by |grad_pre_act[dst] * activation[src]|, then see what grew.
    inputs, label = set_sparse._sample(teacher, rng)
    state = train_step(
        state, px.StepInputs(inputs=inputs, targets=set_sparse._one_hot(label))
    ).state
    act = np.asarray(state.units[px.ACTIVATION.name])
    gpa = np.asarray(state.units[mlp_xor.GradPreAct.name])
    lvl = np.asarray(state.units[px.LEVEL.name])
    before = _live_pairs(state)
    num = static.num_units
    pool = np.array(
        [
            abs(float(gpa[d]) * float(act[s]))
            for s in range(num)
            for d in range(num)
            if lvl[d] > lvl[s] and (s, d) not in before
        ]
    )

    state = churn_step(
        state, px.StepInputs(inputs=jnp.asarray(inputs), targets=None)
    ).state
    grown = _live_pairs(state) - before
    grown_grads = np.array([abs(float(gpa[d]) * float(act[s])) for (s, d) in grown])

    acc_after, state = set_sparse.evaluate(eval_step, static, state, teacher, rng, 512)
    return {
        "budget": sum(set_sparse._BUDGETS),
        "live": live,
        "grown_mean": float(grown_grads.mean()),
        "pool_mean": float(pool.mean()),
        "n_grown": int(grown_grads.size),
        "acc_before": acc_before,
        "acc_after": acc_after,
    }


def test_rigl_holds_sparsity(rigl: dict[str, object]) -> None:
    assert set(rigl["live"]) == {rigl["budget"]}


def test_rigl_grows_high_gradient_edges(rigl: dict[str, object]) -> None:
    # The defining RigL property: grown edges carry far more gradient than the
    # average candidate. Random (SET) growth would give a ratio near 1.
    assert rigl["n_grown"] > 0
    assert rigl["grown_mean"] > 2.0 * rigl["pool_mean"], (
        rigl["grown_mean"],
        rigl["pool_mean"],
    )


def test_rigl_learns_under_rewiring(rigl: dict[str, object]) -> None:
    assert rigl["acc_after"] > rigl["acc_before"]
    assert rigl["acc_after"] > 0.6
