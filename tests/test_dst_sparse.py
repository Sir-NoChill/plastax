"""Dynamic sparse training (SET + RigL, one switch): both hold sparsity and
learn, and the growth *signature* distinguishes them -- RigL grows high-gradient
edges, SET grows at random.

examples/dst_sparse.py is loaded by file path after mlp_xor (which it imports),
mirroring the other example-backed tests.
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


_load_example("mlp_xor")
dst = _load_example("dst_sparse")
GradPreAct = dst.GradPreAct


def _live_pairs(state: px.NetworkState[None]) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for bucket in state.conns:
        dead = np.asarray(bucket[px.DEAD.name])
        frm = np.asarray(bucket[px.FROM_ID.name])
        to = np.asarray(bucket[px.TO_ID.name])
        for i in np.flatnonzero(~dead):
            pairs.add((int(frm[i]), int(to[i])))
    return pairs


def _train_and_probe(method: str) -> dict[str, object]:
    layers, budgets = dst._DEMO_LAYERS, dst._DEMO_BUDGETS
    optimizer = px.optim.adam(0.05, GradPreAct)
    train_net = dst.make_net(optimizer, method=method, mode="train")
    churn_net = dst.make_net(
        optimizer, method=method, mode="churn", zeta=0.3, max_candidates=max(budgets)
    )
    eval_net = dst.make_net(optimizer, method=method, mode="eval")
    static, state = dst.build_sparse_mlp(train_net, layers, budgets, 0)
    train_step = px.make_step(train_net, static)
    churn_step = px.make_step(churn_net, static)
    eval_step = px.make_step(eval_net, static)

    teacher, rng = dst.teacher_task(layers[0] - 1, layers[-1], 0)
    live: list[int] = []
    acc_before = 0.0
    last = jnp.zeros((layers[0],), dtype=jnp.float32)
    for cycle in range(8):
        for _ in range(40):
            x, label = dst._sample(teacher, rng)
            state = train_step(
                state, px.StepInputs(inputs=x, targets=dst._one_hot(label, layers[-1]))
            ).state
            last = x
        state = churn_step(state, px.StepInputs(inputs=last, targets=None)).state
        live.append(int(px.state.live_conn_count(state)))
        if cycle == 0:
            acc_before, state = dst.evaluate(
                eval_step, static, state, teacher, rng, 256
            )

    # Instrumented churn: score every deeper-absent candidate by the edge
    # gradient |grad_pre_act[dst] * activation[src]| from the columns growth
    # reads, then see what actually grew.
    x, label = dst._sample(teacher, rng)
    state = train_step(
        state, px.StepInputs(inputs=x, targets=dst._one_hot(label, layers[-1]))
    ).state
    act = np.asarray(state.units[px.ACTIVATION.name])
    gpa = np.asarray(state.units[GradPreAct.name])
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
    state = churn_step(state, px.StepInputs(inputs=x, targets=None)).state
    grown = _live_pairs(state) - before
    grown_g = np.array([abs(float(gpa[d]) * float(act[s])) for (s, d) in grown])
    acc_after, state = dst.evaluate(eval_step, static, state, teacher, rng, 512)
    return {
        "budget": sum(budgets),
        "live": live,
        "grown_ratio": float(grown_g.mean() / pool.mean()),
        "n_grown": int(grown_g.size),
        "acc_before": acc_before,
        "acc_after": acc_after,
    }


@pytest.fixture(scope="module", params=["set", "rigl"])
def trained(request: pytest.FixtureRequest) -> dict[str, object]:
    result = _train_and_probe(request.param)
    result["method"] = request.param
    return result


def test_holds_sparsity(trained: dict[str, object]) -> None:
    assert set(trained["live"]) == {trained["budget"]}


def test_learns(trained: dict[str, object]) -> None:
    assert trained["acc_after"] > trained["acc_before"]
    assert trained["acc_after"] > 0.6


def test_growth_signature(trained: dict[str, object]) -> None:
    # RigL grows the highest-gradient absent edges (mean |dL/dw| far above the
    # candidate pool's); SET grows at random (mean ~ the pool's).
    assert trained["n_grown"] > 0
    if trained["method"] == "rigl":
        assert trained["grown_ratio"] > 2.0, trained["grown_ratio"]
    else:
        assert trained["grown_ratio"] < 1.8, trained["grown_ratio"]
