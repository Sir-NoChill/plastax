"""S1 (SPARSE_PLAN.md): per-unit-quantile SET holds sparsity, rewires, and learns.

examples/set_sparse.py supplies the SET policies (magnitude-stats forward, prune,
random regrow); it is loaded by file path, and because it does ``from mlp_xor
import ...`` at module load, mlp_xor is loaded into sys.modules first (mirroring
test_optim_sparse.py's example-loading).

The invariants under test are exactly S1's acceptance criteria: a churn conserves
the live-edge count (prune count == grow count, so sparsity is pinned), the churn
genuinely rewires (a nonzero, ~zeta fraction of edges turns over), and training
under rewiring learns the task -- all with no writes to global state.
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


_load_example("mlp_xor")  # set_sparse imports the sigmoid/MSE traits from it
set_sparse = _load_example("set_sparse")


def _live_pairs(state: px.NetworkState[None]) -> set[tuple[int, int]]:
    """Return the set of (from_id, to_id) pairs of every live edge."""
    pairs: set[tuple[int, int]] = set()
    for bucket in state.conns:
        dead = np.asarray(bucket[px.DEAD.name])
        frm = np.asarray(bucket[px.FROM_ID.name])
        to = np.asarray(bucket[px.TO_ID.name])
        for i in np.flatnonzero(~dead):
            pairs.add((int(frm[i]), int(to[i])))
    return pairs


@pytest.fixture(scope="module")
def rewired() -> dict[str, object]:
    """Train the SET sparse MLP with rewiring, recording sparsity and turnover.

    Runs the example's own policies and task end to end (train N steps, then one
    churn) for a handful of cycles, capturing the post-churn live count and the
    (pruned, grown) turnover each cycle plus accuracy before/after training.
    """
    zeta = 0.3
    optimizer = px.optim.adam(0.05, set_sparse.GradPreAct)
    train_net = set_sparse.make_net(optimizer, mode="train")
    churn_net = set_sparse.make_net(
        optimizer, mode="churn", zeta=zeta, max_candidates=max(set_sparse._BUDGETS)
    )
    eval_net = set_sparse.make_net(optimizer, mode="eval")

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
    turnover: list[tuple[int, int]] = []
    acc_before = 0.0
    for cycle in range(8):
        for _ in range(40):
            inputs, label = set_sparse._sample(teacher, rng)
            state = train_step(
                state, px.StepInputs(inputs=inputs, targets=set_sparse._one_hot(label))
            ).state
        before = _live_pairs(state)
        state = churn_step(state, px.StepInputs(inputs=dummy, targets=None)).state
        after = _live_pairs(state)
        live.append(len(after))
        turnover.append((len(before - after), len(after - before)))
        if cycle == 0:
            acc_before, state = set_sparse.evaluate(
                eval_step, static, state, teacher, rng, 256
            )
    acc_after, state = set_sparse.evaluate(eval_step, static, state, teacher, rng, 512)
    return {
        "budget": sum(set_sparse._BUDGETS),
        "live": live,
        "turnover": turnover,
        "acc_before": acc_before,
        "acc_after": acc_after,
        "zeta": zeta,
    }


def test_sparsity_is_pinned(rewired: dict[str, object]) -> None:
    # every post-churn live count equals the arena budget: regrow refills exactly
    # what prune freed, so sparsity never drifts.
    assert set(rewired["live"]) == {rewired["budget"]}


def test_churn_conserves_count_and_rewires(rewired: dict[str, object]) -> None:
    turnover: list[tuple[int, int]] = rewired["turnover"]  # type: ignore[assignment]
    for pruned, grown in turnover:
        assert pruned == grown, "regrow must refill exactly the pruned count"
        assert pruned > 0, "a churn that prunes nothing is not rewiring"


def test_pruned_fraction_estimates_zeta(rewired: dict[str, object]) -> None:
    # the purely per-unit half-normal threshold should prune ~zeta of live edges
    # overall -- the whole point of the local estimate. Not exact (distributional
    # model + estimation), so a band around zeta=0.3, well clear of 0 or "all".
    turnover: list[tuple[int, int]] = rewired["turnover"]  # type: ignore[assignment]
    budget: int = rewired["budget"]  # type: ignore[assignment]
    fractions = [pruned / budget for pruned, _ in turnover]
    mean_fraction = sum(fractions) / len(fractions)
    assert 0.2 <= mean_fraction <= 0.4, fractions


def test_set_learns_under_rewiring(rewired: dict[str, object]) -> None:
    # accuracy climbs well above chance (1/4) despite continuous rewiring.
    assert rewired["acc_after"] > rewired["acc_before"]
    assert rewired["acc_after"] > 0.6
