"""S0 (SPARSE_PLAN.md): a regrown connection's optimizer state is zeroed.

The add_conn phase resets a grown edge's untouched fields to their FieldSpec
default, so a stateful optimizer (adam) starts a regrown edge's moments and step
counter at 0 -- never inheriting the pruned edge that held the slot. This is the
A.2 foundation for RigL/SET-style regrow-zeroing.

examples/mlp_xor.py supplies the sigmoid/MSE traits (loaded by file path, like
test_mlp_xor.py).
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


mlp_xor = _load_example("mlp_xor")

_GROWN_WEIGHT = 7.0  # distinctive init marker so grown edges are identifiable
_ADAM = px.optim.adam(0.1, mlp_xor.GradPreAct)


class _PruneFromUnit0(px.PruneConn):
    """Tombstone every edge whose source is unit 0 -- deterministic, and frees
    the low-index bucket-0 slots that add_conn then reclaims."""

    def predicate(
        self, u: px.UnitView, c: px.ConnView, cid: px.ConnIdx, g: object
    ) -> jax.Array:
        del u, g
        return c[px.FROM_ID, cid] == jnp.int32(0)


class _GrowAhead(px.AddConn[None]):
    """Grow level-ahead edges (dst deeper than src, so no resort), each marked
    with a distinctive weight. Optimizer columns are untouched, so the phase
    resets them to their FieldSpec default."""

    max_candidates = 3

    def score(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: object
    ) -> jax.Array:
        del g
        ahead = u[px.LEVEL, dst] > u[px.LEVEL, src]
        return jnp.where(ahead, jnp.float32(1.0), -jnp.inf)

    def init(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: object
    ) -> px.ConnWrite:
        del u, src, dst, g
        return px.ConnWrite.of((px.WEIGHT, jnp.float32(_GROWN_WEIGHT)))


class _TrainNet(px.Network[None]):
    forward_pass = mlp_xor.SigmoidForward()
    backward_pass = mlp_xor.SigmoidBackward()
    loss = mlp_xor.MSELoss()
    update_conn = _ADAM.update_conn()
    extra_unit_fields = (mlp_xor.GradPreAct, mlp_xor.LossGrad)
    extra_conn_fields = _ADAM.state_fields
    propagation = px.Propagation.TOPOLOGICAL


class _ChurnNet(px.Network[None]):
    forward_pass = mlp_xor.SigmoidForward()
    backward_pass = mlp_xor.SigmoidBackward()
    loss = mlp_xor.MSELoss()
    update_conn = _ADAM.update_conn()
    prune_conn = _PruneFromUnit0()
    add_conn = _GrowAhead()
    extra_unit_fields = (mlp_xor.GradPreAct, mlp_xor.LossGrad)
    extra_conn_fields = _ADAM.state_fields
    propagation = px.Propagation.TOPOLOGICAL
    neighbourhood = 2  # let input->output skip edges be growable


def _snapshot(state: px.NetworkState[None]) -> list[dict[str, np.ndarray]]:
    return [
        {
            "dead": np.asarray(b[px.DEAD.name]),
            "w": np.asarray(b[px.WEIGHT.name]),
            "m": np.asarray(b["opt/m"]),
            "v": np.asarray(b["opt/v"]),
            "t": np.asarray(b["opt/t"]),
        }
        for b in state.conns
    ]


def test_regrown_edge_optimizer_state_is_zero() -> None:
    key = jax.random.PRNGKey(0)
    topology = px.topology.sequential(
        px.topology.input_units(3),
        px.topology.dense(3, 3),
        px.topology.dense(3, 2),
    )
    static, state = px.NetworkBuilder.from_topology(
        _TrainNet, topology, key, globals_=None
    )

    # Train so every live edge accumulates non-zero adam moments.
    train = px.make_step(_TrainNet, static)
    rng = np.random.default_rng(0)
    for _ in range(15):
        x = jnp.asarray(rng.standard_normal(3), dtype=jnp.float32)
        y = jnp.asarray(rng.standard_normal(2), dtype=jnp.float32)
        state = train(state, px.StepInputs(inputs=x, targets=y)).state

    before = _snapshot(state)
    assert any((np.abs(b["m"][~b["dead"]]) > 1e-6).any() for b in before), (
        "no adam state accumulated -- the test would be vacuous"
    )

    # One churn step: prune unit-0 edges (freeing their trained slots), then grow
    # marked edges into the freed slots (prune runs before add in the same step).
    churn = px.make_step(_ChurnNet, static)
    x = jnp.asarray(rng.standard_normal(3), dtype=jnp.float32)
    y = jnp.asarray(rng.standard_normal(2), dtype=jnp.float32)
    state = churn(state, px.StepInputs(inputs=x, targets=y)).state
    after = _snapshot(state)

    reused_a_trained_slot = False
    for pre, post in zip(before, after, strict=True):
        grown = (~post["dead"]) & np.isclose(post["w"], _GROWN_WEIGHT)
        # every regrown edge starts with zero adam state
        np.testing.assert_allclose(post["m"][grown], 0.0, atol=0.0)
        np.testing.assert_allclose(post["v"][grown], 0.0, atol=0.0)
        np.testing.assert_allclose(post["t"][grown], 0.0, atol=0.0)
        # ... including where it reused a slot that held a trained (nonzero-m) edge
        if (grown & (~pre["dead"]) & (np.abs(pre["m"]) > 1e-6)).any():
            reused_a_trained_slot = True

    assert reused_a_trained_slot, (
        "no grown edge reclaimed a slot that had held a trained edge -- the "
        "no-stale-leak claim is untested (loosen the setup)"
    )
