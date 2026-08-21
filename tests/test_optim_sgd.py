"""plastax.optim.sgd matches optax.sgd to float32 precision (Track A.1).

Oracle test: a dense MLP built two ways -- plastax (`optim.sgd`) and jax+optax
(`optax.sgd`) -- from identical initial weights, fed identical synthetic samples
online (batch-1), must agree on the per-step loss and the final weights to
float32 tolerance. That pins `optim.sgd`'s weight update to the reference SGD.

optax is a test-only oracle, never a runtime dependency, so this module is
marked `slow` and skips when optax is absent. examples/mlp_xor.py supplies the
sigmoid forward/backward and MSE-loss traits; it is loaded by file path
(examples/ is not an installed package), matching test_mlp_xor.py.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import plastax as px

optax = pytest.importorskip("optax")

pytestmark = pytest.mark.slow

_EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def _load_example(name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, _EXAMPLES_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


mlp_xor = _load_example("mlp_xor")

N_IN, H, K = 12, 6, 4
LR = 0.1
STEPS = 60
# max |Δ| below was ~6e-8 (float32 rounding); these bounds fail on any real
# divergence while tolerating the differing reduction order of the two backends.
RTOL, ATOL = 1e-4, 1e-5


def _const(mat: np.ndarray) -> Callable[[jax.Array, tuple[int, ...]], jax.Array]:
    """A topology initializer that ignores the key and returns fixed weights."""

    def init(key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
        del key, shape
        return jnp.asarray(mat)

    return init


def _build_plastax(
    w1: np.ndarray, w2: np.ndarray
) -> tuple[type[px.Network[None]], px.NetworkStatic, px.NetworkState[None]]:
    class _MLP(px.Network[None]):
        forward_pass = mlp_xor.SigmoidForward()
        backward_pass = mlp_xor.SigmoidBackward()
        loss = mlp_xor.MSELoss()
        update_conn = px.optim.sgd(LR, mlp_xor.GradPreAct).update_conn()
        extra_unit_fields = (mlp_xor.GradPreAct, mlp_xor.LossGrad)
        propagation = px.Propagation.TOPOLOGICAL

    topo = px.topology.sequential(
        px.topology.input_units(N_IN),
        px.topology.dense(N_IN, H, init=_const(w1)),
        px.topology.dense(H, K, init=_const(w2)),
    )
    static, state = px.NetworkBuilder.from_topology(
        _MLP, topo, jax.random.PRNGKey(0), globals_=None
    )
    return _MLP, static, state


def _plastax_weights(state: px.NetworkState[None]) -> tuple[np.ndarray, np.ndarray]:
    """Reassemble the live edge weights into the two dense weight matrices.

    Unit ids are laid out inputs [0, N_IN), hidden [N_IN, N_IN+H), outputs
    after that, so an edge's endpoints identify which matrix cell it is.
    """
    fr, to, we = [], [], []
    for bucket in state.conns:
        live = ~np.asarray(bucket[px.DEAD.name])
        fr.append(np.asarray(bucket[px.FROM_ID.name])[live])
        to.append(np.asarray(bucket[px.TO_ID.name])[live])
        we.append(np.asarray(bucket[px.WEIGHT.name])[live])
    from_ids, to_ids, weights = (
        np.concatenate(fr),
        np.concatenate(to),
        np.concatenate(we),
    )
    w1 = np.zeros((N_IN, H), np.float32)
    w2 = np.zeros((H, K), np.float32)
    for f, t, w in zip(from_ids, to_ids, weights, strict=True):
        if t < N_IN + H:
            w1[f, t - N_IN] = w
        else:
            w2[f - N_IN, t - N_IN - H] = w
    return w1, w2


def test_sgd_matches_optax_online() -> None:
    rng = np.random.default_rng(0)
    w1 = (rng.standard_normal((N_IN, H)) * 0.3).astype(np.float32)
    w2 = (rng.standard_normal((H, K)) * 0.3).astype(np.float32)

    cls, static, state = _build_plastax(w1, w2)
    step = px.make_step(cls, static)

    params = {"W1": jnp.asarray(w1), "W2": jnp.asarray(w2)}
    opt = optax.sgd(LR)
    opt_state = opt.init(params)

    def loss_fn(p: dict[str, jax.Array], x: jax.Array, y: jax.Array) -> jax.Array:
        h = jax.nn.sigmoid(x @ p["W1"])
        out = jax.nn.sigmoid(h @ p["W2"])
        diff = out - y
        return 0.5 * jnp.sum(diff * diff)

    @jax.jit
    def jax_step(
        p: dict[str, jax.Array], os: optax.OptState, x: jax.Array, y: jax.Array
    ) -> tuple[dict[str, jax.Array], optax.OptState, jax.Array]:
        loss, grad = jax.value_and_grad(loss_fn)(p, x, y)
        updates, os = opt.update(grad, os, p)
        return optax.apply_updates(p, updates), os, loss

    xs = (rng.standard_normal((STEPS, N_IN)) * 0.5).astype(np.float32)
    ys = rng.integers(0, K, size=STEPS)

    for t in range(STEPS):
        x = jnp.asarray(xs[t])
        y = jax.nn.one_hot(ys[t], K, dtype=jnp.float32)
        result = step(state, px.StepInputs(inputs=x, targets=y))
        state = result.state
        params, opt_state, jax_loss = jax_step(params, opt_state, x, y)
        np.testing.assert_allclose(
            float(result.loss), float(jax_loss), rtol=RTOL, atol=ATOL
        )

    pw1, pw2 = _plastax_weights(state)
    np.testing.assert_allclose(pw1, np.asarray(params["W1"]), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(pw2, np.asarray(params["W2"]), rtol=RTOL, atol=ATOL)
