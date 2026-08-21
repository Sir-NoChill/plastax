"""plastax.optim optimizers match their optax reference to float32 (Track A.1).

Oracle test: a dense MLP built two ways -- plastax (plastax.optim) and jax+optax
-- from identical initial weights, fed identical samples online (batch-1), must
agree on the per-step loss and the final weights to float32. Each plastax
optimizer is paired with the optax transform it should reproduce (`_CASES`; add
a row per new optimizer), and every optimizer is checked both on small synthetic
data and on a real MNIST slice.

optax is a test-only oracle, never a runtime dependency, so this module is
marked slow and skips when optax is absent. examples/{mlp_xor,mnist_sgd}.py are
loaded by file path (examples/ is not an installed package), matching
test_mlp_xor.py; the MNIST test skips when the cached IDX files are absent (e.g.
in CI), so it runs locally without adding a dataset dependency to CI.
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
from plastax._types import FieldSpec

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
mnist_sgd = _load_example("mnist_sgd")  # data loader reuse; needs mlp_xor loaded first

LR, MU, ADAM_LR, RMS_LR = 0.1, 0.9, 0.01, 0.001
STEPS = 60
# max |Δ| observed ~1e-7 (float32 rounding); bounds fail on any real divergence
# while tolerating the differing reduction order of the two backends.
RTOL, ATOL = 1e-4, 1e-5

OptFactory = Callable[[FieldSpec[np.float32]], px.optim.Optimizer]

# Each plastax optimizer paired with the optax transform it must reproduce.
_CASES: list[tuple[str, OptFactory, object]] = [
    ("sgd", lambda gf: px.optim.sgd(LR, gf), optax.sgd(LR)),
    ("momentum", lambda gf: px.optim.momentum(LR, MU, gf), optax.sgd(LR, momentum=MU)),
    ("adam", lambda gf: px.optim.adam(ADAM_LR, gf), optax.adam(ADAM_LR)),
    ("adamw", lambda gf: px.optim.adamw(ADAM_LR, gf), optax.adamw(ADAM_LR)),
    ("rmsprop", lambda gf: px.optim.rmsprop(RMS_LR, gf), optax.rmsprop(RMS_LR)),
]
_IDS = [case[0] for case in _CASES]
_PARAMS = [(case[1], case[2]) for case in _CASES]


def _const(mat: np.ndarray) -> Callable[[jax.Array, tuple[int, ...]], jax.Array]:
    """A topology initializer that ignores the key and returns fixed weights."""

    def init(key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
        del key, shape
        return jnp.asarray(mat)

    return init


def _build_plastax(
    optimizer: px.optim.Optimizer, sizes: list[int], weights: list[np.ndarray]
) -> tuple[type[px.Network[None]], px.NetworkStatic, px.NetworkState[None]]:
    class _MLP(px.Network[None]):
        forward_pass = mlp_xor.SigmoidForward()
        backward_pass = mlp_xor.SigmoidBackward()
        loss = mlp_xor.MSELoss()
        update_conn = optimizer.update_conn()
        extra_unit_fields = (mlp_xor.GradPreAct, mlp_xor.LossGrad)
        extra_conn_fields = optimizer.state_fields
        propagation = px.Propagation.TOPOLOGICAL

    blocks = [px.topology.input_units(sizes[0])]
    for i, w in enumerate(weights):
        blocks.append(px.topology.dense(sizes[i], sizes[i + 1], init=_const(w)))
    topo = px.topology.sequential(*blocks)
    static, state = px.NetworkBuilder.from_topology(
        _MLP, topo, jax.random.PRNGKey(0), globals_=None
    )
    return _MLP, static, state


def _plastax_weights(
    state: px.NetworkState[None], sizes: list[int]
) -> list[np.ndarray]:
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


def _glorot(sizes: list[int], rng: np.random.Generator) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for a, b in zip(sizes[:-1], sizes[1:], strict=True):
        limit = float(np.sqrt(6.0 / (a + b)))
        out.append(rng.uniform(-limit, limit, size=(a, b)).astype(np.float32))
    return out


def _assert_parity(
    make_plastax: OptFactory,
    optax_opt: optax.GradientTransformation,
    sizes: list[int],
    weights: list[np.ndarray],
    xs: np.ndarray,
    ys: np.ndarray,
) -> None:
    """Run one optimizer both ways online and assert loss + weights agree."""
    cls, static, state = _build_plastax(
        make_plastax(mlp_xor.GradPreAct), sizes, weights
    )
    step = px.make_step(cls, static)

    params = [jnp.asarray(w) for w in weights]
    opt_state = optax_opt.init(params)

    def loss_fn(p: list[jax.Array], x: jax.Array, y: jax.Array) -> jax.Array:
        a = x
        for w in p:
            a = jax.nn.sigmoid(a @ w)
        return 0.5 * jnp.sum((a - y) ** 2)

    @jax.jit
    def jax_step(
        p: list[jax.Array], os: optax.OptState, x: jax.Array, y: jax.Array
    ) -> tuple[list[jax.Array], optax.OptState, jax.Array]:
        loss, grad = jax.value_and_grad(loss_fn)(p, x, y)
        updates, os = optax_opt.update(grad, os, p)
        return optax.apply_updates(p, updates), os, loss

    for x, y in zip(xs, ys, strict=True):
        xj, yj = jnp.asarray(x), jnp.asarray(y)
        result = step(state, px.StepInputs(inputs=xj, targets=yj))
        state = result.state
        params, opt_state, jax_loss = jax_step(params, opt_state, xj, yj)
        np.testing.assert_allclose(
            float(result.loss), float(jax_loss), rtol=RTOL, atol=ATOL
        )

    for got, want in zip(_plastax_weights(state, sizes), params, strict=True):
        np.testing.assert_allclose(got, np.asarray(want), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(("make_plastax", "optax_opt"), _PARAMS, ids=_IDS)
def test_optimizer_matches_optax_online(
    make_plastax: OptFactory, optax_opt: optax.GradientTransformation
) -> None:
    rng = np.random.default_rng(0)
    sizes = [12, 6, 4]
    weights = _glorot(sizes, rng)
    xs = (rng.standard_normal((STEPS, sizes[0])) * 0.5).astype(np.float32)
    ys = np.asarray(jax.nn.one_hot(rng.integers(0, sizes[-1], size=STEPS), sizes[-1]))
    _assert_parity(make_plastax, optax_opt, sizes, weights, xs, ys)


_MNIST_RAW = Path("/tmp/data/MNIST/raw")
_MNIST_PRESENT = all(
    (_MNIST_RAW / f).exists() or (_MNIST_RAW / f"{f}.gz").exists()
    for f in ("train-images-idx3-ubyte", "train-labels-idx1-ubyte")
)


@pytest.mark.skipif(not _MNIST_PRESENT, reason="MNIST IDX cache absent (e.g. CI)")
@pytest.mark.parametrize(("make_plastax", "optax_opt"), _PARAMS, ids=_IDS)
def test_optimizer_matches_optax_on_mnist(
    make_plastax: OptFactory, optax_opt: optax.GradientTransformation
) -> None:
    xtr, ytr, _, _ = mnist_sgd.load_mnist()
    images = mnist_sgd.preprocess(xtr, pool=2)  # 14x14 = 196 inputs
    n_steps = 40
    sizes = [images.shape[1], 16, 10]
    rng = np.random.default_rng(0)
    weights = _glorot(sizes, rng)
    xs = images[:n_steps]
    ys = np.asarray(jax.nn.one_hot(ytr[:n_steps], 10))
    _assert_parity(make_plastax, optax_opt, sizes, weights, xs, ys)
