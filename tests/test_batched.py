"""make_step(batch_size=B): the batched step (fast checks).

The optax oracle for batched optimizers lives in test_optim.py (slow). Here: a
batch of one is the streaming step; a rule without the exact per_sample /
incoming_batched pair falls back to the mean of its per-sample writes, which
for a linear rule equals the exact path; PIPELINE nets and non-positive batch
sizes are rejected; and structural phases (prune, proposal growth) run under a
batch.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
import types
from pathlib import Path

import jax
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
_SIZES = [6, 5, 3]
_LR = 0.2


@dataclasses.dataclass(frozen=True)
class _PlainSGD:
    """SGD with no per_sample / incoming_batched: the mean-of-writes path."""

    def incoming(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: None,
    ) -> px.ConnWrite:
        del g
        grad = u[mlp_xor.GradPreAct, dst] * u[px.ACTIVATION, src]
        return px.ConnWrite.of((px.WEIGHT, c[px.WEIGHT, cid] - _LR * grad))

    def outgoing(
        self,
        u: px.UnitView,
        src: px.UnitIdx,
        dst: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: None,
    ) -> px.ConnWrite:
        del u, src, dst, c, cid, g
        return px.ConnWrite.of()


def _mlp(
    update: px.UpdateConn[None], extra_conn: tuple[px.FieldSpec[np.generic], ...] = ()
) -> tuple[type[px.Network[None]], px.NetworkStatic, px.NetworkState[None]]:
    class _MLP(px.Network[None]):
        forward_pass = mlp_xor.SigmoidForward()
        backward_pass = mlp_xor.SigmoidBackward()
        loss = mlp_xor.MSELoss()
        update_conn = update
        extra_unit_fields = (mlp_xor.GradPreAct, mlp_xor.LossGrad)
        extra_conn_fields = extra_conn
        propagation = px.Propagation.TOPOLOGICAL

    rng = np.random.default_rng(0)
    blocks = [px.topology.input_units(_SIZES[0])]
    for a, b in zip(_SIZES[:-1], _SIZES[1:], strict=True):
        w = (rng.standard_normal((a, b)) * 0.5).astype(np.float32)
        blocks.append(px.topology.dense(a, b, init=lambda k, s, w=w: jnp.asarray(w)))
    static, state = px.NetworkBuilder.from_topology(
        _MLP, px.topology.sequential(*blocks), jax.random.PRNGKey(0), globals_=None
    )
    return _MLP, static, state


def _weights(state: px.NetworkState[None]) -> np.ndarray:
    return np.concatenate([np.asarray(b[px.WEIGHT.name]) for b in state.conns])


def _data(batch: int, steps: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    xs = rng.standard_normal((steps, batch, _SIZES[0])).astype(np.float32)
    ys = np.asarray(
        jax.nn.one_hot(rng.integers(0, _SIZES[-1], (steps, batch)), _SIZES[-1])
    )
    return xs, ys


@pytest.mark.parametrize(
    "make",
    [
        lambda: px.optim.sgd(_LR, mlp_xor.GradPreAct),
        lambda: px.optim.adam(0.01, mlp_xor.GradPreAct),
    ],
)
def test_a_batch_of_one_is_the_streaming_step(make: object) -> None:
    opt = make()  # type: ignore[operator]
    net, static, state = _mlp(opt.update_conn(), opt.state_fields)
    stream = px.make_step(net, static)
    batched = px.make_step(net, static, batch_size=1)
    xs, ys = _data(1, 5)
    s_stream, s_batch = state, jax.tree.map(jnp.copy, state)
    for x, y in zip(xs, ys, strict=True):
        r1 = stream(
            s_stream, px.StepInputs(inputs=jnp.asarray(x[0]), targets=jnp.asarray(y[0]))
        )
        r2 = batched(
            s_batch, px.StepInputs(inputs=jnp.asarray(x), targets=jnp.asarray(y))
        )
        s_stream, s_batch = r1.state, r2.state
        np.testing.assert_allclose(float(r2.loss), float(r1.loss), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(
        _weights(s_batch), _weights(s_stream), rtol=1e-6, atol=1e-7
    )


def test_mean_of_writes_equals_the_exact_path_for_a_linear_rule() -> None:
    net_plain, static, state = _mlp(_PlainSGD())
    net_exact, static_e, state_e = _mlp(
        px.optim.sgd(_LR, mlp_xor.GradPreAct).update_conn()
    )
    plain = px.make_step(net_plain, static, batch_size=4)
    exact = px.make_step(net_exact, static_e, batch_size=4)
    xs, ys = _data(4, 6)
    for x, y in zip(xs, ys, strict=True):
        inputs = px.StepInputs(inputs=jnp.asarray(x), targets=jnp.asarray(y))
        state = plain(state, inputs).state
        state_e = exact(state_e, inputs).state
    np.testing.assert_allclose(_weights(state), _weights(state_e), rtol=1e-5, atol=1e-6)


def test_batched_step_is_the_mean_of_per_sample_sgd_steps() -> None:
    # For SGD, one batched step from S equals the mean of B streaming steps
    # each taken from the same S.
    opt = px.optim.sgd(_LR, mlp_xor.GradPreAct)
    net, static, state = _mlp(opt.update_conn())
    xs, ys = _data(4, 1)
    stream = px.make_step(net, static)
    singles = []
    for b in range(4):
        r = stream(
            jax.tree.map(jnp.copy, state),
            px.StepInputs(inputs=jnp.asarray(xs[0, b]), targets=jnp.asarray(ys[0, b])),
        )
        singles.append(_weights(r.state))
    batched = px.make_step(net, static, batch_size=4)
    r = batched(
        state, px.StepInputs(inputs=jnp.asarray(xs[0]), targets=jnp.asarray(ys[0]))
    )
    np.testing.assert_allclose(
        _weights(r.state), np.mean(singles, axis=0), rtol=1e-5, atol=1e-6
    )


def test_batch_size_is_validated() -> None:
    net, static, _ = _mlp(_PlainSGD())
    with pytest.raises(ValueError, match="batch_size"):
        px.make_step(net, static, batch_size=0)

    class _Pipe(px.Network[None]):
        forward_pass = mlp_xor.SigmoidForward()
        propagation = px.Propagation.PIPELINE

    with pytest.raises(ValueError, match="PIPELINE"):
        px.make_step(_Pipe, static, batch_size=2)


def test_structural_phases_run_once_per_batched_step() -> None:
    spec = importlib.util.spec_from_file_location(
        "_churn", Path(__file__).parent / "test_inplace_churn.py"
    )
    assert spec is not None and spec.loader is not None
    churn = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(churn)
    static, state = churn._build_with(churn._GrowingNet, headroom=0.5)
    step = px.make_step(churn._GrowingNet, static, batch_size=3)
    x = jnp.asarray(np.random.default_rng(2).standard_normal((3, churn._WIDTH)))
    live0 = int(px.state.live_conn_count(state))
    reference = jax.tree.map(jnp.copy, state)  # step donates `state`
    result = step(state, px.StepInputs(inputs=x.astype(jnp.float32), targets=None))
    grown = int(px.state.live_conn_count(result.state)) - live0
    # One growth pass (<= max_candidates per bucket), not one per sample.
    assert 0 < grown <= churn._FanoutGrow.max_candidates * len(static.level_capacities)
    # The stored units are the batch mean of the per-sample forwards.
    per_sample = []
    fwd = px.make_step(churn._ForwardOnly, static)
    for b in range(3):
        r = fwd(
            jax.tree.map(jnp.copy, reference),
            px.StepInputs(inputs=x[b].astype(jnp.float32), targets=None),
        )
        per_sample.append(np.asarray(r.state.units[px.ACTIVATION.name]))
    np.testing.assert_allclose(
        np.asarray(result.state.units[px.ACTIVATION.name]),
        np.mean(per_sample, axis=0),
        rtol=1e-5,
        atol=1e-6,
    )
