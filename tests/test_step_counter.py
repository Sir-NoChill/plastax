"""`NetworkState.step`: one increment per step, batched counts once."""

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
_SIZES = (2, 3, 1)


def _mlp() -> tuple[type[px.Network[None]], px.NetworkStatic, px.NetworkState[None]]:
    base = mlp_xor.make_net(px.optim.sgd(0.2, mlp_xor.GradPreAct), train=True)

    class net(base):  # type: ignore[valid-type, misc]
        batch_reduction = px.MeanFloatFirstRest()

    rng = np.random.default_rng(0)
    blocks = [px.topology.input_units(_SIZES[0])]
    for a, b in zip(_SIZES[:-1], _SIZES[1:], strict=True):
        w = (rng.standard_normal((a, b)) * 0.5).astype(np.float32)
        blocks.append(px.topology.dense(a, b, init=lambda k, s, w=w: jnp.asarray(w)))
    static, state = px.NetworkBuilder.from_topology(
        net, px.topology.sequential(*blocks), jax.random.PRNGKey(0), globals_=None
    )
    return net, static, state


def test_step_starts_at_zero_and_counts_completed_steps() -> None:
    net, static, state = _mlp()
    assert state.step.dtype == jnp.int32
    assert int(state.step) == 0

    step = px.make_step(net, static)
    x = jnp.asarray([0.0, 1.0], dtype=jnp.float32)
    t = jnp.asarray([1.0], dtype=jnp.float32)
    for want in (1, 2, 3):
        state = step(state, px.StepInputs(inputs=x, targets=t)).state
        assert int(state.step) == want


def test_batched_step_counts_as_one_step() -> None:
    net, static, state = _mlp()
    batched = px.make_step(net, static, batch_size=4)
    x = jnp.zeros((4, 2), dtype=jnp.float32)
    t = jnp.ones((4, 1), dtype=jnp.float32)
    state = batched(state, px.StepInputs(inputs=x, targets=t)).state
    assert int(state.step) == 1
    state = batched(state, px.StepInputs(inputs=x, targets=t)).state
    assert int(state.step) == 2
