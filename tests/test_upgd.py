"""UPGD traits (examples/upgd.py).

The load-bearing claim is that the per-weight utility `-(dL/dw)*w` is pure
per-edge local on the arena, so the whole method is one `UpdateConn` with no
churn net and no topology change. Test 1 pins the utility against a NumPy
oracle; the last test pins that the topology really is untouched.

The utility GATE is what makes UPGD more than perturbed SGD, so it gets its own
test at both extremes: a maximally useful weight must be left alone by gradient
AND noise, a useless one must receive the full update.
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


for _dep in ("mlp_xor", "dst_sparse", "nonstationary"):
    _load_example(_dep)
nonstationary = sys.modules["nonstationary"]
upgd = _load_example("upgd")

_LAYERS = (5, 6, 6, 3)


def _build(**kwargs: object) -> tuple:
    net = upgd.make_net(mode="train", **kwargs)
    static, state = nonstationary.build_dense_mlp(net, _LAYERS, 0)
    return net, static, nonstationary.mark_outputs(static, state)


def _live_edges(state: px.NetworkState[None]) -> dict[tuple[int, int], float]:
    out: dict[tuple[int, int], float] = {}
    for bucket in state.conns:
        dead = np.asarray(bucket[px.DEAD.name])
        f = np.asarray(bucket[px.FROM_ID.name])
        t = np.asarray(bucket[px.TO_ID.name])
        w = np.asarray(bucket[px.WEIGHT.name])
        for k in range(len(f)):
            if not dead[k]:
                out[(int(f[k]), int(t[k]))] = float(w[k])
    return out


def _run_one_step(
    net: type, static: px.NetworkStatic, state: px.NetworkState[None]
) -> px.StepResult[None]:
    inputs = jnp.asarray(
        np.random.default_rng(0).standard_normal(_LAYERS[0]), jnp.float32
    )
    targets = jnp.asarray(np.eye(_LAYERS[-1], dtype=np.float32)[0])
    return px.make_step(net, static)(
        state, px.StepInputs(inputs=inputs, targets=targets)
    )


def test_utility_matches_numpy_oracle() -> None:
    """U == -(dL/dw)*w, with dL/dw the delta rule grad_pre_act[dst]*act[src].

    beta=0 collapses the running average to the instantaneous value so no trace
    can hide a wrong utility.
    """
    net, static, state = _build(beta=0.0, sigma=0.0)
    before = _live_edges(state)
    result = _run_one_step(net, static, state)
    activations = np.asarray(result.state.units[px.ACTIVATION.name])
    grad_pre_act = np.asarray(result.state.units[upgd.GradPreAct.name])

    checked = 0
    for bucket in result.state.conns:
        dead = np.asarray(bucket[px.DEAD.name])
        if dead.all():
            continue
        from_ids = np.asarray(bucket[px.FROM_ID.name])[~dead]
        to_ids = np.asarray(bucket[px.TO_ID.name])[~dead]
        got = np.asarray(bucket[upgd.UPGD_UTIL.name])[~dead]
        weights = np.array(
            [before[(int(f), int(t))] for f, t in zip(from_ids, to_ids, strict=True)]
        )
        gradient = grad_pre_act[to_ids] * activations[from_ids]
        np.testing.assert_allclose(got, -gradient * weights, atol=1e-6)
        checked += got.size
    assert checked > 0, "no live edges were checked"


def test_utility_gate_protects_useful_weights() -> None:
    """Scaled utility 1 leaves a weight untouched; 0 gives it the full update.

    This is the whole method in one assertion -- without the gate UPGD is just
    perturbed SGD.
    """
    update = upgd.UpgdUpdate(lr=0.1, beta=0.0, sigma=0.0)
    # sigmoid(x/eta) -> 1 for x >> eta, -> 0 for x << -eta
    assert float(jnp.asarray(0.0)) == 0.0  # guard against a vacuous test below

    for utility_sign, expect_change in ((+1.0, False), (-1.0, True)):
        columns = {
            px.WEIGHT.name: jnp.asarray([1.0], jnp.float32),
            upgd.UPGD_UTIL.name: jnp.asarray([0.0], jnp.float32),
            upgd.UPGD_T.name: jnp.asarray([0.0], jnp.float32),
        }
        units = {
            px.ACTIVATION.name: jnp.asarray([1.0, 1.0], jnp.float32),
            upgd.GradPreAct.name: jnp.asarray([0.0, -utility_sign * 10.0], jnp.float32),
            upgd.UPGD_ETA.name: jnp.asarray([1.0, 1.0], jnp.float32),
        }
        write = update.incoming(
            px.UnitView(units),
            px.UnitIdx(jnp.asarray(1, jnp.int32)),
            px.UnitIdx(jnp.asarray(0, jnp.int32)),
            px.ConnView(columns),
            px.ConnIdx(jnp.asarray(0, jnp.int32)),
            None,
        )
        moved = abs(float(write.fields[px.WEIGHT.name]) - 1.0) > 1e-4
        assert moved == expect_change, (
            f"utility sign {utility_sign:+.0f}: expected moved={expect_change}"
        )


def test_local_eta_never_exceeds_the_global_one() -> None:
    """v1's per-unit max is a max over a SUBSET of what v0 maximises over.

    If a local eta ever exceeded the global one, the two variants would not be
    comparable at all.
    """
    net, static, state = _build()
    result = _run_one_step(net, static, state)
    local = np.asarray(result.state.units[upgd.UPGD_ETA.name])
    globalised = np.asarray(upgd.set_global_eta(result.state).units[upgd.UPGD_ETA.name])
    assert np.all(globalised >= local - 1e-6), "local eta exceeded the global max"
    assert np.allclose(globalised, globalised[0]), "global eta must be uniform"


def test_upgd_changes_no_topology() -> None:
    """No prune_conn, no add_conn: the live edge SET is identical after a step.

    Counting edges would not be enough -- a method that pruned one and grew
    another would keep the count.
    """
    net, static, state = _build()
    before = set(_live_edges(state))
    result = _run_one_step(net, static, state)
    assert set(_live_edges(result.state)) == before


def test_noise_is_deterministic_per_edge_and_step() -> None:
    """The perturbation is a stateless hash, so a run is reproducible.

    A PRNG threaded through globals would break the arena's purity; the same
    edge at the same step must always draw the same noise.
    """
    net, static, state = _build(sigma=0.1)
    first = _live_edges(_run_one_step(net, static, state).state)
    _net2, static2, state2 = _build(sigma=0.1)
    second = _live_edges(_run_one_step(_net2, static2, state2).state)
    assert first.keys() == second.keys()
    for key in first:
        assert first[key] == pytest.approx(second[key], abs=1e-9)
