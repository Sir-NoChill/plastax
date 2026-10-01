"""Neuroplastic Expansion traits (examples/ne.py).

Pins the three constraints the paper imposes that a plausible-looking
implementation would quietly drop: the first and last transitions stay DENSE and
untouched, growth follows the cosine envelope and shuts down, and a dormant unit
loses its incoming edges. Also pins that growth never overflows the arena, since
an overflow forces a host-side rebuild and a retrace mid-run.
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
ne = _load_example("ne")

_LAYERS = (17, 32, 32, 4)


def _bucket_live(state: px.NetworkState[None]) -> list[int]:
    """Live-edge count per bucket."""
    return [int((~np.asarray(b[px.DEAD.name])).sum()) for b in state.conns]


def _build(**kwargs: object) -> tuple:
    optimizer = px.optim.adam(0.001, ne.GradPreAct)
    train = ne.make_net(optimizer, mode="train")
    static, state = ne.build_ne_net(train, _LAYERS, seed=0, **kwargs)
    return optimizer, static, state


def test_ends_are_dense_and_interior_is_sparse() -> None:
    """The encoder and decoder start fully connected; only the interior is sparse."""
    _optimizer, _static, state = _build(initial_density=0.2)
    live = _bucket_live(state)
    assert live[0] == _LAYERS[0] * _LAYERS[1], "first transition must be dense"
    assert live[-1] == _LAYERS[-2] * _LAYERS[-1], "last transition must be dense"
    interior_dense = _LAYERS[1] * _LAYERS[2]
    assert live[1] < 0.35 * interior_dense, f"interior not sparse: {live[1]}"


def test_elastic_marks_only_interior_destinations() -> None:
    """Only units whose INCOMING transition is interior may be modified."""
    _optimizer, static, state = _build()
    flags = np.asarray(state.units[ne.NE_ELASTIC.name])
    # layers = (17, 32, 32, 4): unit ids 49..80 are the second hidden layer
    start = _LAYERS[0] + _LAYERS[1]
    expected = np.zeros(static.num_units, dtype=np.float32)
    expected[start : start + _LAYERS[2]] = 1.0
    np.testing.assert_array_equal(flags, expected)


def test_shallow_net_is_rejected() -> None:
    """A net with no interior transition must fail loudly, not silently no-op.

    With one hidden layer every transition is an end, so NE keeping the ends
    dense would leave it nothing to grow -- an easy way to run a null arm and
    report it as a result.
    """
    optimizer = px.optim.adam(0.001, ne.GradPreAct)
    train = ne.make_net(optimizer, mode="train")
    with pytest.raises(ValueError, match="interior"):
        ne.build_ne_net(train, (17, 32, 4), seed=0)


def test_cosine_growth_rate_anneals_to_zero() -> None:
    """The growth schedule starts at alpha and shuts down exactly at T_end."""
    alpha, horizon = 0.5, 100
    schedule = [ne.cosine_growth_rate(t, horizon, alpha) for t in range(horizon + 1)]
    assert schedule[0] == pytest.approx(alpha)
    assert schedule[-1] == pytest.approx(0.0, abs=1e-12)
    assert all(a >= b for a, b in zip(schedule[:-1], schedule[1:], strict=True))
    assert ne.cosine_growth_rate(5, 0, alpha) == 0.0


def test_growth_expands_interior_without_touching_the_ends() -> None:
    """Churn grows the interior and leaves the dense transitions alone.

    The end buckets are the check that matters: a growth policy that ignored the
    elastic gate would happily add edges to the encoder, and the interior count
    alone would not reveal it.
    """
    optimizer, static, state = _build(initial_density=0.2, terminal_density=1.0)
    churn = ne.make_net(optimizer, mode="churn", max_candidates=1024, shortlist=32)
    step = px.make_step(churn, static)
    before = _bucket_live(state)
    state = ne.set_growth_rate(state, 0.5)
    state = ne.set_prune_probability(state, omega=0.0, expected_grown=0.0)
    result = step(
        state, px.StepInputs(inputs=jnp.zeros((_LAYERS[0],), jnp.float32), targets=None)
    )
    after = _bucket_live(result.state)
    assert not bool(result.overflow), "growth must fit the reserved headroom"
    assert after[1] > before[1], f"interior did not grow: {before[1]} -> {after[1]}"
    assert after[0] == before[0], "encoder transition must not change"
    assert after[-1] == before[-1], "decoder transition must not change"


def test_dormant_unit_loses_its_incoming_edges() -> None:
    """A fully dormant elastic unit is pruned; a live one is not.

    NE's dormancy premise cannot be validated end-to-end in the Stage-0 harness
    (drift measurably REDUCES dormancy there), so the mechanism is pinned
    directly by forcing the statistic instead.
    """
    optimizer, static, state = _build(initial_density=0.5)
    churn = ne.make_net(optimizer, mode="churn", tau=1e-6, max_candidates=8)
    step = px.make_step(churn, static)

    elastic = np.flatnonzero(np.asarray(state.units[ne.NE_ELASTIC.name]) > 0.5)
    victim, survivor = int(elastic[0]), int(elastic[1])
    ema = np.full(static.num_units, 1.0, dtype=np.float32)
    ema[victim] = 0.0
    state.units = {**state.units, ne.ACT_EMA.name: jnp.asarray(ema)}
    state = ne.set_growth_rate(state, 0.0)
    state = ne.set_prune_probability(state, omega=1.0, expected_grown=1e9)

    result = step(
        state, px.StepInputs(inputs=jnp.zeros((_LAYERS[0],), jnp.float32), targets=None)
    )
    bucket = result.state.conns[1]
    dead = np.asarray(bucket[px.DEAD.name])
    to_ids = np.asarray(bucket[px.TO_ID.name])
    assert int(np.sum((~dead) & (to_ids == victim))) == 0, "dormant unit kept edges"
    assert int(np.sum((~dead) & (to_ids == survivor))) > 0, "pruned a live unit"


def test_prune_probability_expresses_the_omega_cap() -> None:
    """The cap becomes a per-unit coin whose expectation is the budget."""
    _optimizer, static, state = _build()
    elastic = np.asarray(state.units[ne.NE_ELASTIC.name]) > 0.5
    ema = np.where(elastic, 0.0, 1.0).astype(np.float32)  # every elastic unit dormant
    state.units = {**state.units, ne.ACT_EMA.name: jnp.asarray(ema)}
    dormant = int(elastic.sum())

    state = ne.set_prune_probability(state, omega=0.5, expected_grown=dormant)
    probability = np.asarray(state.units[ne.NE_PRUNE_P.name])
    assert probability[elastic][0] == pytest.approx(0.5)
    assert np.all(probability[~elastic] == 0.0), "non-elastic units are never pruned"

    # a budget larger than the dormant population saturates at certainty
    state = ne.set_prune_probability(state, omega=1.0, expected_grown=10 * dormant)
    assert np.asarray(state.units[ne.NE_PRUNE_P.name])[elastic][0] == pytest.approx(1.0)
    del static
