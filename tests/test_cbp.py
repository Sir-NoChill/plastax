"""Continual Backprop traits (examples/cbp.py).

The load-bearing claim is that CBP's contribution utility -- written in the
literature as a dense column norm of the weight matrix -- is natively one
`BackwardPass` monoid reduction over a unit's outgoing edges. Test 1 pins that
against a NumPy oracle.

The rest pin the two things that were actually wrong when this was first built:
a value-threshold selection rule collapses when the utility distribution has
mass at a point (every ReLU-dormant unit has utility exactly 0, so the rho
quantile IS 0 and strict `<` selects none of them), and the v1 local rule has no
rate control for the same reason -- `0 < scale * mean` holds for any positive
scale.
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
cbp = _load_example("cbp")

_LAYERS = (5, 6, 3)


def _outgoing_abs_weight(
    static: px.NetworkStatic, state: px.NetworkState[None]
) -> np.ndarray:
    """NumPy oracle for sum_{j in out(i)} |w_ij| over live edges."""
    totals = np.zeros(static.num_units)
    for bucket in state.conns:
        dead = np.asarray(bucket[px.DEAD.name])
        from_ids = np.asarray(bucket[px.FROM_ID.name])[~dead]
        weights = np.asarray(bucket[px.WEIGHT.name])[~dead]
        np.add.at(totals, from_ids, np.abs(weights))
    return totals


def _incoming_abs_weight(
    static: px.NetworkStatic, state: px.NetworkState[None]
) -> np.ndarray:
    """NumPy oracle for the adaptation term sum_{j in in(i)} |w_ji|."""
    totals = np.zeros(static.num_units)
    for bucket in state.conns:
        dead = np.asarray(bucket[px.DEAD.name])
        to_ids = np.asarray(bucket[px.TO_ID.name])[~dead]
        weights = np.asarray(bucket[px.WEIGHT.name])[~dead]
        np.add.at(totals, to_ids, np.abs(weights))
    return totals


def test_utility_matches_numpy_oracle() -> None:
    """utility == (|h - f_hat| * sum|w_out|) / sum|w_in|, the paper's form.

    Both reductions are checked at once, and they run in OPPOSITE directions:
    sum|w_out| is a BackwardPass reduction into the source, sum|w_in| a
    ForwardPass reduction into the destination. decay=0 collapses the running
    average to the instantaneous value so no EMA can hide a wrong reduction, and
    at the first churn the activation average is still 0, so f_hat is 0 and the
    mean correction is inert.
    """
    optimizer = px.optim.adam(0.01, cbp.GradPreAct)
    static, state, train_net = cbp.build(optimizer, _LAYERS, 0)
    churn_net = cbp.make_net(optimizer, mode="churn", decay=0.0, maturity=0)
    inputs = jnp.asarray(
        np.random.default_rng(0).standard_normal(_LAYERS[0]), jnp.float32
    )
    targets = jnp.asarray(np.eye(_LAYERS[-1], dtype=np.float32)[0])
    state = px.make_step(train_net, static)(
        state, px.StepInputs(inputs=inputs, targets=targets)
    ).state

    activations = np.asarray(state.units[px.ACTIVATION.name])
    expected = (np.abs(activations) * _outgoing_abs_weight(static, state)) / np.maximum(
        _incoming_abs_weight(static, state), 1e-8
    )

    state = cbp.set_oracle_threshold(static, state, 0.05)
    state = px.make_step(churn_net, static)(
        state, px.StepInputs(inputs=inputs, targets=None)
    ).state
    got = np.asarray(state.units[cbp.CBP_UTIL.name])
    hidden = cbp.hidden_ids(static)
    np.testing.assert_allclose(got[hidden], expected[hidden], atol=1e-6)


def _force_reset(
    static: px.NetworkStatic, state: px.NetworkState[None], unit: int
) -> px.NetworkState[None]:
    """Mark one unit mature and unconditionally selected."""
    ages = np.asarray(state.units[cbp.CBP_AGE.name]).copy()
    ages[unit] = 10_000
    thresh = np.full(static.num_units, -1.0, dtype=np.float32)
    thresh[unit] = np.inf
    state.units = {
        **state.units,
        cbp.CBP_AGE.name: jnp.asarray(ages),
        cbp.CBP_THRESH.name: jnp.asarray(thresh),
    }
    return state


def test_reset_zeroes_outgoing_weights_and_optimizer_state() -> None:
    """A replaced unit stops driving the network and starts adam afresh."""
    optimizer = px.optim.adam(0.01, cbp.GradPreAct)
    static, state, train_net = cbp.build(optimizer, _LAYERS, 0)
    churn_net = cbp.make_net(optimizer, mode="churn", decay=0.0, maturity=5)
    inputs = jnp.asarray(np.ones(_LAYERS[0]), jnp.float32)
    targets = jnp.asarray(np.eye(_LAYERS[-1], dtype=np.float32)[0])
    train_step = px.make_step(train_net, static)
    for _ in range(3):  # accumulate non-zero adam moments to prove they clear
        state = train_step(state, px.StepInputs(inputs=inputs, targets=targets)).state

    victim = int(cbp.hidden_ids(static)[0])
    state = _force_reset(static, state, victim)
    state = px.make_step(churn_net, static)(
        state, px.StepInputs(inputs=inputs, targets=None)
    ).state

    assert victim in cbp.reset_ids(state)
    for bucket in state.conns:
        dead = np.asarray(bucket[px.DEAD.name])
        from_ids = np.asarray(bucket[px.FROM_ID.name])
        to_ids = np.asarray(bucket[px.TO_ID.name])
        weights = np.asarray(bucket[px.WEIGHT.name])
        out_edges = (~dead) & (from_ids == victim)
        np.testing.assert_allclose(weights[out_edges], 0.0, atol=0.0)
        in_edges = (~dead) & (to_ids == victim)
        for spec in optimizer.state_fields:
            column = np.asarray(bucket[spec.name])
            np.testing.assert_allclose(column[in_edges], 0.0, atol=0.0)


def test_immature_unit_is_never_reset() -> None:
    """Age gating protects fresh units even with an unreachable threshold."""
    optimizer = px.optim.adam(0.01, cbp.GradPreAct)
    static, state, _ = cbp.build(optimizer, _LAYERS, 0)
    churn_net = cbp.make_net(optimizer, mode="churn", decay=0.0, maturity=5)
    thresh = np.full(static.num_units, np.inf, dtype=np.float32)
    state.units = {**state.units, cbp.CBP_THRESH.name: jnp.asarray(thresh)}
    # every age starts at 0, i.e. below maturity
    state = px.make_step(churn_net, static)(
        state,
        px.StepInputs(inputs=jnp.ones((_LAYERS[0],), jnp.float32), targets=None),
    ).state
    assert cbp.reset_ids(state) == set()


def test_oracle_selection_survives_utility_ties_at_zero() -> None:
    """v0 selects a rank-controlled count even when most utilities are 0.

    This is the defect the rank-based rule fixes: with a value threshold, the
    rho-quantile of a distribution with mass at 0 IS 0, and `utility < 0` marks
    nothing -- the replacement rate silently collapses toward zero exactly when
    there is most to replace.
    """
    optimizer = px.optim.adam(0.01, cbp.GradPreAct)
    static, state, _ = cbp.build(optimizer, (5, 20, 3), 0)
    hidden = cbp.hidden_ids(static)
    utility = np.zeros(static.num_units, dtype=np.float32)
    utility[hidden[:15]] = 0.0  # a large tie at zero
    utility[hidden[15:]] = np.linspace(1.0, 2.0, hidden.size - 15)
    ages = np.full(static.num_units, 100, dtype=np.int32)
    state.units = {
        **state.units,
        cbp.CBP_UTIL.name: jnp.asarray(utility),
        cbp.CBP_AGE.name: jnp.asarray(ages),
    }
    state = cbp.set_oracle_threshold(static, state, 0.25, maturity=5)
    selected = np.flatnonzero(np.isinf(np.asarray(state.units[cbp.CBP_THRESH.name])))
    assert len(selected) == 5, f"expected 25% of 20 hidden units, got {len(selected)}"
    assert set(selected.tolist()) <= set(hidden[:15].tolist()), (
        "selection must come from the zero-utility tie, not above it"
    )


def test_jaccard_does_not_score_empty_agreement() -> None:
    """Two rules that both did nothing are not evidence that they agree."""
    assert cbp.jaccard(set(), set()) is None
    assert cbp.jaccard({1, 2}, {1, 2}) == 1.0
    assert cbp.jaccard({1}, {1, 2, 3}) == 1 / 3


@pytest.mark.slow
def test_v1_local_threshold_agrees_with_the_oracle_on_rate() -> None:
    """The two-hop local bar now matches v0's replacement RATE, not just its rank.

    History, because the numbers are the point. v1 first compared each unit
    against a fraction of its neighbourhood's mean utility -- a threshold on a
    VALUE, which cannot express "the lowest rho". It fired 4-5x more resets than
    v0 and the agreement gate sat at 0.20 here (0.393 at (128,128)). Replacing
    the bar with the analytic log-normal rho-quantile of the neighbourhood's log
    moments, and carrying v0's fractional count the way the authors' code does,
    took the rate to within ~10% and the gate to ~0.80.

    RL_PLAN's bar is 0.9 and is still NOT met. What remains is genuine ranking
    disagreement on the marginal unit, not rate: at moment_decay 0.5 the two
    fire within 4% of each other and the gate does not move. Asserted as a
    regression floor rather than an equality, since a further improvement should
    pass.
    """
    scores, sizes = cbp.jaccard_gate(
        d=16, hidden_layers=(32, 32), classes=4, num_cycles=30, steps_per_cycle=50
    )
    settled = scores[8:]
    assert settled, "no churn scored -- the gate measured nothing"
    mean_score = float(np.mean(settled))
    assert mean_score > 0.7, f"local threshold regressed to {mean_score:.3f}"

    n0 = float(np.mean([a for a, _ in sizes[8:]]))
    n1 = float(np.mean([b for _, b in sizes[8:]]))
    assert n1 < 1.3 * n0, f"v1 over-fires again: {n1:.2f} vs v0 {n0:.2f}"
    assert n1 > 0.7 * n0, f"v1 under-fires: {n1:.2f} vs v0 {n0:.2f}"


def test_oracle_count_carries_its_fractional_remainder() -> None:
    """v0 hits rho on average instead of flooring it away every churn.

    gnt.py:151-153 carries the fractional part of `rho * |eligible|` across
    steps. Flooring instead is a STANDING rate error, not rounding noise: at
    rho=0.05 over 26 mature units the target is 1.3 per churn and a floor gives
    1, a 23% under-replacement that never corrects. The accumulator has to be
    threaded per level, so the test uses two levels with different populations.
    """
    optimizer = px.optim.sgd(0.1, cbp.GradPreAct)
    static, state, _ = cbp.build(optimizer, (5, 12, 7, 3), 0)
    ages = np.full(static.num_units, 50, dtype=np.int32)
    state.units = {**state.units, cbp.CBP_AGE.name: jnp.asarray(ages)}
    rng = np.random.default_rng(0)

    replaceable = cbp.hidden_ids(static)
    levels = np.asarray(state.units[px.LEVEL.name])
    hidden_levels = np.unique(levels[replaceable])
    per_level = [int((levels[replaceable] == lv).sum()) for lv in hidden_levels]
    rho, churns = 0.3, 60
    accumulator: dict[int, float] = {}
    selected = 0
    for _ in range(churns):
        utility = jnp.asarray(rng.random(static.num_units).astype(np.float32))
        state.units = {**state.units, cbp.CBP_UTIL_HAT.name: utility}
        marked = cbp.set_oracle_threshold(
            static, state, rho, maturity=5, accumulator=accumulator
        )
        selected += int((np.asarray(marked.units[cbp.CBP_THRESH.name]) > 0).sum())

    target = sum(rho * n for n in per_level) * churns
    floored = sum(int(rho * n) for n in per_level) * churns
    assert target > floored, "test is vacuous unless the count has a fractional part"
    assert abs(selected - target) <= len(per_level), (
        f"carried count {selected} strayed from the target {target:.1f}"
    )
