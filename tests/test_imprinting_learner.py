"""The imprinting learner's structural rules (examples/imprinting_learner.py).

Each rule is checked on a hand-built net where the thesis's behaviour is
known exactly: a memory unit's register, a pattern unit's threshold over its
inputs' previous activations, one generation step (which parents spawn,
what the children are wired to), removal of an idle feature's output edge,
and the ghost rule (a unit survives while something reads it, and goes the
step after its last reader does). The example is loaded by file path.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import math
import sys
import types
from pathlib import Path

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


il = _load_example("imprinting_learner")
swifttd = sys.modules["swifttd"]

_N_OBS = 4
_GAMMA = _N_OBS  # the control unit
_OUT = _N_OBS + 1


def _hp(**kw: object) -> object:
    return il.ImprintingParams(
        swifttd=swifttd.SwiftTDParams(gamma=0.9, alpha_init=0.01),
        unit_capacity=16,
        shortlist_size=16,
        **kw,
    )


def _net_and_builder(hp: object) -> tuple[type, px.NetworkBuilder]:
    net = il.make_net(hp, _N_OBS)
    b = px.NetworkBuilder(net, {"step": jnp.int32(0)})
    for _ in range(_N_OBS):
        b.add_unit(**{il.TRACK_THR.name: 0.0})
    b.add_unit(**{il.KIND.name: il.KIND_CONTROL})
    b.add_unit(**{il.KIND.name: il.KIND_OUTPUT})
    for i in range(_N_OBS):
        b.mark_input(i)
    b.mark_input(_GAMMA)
    b.mark_output(_OUT)
    return net, b


def _step(
    driver: px.Driver, obs: list[float], reward: float = 0.0, gamma: float = 0.9
) -> None:
    driver.step(
        px.StepInputs(
            inputs=jnp.asarray([*obs, gamma], jnp.float32),
            targets=jnp.asarray([reward], jnp.float32),
        )
    )


def _units(driver: px.Driver) -> dict[str, np.ndarray]:
    return {k: np.asarray(v) for k, v in driver.state.units.items()}


def _live_edges(driver: px.Driver) -> set[tuple[int, int]]:
    c = driver.state.conns[0]
    dead = np.asarray(c["dead"])
    src = np.asarray(c["from_id"])
    dst = np.asarray(c["to_id"])
    return {(int(s), int(d)) for s, d, x in zip(src, dst, dead, strict=True) if not x}


def test_memory_unit_register_relays_its_input_after_the_delay() -> None:
    """k1 = 2, k2 = 1: active for two steps starting two steps after the trigger."""
    hp = _hp()
    net, b = _net_and_builder(hp)
    k1, k2 = 2, 1
    pattern = ((1 << k1) - 1) << k2
    mem = b.add_unit(
        **{
            il.KIND.name: il.KIND_MEMORY,
            il.PULSE_PATTERN.name: pattern,
            il.MEM_SRC.name: 0,
            il.TRACK_THR.name: 1e-3,
        }
    )
    b.add_conn(0, mem, weight=1.0)
    b.add_conn(mem, _OUT, **{swifttd.BETA.name: math.log(0.01)})
    static, state = b.finalize()
    driver = px.Driver(net, static, il.place_output(state, _OUT))
    trace = []
    # The input is active on step 0 only; the memory unit sees it (through
    # act_prev) on step 1, so bits k2..k2+k1-1 are set then and read out on
    # steps 1+k2+1 .. 1+k2+k1.
    for t in range(8):
        _step(driver, [1.0 if t == 0 else 0.0, 0.0, 0.0, 0.0])
        trace.append(int(_units(driver)["activation"][mem]))
    assert trace == [0, 0, 0, 1, 1, 0, 0, 0]


def test_pattern_unit_fires_on_the_fraction_of_previous_inputs() -> None:
    """Threshold 0.7 over three inputs: needs all three (2/3 < 0.7)."""
    hp = _hp()
    net, b = _net_and_builder(hp)
    pat = b.add_unit(
        **{il.KIND.name: il.KIND_PATTERN, il.THRESH.name: 0.7, il.TRACK_THR.name: 1e-3}
    )
    for i in range(3):
        b.add_conn(i, pat, weight=1.0)
    b.add_conn(pat, _OUT, **{swifttd.BETA.name: math.log(0.01)})
    static, state = b.finalize()
    driver = px.Driver(net, static, il.place_output(state, _OUT))
    frames = [[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]]
    trace = []
    for f in frames:
        _step(driver, [float(x) for x in f])
        trace.append(int(_units(driver)["activation"][pat]))
    # One step of lag: the pattern reads the previous frame.
    assert trace == [0, 1, 0, 1, 0]


def _tenured_net(
    hp: object, weights: list[float]
) -> tuple[type, px.NetworkStatic, object]:
    """Observation units wired to the output with the given weights."""
    net, b = _net_and_builder(hp)
    for i, w in enumerate(weights):
        b.add_conn(
            i, _OUT, weight=w, **{swifttd.BETA.name: math.log(hp.swifttd.alpha_init)}
        )
    static, state = b.finalize()
    return net, static, il.place_output(state, _OUT)


def test_generation_wires_pattern_and_memory_children_as_the_thesis_says() -> None:
    """With a budget above the eligible count every active tenured unit spawns."""
    # k_mem + k_pat = 40 >> eligible; eta large so tau never binds the budget.
    hp = _hp(k_mem=20, k_pat=20)
    hp = dataclasses.replace(hp, swifttd=dataclasses.replace(hp.swifttd, eta=10.0))
    # Units 0, 1, 2 tenured (|w| >= 0.01, one negative), unit 3 idle.
    net, static, state = _tenured_net(hp, [0.5, -0.5, 0.02, 0.0])
    driver = px.Driver(net, static, state)
    # Step 0: the backward mirrors w_out before the structural tail, units
    # 0..2 are active and tenured, so all three spawn (p = budget / n_elig
    # >= 1) and growth wires the children in the same step.
    _step(driver, [1.0, 1.0, 1.0, 1.0])
    assert int(driver.state.units_added) == 3
    u = _units(driver)
    children = [i for i in range(_OUT + 1, hp.unit_capacity) if not u["pruned"][i]]
    assert len(children) == 3
    edges = _live_edges(driver)
    eligible = {0, 1, 2}
    for child in children:
        kind = int(u[il.KIND.name][child])
        assert int(u[il.BORN.name][child]) == 0
        assert (child, _OUT) in edges
        if kind == il.KIND_PATTERN:
            assert {s for s, d in edges if d == child} == eligible
            assert int(u["activation"][child]) == 1
            assert any(abs(float(u[il.THRESH.name][child]) - k) < 1e-6 for k in hp.k0)
        else:
            assert kind == il.KIND_MEMORY
            src = int(u[il.MEM_SRC.name][child])
            assert src in eligible
            assert {s for s, d in edges if d == child} == {src}
            assert int(u["activation"][child]) == 0
            assert (
                int(u[il.PULSE.name][child])
                == int(u[il.PULSE_PATTERN.name][child])
                != 0
            )
    # New output edges start at w = 0, beta = ln(alpha_init).
    c = driver.state.conns[0]
    for k in range(len(c["dead"])):
        if (
            not bool(c["dead"][k])
            and int(c["to_id"][k]) == _OUT
            and int(c["from_id"][k]) in children
        ):
            assert float(c["weight"][k]) == 0.0
            assert float(c["swifttd/beta"][k]) == np.float32(
                math.log(hp.swifttd.alpha_init)
            )


def test_generation_is_gated_by_the_correction_ratio_and_the_eligible_set() -> None:
    hp = _hp(k_mem=20, k_pat=20)
    # tau_hat = 3 active * alpha_init (0.01) = 0.03; eta = 0.035 leaves no
    # room for one more alpha_init -> no spawn.
    tight = dataclasses.replace(hp, swifttd=dataclasses.replace(hp.swifttd, eta=0.035))
    net, static, state = _tenured_net(tight, [0.5, 0.5, 0.5, 0.0])
    driver = px.Driver(net, static, state)
    for _ in range(3):
        _step(driver, [1.0, 1.0, 1.0, 0.0])
        assert int(driver.state.units_added) == 0
    # Same net with room, but no active tenured unit -> no spawn either.
    roomy = dataclasses.replace(hp, swifttd=dataclasses.replace(hp.swifttd, eta=10.0))
    net, static, state = _tenured_net(roomy, [0.5, 0.5, 0.5, 0.0])
    driver = px.Driver(net, static, state)
    for _ in range(3):
        _step(driver, [0.0, 0.0, 0.0, 1.0])  # only the idle unit is active
        assert int(driver.state.units_added) == 0


def test_idle_feature_loses_its_output_edge_then_the_unit_when_unread() -> None:
    hp = _hp()
    net, b = _net_and_builder(hp)
    # A generated unit, idle (w below the track threshold) and with a faded
    # trace, that another generated unit reads.
    ghost = b.add_unit(
        **{il.KIND.name: il.KIND_PATTERN, il.THRESH.name: 0.5, il.TRACK_THR.name: 1e-3}
    )
    reader = b.add_unit(
        **{il.KIND.name: il.KIND_PATTERN, il.THRESH.name: 0.5, il.TRACK_THR.name: 1e-3}
    )
    b.add_conn(0, ghost, weight=1.0)
    b.add_conn(ghost, reader, weight=1.0)
    # The ghost's trace is live and just above the fade threshold
    # (e^beta eps_z = 1e-4); one decay (gamma lambda = 0.81) drops it.
    b.add_conn(
        ghost,
        _OUT,
        weight=0.0,
        **{
            swifttd.BETA.name: math.log(0.01),
            swifttd.Z.name: 1.1e-4,
            il.ELIGIBLE.name: 1.0,
        },
    )
    b.add_conn(reader, _OUT, weight=0.5, **{swifttd.BETA.name: math.log(0.01)})
    static, state = b.finalize()
    driver = px.Driver(net, static, il.place_output(state, _OUT))
    _step(driver, [0.0, 0.0, 0.0, 0.0])
    edges = _live_edges(driver)
    # The ghost's output edge is gone on the first step (idle as its trace fades).
    assert (ghost, _OUT) not in edges
    assert (ghost, reader) in edges and (reader, _OUT) in edges
    assert not bool(_units(driver)["pruned"][ghost])
    # The reader is tenured (0.5) so it stays; the ghost stays while read.
    for _ in range(3):
        _step(driver, [0.0, 0.0, 0.0, 0.0])
    assert not bool(_units(driver)["pruned"][ghost])
    assert (ghost, reader) in _live_edges(driver)


def test_unit_nothing_reads_is_removed_the_step_after_its_last_reader() -> None:
    hp = _hp()
    net, b = _net_and_builder(hp)
    orphan = b.add_unit(
        **{il.KIND.name: il.KIND_PATTERN, il.THRESH.name: 0.5, il.TRACK_THR.name: 1e-3}
    )
    b.add_conn(0, orphan, weight=1.0)
    b.add_conn(
        orphan,
        _OUT,
        weight=0.0,
        **{
            swifttd.BETA.name: math.log(0.01),
            swifttd.Z.name: 1.1e-4,
            il.ELIGIBLE.name: 1.0,
        },
    )
    static, state = b.finalize()
    driver = px.Driver(net, static, il.place_output(state, _OUT))
    _step(driver, [0.0, 0.0, 0.0, 0.0])
    # Step 0: output edge pruned; out_deg (from the backward, before the
    # prune) still 1, so the unit survives.
    assert (orphan, _OUT) not in _live_edges(driver)
    assert not bool(_units(driver)["pruned"][orphan])
    _step(driver, [0.0, 0.0, 0.0, 0.0])
    # Step 1: out_deg 0 -> pruned, and its incoming edge tombstoned.
    assert bool(_units(driver)["pruned"][orphan])
    assert (0, orphan) not in _live_edges(driver)


def test_a_feature_that_was_never_active_is_not_removed() -> None:
    """No trace, no removal: a fresh memory feature waits out its delay."""
    hp = _hp()
    net, b = _net_and_builder(hp)
    mem = b.add_unit(
        **{
            il.KIND.name: il.KIND_MEMORY,
            il.PULSE_PATTERN.name: 1 << 5,
            il.MEM_SRC.name: 0,
            il.TRACK_THR.name: 1e-3,
        }
    )
    b.add_conn(0, mem, weight=1.0)
    b.add_conn(mem, _OUT, weight=0.0, **{swifttd.BETA.name: math.log(0.01)})
    static, state = b.finalize()
    driver = px.Driver(net, static, il.place_output(state, _OUT))
    for _ in range(6):
        _step(driver, [0.0, 0.0, 0.0, 0.0])
    assert (mem, _OUT) in _live_edges(driver)
    assert not bool(_units(driver)["pruned"][mem])


def test_observation_features_are_never_removed() -> None:
    hp = _hp()
    net, static, state = _tenured_net(hp, [0.0, 0.0, 0.0, 0.0])
    driver = px.Driver(net, static, state)
    for _ in range(5):
        _step(driver, [0.0, 0.0, 0.0, 0.0])
    assert {(i, _OUT) for i in range(_N_OBS)} <= _live_edges(driver)
    assert not np.any(_units(driver)["pruned"][: _OUT + 1])
