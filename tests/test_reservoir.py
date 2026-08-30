"""Wide sparse reservoir (examples/reservoir.py).

Three properties carry this example, and each has a way of silently looking
fine while being wrong:

* the analytic spectral scaling `sigma = rho / sqrt(fan_in)` -- the whole reason
  a 10^6-unit reservoir can be built without an eigendecomposition;
* the echo state property, which a mis-scaled reservoir loses by blowing up;
* an OUT-OF-SAMPLE forgetting curve. Scored in-sample, a readout with F
  features and only a few times F rows reconstructs the delayed input partly
  from noise, so the curve flattens onto a spurious floor instead of decaying
  and the reported memory capacity inflates several-fold.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

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


reservoir = _load_example("reservoir")

_UNITS = 4000
_FAN_IN = 16


def _build(rho: float, seed: int = 0) -> tuple[px.NetworkStatic, px.NetworkState[None]]:
    return reservoir.build_reservoir(
        reservoir.Reservoir,
        _UNITS,
        num_inputs=1,
        fan_in=_FAN_IN,
        spectral_radius=rho,
        seed=seed,
    )


def test_fan_in_is_exact() -> None:
    """Every reservoir unit gets exactly fan_in recurrent edges plus the input.

    Pipeline mode keeps one flat bucket, so this also confirms the recurrent
    edges (cycles and all) survived construction -- topological propagation
    would have rejected them outright.
    """
    static, state = _build(0.9)
    bucket = state.conns[0]
    live = ~np.asarray(bucket[px.DEAD.name])
    to_ids = np.asarray(bucket[px.TO_ID.name])[live]
    counts = np.bincount(to_ids, minlength=static.num_units)
    # unit 0 is the input (no incoming); reservoir units take fan_in + 1 input.
    assert counts[0] == 0
    assert np.all(counts[1:] == _FAN_IN + 1)


def test_spectral_radius_matches_analytic_scaling() -> None:
    """Power iteration through the network recovers the requested radius.

    `sigma = rho / sqrt(fan_in)` comes from the circular law; if that scaling
    were wrong the reservoir would still build and run, just with the wrong
    dynamics, so this is the check that keeps the O(1) shortcut honest.
    """
    for rho in (0.5, 0.9):
        static, state = _build(rho)
        estimated = reservoir.spectral_radius(static, state, 1, iterations=40)
        assert abs(estimated - rho) < 0.15 * rho, (
            f"rho={rho}: power iteration gave {estimated}"
        )


def test_echo_state_property_impulse_decays() -> None:
    """A sub-unit spectral radius makes an input impulse fade, not blow up."""
    static, state = _build(0.9)
    probe = np.arange(1, min(_UNITS, 256) + 1, dtype=np.int32)
    impulse = np.zeros((40, 1), dtype=np.float32)
    impulse[0, 0] = 1.0
    states, _, _ = reservoir.drive(static, state, impulse, probe)
    norms = np.linalg.norm(states, axis=1)
    assert norms[0] > 0.0
    assert norms[-1] < 0.25 * norms[0], f"impulse did not fade: {norms[::10]}"
    assert np.all(np.isfinite(norms))


def test_forgetting_curve_decays_out_of_sample() -> None:
    """r^2 is ~1 at delay 1 and collapses at long delay, with no floor.

    The long-delay assertion is the real content: an in-sample fit leaves r^2
    stuck around 0.3 out to arbitrary delays, which is what a spuriously large
    memory capacity is made of.
    """
    static, state = _build(0.95)
    probe = np.arange(1, 129, dtype=np.int32)
    rng = np.random.default_rng(1)
    steps = reservoir.WASHOUT + 1500
    signal = rng.uniform(-1.0, 1.0, size=(steps, 1)).astype(np.float32)
    states, _, _ = reservoir.drive(static, state, signal, probe)
    curve = reservoir.memory_capacity(states, signal, reservoir.WASHOUT)
    assert curve[0] > 0.9, f"delay 1 should be near-perfectly recalled: {curve[0]}"
    assert curve[-1] < 0.05, f"long-delay r^2 floor detected: {curve[-1]}"
    assert float(curve.sum()) < probe.size, "MC cannot exceed the readout width"
