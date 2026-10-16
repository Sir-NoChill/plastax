"""SwiftTD as traits (examples/swifttd.py) against the reference implementation.

The trait version reorders SwiftTD's step -- the three mid-step sums are
taken one step late by the forward pass and the connection update runs the
previous step's second loop before this step's first loop -- so the check is
that this reordering is the same arithmetic as the reference's `Step`
(ported to float32 NumPy in the example): the prediction traces agree to
float32 round-off on a random binary stream with rewards, and on the 5-state
random walk run as one stream with the per-step termination `gamma_t`. The
example is loaded by file path (examples/ is not a package).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

_EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def _load_example(name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, _EXAMPLES_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


swifttd = _load_example("swifttd")

# External-oracle tolerance (float32 reference, segment-sum vs np.dot order).
_RTOL, _ATOL = 1e-4, 1e-5


def test_trait_swifttd_matches_the_reference_on_a_random_stream() -> None:
    hp = swifttd.SwiftTDParams(gamma=0.9, lambda_=0.9, theta=1e-2, alpha_init=1e-2)
    rng = np.random.default_rng(1)
    steps, n = 400, 8
    phis = (rng.random((steps, n)) < 0.3).astype(np.float32)
    rewards = ((rng.random(steps) < 0.08) * np.sign(rng.standard_normal(steps))).astype(
        np.float32
    )
    got, want, driver = swifttd.run_stream(hp, phis, rewards)
    np.testing.assert_allclose(got, want, rtol=_RTOL, atol=_ATOL)
    # The learner did something: predictions are not identically zero and the
    # overshoot bound was exercised (eta is small against 0.3 * 8 features).
    assert np.max(np.abs(want)) > 0.05
    w = swifttd.output_weights(driver, n)
    assert np.any(w != 0.0)


def test_trait_swifttd_matches_the_reference_with_per_step_termination() -> None:
    hp = swifttd.SwiftTDParams(gamma=1.0, lambda_=0.9, theta=1e-2, alpha_init=0.05)
    phis, rewards, gammas = swifttd.random_walk_stream(300, seed=2)
    assert np.any(gammas == 0.0)
    got, want, driver = swifttd.run_stream(hp, phis, rewards, gammas)
    np.testing.assert_allclose(got, want, rtol=_RTOL, atol=_ATOL)
    # 300 episodes is enough for the ordering of the values to be right.
    values = swifttd.output_weights(driver, 5)
    assert np.all(np.diff(values) > 0.0)
    assert 0.0 < values[0] < values[4] < 1.2


def test_control_unit_has_no_edges_and_is_not_a_feature() -> None:
    hp = swifttd.SwiftTDParams()
    n = 3
    net = swifttd.make_net(hp, n)
    static, state, out = swifttd.build(net, n, hp)
    assert tuple(static.input_ids) == (0, 1, 2, n)
    assert tuple(static.output_ids) == (out,)
    conns = state.conns[0]
    src = np.asarray(conns["from_id"])
    dead = np.asarray(conns["dead"])
    assert set(src[~dead].tolist()) == {0, 1, 2}
    assert int(state.units[swifttd.KIND.name][n]) == swifttd.KIND_CONTROL
