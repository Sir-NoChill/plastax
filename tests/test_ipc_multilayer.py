"""Streaming-iPC acceptance test (M4/M5 milestone: "the ipc_multilayer
example is the flagship PIPELINE oracle target"). Runs the real example end
to end -- forward, backward, update_conn plus the host-side value-node
dynamics -- and asserts it learns, matching the C++ oracle's own PASS
criterion (final iPC error window below the predict-previous baseline).

Loaded by file path, same rationale as test_mlp_xor.py: examples/ has no
__init__.py and is not part of the installed distribution.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import jax

import plastax as px

_EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def _load_example(name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, _EXAMPLES_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ipc = _load_example("ipc_multilayer")


def test_ipc_learns_and_beats_the_predict_previous_baseline() -> None:
    """The oracle's PASS condition (ipc_multilayer.cpp's final line): the
    windowed iPC prediction error must fall below the predict-previous
    baseline. 1200 steps is a fraction of the example's 2000 but already
    well into the learned regime; the 2x margin makes this decisive rather
    than a boundary pass (the C++ oracle runs ~4x below baseline, and the
    port tracks the same ratio)."""
    key = jax.random.PRNGKey(ipc.SEED)
    static, state = ipc.build_net(key)
    ipc_err, baseline = ipc.run(static, state, total_steps=1200, verbose=False)

    assert baseline > 0.0
    assert ipc_err < baseline
    assert ipc_err * 2.0 < baseline, (
        f"iPC error {ipc_err:.4f} not decisively below baseline {baseline:.4f}"
    )


def test_ipc_run_is_deterministic_for_a_fixed_seed() -> None:
    """Both randomness sources are seeded (jax PRNG for LeCun init, numpy
    default_rng for the data stream), and the step itself is pure array
    arithmetic, so two runs from the same seed return bit-identical windows."""

    def run_once() -> tuple[float, float]:
        static, state = ipc.build_net(jax.random.PRNGKey(ipc.SEED))
        return ipc.run(static, state, total_steps=300, seed=ipc.SEED, verbose=False)

    assert run_once() == run_once()


def test_ipc_net_uses_pipeline_propagation() -> None:
    """iPC needs a single simultaneous snapshot of f(ValueNode) across every
    layer (the defining reason ipc_multilayer.cpp selects Propagation::
    Pipeline): one flat conn bucket, not the per-level topological walk."""
    static, _ = ipc.build_net(jax.random.PRNGKey(ipc.SEED))
    assert static.propagation is px.Propagation.PIPELINE
    assert len(static.level_capacities) == 1
    assert len(static.output_ids) == 1
