"""Streaming-iPC acceptance test: the ipc_multilayer example is the
flagship PIPELINE oracle target. Runs the real example end
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
