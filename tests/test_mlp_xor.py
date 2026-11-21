"""XOR training acceptance test: the mlp_xor example trains XOR to
convergence. End to end through
build_phases/make_step -- forward, loss, backward, update_conn all wired
together via a real gradient-descent training loop, not a hand-fed
reference. The example is now optimizer-parameterized; the fast tests pin the
sgd path and a slow test checks every optimizer the showcase provides.

examples/mlp_xor.py is not part of the plastax distribution (examples/ has
no __init__.py and is not installed), so it is loaded directly by file path
rather than imported as a module -- deliberately not touching
pyproject.toml's pytest config to put examples/ on sys.path, which would be
a much larger-blast-radius change than this one test file needs.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import jax
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

# XOR truth table (mirrors mlp_xor._XOR_TARGETS as classes, spelled out here
# so the assertions below read as "the 4 patterns are classified correctly"
# without a reader needing to cross-reference the example module).
_WANT_CLASS = (0, 1, 1, 0)


def _sgd() -> px.optim.Optimizer:
    return px.optim.sgd(0.5, mlp_xor.GradPreAct)


def test_xor_training_converges_and_classifies_all_patterns_correctly() -> None:
    opt = _sgd()
    key = jax.random.PRNGKey(mlp_xor.SEED)
    net, static, state = mlp_xor.build_net(opt, key)
    state, final_loss = mlp_xor.train(
        net, static, state, num_epochs=5000, verbose=False
    )

    # "loss drops below a small threshold" (task acceptance): the sgd path
    # converges to ~9e-4 (uv run python examples/mlp_xor.py); 0.05 is a
    # generous but decisive threshold -- well below the ~0.5 starting loss,
    # well above float32 noise -- so this only passes on real convergence.
    assert final_loss < 0.05

    _, ok, predictions = mlp_xor.evaluate(opt, static, state)
    assert ok is True
    assert len(predictions) == 4
    for pred, want in zip(predictions, _WANT_CLASS, strict=True):
        got_class = 1 if pred > 0.5 else 0
        assert got_class == want, f"prediction {pred} misclassified (want class {want})"
        # Not just on the right side of 0.5: confidently so, ruling out a
        # network that "passes" by sitting at the decision boundary.
        assert abs(pred - 0.5) > 0.3, f"prediction {pred} too close to the boundary"


@pytest.mark.slow
def test_every_showcase_optimizer_learns_xor() -> None:
    """The example's headline claim: swapping the optimizer is a one-line
    change and every optimizer plastax.optim provides learns XOR through the
    same traits. Mirrors examples/mlp_xor.py's main(); slow (trains five nets
    to convergence)."""
    for name, optimizer, epochs in mlp_xor.showcase():
        net, static, state = mlp_xor.build_net(
            optimizer, jax.random.PRNGKey(mlp_xor.SEED)
        )
        state, loss = mlp_xor.train(
            net, static, state, num_epochs=epochs, verbose=False
        )
        _, ok, _ = mlp_xor.evaluate(optimizer, static, state)
        assert ok, f"{name} failed to learn XOR (final loss {loss:.4f})"
