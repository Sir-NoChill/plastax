"""Running a network: evaluation modes, propagation modes, formats.

Design document sections 4 and 5.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from jaxtyping import Array, Float

import plastax as px

from . import support

Scalar = Float[Array, ""]
f32 = np.float32

X = np.array([1.0, -2.0, 3.0, 0.5], f32)
T = np.array([0.5, -1.0], f32)
W1 = (np.arange(12, dtype=f32).reshape(3, 4) - 5.0) / 10.0
W2 = np.array([[0.5, -1.0, 2.0], [1.5, 0.25, -0.75]], f32)


@px.policy
class CountsForwardPasses(px.ForwardUnitPolicy[Scalar, px.Globals]):
    """A unit whose state advances on every forward pass."""

    combine = px.monoid.sum_
    passes: px.UnitField[np.int32] = 0

    def apply(self, u: px.UnitView, i: px.UnitIdx, g: px.Globals, acc: Scalar) -> Any:
        return px.UnitWrite.of(
            (px.ACTIVATION, acc), (self.passes, u[self.passes, i] + 1)
        )


def two_layers(
    *, counter: Any = None, propagation: str = "TOPOLOGICAL", fmt: str = "AUTO"
) -> tuple[support.Runner, Any, Any, Any]:
    """x -> fc1 -> fc2. Returns (runner, fc1, fc2, ref to the hidden units)."""
    hidden = px.UnitsRef("hidden")
    hint = px.Hint(format=getattr(px.Format, fmt))
    fc1 = px.layers.Dense(3, bias=False, init=px.init.array(W1), hint=hint)
    fc2 = px.layers.Dense(2, bias=False, init=px.init.array(W2), hint=hint)

    def call(m: Any, x: Any) -> Any:
        h = m.fc1(x)
        if counter is not None:
            h = h.with_policy(
                px.policies.Unit(
                    forward=counter, backward=px.policies.PassThroughBackward()
                )
            )
        return m.fc2(h.bind(hidden))

    net = support.network(
        support.topology(call, fc1=fc1, fc2=fc2),
        input_shape=(4,),
        config=px.Config(
            propagation=getattr(px.Propagation, propagation),
            representation_optimization=False,
            seed=0,
        ),
    )
    return support.Runner(net), fc1, fc2, hidden


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def test_forward_only_run_returns_the_outputs() -> None:
    r, _, _, _ = two_layers()

    support.close(r.predict(X).outputs, W2 @ W1 @ X)


def test_discarding_writes_leaves_the_state_as_it_was() -> None:
    counter = CountsForwardPasses()
    r, fc1, fc2, hidden = two_layers(counter=counter)
    for _ in range(3):
        r.predict(X)

    np.testing.assert_array_equal(r.field(counter.passes, hidden), 0)
    support.close(r.field(px.ACTIVATION, hidden), np.zeros(3, f32))
    support.close(r.weights(fc1), W1)
    support.close(r.weights(fc2), W2)


def test_keeping_writes_advances_what_the_forward_policies_wrote() -> None:
    counter = CountsForwardPasses()
    r, fc1, fc2, hidden = two_layers(counter=counter)
    for _ in range(3):
        r.predict(X, keep=True)

    np.testing.assert_array_equal(r.field(counter.passes, hidden), 3)
    support.close(r.field(px.ACTIVATION, hidden), W1 @ X)
    # Only the forward policies ran: nothing was trained.
    support.close(r.weights(fc1), W1)
    support.close(r.weights(fc2), W2)


def test_full_step_runs_the_forward_policies_once_and_trains() -> None:
    counter = CountsForwardPasses()
    r, fc1, fc2, hidden = two_layers(counter=counter)
    r.step(X, T)

    np.testing.assert_array_equal(r.field(counter.passes, hidden), 1)
    assert not np.allclose(r.weights(fc1), W1)
    assert not np.allclose(r.weights(fc2), W2)


# --------------------------------------------------------------------------- #
# Propagation
# --------------------------------------------------------------------------- #
def test_topological_mode_carries_an_input_to_the_output_in_one_step() -> None:
    r, _, _, _ = two_layers(propagation="TOPOLOGICAL")

    support.close(r.predict(X, keep=True).outputs, W2 @ W1 @ X)


@pytest.mark.open_question(
    "In pipeline mode every unit reads the previous step's values. Assumed: a unit "
    "that "
    "reads an input sees the value fed in this step, so the delay is one step per "
    "layer "
    "after the first."
)
def test_pipeline_mode_carries_an_input_one_layer_per_step() -> None:
    """Every unit reads the previous step's values, so two layers take two steps."""
    r, _, _, hidden = two_layers(propagation="PIPELINE")

    first = r.predict(X, keep=True).outputs
    support.close(first, np.zeros(2, f32))  # the hidden units had not fired yet
    support.close(r.field(px.ACTIVATION, hidden), W1 @ X)

    second = r.predict(X, keep=True).outputs
    support.close(second, W2 @ W1 @ X)


def test_pipeline_mode_settles_to_the_topological_result_for_a_constant_input() -> None:
    r, _, _, _ = two_layers(propagation="PIPELINE")
    for _ in range(4):
        out = r.predict(X, keep=True).outputs

    support.close(out, W2 @ W1 @ X)


# --------------------------------------------------------------------------- #
# Formats
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fmt", ["AUTO", "COO", "CSR", "DENSE"])
def test_every_format_gives_the_same_forward_pass_and_training_step(fmt: str) -> None:
    r, fc1, fc2, _ = two_layers(fmt=fmt)
    result = r.step(X, T)

    seed = W2 @ W1 @ X - T
    support.close(result.outputs, W2 @ W1 @ X)
    support.close(r.weights(fc2), W2 - 0.1 * np.outer(seed, W1 @ X))
    support.close(r.weights(fc1), W1 - 0.1 * np.outer(W2.T @ seed, X))


@pytest.mark.parametrize("fmt", ["COO", "CSR", "DENSE"])
def test_layer_is_stored_in_the_format_its_hint_names(fmt: str) -> None:
    """With representation optimization off, the framework does not change a format."""
    r, fc1, _, _ = two_layers(fmt=fmt)
    r.step(X, T)

    assert px.inspect.format(r.net, r.state, fc1) is getattr(px.Format, fmt)


def test_layer_hint_overrides_the_network_format() -> None:
    fc = px.layers.Dense(
        3, bias=False, init=px.init.array(W1), hint=px.Hint(format=px.Format.CSR)
    )
    model = support.topology(lambda m, x: m.fc(x), fc=fc)
    net = support.network(
        model,
        input_shape=(4,),
        config=px.Config(
            format=px.Format.COO, propagation=px.Propagation.TOPOLOGICAL, seed=0
        ),
    )
    r = support.Runner(net)

    assert px.inspect.format(r.net, r.state, fc) is px.Format.CSR
