"""Global state: hyperparameters, the step counter, and `reset_global`.

Design document sections 3.4 and 3.5. Global state is read-only to every policy. Numeric
hyperparameters in it are run-time values; anything used in plain Python is static.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any

import numpy as np
import pytest
from jaxtyping import Array, Float

import plastax as px

from . import support

Scalar = Float[Array, ""]
f32 = np.float32

X = np.array([1.0, -2.0, 3.0, 0.5], f32)
W = (np.arange(12, dtype=f32).reshape(3, 4) - 5.0) / 10.0

# Appended to each time `Scaled.apply` is traced. A compiled step does not run this
# again.
TRACES: list[str] = []


@dataclasses.dataclass(frozen=True)
class Params:
    scale: float = 2.0


@px.globals
class Globals:
    hp: Params


@px.globals
class Counting:
    hp: Params
    resets: int = 0


@px.policy
class Scaled(px.ForwardUnitPolicy[Scalar, Globals]):
    """Outputs hp.scale * factor * sum; `factor` is a static attribute."""

    combine = px.monoid.sum_
    factor: float = 1.0
    step_seen: px.UnitField[np.int32] = -1

    def apply(self, u: px.UnitView, i: px.UnitIdx, g: Globals, acc: Scalar) -> Any:
        TRACES.append("apply")
        return px.UnitWrite.of(
            (px.ACTIVATION, g.hp.scale * self.factor * acc), (self.step_seen, g.step)
        )


@px.policy
class CountResets(px.ResetGlobalPolicy[Counting]):
    def reset(self, g: Counting) -> Counting:
        return dataclasses.replace(g, resets=g.resets + 1)


@px.policy
class ReadsResets(px.ForwardUnitPolicy[Scalar, Counting]):
    combine = px.monoid.sum_

    def apply(self, u: px.UnitView, i: px.UnitIdx, g: Counting, acc: Scalar) -> Any:
        return px.UnitWrite.of((px.ACTIVATION, acc * 0.0 + g.resets))


def scaled_network(policy: Any, **attrs: Any) -> tuple[Any, Any]:
    out = px.UnitsRef("out")
    unit = px.policies.Unit(forward=policy, backward=px.policies.PassThroughBackward())
    fc = px.layers.Dense(3, bias=False, init=px.init.array(W))
    model = support.topology(lambda m, x: m.fc(x).with_policy(unit).bind(out), fc=fc)
    return support.network(model, input_shape=(4,), **attrs), out


# --------------------------------------------------------------------------- #
# Hyperparameters
# --------------------------------------------------------------------------- #
def test_policy_reads_a_hyperparameter_from_global_state() -> None:
    net, _ = scaled_network(Scaled())
    r = support.Runner(net, globals=Globals(hp=Params(scale=2.0)))

    support.close(r.predict(X).outputs, 2.0 * (W @ X))


def test_changing_a_hyperparameter_does_not_recompile() -> None:
    """Two states with different hyperparameters run through one compiled step."""
    import jax

    net, _ = scaled_network(Scaled())
    step = px.make_step(net)
    key = jax.random.PRNGKey(0)
    inputs = px.StepInputs(inputs=X, targets=np.zeros(3, f32))

    TRACES.clear()
    first = step(net.init(key, globals=Globals(hp=Params(scale=2.0))), inputs)
    traced_once = len(TRACES)
    second = step(net.init(key, globals=Globals(hp=Params(scale=-3.0))), inputs)

    support.close(first.outputs, 2.0 * (W @ X))
    support.close(second.outputs, -3.0 * (W @ X))
    assert traced_once > 0
    assert len(TRACES) == traced_once


def test_static_attribute_is_fixed_when_the_step_is_compiled() -> None:
    net, _ = scaled_network(Scaled(factor=10.0))
    r = support.Runner(net, globals=Globals(hp=Params(scale=2.0)))

    support.close(r.predict(X).outputs, 20.0 * (W @ X))


# --------------------------------------------------------------------------- #
# What the framework supplies
# --------------------------------------------------------------------------- #
def test_step_counter_advances_by_one_each_step() -> None:
    policy = Scaled()
    net, out = scaled_network(policy)
    r = support.Runner(net, globals=Globals(hp=Params()))

    seen = []
    for _ in range(3):
        r.step(X, np.zeros(3, f32))
        seen.append(int(r.field(policy.step_seen, out)[0]))

    assert seen[1] - seen[0] == 1
    assert seen[2] - seen[1] == 1


def test_forward_only_run_does_not_advance_the_step_counter() -> None:
    policy = Scaled()
    net, out = scaled_network(policy)
    r = support.Runner(net, globals=Globals(hp=Params()))
    r.step(X, np.zeros(3, f32))
    before = int(r.field(policy.step_seen, out)[0])

    r.predict(X)  # writes discarded
    r.step(X, np.zeros(3, f32))

    assert int(r.field(policy.step_seen, out)[0]) == before + 1


# --------------------------------------------------------------------------- #
# reset_global: the one way global state changes
# --------------------------------------------------------------------------- #
def test_network_without_reset_global_builds_without_a_warning() -> None:
    net, _ = scaled_network(Scaled())

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        support.Runner(net, globals=Globals(hp=Params()))


def test_reset_global_warns_when_the_network_is_built() -> None:
    net, _ = scaled_network(ReadsResets(), reset_global=CountResets())

    with pytest.warns(px.PerformanceWarning, match="reset_global"):
        support.Runner(net, globals=Counting(hp=Params()))


@pytest.mark.filterwarnings("ignore::plastax.PerformanceWarning")
def test_reset_global_runs_as_the_last_phase_of_every_step() -> None:
    net, _ = scaled_network(ReadsResets(), reset_global=CountResets())
    r = support.Runner(net, globals=Counting(hp=Params()))

    outputs = [float(r.step(X, np.zeros(3, f32)).outputs[0]) for _ in range(3)]

    # The forward pass of step n sees the resets of the n steps before it.
    assert outputs == [0.0, 1.0, 2.0]
