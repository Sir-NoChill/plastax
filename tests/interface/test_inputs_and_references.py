"""Named inputs and references to units.

Design document sections 2.4 and 2.5. The errors for inputs and references that do not
resolve are in test_build_errors.py.
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

# An input is declared once, as a module-level object, and used wherever it is meant.
FRAME = px.Input("frame")
GAIN = px.Input("gain")  # a single unit, so it is also a reference to that unit
PICKED = px.UnitRef("picked")  # a unit that is not an input

WF = (np.arange(12, dtype=f32).reshape(3, 4) - 5.0) / 10.0
FRAME_VALUE = np.array([1.0, -2.0, 3.0, 0.5], f32)


class TwoInputs(px.layers.Module):
    """out = fc(frame); `gain` is declared, given a unit, and otherwise unused."""

    def setup(self) -> None:
        self.fc = px.layers.Dense(3, bias=False, init=px.init.array(WF))

    def __call__(self, gain: Any, frame: Any) -> Any:  # not the order of `inputs`
        return self.fc(frame)


@px.policy
class TimesReferencedUnit(px.ForwardConnPolicy[Scalar, px.Globals]):
    """A weighted sum scaled by the activation of one particular unit."""

    scale_by: px.UnitRef

    def map(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: px.Globals,
    ) -> Scalar:
        return (
            c[px.WEIGHT, cid] * u[px.ACTIVATION, src] * u[px.ACTIVATION, self.scale_by]
        )


@px.policy
class RemembersActivation(px.UpdateUnitPolicy[px.Globals]):
    remembered: px.UnitField[f32] = 0.0

    def update(self, u: px.UnitView, i: px.UnitIdx, g: px.Globals) -> Any:
        return px.UnitWrite.of((self.remembered, u[px.ACTIVATION, i]))


def scaled_by(ref: Any) -> Any:
    return px.policies.Conn(
        forward=TimesReferencedUnit(scale_by=ref),
        backward=px.policies.WeightedSumBackward(),
        update=None,
        prune=None,
        add=None,
    )


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def test_inputs_are_matched_to_call_arguments_by_name() -> None:
    net = support.network(TwoInputs(), inputs={FRAME: (4,), GAIN: ()})
    r = support.Runner(net)

    out = r.predict({FRAME: FRAME_VALUE, GAIN: 7.0}).outputs

    support.close(out, WF @ FRAME_VALUE)


def test_input_units_hold_the_data_fed_under_their_input() -> None:
    net = support.network(TwoInputs(), inputs={FRAME: (4,), GAIN: ()})
    r = support.Runner(net)
    r.predict({FRAME: FRAME_VALUE, GAIN: 7.0}, keep=True)

    support.close(r.field(px.ACTIVATION, FRAME), FRAME_VALUE)
    support.close(r.field(px.ACTIVATION, GAIN), np.float32(7.0))


def test_one_input_may_be_given_as_input_shape() -> None:
    fc = px.layers.Dense(3, bias=False, init=px.init.array(WF))
    model = support.topology(lambda m, x: m.fc(x), fc=fc)
    r = support.Runner(support.network(model, input_shape=(4,)))

    support.close(r.predict(FRAME_VALUE).outputs, WF @ FRAME_VALUE)


def test_input_unit_is_an_ordinary_unit_with_a_policy_set() -> None:
    """An input unit runs the policies of its set; only its activation comes from
    data.
    """
    remembers = RemembersActivation()
    watched = px.policies.Unit(update=remembers)

    class Watched(px.layers.Module):
        def setup(self) -> None:
            self.fc = px.layers.Dense(3, bias=False, init=px.init.array(WF))

        def __call__(self, frame: Any) -> Any:
            return self.fc(frame.with_policy(watched))

    r = support.Runner(support.network(Watched(), inputs={FRAME: (4,)}))
    r.step({FRAME: FRAME_VALUE}, np.zeros(3, f32))

    support.close(r.field(remembers.remembered, FRAME), FRAME_VALUE)


# --------------------------------------------------------------------------- #
# References
# --------------------------------------------------------------------------- #
def test_single_unit_input_is_a_reference_a_policy_can_hold() -> None:
    class Gained(px.layers.Module):
        def setup(self) -> None:
            self.fc = px.layers.Dense(
                3, bias=False, init=px.init.array(WF), conn_policy=scaled_by(GAIN)
            )

        def __call__(self, frame: Any, gain: Any) -> Any:
            return self.fc(frame)

    r = support.Runner(support.network(Gained(), inputs={FRAME: (4,), GAIN: ()}))

    for gain in (2.0, -0.5):
        out = r.predict({FRAME: FRAME_VALUE, GAIN: gain}).outputs
        support.close(out, gain * (WF @ FRAME_VALUE))


@pytest.mark.open_question(
    "A policy reads a referenced unit that is computed in the same step. Nothing tells "
    "the topological order that the reader depends on it; here both sit one level "
    "after "
    "the input. Either a reference adds an ordering dependency or this is an error."
)
def test_reference_bound_in_the_topology_resolves_to_that_unit() -> None:
    """`picked` is the single unit of a hidden layer; a later layer is scaled by it."""
    w_pick = np.array([[1.0, 0.0, 0.0, 2.0]], f32)

    class Picks(px.layers.Module):
        def setup(self) -> None:
            self.pick = px.layers.Dense(1, bias=False, init=px.init.array(w_pick))
            self.fc = px.layers.Dense(
                3, bias=False, init=px.init.array(WF), conn_policy=scaled_by(PICKED)
            )

        def __call__(self, frame: Any) -> Any:
            self.pick(frame).bind(PICKED)
            return self.fc(frame)

    net = support.network(Picks(), inputs={FRAME: (4,)}, conn_policy=support.fixed())
    out = support.Runner(net).predict({FRAME: FRAME_VALUE}).outputs

    support.close(out, float(w_pick[0] @ FRAME_VALUE) * (WF @ FRAME_VALUE))


@pytest.mark.open_question(
    "A policy reads a referenced unit that is computed in the same step. Nothing tells "
    "the topological order that the reader depends on it; here both sit one level "
    "after "
    "the input. Either a reference adds an ordering dependency or this is an error."
)
def test_referenced_unit_keeps_its_meaning_when_units_are_laid_out_differently() -> (
    None
):
    """The reference is a name, not a position: adding a layer before it changes
    nothing a policy sees through it.
    """
    w_pick = np.array([[1.0, 0.0, 0.0, 2.0]], f32)

    class PicksAfterAnotherLayer(px.layers.Module):
        def setup(self) -> None:
            self.unrelated = px.layers.Dense(5, bias=False)
            self.pick = px.layers.Dense(1, bias=False, init=px.init.array(w_pick))
            self.fc = px.layers.Dense(
                3, bias=False, init=px.init.array(WF), conn_policy=scaled_by(PICKED)
            )

        def __call__(self, frame: Any) -> Any:
            self.unrelated(frame)
            self.pick(frame).bind(PICKED)
            return self.fc(frame)

    net = support.network(
        PicksAfterAnotherLayer(), inputs={FRAME: (4,)}, conn_policy=support.fixed()
    )
    out = support.Runner(net).predict({FRAME: FRAME_VALUE}).outputs

    support.close(out, float(w_pick[0] @ FRAME_VALUE) * (WF @ FRAME_VALUE))
