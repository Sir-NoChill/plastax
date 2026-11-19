"""Pieces of the imprinting learner from `docs/design/sketches/imprinting_learner.py`.

Two parts are isolated here:

- the TD-error loss, which reads a referenced unit (the discount) and the readout's
  previous prediction;
- the feature unit, one policy holding the pattern and the memory behaviour, selected by
a
  field.

The creation and removal of units is in test_unit_lifecycle.py. SwiftTD's update is not
extracted: checking it needs a reference implementation of the algorithm, which is a
parity check and out of scope for now.

The policies follow the sketch. One simplification: an imprint connection here reads its
source's current activation, where the sketch reads the previous step's, so that the
tests need no extra bookkeeping units.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, Float, Int32

import plastax as px

from . import support

Scalar = Float[Array, ""]
Pair = tuple[Scalar, Scalar]
f32 = np.float32

# --------------------------------------------------------------------------- #
# The TD-error loss
# --------------------------------------------------------------------------- #
FRAME = px.Input("frame")
GAMMA = px.Input("gamma")  # one unit carrying this step's discount
V_OLD = px.UnitField[f32]("v_old", default=0.0)
DELTA = px.UnitField[f32]("delta", default=0.0)

W_READOUT = np.array([[0.5, -1.0, 2.0]], f32)


@px.policy
class RememberPrediction(px.UpdateUnitPolicy[px.Globals]):
    def update(self, u: px.UnitView, i: px.UnitIdx, g: px.Globals) -> Any:
        return px.UnitWrite.of((V_OLD, u[px.ACTIVATION, i]))


@px.policy
class TDErrorLoss(px.LossPolicy[px.Globals]):
    """delta = r + gamma_t * v - v_old, written to the readout."""

    gamma: px.UnitRef
    seed: px.UnitField[f32]

    def calculate_loss(
        self,
        u: px.UnitView,
        outputs: Int32[Array, " n"],
        targets: Float[Array, " n"],
        g: px.Globals,
    ) -> tuple[Float[Array, ""], Float[Array, " n"]]:
        v = u.gather(px.ACTIVATION, outputs)
        delta = targets + u[px.ACTIVATION, self.gamma] * v - u.gather(V_OLD, outputs)
        return jnp.sum(0.5 * delta * delta), delta


class Predictor(px.layers.Module):
    def setup(self) -> None:
        self.readout = px.layers.Dense(1, bias=False, init=px.init.array(W_READOUT))

    def __call__(self, frame: Any, gamma: Any) -> Any:
        p = px.policies
        readout = p.Unit(
            forward=p.SumThenActivate(),
            backward=p.PassThroughBackward(),
            update=RememberPrediction(),
        )
        return self.readout(frame).with_policy(readout)


def predictor() -> support.Runner:
    net = support.network(
        Predictor(),
        inputs={FRAME: (3,), GAMMA: ()},
        conn_policy=support.fixed(),
        loss=TDErrorLoss(gamma=GAMMA, seed=DELTA),
    )
    return support.Runner(net)


def test_td_error_uses_the_discount_unit_and_the_previous_prediction() -> None:
    frames = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0]], f32)
    rewards = np.array([1.0, -0.5, 2.0], f32)
    gammas = np.array([0.9, 0.5, 0.0], f32)
    r = predictor()

    v_old = 0.0
    for frame, reward, gamma in zip(frames, rewards, gammas, strict=True):
        result = r.step({FRAME: frame, GAMMA: gamma}, np.array([reward], f32))

        v = float(W_READOUT[0] @ frame)
        delta = reward + gamma * v - v_old
        support.close(result.outputs, [v])
        support.close(result.loss, 0.5 * delta * delta)
        v_old = v


def test_td_error_is_written_to_the_readout() -> None:
    readout = px.UnitsRef("readout")

    class Bound(Predictor):
        def __call__(self, frame: Any, gamma: Any) -> Any:
            return super().__call__(frame, gamma).bind(readout)

    net = support.network(
        Bound(),
        inputs={FRAME: (3,), GAMMA: ()},
        conn_policy=support.fixed(),
        loss=TDErrorLoss(gamma=GAMMA, seed=DELTA),
    )
    r = support.Runner(net)
    frame = np.array([1.0, 0.0, 1.0], f32)
    r.step({FRAME: frame, GAMMA: 0.9}, np.array([1.0], f32))

    v = float(W_READOUT[0] @ frame)
    support.close(r.field(DELTA, readout), [1.0 + 0.9 * v])
    support.close(r.field(V_OLD, readout), [v])


# --------------------------------------------------------------------------- #
# The feature unit: pattern or memory, chosen by a field
# --------------------------------------------------------------------------- #
UNIT_INIT = pytest.mark.open_question(
    "A feature's threshold and pulse are fields of the unit. In the sketch a spawning "
    "policy writes them; here the topology makes the unit, which needs initial unit "
    "field values. `px.policies.Unit(init=...)` is a guess at that form."
)


@px.policy
class ImprintForward(px.ForwardConnPolicy[Pair, px.Globals]):
    """Contributes (its source's activation, 1)."""

    def map(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: px.Globals,
    ) -> Pair:
        a = u[px.ACTIVATION, src]
        return (a, a * 0.0 + 1.0)


@px.policy
class FeatureForward(px.ForwardUnitPolicy[Pair, px.Globals]):
    """pattern: fires when at least `thresh` of its inputs are active.
    memory: a shift register; bit 0 is 'active now', an active input reloads the pulse.
    """

    combine = (px.monoid.sum_, px.monoid.sum_)  # (active inputs, inputs)
    is_mem: px.UnitField[np.bool_] = False
    thresh: px.UnitField[f32] = 0.0
    pulse: px.UnitField[np.uint32] = 0
    pulse_pattern: px.UnitField[np.uint32] = 0  # ((1 << k1) - 1) << k2

    def apply(self, u: px.UnitView, i: px.UnitIdx, g: px.Globals, acc: Pair) -> Any:
        active, count = acc
        is_mem = u[self.is_mem, i]
        fires = (count > 0.0) & (active >= u[self.thresh, i] * count)
        pulse = u[self.pulse, i]
        reload = jnp.where(active > 0.0, u[self.pulse_pattern, i], np.uint32(0))
        return px.UnitWrite.of(
            (
                px.ACTIVATION,
                jnp.where(is_mem, (pulse & 1).astype(f32), fires.astype(f32)),
            ),
            (self.pulse, jnp.where(is_mem, (pulse >> 1) | reload, pulse)),
        )


def feature_network(
    n_inputs: int, wiring: Any = None, **unit_fields: Any
) -> tuple[support.Runner, FeatureForward]:
    """`n_inputs` inputs imprinted on by one feature unit with the given field
    values.
    """
    forward = FeatureForward()
    p = px.policies
    feature = p.Unit(
        forward=forward,
        backward=p.PassThroughBackward(),
        init={getattr(forward, name): value for name, value in unit_fields.items()},
    )
    imprint = p.Conn(
        forward=ImprintForward(),
        backward=p.WeightedSumBackward(),
        update=None,
        prune=None,
        add=None,
    )
    layer_args = {} if wiring is None else {"connectivity": wiring}
    fc = px.layers.Dense(
        1,
        bias=False,
        init=px.init.constant(1.0),
        conn_policy=imprint,
        unit_policy=feature,
        **layer_args,
    )
    model = support.topology(lambda m, x: m.fc(x), fc=fc)
    net = support.network(model, input_shape=(n_inputs,))
    return support.Runner(net), forward


@UNIT_INIT
@pytest.mark.parametrize(
    ("thresh", "inputs", "fires"),
    [
        (0.6, [1, 1, 1, 0], True),  # 3 of 4 = 0.75
        (0.6, [1, 1, 0, 0], False),  # 2 of 4 = 0.5
        (0.5, [1, 1, 0, 0], True),  # exactly at the threshold
        (0.9, [1, 1, 1, 1], True),
        (0.9, [0, 0, 0, 0], False),
        (0.0, [0, 0, 0, 0], True),  # a threshold of zero is always met
    ],
)
def test_pattern_fires_when_enough_of_its_inputs_are_active(
    thresh: float, inputs: list[int], fires: bool
) -> None:
    r, _ = feature_network(4, is_mem=False, thresh=thresh)

    out = r.predict(np.array(inputs, f32)).outputs

    support.close(out, [1.0 if fires else 0.0])


@UNIT_INIT
def test_pattern_with_no_inputs_never_fires() -> None:
    """All of its imprint connections are gone: the count is zero."""
    none = px.layers.pairs(src=np.array([], np.int64), dst=np.array([], np.int64))
    r, _ = feature_network(4, wiring=none, is_mem=False, thresh=0.0)

    support.close(r.predict(np.ones(4, f32)).outputs, [0.0])


def memory_activity(triggers: list[int], k1: int, k2: int) -> list[float]:
    """The memory unit's activation at each step, given its input at each step."""
    pattern = np.uint32(((1 << k1) - 1) << k2)
    r, _ = feature_network(1, is_mem=True, pulse_pattern=pattern)
    return [
        float(r.predict(np.array([t], f32), keep=True).outputs[0]) for t in triggers
    ]


@UNIT_INIT
def test_memory_is_active_for_k1_steps_after_a_delay_of_k2() -> None:
    # Triggered at step 0 with length 2 and delay 1: active at steps 2 and 3.
    assert memory_activity([1, 0, 0, 0, 0, 0], k1=2, k2=1) == [0, 0, 1, 1, 0, 0]


@UNIT_INIT
def test_memory_with_no_delay_is_active_from_the_next_step() -> None:
    assert memory_activity([1, 0, 0, 0, 0], k1=3, k2=0) == [0, 1, 1, 1, 0]


@UNIT_INIT
def test_memory_triggered_again_while_running_extends_its_pulse() -> None:
    # Triggers at steps 0 and 1 overlap: active at steps 2, 3 and 4.
    assert memory_activity([1, 1, 0, 0, 0, 0], k1=2, k2=1) == [0, 0, 1, 1, 1, 0]


@UNIT_INIT
def test_memory_that_is_never_triggered_stays_inactive() -> None:
    assert memory_activity([0, 0, 0, 0], k1=2, k2=1) == [0, 0, 0, 0]


@UNIT_INIT
def test_one_policy_serves_both_kinds_side_by_side() -> None:
    """Two feature units read the same input; `is_mem` alone tells them apart."""
    forward = FeatureForward()
    p = px.policies
    imprint = p.Conn(
        forward=ImprintForward(),
        backward=p.WeightedSumBackward(),
        update=None,
        prune=None,
        add=None,
    )

    def feature(**fields: Any) -> Any:
        return p.Unit(
            forward=forward,
            backward=p.PassThroughBackward(),
            init={getattr(forward, name): value for name, value in fields.items()},
        )

    def layer(unit: Any) -> Any:
        return px.layers.Dense(
            1,
            bias=False,
            init=px.init.constant(1.0),
            conn_policy=imprint,
            unit_policy=unit,
        )

    model = support.topology(
        lambda m, x: (m.pattern(x), m.memory(x)),
        pattern=layer(feature(is_mem=False, thresh=1.0)),
        memory=layer(feature(is_mem=True, pulse_pattern=np.uint32(0b1))),
    )
    r = support.Runner(support.network(model, input_shape=(1,)))

    seen = [r.predict(np.array([t], f32), keep=True).outputs for t in (1, 0, 0)]
    pattern = [float(p_out[0]) for p_out, _ in seen]
    memory = [float(m_out[0]) for _, m_out in seen]

    assert pattern == [1, 0, 0]  # fires with its input
    assert memory == [0, 1, 0]  # fires one step after its input
