"""Errors raised when a network is built, and shape errors raised where they are
written.

Design document section 6. Every test builds a network that should be refused.

The error types are not decided. These tests assume `px.BuildError` for errors raised
when the network is built and `px.ShapeError` for shape mismatches, and match one word
of the message that names what the user got wrong.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, Bool, Float

import plastax as px

from . import support

Scalar = Float[Array, ""]
Pair = tuple[Scalar, Scalar]
Flag = Bool[Array, ""]
f32 = np.float32

FRAME = px.Input("frame")
EXTRA = px.Input("extra")
REF = px.UnitRef("ref")


def build(model: Any, **attrs: Any) -> support.Runner:
    attrs.setdefault("input_shape", (4,))
    if "inputs" in attrs:
        del attrs["input_shape"]
    return support.Runner(support.network(model, **attrs))


def single(**layer_args: Any) -> Any:
    """x -> fc."""
    fc = px.layers.Dense(3, bias=False, **layer_args)
    return support.topology(lambda m, x: m.fc(x), fc=fc)


# --------------------------------------------------------------------------- #
# Policies used to provoke the errors
# --------------------------------------------------------------------------- #
@px.policy
class PairForward(px.ForwardConnPolicy[Pair, px.Globals]):
    """A connection that hands its unit a pair."""

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
        return (a, a)


@px.policy
class Always(px.PruneUnitPolicy[px.Globals]):
    def predicate(self, u: px.UnitView, i: px.UnitIdx, g: px.Globals) -> Flag:
        return u[px.ACTIVATION, i] == u[px.ACTIVATION, i]


@px.policy
class BranchesOnAValue(px.ForwardUnitPolicy[Scalar, px.Globals]):
    """Uses a Python `if` on a run-time value, which cannot be compiled."""

    combine = px.monoid.sum_

    def apply(self, u: px.UnitView, i: px.UnitIdx, g: px.Globals, acc: Scalar) -> Any:
        if acc > 0:
            return px.UnitWrite.of((px.ACTIVATION, acc))
        return px.UnitWrite.of((px.ACTIVATION, -acc))


@px.policy
class SpawnAlways(px.AddUnitPolicy[px.Globals]):
    child: px.policies.Unit

    def spawn(self, u: px.UnitView, parent: px.UnitIdx, g: px.Globals) -> Flag:
        return u[px.ACTIVATION, parent] == u[px.ACTIVATION, parent]

    def init(
        self, u: px.UnitView, child: px.UnitIdx, parent: px.UnitIdx, g: px.Globals
    ) -> Any:
        return px.UnitWrite.of((px.ACTIVATION, 0.0))


@px.policy
class ConnectEverything(px.ScoreAddConnPolicy[px.Globals]):
    candidates = px.Candidates.EXHAUSTIVE
    selection = px.Selection.ALL
    creates: px.policies.Conn
    into: px.policies.Unit

    def score(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: px.Globals
    ) -> Scalar:
        return jnp.float32(0.0)

    def init(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: px.Globals
    ) -> Any:
        return px.ConnWrite.of((px.WEIGHT, 0.0))


def pair_connections() -> Any:
    return px.policies.Conn(
        forward=PairForward(),
        backward=px.policies.WeightedSumBackward(),
        update=None,
        prune=None,
        add=None,
    )


# --------------------------------------------------------------------------- #
# Shapes: raised at the line that wrote them
# --------------------------------------------------------------------------- #
def raised_in_this_file(excinfo: pytest.ExceptionInfo[BaseException]) -> bool:
    return any(str(entry.path) == __file__ for entry in excinfo.traceback)


def test_adding_handles_of_different_shapes() -> None:
    model = support.topology(
        lambda m, x: m.a(x) + m.b(x),
        a=px.layers.Dense(3, bias=False),
        b=px.layers.Dense(2, bias=False),
    )

    with pytest.raises(px.ShapeError) as excinfo:
        build(model)
    assert raised_in_this_file(excinfo)


def test_reshape_to_a_different_number_of_units() -> None:
    model = support.topology(lambda m, x: x.reshape(3, 2))

    with pytest.raises(px.ShapeError, match="reshape") as excinfo:
        build(model)
    assert raised_in_this_file(excinfo)


def test_convolution_over_a_handle_with_no_grid() -> None:
    model = support.topology(
        lambda m, x: m.conv(x), conv=px.layers.Conv(2, 3, padding="same")
    )

    with pytest.raises(px.ShapeError):
        build(model)


def test_initializer_array_of_the_wrong_shape() -> None:
    with pytest.raises(px.ShapeError):
        build(single(init=px.init.array(np.zeros((2, 4), f32))))


# --------------------------------------------------------------------------- #
# Policy defaults
# --------------------------------------------------------------------------- #
def test_none_connection_default_with_a_layer_that_names_no_set() -> None:
    with pytest.raises(px.BuildError, match="conn_policy"):
        build(single(), conn_policy=None)


def test_none_loss_with_an_output_handle_that_has_no_loss() -> None:
    with pytest.raises(px.BuildError, match="loss"):
        build(single(), loss=None)


def test_none_unit_default_with_a_handle_that_names_no_set() -> None:
    with pytest.raises(px.BuildError, match="unit_policy"):
        build(single(), unit_policy=None)


# --------------------------------------------------------------------------- #
# Inputs and references
# --------------------------------------------------------------------------- #
class TakesFrame(px.layers.Module):
    def setup(self) -> None:
        self.fc = px.layers.Dense(3, bias=False)

    def __call__(self, frame: Any) -> Any:
        return self.fc(frame)


class TakesFrameAndExtra(px.layers.Module):
    def setup(self) -> None:
        self.fc = px.layers.Dense(3, bias=False)

    def __call__(self, frame: Any, extra: Any) -> Any:
        return self.fc(frame)


def test_input_declared_with_no_matching_argument() -> None:
    with pytest.raises(px.BuildError, match="extra"):
        build(TakesFrame(), inputs={FRAME: (4,), EXTRA: (2,)})


def test_argument_with_no_declared_input() -> None:
    with pytest.raises(px.BuildError, match="extra"):
        build(TakesFrameAndExtra(), inputs={FRAME: (4,)})


def reads_ref() -> Any:
    """A connection set whose forward policy holds REF."""

    @px.policy
    class ReadsRef(px.ForwardConnPolicy[Scalar, px.Globals]):
        ref: px.UnitRef

        def map(
            self,
            u: px.UnitView,
            dst: px.UnitIdx,
            src: px.UnitIdx,
            c: px.ConnView,
            cid: px.ConnIdx,
            g: px.Globals,
        ) -> Scalar:
            return u[px.ACTIVATION, src] * u[px.ACTIVATION, self.ref]

    return px.policies.Conn(
        forward=ReadsRef(ref=REF),
        backward=px.policies.WeightedSumBackward(),
        update=None,
        prune=None,
        add=None,
    )


def test_reference_that_is_never_bound() -> None:
    with pytest.raises(px.BuildError, match="ref"):
        build(single(conn_policy=reads_ref()))


def test_reference_bound_twice() -> None:
    def call(m: Any, x: Any) -> Any:
        m.one(x).bind(REF)
        m.other(x).bind(REF)
        return m.fc(x)

    model = support.topology(
        call,
        one=px.layers.Dense(1, bias=False),
        other=px.layers.Dense(1, bias=False),
        fc=px.layers.Dense(3, bias=False, conn_policy=reads_ref()),
    )

    with pytest.raises(px.BuildError, match="ref"):
        build(model)


def test_reference_bound_to_more_than_one_unit() -> None:
    def call(m: Any, x: Any) -> Any:
        m.many(x).bind(REF)
        return m.fc(x)

    model = support.topology(
        call,
        many=px.layers.Dense(2, bias=False),
        fc=px.layers.Dense(3, bias=False, conn_policy=reads_ref()),
    )

    with pytest.raises(px.BuildError, match="ref"):
        build(model)


def test_referenced_unit_that_its_policy_set_can_prune() -> None:
    p = px.policies
    prunable = p.Unit(
        forward=p.SumThenActivate(), backward=p.PassThroughBackward(), prune=Always()
    )

    def call(m: Any, x: Any) -> Any:
        m.one(x).with_policy(prunable).bind(REF)
        return m.fc(x)

    model = support.topology(
        call,
        one=px.layers.Dense(1, bias=False),
        fc=px.layers.Dense(3, bias=False, conn_policy=reads_ref()),
    )

    with pytest.raises(px.BuildError, match="prune"):
        build(model)


# --------------------------------------------------------------------------- #
# Accumulators
# --------------------------------------------------------------------------- #
def test_connection_attached_to_a_unit_with_a_different_accumulator() -> None:
    """Pair-producing connections into summing units, which expect a scalar."""
    with pytest.raises(px.BuildError, match="accumulator"):
        build(single(conn_policy=pair_connections()))


def test_growth_rule_whose_connections_do_not_fit_what_it_connects_into() -> None:
    plain = px.policies.summing()
    growth = ConnectEverything(creates=pair_connections(), into=plain)

    with pytest.raises(px.BuildError, match="accumulator"):
        build(
            single(),
            unit_policy=plain,
            add_conn=growth,
            config=px.Config(propagation=px.Propagation.TOPOLOGICAL, seed=0),
        )


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def test_loss_axis_that_the_outputs_do_not_have() -> None:
    loss = px.SoftmaxCrossEntropyLossPolicy(seed=px.policies.SEED, axis=2)

    with pytest.raises(px.BuildError, match="axis"):
        build(single(), loss=loss)


# --------------------------------------------------------------------------- #
# Dense layers
# --------------------------------------------------------------------------- #
def test_adding_units_with_no_room_to_add_them() -> None:
    child = px.policies.summing()
    config = px.Config(
        propagation=px.Propagation.TOPOLOGICAL, unit_capacity=4 + 3, seed=0
    )

    with pytest.raises(px.BuildError, match="capacity"):
        build(single(), add_unit=SpawnAlways(child=child), config=config)


@pytest.mark.open_question(
    "Units are never added to a dense block. Whether an add-unit policy in a network "
    "with a dense layer is refused when built, or only when a unit would land in that "
    "block, depends on how a created unit joins a layer (deferred)."
)
def test_adding_units_to_a_layer_hinted_dense() -> None:
    child = px.policies.summing()
    config = px.Config(
        format=px.Format.DENSE,
        propagation=px.Propagation.TOPOLOGICAL,
        unit_capacity=64,
        seed=0,
    )

    with pytest.raises(px.BuildError, match="DENSE"):
        build(single(), add_unit=SpawnAlways(child=child), config=config)


@pytest.mark.open_question(
    "Which hints a layer cannot honour is not listed. Assumed: a layer that creates "
    "connections for units made at run time cannot be dense."
)
def test_hint_the_layer_cannot_honour() -> None:
    readout = px.layers.ReduceAll(1, hint=px.Hint(format=px.Format.DENSE))
    model = support.topology(lambda m, x: m.readout(x), readout=readout)

    with pytest.raises(px.BuildError, match="hint"):
        build(model)


# --------------------------------------------------------------------------- #
# Policies and fields
# --------------------------------------------------------------------------- #
def test_policy_that_cannot_be_compiled_is_named() -> None:
    unit = px.policies.Unit(
        forward=BranchesOnAValue(), backward=px.policies.PassThroughBackward()
    )

    with pytest.raises(px.BuildError, match="BranchesOnAValue"):
        build(single(unit_policy=unit))


def test_two_fields_with_one_name_and_different_types() -> None:
    as_float = px.UnitField[np.float32]("clash", default=0.0)
    as_int = px.UnitField[np.int32]("clash", default=0)

    @px.policy
    class WritesBoth(px.UpdateUnitPolicy[px.Globals]):
        def update(self, u: px.UnitView, i: px.UnitIdx, g: px.Globals) -> Any:
            return px.UnitWrite.of((as_float, 1.0), (as_int, 1))

    p = px.policies
    unit = p.Unit(
        forward=p.SumThenActivate(),
        backward=p.PassThroughBackward(),
        update=WritesBoth(),
    )

    with pytest.raises(px.BuildError, match="clash"):
        build(single(unit_policy=unit))


@pytest.mark.open_question(
    "The design says a policy set never sets the weight. Whether `init` naming "
    "`px.WEIGHT` is an error is not stated; assumed that it is."
)
def test_policy_set_init_that_sets_the_weight() -> None:
    conn = px.policies.Conn(
        forward=px.policies.WeightedSum(),
        backward=px.policies.WeightedSumBackward(),
        update=None,
        prune=None,
        add=None,
        init={px.WEIGHT: 1.0},
    )

    with pytest.raises(px.BuildError, match="weight"):
        build(single(conn_policy=conn))
