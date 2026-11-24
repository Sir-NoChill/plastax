"""The checks that run when a network is built, before anything is materialized.

Plan `plans/20261123T153015Z-layer-policies-stage2.md` (workspace root). Every policy
in a network's resolved table is traced once with abstract values. A connection's
`map` is compared with the `combine` of the units that reduce it: the units it ends in
going forward, the units it starts from going backward. Every write is compared with
the field it is written to. Every fault is found, and one error reports them all.

`Net.faults()` returns the faults without raising; `Net.init` raises a `BuildError`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import plastax as px
import plastax.layers as pxl
from plastax.layers import Dense, Units

KEY = jax.random.key(0)
GRAD = px.FieldSpec("grad", np.float32, 0.0)
SEEN = px.FieldSpec("seen", np.float32, 0.0)
FLAG = px.FieldSpec("flag", np.bool_, False)
MARK = px.FieldSpec("mark", np.bool_, False)  # on connections


# --------------------------------------------------------------------------- #
# Policies that return whatever a test gives them
# --------------------------------------------------------------------------- #
class Sends(px.policies.ForwardConnPolicy, px.policies.BackwardConnPolicy):
    """A connection half whose `map` returns `shape(x)` for a scalar `x`."""

    def __init__(self, shape: Callable[[jax.Array], Any]) -> None:
        self.shape = shape
        self.calls = 0

    def map(
        self,
        u: px.UnitView,
        a: px.UnitIdx,
        b: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: Any,
    ) -> Any:
        self.calls += 1
        return self.shape(c[px.WEIGHT, cid] * u[px.ACTIVATION, b])


class Reduces(px.policies.ForwardUnitPolicy, px.policies.BackwardUnitPolicy):
    """A unit half that reduces with `combine` and writes `field`."""

    def __init__(self, combine: Any, field: px.FieldSpec[Any] = px.ACTIVATION) -> None:
        self.combine = combine
        self.field = field

    def apply(self, u: px.UnitView, i: px.UnitIdx, g: Any, acc: Any) -> px.UnitWrite:
        return px.UnitWrite.of((self.field, u[self.field, i]))


class Writes(px.policies.UpdateUnitPolicy):
    """A unit update that writes what a test gives it."""

    def __init__(self, *pairs: tuple[px.FieldSpec[Any], Any]) -> None:
        self.pairs = pairs

    def update(self, u: px.UnitView, i: px.UnitIdx, g: Any) -> px.UnitWrite:
        return px.UnitWrite.of(*self.pairs)


class Marks(px.policies.UpdateConnPolicy):
    """A connection update that writes what a test gives it on the incoming side."""

    def __init__(self, *pairs: tuple[px.FieldSpec[Any], Any]) -> None:
        self.pairs = pairs

    def incoming(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: Any,
    ) -> px.ConnWrite:
        return px.ConnWrite.of(*self.pairs)

    def outgoing(
        self,
        u: px.UnitView,
        src: px.UnitIdx,
        dst: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: Any,
    ) -> px.ConnWrite:
        return px.ConnWrite.of()


def one(x: jax.Array) -> Any:
    return x


def two(x: jax.Array) -> Any:
    return (x, x)


sum_, pair = px.monoid.sum_, (px.monoid.sum_, px.monoid.sum_)


def two_layers(a: dict[str, Any] | None = None, b: dict[str, Any] | None = None) -> Any:
    """inputs -> `a` -> `b`, two dense layers with no bias, each given keywords."""

    class TwoLayers(pxl.Module):
        def setup(self) -> None:
            self.a = Dense(4, bias=False, **(a or {}))
            self.b = Dense(3, bias=False, **(b or {}))

        def __call__(self, x: Units) -> Units:
            return self.b(self.a(x))

    return TwoLayers()


def network(topology_: Any = None, **fields: Any) -> type[px.Network[None]]:
    class Net(px.Network[None]):
        topology = topology_ if topology_ is not None else two_layers()
        input_shape = (5,)
        forward = fields.pop(
            "forward", px.policies.Forward(conn=Sends(one), unit=Reduces(sum_))
        )
        backward = fields.pop("backward", None)
        update = fields.pop("update", None)
        extra_unit_fields = (GRAD, SEEN, FLAG)
        extra_conn_fields = (MARK,)

    assert not fields
    return Net


def kinds(net: type[px.Network[None]]) -> list[str]:
    return [fault.kind for fault in net.faults()]


# --------------------------------------------------------------------------- #
# Accumulators
# --------------------------------------------------------------------------- #
def test_a_map_and_a_combine_that_agree_are_sound() -> None:
    scalar = network()
    tuples = network(forward=px.policies.Forward(conn=Sends(two), unit=Reduces(pair)))
    keyed = network(
        forward=px.policies.Forward(
            conn=Sends(lambda x: {"w": x, "n": x}),
            unit=Reduces({"w": sum_, "n": px.monoid.max_}),
        )
    )

    assert scalar.faults() == []
    assert tuples.faults() == []
    assert keyed.faults() == []


def test_one_scalar_sent_to_units_that_combine_two_is_a_fault() -> None:
    net = network(forward=px.policies.Forward(conn=Sends(one), unit=Reduces(pair)))

    fault = net.faults()[0]

    assert fault.kind == "accumulator"
    assert fault.where == "a.weight, forward"
    assert fault.policy == "Sends.map"
    assert "one scalar" in fault.message
    assert "a tuple of 2 monoids" in fault.message
    assert "the units of `a.units`" in fault.message


def test_a_dict_with_other_keys_is_a_fault() -> None:
    net = network(
        forward=px.policies.Forward(
            conn=Sends(lambda x: {"w": x, "count": x}),
            unit=Reduces({"w": sum_, "n": sum_}),
        )
    )

    fault = net.faults()[0]

    assert fault.kind == "accumulator"
    assert "keys count, w" in fault.message and "keys n, w" in fault.message


def test_a_map_that_returns_a_vector_where_a_scalar_is_combined_is_a_fault() -> None:
    net = network(
        forward=px.policies.Forward(
            conn=Sends(lambda x: jnp.stack([x, x])), unit=Reduces(sum_)
        )
    )

    assert "an array of shape (2,)" in net.faults()[0].message


def test_a_backward_map_is_compared_with_the_units_it_starts_from() -> None:
    # `a`'s units combine a pair going backward; everything else combines one value.
    # The connections that start at `a`'s units are `b`'s, so it is `b` that must send
    # a pair, and `a`'s own connections, which start at the inputs, that must not.
    backward = px.policies.Backward(conn=Sends(one), unit=Reduces(sum_, GRAD))
    a = {"backward": px.policies.Backward(unit=Reduces(pair, GRAD))}
    sends_pair = {"backward": px.policies.Backward(conn=Sends(two))}

    sound = network(two_layers(a, sends_pair), backward=backward)
    wrong = network(two_layers(a), backward=backward)

    assert [f.kind for f in sound.faults()] == []
    (fault,) = wrong.faults()
    assert fault.where == "b.weight, backward"
    assert "the units of `a.units`" in fault.message
    assert "starts from" in fault.message


def test_connections_that_start_at_the_inputs_meet_the_networks_backward_half() -> None:
    backward = px.policies.Backward(conn=Sends(one), unit=Reduces(sum_, GRAD))
    net = network(
        two_layers({"backward": px.policies.Backward(conn=Sends(two))}),
        backward=backward,
    )

    (fault,) = net.faults()

    assert fault.where == "a.weight, backward"
    assert "the inputs" in fault.message


def test_a_layer_that_sends_nothing_back_is_sound() -> None:
    backward = px.policies.Backward(conn=Sends(one), unit=Reduces(sum_, GRAD))
    net = network(
        two_layers(b={"backward": px.policies.Backward(conn=None)}), backward=backward
    )

    assert net.faults() == []


def test_a_map_sent_to_units_with_no_half_is_sound() -> None:
    backward = px.policies.Backward(conn=Sends(two), unit=Reduces(pair, GRAD))
    net = network(
        two_layers(a={"backward": px.policies.Backward(unit=None)}), backward=backward
    )

    assert net.faults() == []


# --------------------------------------------------------------------------- #
# Writes
# --------------------------------------------------------------------------- #
def test_a_float_written_to_a_bool_field_is_a_fault_naming_policy_and_field() -> None:
    net = network(update=px.policies.Update(unit=Writes((FLAG, jnp.float32(1.0)))))

    (fault,) = net.faults()

    assert fault.kind == "write"
    assert fault.where == "the network, update"
    assert fault.policy == "Writes.update"
    assert "float32" in fault.message and "`flag`  bool" in fault.message


def test_a_python_number_takes_the_dtype_of_its_field() -> None:
    net = network(update=px.policies.Update(unit=Writes((SEEN, 0.0), (FLAG, False))))

    assert net.faults() == []


def test_a_python_float_does_not_fit_a_bool_field() -> None:
    net = network(update=px.policies.Update(unit=Writes((FLAG, 1.0))))

    assert kinds(net) == ["write"]


def test_a_write_to_a_field_the_element_does_not_have_lists_those_it_has() -> None:
    missing = px.FieldSpec("missing", np.float32, 0.0)
    net = network(update=px.policies.Update(unit=Writes((missing, 0.0))))

    (fault,) = net.faults()

    assert "`missing`" in fault.message
    assert "activation, flag, grad, level, seen" in fault.message


def test_a_policy_may_write_the_engines_own_columns() -> None:
    net = network(update=px.policies.Update(conn=Marks((px.DEAD, True), (MARK, True))))

    assert net.faults() == []


def test_a_fault_in_a_layers_policy_names_the_layer() -> None:
    wrong = px.policies.Update(unit=Writes((FLAG, jnp.float32(1.0))))
    net = network(two_layers(b={"update": wrong}))

    (fault,) = net.faults()

    assert fault.where == "b.units, update"


# --------------------------------------------------------------------------- #
# Field names
# --------------------------------------------------------------------------- #
def test_a_write_given_one_field_twice_is_refused_where_it_is_written() -> None:
    with pytest.raises(ValueError, match="the field 'seen' twice"):
        px.UnitWrite.of((SEEN, 0.0), (SEEN, 1.0))


def test_a_field_written_under_another_fields_name_is_a_fault() -> None:
    imposter = px.FieldSpec("seen", np.int32, 0)
    net = network(update=px.policies.Update(unit=Writes((imposter, jnp.int32(1)))))

    (fault,) = net.faults()

    assert fault.kind == "field"
    assert "Two different fields have one name" in fault.message


# --------------------------------------------------------------------------- #
# Tracing
# --------------------------------------------------------------------------- #
def test_a_policy_that_raises_is_a_fault_carrying_its_error() -> None:
    def broken(x: jax.Array) -> Any:
        raise RuntimeError("no such thing")

    net = network(forward=px.policies.Forward(conn=Sends(broken), unit=Reduces(sum_)))

    (fault,) = net.faults()

    assert fault.kind == "trace"
    assert fault.policy == "Sends.map"
    assert "RuntimeError: no such thing" in fault.message


def test_a_policy_used_by_every_layer_is_traced_once() -> None:
    sends = Sends(one)
    net = network(forward=px.policies.Forward(conn=sends, unit=Reduces(sum_)))

    net.faults()

    assert sends.calls == 1


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def test_every_fault_is_found_and_one_error_reports_them_all() -> None:
    net = network(
        forward=px.policies.Forward(conn=Sends(one), unit=Reduces(pair)),
        update=px.policies.Update(unit=Writes((FLAG, jnp.float32(1.0)))),
    )

    # Both layers' connections are mismatched, and the update writes a wrong dtype.
    assert kinds(net) == ["accumulator", "accumulator", "write"]
    with pytest.raises(pxl.BuildError) as refused:
        net.init(KEY)
    message = str(refused.value)
    assert "Net has 3 faults in its policies" in message
    assert "a.weight, forward: Sends.map" in message
    assert "b.weight, forward: Sends.map" in message
    assert "the network, update: Writes.update" in message
    assert "Net.faults()" in message


def test_a_fault_is_reported_before_the_engine_refuses_layer_policies() -> None:
    wrong = px.policies.Update(unit=Writes((FLAG, jnp.float32(1.0))))
    sound = px.policies.Update(unit=Writes((FLAG, True)))

    # Pruning the units of a layer needs a unit capacity, which this network has
    # not: something the engine cannot run.
    class Never(px.policies.PruneUnitPolicy):
        def predicate(self, u: px.UnitView, i: px.UnitIdx, g: Any) -> jax.Array:
            return jnp.bool_(False)

    prune = px.policies.Prune(unit=Never())
    with pytest.raises(pxl.BuildError, match="1 fault in its policies"):
        network(two_layers(b={"update": wrong, "prune": prune})).init(KEY)
    # With the fault fixed, what is left is that the engine cannot run it.
    with pytest.raises(pxl.BuildError, match="cannot be run yet"):
        network(two_layers(b={"update": sound, "prune": prune})).init(KEY)
    # And a unit update on one layer alone is run.
    network(two_layers(b={"update": sound})).init(KEY)


def test_a_sound_network_builds_and_steps() -> None:
    net = network()
    inputs = px.StepInputs(inputs=jnp.ones(5, jnp.float32), targets=None)

    assert net.faults() == []
    assert np.asarray(net.init(KEY).step(inputs, outputs=True).outputs).shape == (3,)
