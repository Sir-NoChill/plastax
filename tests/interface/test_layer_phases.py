"""A phase that applies to some layers only: forward, backward, update and prune.

Plan `plans/20261123T215119Z-layer-policies-stage3.md` (workspace root), parts 3 and
4. In each phase every element has one policy or none; the phase is computed for
the elements that have it. Each test reads a block's connections or units through
`static.layout`, and no test asserts a literal id or slot.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import plastax as px
import plastax.layers as pxl
from plastax.layers import BuildError, Dense, Units
from plastax.phases import phase_scopes, plan_prune_fusion

KEY = jax.random.key(0)
SEEN = px.FieldSpec("seen", np.float32, 0.0)
SLOT = px.FieldSpec("slot", np.int32, -1)
X = jnp.asarray(np.random.default_rng(0).normal(size=5), jnp.float32)


class WeightedSum(px.policies.ForwardConnPolicy):
    def map(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: None,
    ) -> jax.Array:
        return c[px.WEIGHT, cid] * u[px.ACTIVATION, src]


class Summed(px.policies.ForwardUnitPolicy):
    combine = px.monoid.sum_

    def apply(
        self, u: px.UnitView, i: px.UnitIdx, g: None, acc: jax.Array
    ) -> px.UnitWrite:
        return px.UnitWrite.of((px.ACTIVATION, acc))


class Halve(px.policies.UpdateConnPolicy):
    """Every weight it is run for is halved, and is told which slot it is in."""

    def incoming(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: None,
    ) -> px.ConnWrite:
        return px.ConnWrite.of(
            (px.WEIGHT, 0.5 * c[px.WEIGHT, cid]), (SLOT, jnp.asarray(cid, jnp.int32))
        )

    def outgoing(
        self,
        u: px.UnitView,
        src: px.UnitIdx,
        dst: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: None,
    ) -> px.ConnWrite:
        return px.ConnWrite.of()


class HalveOnce(Halve):
    """The same rule in the form an optimizer has: one step from a batch's mean."""

    def per_sample(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: None,
    ) -> jax.Array:
        return jnp.float32(0.5)

    def incoming_batched(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: None,
        stat: jax.Array,
    ) -> px.ConnWrite:
        return px.ConnWrite.of(
            (px.WEIGHT, stat * c[px.WEIGHT, cid]), (SLOT, jnp.asarray(cid, jnp.int32))
        )


class Quarter(px.policies.UpdateConnPolicy):
    """A second, different connection update: every weight is quartered."""

    def incoming(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: None,
    ) -> px.ConnWrite:
        return px.ConnWrite.of((px.WEIGHT, 0.25 * c[px.WEIGHT, cid]))

    def outgoing(
        self,
        u: px.UnitView,
        src: px.UnitIdx,
        dst: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: None,
    ) -> px.ConnWrite:
        return px.ConnWrite.of()


class CountTwo(px.policies.UpdateUnitPolicy):
    def update(self, u: px.UnitView, i: px.UnitIdx, g: None) -> px.UnitWrite:
        return px.UnitWrite.of((SEEN, u[SEEN, i] + 2.0))


class Positive(px.policies.PruneConnPolicy):
    def predicate(
        self, u: px.UnitView, c: px.ConnView, cid: px.ConnIdx, g: None
    ) -> jax.Array:
        return c[px.WEIGHT, cid] > 0.0


class Never(px.policies.PruneUnitPolicy):
    def predicate(self, u: px.UnitView, i: px.UnitIdx, g: None) -> jax.Array:
        return jnp.bool_(False)


class Count(px.policies.UpdateUnitPolicy):
    def update(self, u: px.UnitView, i: px.UnitIdx, g: None) -> px.UnitWrite:
        return px.UnitWrite.of((SEEN, u[SEEN, i] + 1.0))


class Negative(px.policies.PruneConnPolicy):
    def predicate(
        self, u: px.UnitView, c: px.ConnView, cid: px.ConnIdx, g: None
    ) -> jax.Array:
        return c[px.WEIGHT, cid] < 0.0


class Every(px.policies.PruneUnitPolicy):
    """Selects every unit it is asked about."""

    def predicate(self, u: px.UnitView, i: px.UnitIdx, g: None) -> jax.Array:
        return jnp.bool_(True)


def three_layers(**middle: Any) -> pxl.Module:
    """`first` and `head` say nothing; `middle` is a `Dense` given the keywords."""

    class ThreeLayers(pxl.Module):
        def setup(self) -> None:
            self.first = Dense(4)
            self.middle = Dense(3, **middle)
            self.head = Dense(2)

        def __call__(self, x: Units) -> Units:
            return self.head(self.middle(self.first(x)))

    return ThreeLayers()


def network(topology: pxl.Module, **attributes: Any) -> type[px.Network[None]]:
    fields = {
        "topology": topology,
        "input_shape": (5,),
        "forward": px.policies.Forward(conn=WeightedSum(), unit=Summed()),
        "extra_unit_fields": (SEEN,),
        "extra_conn_fields": (SLOT,),
        "capacity_align": 64,
        "batch_reduction": px.policies.MeanFloatFirstRest(),
        **attributes,
    }
    return type("Net", (px.Network,), fields)


class Built:
    """A built network, read block by block."""

    def __init__(self, net: type[px.Network[None]]) -> None:
        self.net = net
        self.static, self.state = net.build(KEY)
        assert self.static.layout is not None
        self.layout = self.static.layout
        self.table = net.policy_table()

    def step(self, batch: int | None = None) -> Built:
        step = px.make_step(self.net, self.static, batch_size=batch)
        x = X if batch is None else jnp.stack([X] * batch)
        self.state = step(self.state, px.StepInputs(inputs=x, targets=None)).state
        return self

    def conns(self, *path: str, column: str = "weight") -> np.ndarray:
        """A column of one block's live connections, in slot order."""
        runs = self.layout.conn_runs[self.table.conn_paths.index(path)]
        parts = []
        for run in runs:
            bucket = self.state.conns[run.bucket]
            live = ~np.asarray(bucket["dead"])[run.start : run.stop]
            parts.append(np.asarray(bucket[column])[run.start : run.stop][live])
        return np.concatenate(parts)

    def units(self, *path: str, column: str) -> np.ndarray:
        start, stop = self.layout.unit_runs[self.table.unit_paths.index(path)]
        return np.asarray(self.state.units[column])[start:stop]

    def others(self, *layer: str) -> list[tuple[str, ...]]:
        """The connection blocks of every other layer."""
        return [p for p in self.table.conn_paths if p[: len(layer)] != layer]


# --------------------------------------------------------------------------- #
# The connection update
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("batch", [None, 2])
@pytest.mark.parametrize("rule", [Halve(), HalveOnce()])
def test_an_update_given_to_one_layer_changes_that_layers_weights_alone(
    rule: Any, batch: int | None
) -> None:
    net = network(three_layers(update=px.policies.Update(conn=rule)))
    before = Built(net)
    after = Built(net).step(batch)

    for path in (("middle", "weight"), ("middle", "bias_weight")):
        np.testing.assert_allclose(after.conns(*path), 0.5 * before.conns(*path))
    for path in after.others("middle"):
        np.testing.assert_array_equal(after.conns(*path), before.conns(*path))


@pytest.mark.parametrize("batch", [None, 2])
@pytest.mark.parametrize("rule", [Halve(), HalveOnce()])
def test_a_layer_given_no_update_keeps_its_weights(
    rule: Any, batch: int | None
) -> None:
    net = network(three_layers(update=None), update=px.policies.Update(conn=rule))
    before = Built(net)
    after = Built(net).step(batch)

    for path in (("middle", "weight"), ("middle", "bias_weight")):
        np.testing.assert_array_equal(after.conns(*path), before.conns(*path))
    for path in after.others("middle"):
        np.testing.assert_allclose(after.conns(*path), 0.5 * before.conns(*path))


@pytest.mark.parametrize("batch", [None, 2])
@pytest.mark.parametrize("rule", [Halve(), HalveOnce()])
def test_a_policy_is_given_each_connections_slot_in_its_bucket(
    rule: Any, batch: int | None
) -> None:
    net = network(three_layers(update=px.policies.Update(conn=rule), capacity=20))
    built = Built(net).step(batch)
    (run,) = built.layout.conn_runs[built.table.conn_paths.index(("middle", "weight"))]

    # The block has 12 connections, at the first 12 slots of its run.
    np.testing.assert_array_equal(
        built.conns("middle", "weight", column="slot"),
        np.arange(run.start, run.start + 12),
    )
    # Nothing was run for the layers that have no update.
    assert (built.conns("first", "weight", column="slot") == -1).all()


def test_a_scope_is_the_runs_of_the_layers_that_have_the_policy() -> None:
    net = network(three_layers(update=px.policies.Update(conn=Halve())))
    built = Built(net)
    (scope,) = phase_scopes(net, built.static).update_conn

    assert scope.spans is not None
    runs = {
        (run.bucket, run.start, run.stop)
        for path in (("middle", "weight"), ("middle", "bias_weight"))
        for run in built.layout.conn_runs[built.table.conn_paths.index(path)]
    }
    spans = {
        (bucket, start, stop)
        for bucket, of_bucket in enumerate(scope.spans)
        for start, stop in of_bucket
    }
    assert spans == runs
    assert scope.spans[2] == ()  # the head's bucket: nothing is traced for it


def test_a_policy_every_layer_has_covers_the_whole_network() -> None:
    net = network(three_layers(), update=px.policies.Update(conn=Halve()))
    scopes = phase_scopes(net, Built(net).static)

    (only,) = scopes.update_conn
    assert only.spans is None
    assert scopes.whole


# --------------------------------------------------------------------------- #
# The unit update
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("batch", [None, 2])
def test_a_unit_update_given_to_one_layer_runs_for_its_units_alone(
    batch: int | None,
) -> None:
    net = network(three_layers(update=px.policies.Update(unit=Count())))
    built = Built(net).step(batch)
    seen = np.asarray(built.state.units["seen"])

    np.testing.assert_array_equal(built.units("middle", "units", column="seen"), 1.0)
    assert seen.sum() == 3.0  # its three units, and no other


@pytest.mark.parametrize("batch", [None, 2])
def test_a_layer_given_no_unit_update_is_left_out_of_the_networks(
    batch: int | None,
) -> None:
    net = network(three_layers(update=None), update=px.policies.Update(unit=Count()))
    built = Built(net).step(batch)
    seen = np.asarray(built.state.units["seen"])

    np.testing.assert_array_equal(built.units("middle", "units", column="seen"), 0.0)
    # Every other unit, the inputs included, was counted once.
    assert seen.sum() == seen.size - 3


# --------------------------------------------------------------------------- #
# Pruning
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("batch", [None, 2])
def test_pruning_given_to_one_layer_removes_that_layers_connections_alone(
    batch: int | None,
) -> None:
    net = network(three_layers(prune=px.policies.Prune(conn=Negative())))
    before = Built(net)
    after = Built(net).step(batch)

    kept = before.conns("middle", "weight")
    assert (kept < 0).any() and (kept >= 0).any()  # the test has something to show
    np.testing.assert_array_equal(after.conns("middle", "weight"), kept[kept >= 0])
    for path in after.others("middle"):
        np.testing.assert_array_equal(after.conns(*path), before.conns(*path))
    # A bias is never pruned, whatever its layer says.
    np.testing.assert_array_equal(
        after.conns("middle", "bias_weight"), before.conns("middle", "bias_weight")
    )


def test_a_layer_given_no_pruning_keeps_every_connection() -> None:
    net = network(three_layers(prune=None), prune=px.policies.Prune(conn=Negative()))
    before = Built(net)
    after = Built(net).step()

    np.testing.assert_array_equal(
        after.conns("middle", "weight"), before.conns("middle", "weight")
    )
    first = before.conns("first", "weight")
    np.testing.assert_array_equal(after.conns("first", "weight"), first[first >= 0])


def test_unit_pruning_given_to_one_layer_tests_that_layers_units_alone() -> None:
    net = network(three_layers(prune=px.policies.Prune(unit=Every())), unit_capacity=32)
    built = Built(net).step()

    # `Every` selects whatever it is asked about, and only the middle's units went.
    assert built.units("middle", "units", column="pruned").all()
    assert not built.units("first", "units", column="pruned").any()
    assert not built.units("head", "units", column="pruned").any()
    # The connections at a pruned unit go with it, in the layers before and after.
    assert built.conns("middle", "weight").size == 0
    assert built.conns("head", "weight").size == 0
    assert built.conns("first", "weight").size == 20


def test_pruning_on_some_layers_is_not_fused_into_the_forward_sweep() -> None:
    net = network(three_layers(prune=None), prune=px.policies.Prune(conn=Negative()))
    plan = plan_prune_fusion(net, Built(net).static, None, engine="xla")

    assert not plan.fused
    assert "some layers" in plan.reason


# --------------------------------------------------------------------------- #
# Two policies in one phase, each on its own layers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("batch", [None, 2])
@pytest.mark.parametrize("rule", [Halve(), HalveOnce()])
def test_two_connection_updates_each_change_their_own_layers(
    rule: Any, batch: int | None
) -> None:
    """The network's rule beside a layer's own, as SGD beside DeepR would be."""
    net = network(
        three_layers(update=px.policies.Update(conn=Quarter())),
        update=px.policies.Update(conn=rule),
    )
    before = Built(net)
    after = Built(net).step(batch)

    # A batched plain update averages each sample's change, which rounds in the
    # last bit.
    for path in (("middle", "weight"), ("middle", "bias_weight")):
        np.testing.assert_allclose(
            after.conns(*path), 0.25 * before.conns(*path), rtol=1e-6
        )
    for path in after.others("middle"):
        np.testing.assert_allclose(
            after.conns(*path), 0.5 * before.conns(*path), rtol=1e-6
        )


def test_two_connection_updates_have_disjoint_scopes() -> None:
    net = network(
        three_layers(update=px.policies.Update(conn=Quarter())),
        update=px.policies.Update(conn=Halve()),
    )
    built = Built(net)
    scopes = phase_scopes(net, built.static).update_conn

    assert sorted(type(scope.policy).__name__ for scope in scopes) == [
        "Halve",
        "Quarter",
    ]
    slots = [
        {
            (bucket, slot)
            for bucket, runs in enumerate(scope.spans or ())
            for start, stop in runs
            for slot in range(start, stop)
        }
        for scope in scopes
    ]
    assert slots[0] and slots[1] and not slots[0] & slots[1]
    # Between them, every connection of the network.
    assert len(slots[0] | slots[1]) == sum(
        run.stop - run.start for runs in built.layout.conn_runs for run in runs
    )


@pytest.mark.parametrize("batch", [None, 2])
def test_two_unit_updates_each_run_for_their_own_units(batch: int | None) -> None:
    net = network(
        three_layers(update=px.policies.Update(unit=CountTwo())),
        update=px.policies.Update(unit=Count()),
    )
    built = Built(net).step(batch)
    seen = np.asarray(built.state.units["seen"])

    np.testing.assert_array_equal(built.units("middle", "units", column="seen"), 2.0)
    assert seen.sum() == (seen.size - 3) + 2.0 * 3  # every other unit counted once


@pytest.mark.parametrize("batch", [None, 2])
def test_two_connection_prunings_each_test_their_own_layers(
    batch: int | None,
) -> None:
    net = network(
        three_layers(prune=px.policies.Prune(conn=Positive())),
        prune=px.policies.Prune(conn=Negative()),
    )
    before = Built(net)
    after = Built(net).step(batch)

    middle = before.conns("middle", "weight")
    np.testing.assert_array_equal(after.conns("middle", "weight"), middle[middle <= 0])
    for path in (("first", "weight"), ("head", "weight")):
        kept = before.conns(*path)
        np.testing.assert_array_equal(after.conns(*path), kept[kept >= 0])


def test_two_unit_prunings_each_test_their_own_units() -> None:
    net = network(
        three_layers(prune=px.policies.Prune(unit=Every())),
        prune=px.policies.Prune(unit=Never()),
        unit_capacity=32,
    )
    built = Built(net).step()

    assert built.units("middle", "units", column="pruned").all()
    assert not built.units("first", "units", column="pruned").any()
    assert built.conns("first", "weight").size == 20


@pytest.mark.parametrize("batch", [None, 2])
def test_two_instances_of_one_policy_give_what_one_instance_gives(
    batch: int | None,
) -> None:
    """Two `Halve()` are two policies to the engine, each run on its own layers."""
    twice = network(
        three_layers(update=px.policies.Update(conn=Halve())),
        update=px.policies.Update(conn=Halve()),
    )
    once = network(three_layers(), update=px.policies.Update(conn=Halve()))

    assert len(phase_scopes(twice, Built(twice).static).update_conn) == 2
    a, b = Built(twice).step(batch), Built(once).step(batch)
    for path in a.table.conn_paths:
        np.testing.assert_array_equal(a.conns(*path), b.conns(*path))


# --------------------------------------------------------------------------- #
# What is still refused
# --------------------------------------------------------------------------- #
def test_the_same_update_on_the_network_and_on_a_layer_is_one_policy() -> None:
    # One object given twice. Two `Halve()` are two policies until a policy is
    # compared by value (`@px.policy`, card `fields-and-hyperparameters`).
    halve = Halve()
    net = network(
        three_layers(update=px.policies.Update(conn=halve)),
        update=px.policies.Update(conn=halve),
    )

    assert phase_scopes(net, Built(net).static).whole


def test_unit_pruning_on_a_layer_needs_a_unit_capacity() -> None:
    net = network(three_layers(prune=px.policies.Prune(unit=Every())))

    with pytest.raises(BuildError, match=r"(?s)middle\.units.*unit_capacity"):
        net.build(KEY)


# --------------------------------------------------------------------------- #
# The backward pass on some layers
# --------------------------------------------------------------------------- #
# A linear network, inputs -> first -> middle -> head, with no bias and squared error.
# Its gradients are written out below as `jax.grad` gives them.
GRAD = px.FieldSpec("grad", np.float32, 0.0)
SEED = px.FieldSpec("seed", np.float32, 0.0)
W1 = np.random.default_rng(1).normal(size=(4, 5)).astype(np.float32)
W2 = np.random.default_rng(2).normal(size=(3, 4)).astype(np.float32)
W3 = np.random.default_rng(3).normal(size=(2, 3)).astype(np.float32)
TARGET = jnp.asarray([0.5, -1.0], jnp.float32)


class LinearForward(WeightedSum):
    linear_input = px.ACTIVATION  # `map` is weight * activation: one sparse product


class SendGrad(px.policies.BackwardConnPolicy):
    linear_input = GRAD

    def map(
        self,
        u: px.UnitView,
        src: px.UnitIdx,
        dst: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: None,
    ) -> jax.Array:
        return c[px.WEIGHT, cid] * u[GRAD, dst]


class KeepGrad(px.policies.BackwardUnitPolicy):
    combine = px.monoid.sum_

    def apply(
        self, u: px.UnitView, i: px.UnitIdx, g: None, acc: jax.Array
    ) -> px.UnitWrite:
        return px.UnitWrite.of((GRAD, acc + u[SEED, i]))


class SquaredError(px.policies.LossPolicy):
    seed_field = SEED

    def calculate_loss(
        self, u: px.UnitView, outputs: jax.Array, targets: jax.Array, g: None
    ) -> tuple[jax.Array, jax.Array]:
        diff = u.gather(px.ACTIVATION, outputs) - targets
        return 0.5 * jnp.sum(diff * diff), diff


_send, _keep = SendGrad(), KeepGrad()
BACKWARD = px.policies.Backward(conn=_send, unit=_keep)


def linear(network_backward: Any = BACKWARD, **middle: Any) -> type[px.Network[None]]:
    class Linear(pxl.Module):
        def setup(self) -> None:
            self.first = Dense(4, bias=False, init=px.init.array(W1))
            self.middle = Dense(3, bias=False, init=px.init.array(W2), **middle)
            self.head = Dense(2, bias=False, init=px.init.array(W3))

        def __call__(self, x: Units) -> Units:
            return self.head(self.middle(self.first(x)))

    return network(
        Linear(),
        forward=px.policies.Forward(conn=LinearForward(), unit=Summed()),
        backward=network_backward,
        loss=SquaredError(),
        extra_unit_fields=(GRAD, SEED, SEEN),
    )


def gradients() -> dict[str, np.ndarray]:
    """dL/d(activation) of each layer's units, for the whole network trained."""

    def loss(h1: jax.Array, h2: jax.Array, y: jax.Array) -> jax.Array:
        # Each layer's output with a zero added, so its gradient can be asked for.
        a1 = W1 @ X + h1
        a2 = W2 @ a1 + h2
        return 0.5 * jnp.sum((W3 @ a2 + y - TARGET) ** 2)

    zeros = (jnp.zeros(4), jnp.zeros(3), jnp.zeros(2))
    g1, g2, g3 = jax.grad(loss, argnums=(0, 1, 2))(*zeros)
    return {"first": np.asarray(g1), "middle": np.asarray(g2), "head": np.asarray(g3)}


def backward_step(
    net: type[px.Network[None]], batch: int | None, layout: str = "auto"
) -> Built:
    built = Built(net)
    step = px.make_step(net, built.static, batch_size=batch, layout=layout)
    x = X if batch is None else jnp.stack([X] * batch)
    t = TARGET if batch is None else jnp.stack([TARGET] * batch)
    built.state = step(built.state, px.StepInputs(inputs=x, targets=t)).state
    return built


WAYS = [(None, "auto"), (2, "edge_list"), (2, "triton")]  # triton: the XLA product here


@pytest.mark.parametrize(("batch", "layout"), WAYS)
def test_the_whole_network_has_the_gradients_of_jax_grad(
    batch: int | None, layout: str
) -> None:
    built = backward_step(linear(), batch, layout)

    for layer, expected in gradients().items():
        np.testing.assert_allclose(
            built.units(layer, "units", column="grad"), expected, rtol=1e-5
        )


@pytest.mark.parametrize(("batch", "layout"), WAYS)
def test_a_layer_with_no_backward_pass_stops_the_gradient(
    batch: int | None, layout: str
) -> None:
    built = backward_step(linear(backward=None), batch, layout)
    expected = gradients()

    # The head is trained as before; the middle is written nothing; and the first
    # layer receives nothing, because the middle's connections send nothing back.
    np.testing.assert_allclose(
        built.units("head", "units", column="grad"), expected["head"], rtol=1e-5
    )
    np.testing.assert_array_equal(built.units("middle", "units", column="grad"), 0.0)
    np.testing.assert_array_equal(built.units("first", "units", column="grad"), 0.0)


@pytest.mark.parametrize(("batch", "layout"), WAYS)
def test_a_layer_with_no_update_still_passes_the_gradient_through(
    batch: int | None, layout: str
) -> None:
    built = backward_step(linear(update=None), batch, layout)

    for layer, expected in gradients().items():
        np.testing.assert_allclose(
            built.units(layer, "units", column="grad"), expected, rtol=1e-5
        )


def test_connections_into_units_with_no_backward_half_are_left_out() -> None:
    """The head sends back to the middle's units, which reduce nothing."""
    net = linear(backward=None)
    built = Built(net)
    scope = phase_scopes(net, built.static).backward_conn

    assert scope is not None and scope.spans is not None
    runs = {
        (run.bucket, run.start, run.stop)
        for path in (("first", "weight"),)
        for run in built.layout.conn_runs[built.table.conn_paths.index(path)]
    }
    spans = {
        (bucket, start, stop)
        for bucket, of_bucket in enumerate(scope.spans)
        for start, stop in of_bucket
    }
    # Not the middle's own connections (they have no backward half), and not the
    # head's (their sources are the middle's units).
    assert spans == runs


@pytest.mark.parametrize(("batch", "layout"), WAYS)
def test_a_backward_pass_given_to_one_layer_alone_writes_its_units_alone(
    batch: int | None, layout: str
) -> None:
    net = linear(network_backward=None, backward=BACKWARD)
    built = backward_step(net, batch, layout)

    # The middle's units are handed nothing (the head sends nothing back) and have
    # no seed, so they are written zero; no other unit is written at all.
    assert phase_scopes(net, built.static).backward_unit is not None
    np.testing.assert_array_equal(np.asarray(built.state.units["grad"]), 0.0)


def test_a_different_backward_pass_on_one_layer_is_refused() -> None:
    other = px.policies.Backward(conn=SendGrad(), unit=_keep)

    with pytest.raises(BuildError) as refused:
        linear(backward=other).build(KEY)

    message = str(refused.value)
    assert "middle.weight  (connections)" in message
    assert "backward" in message and "the network has SendGrad" in message
    assert "two different backward passes" in message


# --------------------------------------------------------------------------- #
# The forward pass on some layers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("batch", "layout"), WAYS)
def test_units_with_no_forward_half_keep_their_value(
    batch: int | None, layout: str
) -> None:
    """Units nothing computes: they hold what they were built with, here zero."""
    net = linear(forward=px.policies.Forward(unit=None))
    built = backward_step(net, batch, layout)

    np.testing.assert_array_equal(
        built.units("middle", "units", column="activation"), 0.0
    )
    np.testing.assert_allclose(
        built.units("first", "units", column="activation"), W1 @ X, rtol=1e-5
    )
    # The head reads the middle's units as they are.
    np.testing.assert_array_equal(
        built.units("head", "units", column="activation"), 0.0
    )


@pytest.mark.parametrize(("batch", "layout"), WAYS)
def test_connections_with_no_forward_half_send_nothing_forward(
    batch: int | None, layout: str
) -> None:
    """Connections used going backward only, as feedback weights are."""
    net = linear(forward=px.policies.Forward(conn=None))
    built = backward_step(net, batch, layout)
    expected = gradients()

    # Nothing arrives at the middle's units, so they are written the sum of nothing.
    np.testing.assert_array_equal(
        built.units("middle", "units", column="activation"), 0.0
    )
    # Going backward the same connections still carry the gradient to the first layer.
    seed = -np.asarray(TARGET)  # the head's output is zero
    np.testing.assert_allclose(
        built.units("first", "units", column="grad"), W2.T @ (W3.T @ seed), rtol=1e-5
    )
    assert expected["head"].shape == seed.shape


def test_connections_into_units_with_no_forward_half_are_left_out() -> None:
    net = linear(forward=px.policies.Forward(unit=None))
    built = Built(net)
    scope = phase_scopes(net, built.static).forward_conn
    (middle,) = built.layout.conn_runs[
        built.table.conn_paths.index(("middle", "weight"))
    ]

    assert scope is not None and scope.spans is not None
    assert (middle.start, middle.stop) not in scope.spans[middle.bucket]
