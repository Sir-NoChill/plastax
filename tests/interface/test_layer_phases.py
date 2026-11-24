"""A phase that applies to some layers only: update and prune.

Plan `plans/20261123T215119Z-layer-policies-stage3.md` (workspace root), part 3. In
each phase every element has one policy or none; the phase is computed for the
elements that have it. Each test reads a block's connections or units through
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


class Quarter(Halve):
    """A second, different connection update."""


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
    scope = phase_scopes(net, built.static).update_conn

    assert scope is not None and scope.spans is not None
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

    assert scopes.update_conn is not None and scopes.update_conn.spans is None
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
# What is still refused
# --------------------------------------------------------------------------- #
def test_two_different_updates_are_refused_naming_the_layer_and_both() -> None:
    net = network(
        three_layers(update=px.policies.Update(conn=Quarter())),
        update=px.policies.Update(conn=Halve()),
    )

    with pytest.raises(BuildError) as refused:
        net.build(KEY)

    message = str(refused.value)
    assert "Net cannot be run yet" in message
    assert "middle.weight  (connections)" in message
    assert "Quarter" in message and "the network has Halve" in message
    assert "Two different policies in one phase" in message
    assert "first" not in message  # a layer that says nothing is not listed


def test_the_same_update_on_the_network_and_on_a_layer_is_one_policy() -> None:
    # One object given twice. Two `Halve()` are two policies until a policy is
    # compared by value (`@px.policy`, card `fields-and-hyperparameters`).
    halve = Halve()
    net = network(
        three_layers(update=px.policies.Update(conn=halve)),
        update=px.policies.Update(conn=halve),
    )

    assert phase_scopes(net, Built(net).static).whole


@pytest.mark.parametrize(
    ("keywords", "reason"),
    [
        ({"backward": None}, "backward pass of its own"),
        ({"forward": None}, "forward pass of its own"),
    ],
)
def test_a_layers_own_pass_is_refused(keywords: dict[str, Any], reason: str) -> None:
    class Back(px.policies.BackwardConnPolicy, px.policies.BackwardUnitPolicy):
        combine = px.monoid.sum_

        def map(self, u: Any, src: Any, dst: Any, c: Any, cid: Any, g: Any) -> Any:
            return c[px.WEIGHT, cid] * u[SEEN, dst]

        def apply(self, u: Any, i: Any, g: Any, acc: Any) -> px.UnitWrite:
            return px.UnitWrite.of((SEEN, acc))

    net = network(three_layers(**keywords), backward=Back())

    with pytest.raises(BuildError, match=reason):
        net.build(KEY)


def test_unit_pruning_on_a_layer_needs_a_unit_capacity() -> None:
    net = network(three_layers(prune=px.policies.Prune(unit=Every())))

    with pytest.raises(BuildError, match=r"(?s)middle\.units.*unit_capacity"):
        net.build(KEY)
