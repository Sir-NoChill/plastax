"""Policies given to a layer: declared, recorded, and resolved against the network.

Plan `plans/20261123T145850Z-layer-policies-stage1.md` (workspace root). A layer takes
the network's five keywords. A primitive creates one kind of element and takes that
half directly; `Dense` creates both and takes pairs. A half that is not given is the
network's (`INHERIT`), and None says the layer has none.

The engine runs one set of policies for the whole network until stage 3, so a network
whose layers differ from it is refused when it is built. Everything before that
refusal is tested here: what a layer accepts, what the record holds, and what each
block resolves to.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import plastax as px
import plastax.layers as pxl
from plastax.layers import CreateConnections, Dense, Units
from plastax.layers._policies import PHASES, conn_policies, split_pairs, unit_policies
from plastax.policies import INHERIT

KEY = jax.random.key(0)
GRAD = px.FieldSpec("grad", np.float32, 0.0)
SEEN = px.FieldSpec("seen", np.float32, 0.0)
NOTHING = dict.fromkeys(PHASES, INHERIT)  # a layer given no policy keyword


# --------------------------------------------------------------------------- #
# One small policy of each kind
# --------------------------------------------------------------------------- #
class WeightedSum(px.policies.ForwardConnPolicy):
    linear_input = px.ACTIVATION

    def map(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: Any,
    ) -> jax.Array:
        return c[px.WEIGHT, cid] * u[px.ACTIVATION, src]


class Summed(px.policies.ForwardUnitPolicy):
    combine = px.monoid.sum_

    def apply(
        self, u: px.UnitView, i: px.UnitIdx, g: Any, acc: jax.Array
    ) -> px.UnitWrite:
        return px.UnitWrite.of((px.ACTIVATION, acc))


class Both(WeightedSum, Summed):
    """One object with `map`, `combine` and `apply`."""


class Decay(px.policies.UpdateConnPolicy):
    def incoming(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: Any,
    ) -> px.ConnWrite:
        return px.ConnWrite.of((px.WEIGHT, 0.5 * c[px.WEIGHT, cid]))

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


class Count(px.policies.UpdateUnitPolicy):
    def update(self, u: px.UnitView, i: px.UnitIdx, g: Any) -> px.UnitWrite:
        return px.UnitWrite.of((SEEN, u[SEEN, i] + 1.0))


class Negative(px.policies.PruneConnPolicy):
    def predicate(
        self, u: px.UnitView, c: px.ConnView, cid: px.ConnIdx, g: Any
    ) -> jax.Array:
        return c[px.WEIGHT, cid] < 0.0


_conn, _unit = WeightedSum(), Summed()
_forward = px.policies.Forward(conn=_conn, unit=_unit)


def two_layers(**middle: Any) -> pxl.Module:
    """`first` says nothing; `middle` is a `Dense` given the keywords."""

    class TwoLayers(pxl.Module):
        def setup(self) -> None:
            self.first = Dense(4)
            self.middle = Dense(3, **middle)

        def __call__(self, x: Units) -> Units:
            return self.middle(self.first(x))

    return TwoLayers()


def network(topology_: pxl.Module, **fields: Any) -> type[px.Network[None]]:
    class Net(px.Network[None]):
        topology = topology_
        input_shape = (5,)
        forward = fields.pop("forward", _forward)
        optimizer = fields.pop("optimizer", None)
        update = fields.pop("update", None)
        prune = fields.pop("prune", None)
        extra_unit_fields = (GRAD, SEEN)

    assert not fields
    return Net


def block(net: type[px.Network[None]], *path: str) -> Any:
    """The resolved policies of the block at `path`."""
    table = net.policy_table()
    for paths, rows in (
        (table.conn_paths, table.conns),
        (table.unit_paths, table.units),
    ):
        if path in paths:
            return rows[paths.index(path)]
    raise KeyError(path)


# --------------------------------------------------------------------------- #
# Declaring
# --------------------------------------------------------------------------- #
def test_a_half_that_is_not_given_is_inherit() -> None:
    prune = px.policies.Prune(conn=Negative())

    assert prune.unit is INHERIT
    assert INHERIT is not None


def test_a_network_pass_with_one_half_is_refused_naming_the_other() -> None:
    with pytest.raises(TypeError, match=r"OneHalf\.forward has no `unit` policy"):

        class OneHalf(px.Network[None]):
            forward = px.policies.Forward(conn=_conn)


def test_a_half_a_network_does_not_give_reads_as_none() -> None:
    net = network(two_layers(), prune=px.policies.Prune(conn=Negative()))

    assert net.resolved().prune_unit is None


# The three messages below are read from the checks themselves. Called through a
# layer, the test run's own argument check (jaxtyping) raises its TypeError first; the
# last test of the three says a layer refuses, whichever check does it.
def test_a_unit_policy_given_to_connections_is_refused_naming_what_is_wanted() -> None:
    with pytest.raises(
        TypeError, match=r"CreateConnections\(forward=.*ForwardConnPolicy"
    ):
        conn_policies("CreateConnections", **{**NOTHING, "forward": _unit})


def test_a_pair_given_to_a_primitive_is_refused_and_says_to_give_the_half() -> None:
    update = px.policies.Update(unit=Count())

    with pytest.raises(TypeError, match=r"CreateUnits\(update=.*creates units only"):
        unit_policies("CreateUnits", **{**NOTHING, "update": update})


def test_a_bare_policy_given_to_dense_is_refused_and_says_to_give_a_pair() -> None:
    with pytest.raises(TypeError, match=r"Dense\(update=.*Update\(conn=, unit=\)"):
        split_pairs("Dense", **{**NOTHING, "update": Decay()})


def test_a_layer_refuses_a_policy_of_the_wrong_kind() -> None:
    with pytest.raises(TypeError):
        CreateConnections(forward=_unit)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Dense(3, update=Decay())  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Recording
# --------------------------------------------------------------------------- #
def test_dense_gives_each_half_of_a_pair_to_its_own_block() -> None:
    decay, count = Decay(), Count()
    topology = two_layers(update=px.policies.Update(conn=decay, unit=count))
    record = pxl.trace(topology, {"x": (5,)})
    conns = {b.path: b.policies for b in record.blocks}
    units = {b.path: b.policies for b in record.unit_blocks}

    assert conns["middle", "weight"].update is decay
    assert units["middle", "units"].update is count
    # A phase the layer did not name is left to the network.
    assert conns["middle", "weight"].prune is INHERIT
    assert units["first", "units"].update is INHERIT


def test_one_object_with_both_halves_given_to_dense_goes_to_both_blocks() -> None:
    both = Both()
    record = pxl.trace(two_layers(forward=both), {"x": (5,)})
    conns = {b.path: b.policies for b in record.blocks}
    units = {b.path: b.policies for b in record.unit_blocks}

    assert conns["middle", "weight"].forward is both
    assert units["middle", "units"].forward is both


def test_a_bias_takes_the_layers_update_and_is_never_pruned_or_regrown() -> None:
    decay = Decay()
    topology = two_layers(
        update=px.policies.Update(conn=decay),
        prune=px.policies.Prune(conn=Negative()),
    )
    record = pxl.trace(topology, {"x": (5,)})
    bias = {b.path: b.policies for b in record.blocks}["middle", "bias_weight"]

    assert bias.update is decay
    assert bias.prune is None
    assert bias.grow is None


# --------------------------------------------------------------------------- #
# Resolving
# --------------------------------------------------------------------------- #
def test_a_layer_given_nothing_resolves_to_the_network() -> None:
    net = network(two_layers(), optimizer=px.optim.sgd(0.1, GRAD))
    table = net.policy_table()

    assert block(net, "first", "weight") == table.network_conns
    assert block(net, "first", "units") == table.network_units
    assert table.overrides() == []


def test_a_half_a_layer_does_not_give_is_the_networks() -> None:
    count, negative = Count(), Negative()
    net = network(
        two_layers(prune=px.policies.Prune(conn=negative)),
        update=px.policies.Update(unit=count),
    )

    assert block(net, "middle", "weight").prune is negative
    # `update` was not named by the layer at all, so both halves are the network's.
    assert block(net, "middle", "units").update is count
    assert block(net, "first", "weight").prune is None


def test_none_on_a_layer_removes_the_networks_policy_for_that_layer_only() -> None:
    net = network(two_layers(update=None), optimizer=px.optim.sgd(0.1, GRAD))
    table = net.policy_table()

    assert table.network_conns.update is not None
    assert block(net, "middle", "weight").update is None
    assert block(net, "first", "weight").update == table.network_conns.update


def test_a_layers_update_replaces_the_optimizers_for_that_layer_only() -> None:
    decay = Decay()
    net = network(
        two_layers(update=px.policies.Update(conn=decay)),
        optimizer=px.optim.sgd(0.1, GRAD),
    )
    table = net.policy_table()

    assert block(net, "middle", "weight").update is decay
    assert block(net, "first", "weight").update == table.network_conns.update


def test_overrides_lists_exactly_the_cells_that_differ() -> None:
    decay, count = Decay(), Count()
    net = network(two_layers(update=px.policies.Update(conn=decay, unit=count)))

    found = [
        (o.path, o.kind, o.phase, o.policy) for o in net.policy_table().overrides()
    ]

    assert found == [
        (("middle", "weight"), "connections", "update", decay),
        (("middle", "bias_weight"), "connections", "update", decay),
        (("middle", "units"), "units", "update", count),
    ]


def test_the_table_prints_as_markdown_with_one_row_per_differing_block() -> None:
    net = network(
        two_layers(update=px.policies.Update(conn=Decay()), bias=False),
        optimizer=px.optim.sgd(0.1, GRAD),
    )

    lines = str(net.policy_table()).splitlines()
    cells = [[cell.strip() for cell in line.strip("|").split("|")] for line in lines]

    assert cells[0] == [
        "layer",
        "elements",
        "forward",
        "backward",
        "update",
        "prune",
        "grow",
    ]
    assert set(lines[1]) == {"|", "-"}
    assert cells[2][:5] == [
        "(network)",
        "connections",
        "WeightedSum",
        "-",
        "_SGDUpdateConn",
    ]
    assert cells[3][:5] == ["(network)", "units", "Summed", "-", "-"]
    # `.` is the network's policy. `first` differs nowhere and has no row.
    assert cells[4:] == [["middle.weight", "connections", ".", ".", "Decay", ".", "."]]


# --------------------------------------------------------------------------- #
# Running, and refusing what the engine cannot run yet
# --------------------------------------------------------------------------- #
def test_a_network_whose_layers_say_nothing_runs_as_before() -> None:
    net = network(two_layers())
    inputs = px.StepInputs(inputs=jnp.ones(5, jnp.float32), targets=None)

    out = net.init(KEY).step(inputs, outputs=True).outputs

    assert np.asarray(out).shape == (3,)


def test_a_layer_given_the_networks_own_policy_is_not_an_override() -> None:
    net = network(two_layers(forward=_forward))

    assert net.policy_table().overrides() == []
    net.init(KEY)


def test_a_network_with_a_layer_policy_is_refused_naming_layer_and_policies() -> None:
    net = network(
        two_layers(update=px.policies.Update(conn=Decay()), bias=False),
        optimizer=px.optim.sgd(0.1, GRAD),
    )

    with pytest.raises(pxl.BuildError) as refused:
        net.init(KEY)

    message = str(refused.value)
    assert "Net cannot be run yet" in message
    assert "middle.weight  (connections)" in message
    assert "update" in message and "Decay" in message
    assert "the network has _SGDUpdateConn" in message
    assert "Net.policy_table()" in message
    # The table itself is still there to read.
    assert net.policy_table().overrides()
