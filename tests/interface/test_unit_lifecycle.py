"""Creating and removing units at run time.

Design document sections 2.2, 2.7 and 3.10; the rules come from
`docs/design/sketches/imprinting_learner.py`, reduced to their mechanics.

The network is three inputs and one output. In a step, every input above 0.5 spawns one
child unit, and a growth rule triggered by new units wires each child to its parent. A
child outputs ten times its weighted input.
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
Flag = Bool[Array, ""]
f32 = np.float32

BORN = px.UnitField[np.int32]("born", default=-1)  # the step a unit was created
PARENT = px.UnitField[np.int32]("parent", default=-1)

SPAWNING = np.array([1.0, -1.0, 2.0], f32)  # inputs 0 and 2 spawn
QUIET = np.array([0.4, 0.0, 0.3], f32)  # nothing spawns
WIRE_WEIGHT = 3.0
N_INITIAL_UNITS = 4  # three inputs and the output


@px.policy
class TenTimes(px.ForwardUnitPolicy[Scalar, px.Globals]):
    combine = px.monoid.sum_

    def apply(self, u: px.UnitView, i: px.UnitIdx, g: px.Globals, acc: Scalar) -> Any:
        return px.UnitWrite.of((px.ACTIVATION, 10.0 * acc))


@px.policy
class SpawnFromLoudInputs(px.AddUnitPolicy[px.Globals]):
    """A unit the topology made spawns a child when its activation is above 0.5."""

    child: px.policies.Unit  # the policy set of every unit this rule creates

    def spawn(self, u: px.UnitView, parent: px.UnitIdx, g: px.Globals) -> Flag:
        return (u[BORN, parent] < 0) & (u[px.ACTIVATION, parent] > 0.5)

    def init(
        self, u: px.UnitView, child: px.UnitIdx, parent: px.UnitIdx, g: px.Globals
    ) -> Any:
        return px.UnitWrite.of((BORN, g.step), (PARENT, parent))


@px.policy
class WireChildToParent(px.ScoreAddConnPolicy[px.Globals]):
    """Connects each unit created this step to its parent."""

    candidates = px.Candidates.EXHAUSTIVE
    selection = px.Selection.ALL
    trigger = px.Trigger.ON_UNITS_ADDED
    max_level_gap = 8
    creates: px.policies.Conn
    into: px.policies.Unit

    def score(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: px.Globals
    ) -> Scalar:
        wanted = (u[BORN, dst] == g.step) & (src == u[PARENT, dst])
        return jnp.where(wanted, 0.0, -jnp.inf)

    def init(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: px.Globals
    ) -> Any:
        return px.ConnWrite.of((px.WEIGHT, WIRE_WEIGHT))


@px.policy
class BornBeforeThisStep(px.PruneUnitPolicy[px.Globals]):
    """Removes a created unit, but not in the step it was created."""

    def predicate(self, u: px.UnitView, i: px.UnitIdx, g: px.Globals) -> Flag:
        return (u[BORN, i] >= 0) & (u[BORN, i] < g.step)


@px.policy
class FromLoudSource(px.PruneConnPolicy[px.Globals]):
    def predicate(
        self, u: px.UnitView, c: px.ConnView, cid: px.ConnIdx, g: px.Globals
    ) -> Flag:
        return u[px.ACTIVATION, c[px.FROM_ID, cid]] > 10.0


class Growing:
    """The network described at the top of this file."""

    def __init__(
        self,
        *,
        prune_children: bool = False,
        unit_capacity: int = 16,
        readout: Any = None,
    ) -> None:
        p = px.policies
        self.child = p.Unit(
            forward=TenTimes(),
            backward=p.PassThroughBackward(),
            prune=BornBeforeThisStep() if prune_children else None,
        )
        self.inputs = px.UnitsRef("inputs")
        self.out = readout or px.layers.Dense(
            1, bias=False, init=px.init.constant(1.0), conn_policy=support.fixed()
        )
        model = support.topology(lambda m, x: m.out(x.bind(self.inputs)), out=self.out)
        net = support.network(
            model,
            input_shape=(3,),
            conn_policy=None,
            add_unit=SpawnFromLoudInputs(child=self.child),
            add_conn=WireChildToParent(creates=support.fixed(), into=self.child),
            config=px.Config(
                propagation=px.Propagation.TOPOLOGICAL,
                unit_capacity=unit_capacity,
                seed=0,
            ),
        )
        self.runner = support.Runner(net)

    def children(self, field: Any) -> np.ndarray:
        """A field over the created units, in unit order."""
        born = self.runner.field(BORN, None)
        return self.runner.field(field, None)[born >= 0]


# --------------------------------------------------------------------------- #
# Creating units
# --------------------------------------------------------------------------- #
def test_spawn_creates_one_unit_per_parent_that_asks() -> None:
    network = Growing()
    r = network.runner
    assert r.num_units() == N_INITIAL_UNITS

    r.step(SPAWNING, np.zeros(1, f32))

    assert r.num_units() == N_INITIAL_UNITS + 2


def test_child_fields_come_from_the_spawning_policys_init() -> None:
    network = Growing()
    r = network.runner
    r.step(SPAWNING, np.zeros(1, f32))

    input_ids = r.unit_ids(network.inputs)
    assert sorted(network.children(PARENT).tolist()) == sorted(
        [int(input_ids[0]), int(input_ids[2])]
    )
    assert len(set(network.children(BORN).tolist())) == 1


def test_new_units_are_wired_in_the_step_that_creates_them() -> None:
    network = Growing()
    r = network.runner
    before = r.num_connections()
    initial_pairs = set(r.pairs())
    r.step(SPAWNING, np.zeros(1, f32))

    assert r.num_connections() == before + 2
    input_ids = r.unit_ids(network.inputs)
    new_pairs = set(r.pairs()) - initial_pairs
    assert {src for src, _ in new_pairs} == {int(input_ids[0]), int(input_ids[2])}
    assert len({dst for _, dst in new_pairs}) == 2  # one child each


def test_child_runs_the_policy_set_named_by_the_spawning_policy() -> None:
    """A child is not an input like its parent: it has the `child` set's forward."""
    network = Growing()
    r = network.runner
    r.step(SPAWNING, np.zeros(1, f32))
    r.predict(QUIET, keep=True)

    # Children of inputs 0 and 2: ten times the wire weight times the parent's value.
    expected = sorted([10.0 * WIRE_WEIGHT * QUIET[0], 10.0 * WIRE_WEIGHT * QUIET[2]])
    support.close(np.sort(network.children(px.ACTIVATION)), np.array(expected, f32))


def test_units_already_created_do_not_spawn_again_when_nothing_asks() -> None:
    network = Growing()
    r = network.runner
    r.step(SPAWNING, np.zeros(1, f32))
    for _ in range(3):
        r.step(QUIET, np.zeros(1, f32))

    assert r.num_units() == N_INITIAL_UNITS + 2


def test_unit_capacity_bounds_how_many_units_exist() -> None:
    network = Growing(unit_capacity=N_INITIAL_UNITS + 1)
    r = network.runner
    r.step(SPAWNING, np.zeros(1, f32))

    assert r.num_units() == N_INITIAL_UNITS + 1


# --------------------------------------------------------------------------- #
# Removing units
# --------------------------------------------------------------------------- #
def test_unit_is_not_removed_in_the_step_that_creates_it() -> None:
    r = Growing(prune_children=True).runner
    r.step(SPAWNING, np.zeros(1, f32))

    assert r.num_units() == N_INITIAL_UNITS + 2


def test_removed_unit_takes_its_connections_with_it() -> None:
    r = Growing(prune_children=True).runner
    connections = r.num_connections()
    r.step(SPAWNING, np.zeros(1, f32))
    r.step(QUIET, np.zeros(1, f32))

    assert r.num_units() == N_INITIAL_UNITS
    assert r.num_connections() == connections


def test_slots_of_removed_units_are_used_again() -> None:
    """With room for two created units, two can be created again after two are
    removed.
    """
    r = Growing(prune_children=True, unit_capacity=N_INITIAL_UNITS + 2).runner
    for _ in range(4):
        r.step(
            SPAWNING, np.zeros(1, f32)
        )  # removes last step's pair, creates a new one

        assert r.num_units() == N_INITIAL_UNITS + 2


# --------------------------------------------------------------------------- #
# ReduceAll: a readout over every unit, including those created later
# --------------------------------------------------------------------------- #
REDUCE_ALL = pytest.mark.open_question(
    "Which units `ReduceAll` counts as features is not written down. Assumed: the "
    "inputs and every unit created at run time, not its own units."
)


def reduce_all(prune: Any = None) -> Any:
    p = px.policies
    conn = p.Conn(
        forward=p.WeightedSum(),
        backward=p.WeightedSumBackward(),
        update=None,
        prune=prune,
        add=None,
    )
    return px.layers.ReduceAll(1, init=px.init.constant(1.0), conn_policy=conn)


@REDUCE_ALL
def test_reduce_all_connects_every_unit_that_exists_at_the_start() -> None:
    network = Growing(readout=reduce_all())
    r = network.runner

    assert len(r.pairs(network.out)) == 3
    support.close(r.predict(QUIET).outputs, [QUIET.sum()])


@REDUCE_ALL
def test_reduce_all_connects_a_unit_when_it_is_created() -> None:
    network = Growing(readout=reduce_all())
    r = network.runner
    r.step(SPAWNING, np.zeros(1, f32))

    assert len(r.pairs(network.out)) == 3 + 2
    children = 10.0 * WIRE_WEIGHT * (QUIET[0] + QUIET[2])
    support.close(r.predict(QUIET).outputs, [QUIET.sum() + children])


@REDUCE_ALL
def test_pruned_reduce_all_connection_is_never_restored() -> None:
    """A unit whose readout connection was pruned stays, unread by the readout."""
    network = Growing(readout=reduce_all(prune=FromLoudSource()))
    r = network.runner
    r.step(SPAWNING, np.zeros(1, f32))
    assert len(r.pairs(network.out)) == 5

    # With QUIET the child of input 0 reaches 12, above the rule's 10; the child of
    # input 2 reaches 9. One readout connection is pruned.
    r.step(QUIET, np.zeros(1, f32))
    assert len(r.pairs(network.out)) == 4
    for _ in range(3):
        r.step(QUIET, np.zeros(1, f32))

    assert len(r.pairs(network.out)) == 4
    assert r.num_units() == N_INITIAL_UNITS + 2  # the unit itself is still there


# --------------------------------------------------------------------------- #
# Deferred
# --------------------------------------------------------------------------- #
@pytest.mark.open_question(
    "A unit created at run time can join a layer's population (its sources or its "
    "destinations). The mechanism is deferred: card unit-addition-into-layers."
)
def test_created_unit_joins_a_layers_population() -> None:
    pytest.skip("no interface to test against yet")
