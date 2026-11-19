"""Growing connections.

Design document sections 3.6 and 3.10. Growth has two forms:

- carried by a connection policy set (`add=`): confined to the layer the set is attached
  to, and the new connections belong to that set;
- the network's `add_conn`: may connect any two units, and names what it creates and
what
  it may connect into.

A layer with `capacity=INITIAL` has exactly as many connection slots as it starts with,
so growth can only refill what pruning freed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from jaxtyping import Array, Bool

import plastax as px

from . import support

Flag = Bool[Array, ""]
f32 = np.float32

BORN = px.ConnField[np.int32](
    "born", default=-1
)  # the step a grown connection was made


# --------------------------------------------------------------------------- #
# Growth confined to a layer
# --------------------------------------------------------------------------- #
N_IN, N_HIDDEN, FAN_IN = 8, 4, 3
INITIAL = N_HIDDEN * FAN_IN
X = np.array([1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0], f32)
GROWN_WEIGHT = 0.5


@px.policy
class OriginalFromNegativeSource(px.PruneConnPolicy[px.Globals]):
    """Prunes a connection the topology made whose source is negative.

    Grown connections are left alone, so with a fixed input all pruning happens in the
    first step.
    """

    def predicate(
        self, u: px.UnitView, c: px.ConnView, cid: px.ConnIdx, g: px.Globals
    ) -> Flag:
        return (c[BORN, cid] < 0) & (u[px.ACTIVATION, c[px.FROM_ID, cid]] < 0.0)


@px.policy
class Refill(px.ProposeAddConnPolicy[px.Globals]):
    """Proposes random connections inside its layer."""

    proposer = px.Proposer.PER_LAYER
    proposals_per_proposer: int = 64
    dedupe_live = True
    dedupe_step = True
    on_overflow = px.Overflow.DROP

    def propose(
        self, u: px.UnitView, layer: px.LayerView, j: px.ProposalIdx, g: px.Globals
    ) -> Any:
        rng = self.rng(g, j)
        return px.Proposal(
            src=layer.random_source(rng),
            dst=layer.random_destination(rng),
            score=rng.uniform(),
        )

    def init(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: px.Globals
    ) -> Any:
        return px.ConnWrite.of((px.WEIGHT, GROWN_WEIGHT), (BORN, g.step))


def rewired(*, proposals: int = 64, update: Any = None) -> Any:
    p = px.policies
    return p.Conn(
        forward=p.WeightedSum(),
        backward=p.WeightedSumBackward(),
        update=update,
        prune=OriginalFromNegativeSource(),
        add=Refill(proposals_per_proposer=proposals),
    )


def rewiring_network(policy: Any, fmt: str = "COO") -> tuple[support.Runner, Any, Any]:
    """x -> hidden (sparse, rewired, count conserved) -> out (fixed)."""
    hidden = px.layers.Dense(
        N_HIDDEN,
        connectivity=px.layers.sparse(fan_in=FAN_IN, capacity=px.Capacity.INITIAL),
        bias=False,
        init=px.init.constant(1.0),
        conn_policy=policy,
        hint=px.Hint(format=getattr(px.Format, fmt)),
    )
    out = px.layers.Dense(2, bias=False, conn_policy=support.fixed())
    model = support.topology(lambda m, x: m.out(m.hidden(x)), hidden=hidden, out=out)
    net = support.network(model, input_shape=(N_IN,), conn_policy=None, add_conn=None)
    return support.Runner(net), hidden, out


def from_negative_source(r: support.Runner, layer: Any) -> int:
    """How many of the layer's connections read a negative input."""
    sources = np.asarray(px.inspect.sources(r.net, layer)).reshape(-1)
    negative = {int(s) for s, x in zip(sources, X, strict=True) if x < 0}
    return sum(src in negative for src, _ in r.pairs(layer))


@pytest.mark.parametrize("fmt", ["COO", "CSR", "DENSE"])
def test_layer_keeps_its_connection_count(fmt: str) -> None:
    r, hidden, _ = rewiring_network(rewired(), fmt)
    assert len(r.pairs(hidden)) == INITIAL
    to_prune = from_negative_source(r, hidden)
    assert to_prune > 0  # the case is not vacuous

    for _ in range(5):
        r.step(X, np.zeros(2, f32))
        assert len(r.pairs(hidden)) == INITIAL

    born = r.conn_field(hidden, BORN)
    assert (born >= 0).sum() == to_prune


def test_growth_stays_inside_its_layer() -> None:
    r, hidden, out = rewiring_network(rewired())
    for _ in range(5):
        r.step(X, np.zeros(2, f32))

    sources = set(np.asarray(px.inspect.sources(r.net, hidden)).reshape(-1).tolist())
    dests = set(np.asarray(px.inspect.destinations(r.net, hidden)).reshape(-1).tolist())
    assert all(src in sources and dst in dests for src, dst in r.pairs(hidden))
    assert len(r.pairs(out)) == N_HIDDEN * 2  # the other layer has no growth of its own
    assert r.num_connections() == INITIAL + N_HIDDEN * 2


def test_growth_never_makes_a_second_connection_between_two_units() -> None:
    r, hidden, _ = rewiring_network(rewired())
    for _ in range(5):
        r.step(X, np.zeros(2, f32))

    pairs = r.pairs(hidden)
    assert len(set(pairs)) == len(pairs)


def test_grown_connection_takes_its_values_from_the_growth_policys_init() -> None:
    r, hidden, _ = rewiring_network(rewired())
    r.step(X, np.zeros(2, f32))

    born, weight = r.conn_field(hidden, BORN), r.conn_field(hidden, px.WEIGHT)
    assert (born >= 0).any()
    support.close(weight[born >= 0], np.full(int((born >= 0).sum()), GROWN_WEIGHT, f32))
    support.close(weight[born < 0], np.ones(int((born < 0).sum()), f32))
    assert len(set(born[born >= 0].tolist())) == 1  # all made in the same step


def test_grown_connection_belongs_to_the_set_that_grew_it() -> None:
    """The set's update policy trains the connections the set's growth created."""
    sgd = px.optim.sgd(lr=0.1, grad=px.policies.GRAD)
    r, hidden, _ = rewiring_network(rewired(update=sgd))
    r.step(X, np.array([10.0, -10.0], f32))
    first_step = r.conn_field(hidden, BORN).max()
    r.step(X, np.array([10.0, -10.0], f32))

    born, weight = r.conn_field(hidden, BORN), r.conn_field(hidden, px.WEIGHT)
    grown_first = born == first_step
    assert grown_first.any()
    assert not np.allclose(weight[grown_first], GROWN_WEIGHT)


def test_shortfall_is_refilled_on_later_steps() -> None:
    """One proposal a step cannot refill everything the first step pruned."""
    r, hidden, _ = rewiring_network(rewired(proposals=1))
    to_prune = from_negative_source(r, hidden)
    assert to_prune >= 2  # the case is not vacuous

    r.step(X, np.zeros(2, f32))
    after_first = len(r.pairs(hidden))
    assert INITIAL - to_prune <= after_first <= INITIAL - to_prune + 1

    counts = []
    for _ in range(200):
        r.step(X, np.zeros(2, f32))
        counts.append(len(r.pairs(hidden)))

    assert all(b >= a for a, b in zip(counts, counts[1:], strict=False))
    assert max(counts) <= INITIAL
    assert counts[-1] == INITIAL


# --------------------------------------------------------------------------- #
# Network-level growth
# --------------------------------------------------------------------------- #
LOUD = np.array([10.0, 20.0, 30.0], f32)  # the inputs: the only units above 5
W_HIDDEN = np.full((2, 3), 1e-3, f32)
W_OUT = np.array([[1.0, -1.0]], f32)
SKIP_WEIGHT = 2.0
SKIP_LR = 1e-5  # small, so training never makes a hidden unit loud

TARGET = px.UnitRef("target")


@px.policy
class LoudUnitsToTarget(px.ProposeAddConnPolicy[px.Globals]):
    """Every unit whose activation is above 5 proposes a connection to one unit."""

    proposer = px.Proposer.PER_UNIT
    proposals_per_proposer = 1
    max_level_gap = 8
    dedupe_live = True
    target: px.UnitRef
    creates: px.policies.Conn
    into: px.policies.Unit

    def propose(
        self, u: px.UnitView, i: px.UnitIdx, j: px.ProposalIdx, g: px.Globals
    ) -> Any:
        loud = u[px.ACTIVATION, i] > 5.0
        return px.Proposal(src=i, dst=self.target, score=np.where(loud, 1.0, -np.inf))

    def init(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: px.Globals
    ) -> Any:
        return px.ConnWrite.of((px.WEIGHT, SKIP_WEIGHT))


class SkipNetwork:
    """x (3, loud) -> hidden (2, quiet) -> out (1). Growth adds skips into `target`.

    The output unit has the unit policy set `special`, which is what the growth rule may
    connect into. With `target="out"` the rule aims there. With `target="lone"` it aims
    at a unit of the default set instead, which it may not connect into.
    """

    def __init__(
        self, *, target: str = "out", trained: bool = False, fmt: str = "COO"
    ) -> None:
        plain = px.policies.summing()
        special = px.policies.summing()  # the same behaviour, a distinct set
        self.hidden = px.layers.Dense(2, bias=False, init=px.init.array(W_HIDDEN))
        self.out = px.layers.Dense(1, bias=False, init=px.init.array(W_OUT))

        def call(m: Any, x: Any) -> Any:
            lone = m.lone(x)
            y = m.out(m.hidden(x)).with_policy(special)
            (y if target == "out" else lone).bind(TARGET)
            return y

        model = support.topology(
            call,
            hidden=self.hidden,
            out=self.out,
            lone=px.layers.Dense(1, bias=False, init=px.init.constant(1e-3)),
        )
        net = support.network(
            model,
            input_shape=(3,),
            conn_policy=support.learned(SKIP_LR) if trained else support.fixed(),
            unit_policy=plain,
            add_conn=LoudUnitsToTarget(
                target=TARGET, creates=support.fixed(), into=special
            ),
            config=px.Config(
                format=getattr(px.Format, fmt),
                propagation=px.Propagation.TOPOLOGICAL,
                seed=0,
            ),
        )
        self.runner = support.Runner(net)


def with_skips() -> np.ndarray:
    return W_OUT @ (W_HIDDEN @ LOUD) + SKIP_WEIGHT * LOUD.sum()


def test_network_level_growth_connects_units_no_layer_connects() -> None:
    r = SkipNetwork().runner
    before = r.num_connections()
    r.step(LOUD, np.zeros(1, f32))

    assert r.num_connections() == before + 3
    support.close(r.predict(LOUD).outputs, with_skips())


def test_network_level_growth_does_not_repeat_a_live_connection() -> None:
    r = SkipNetwork().runner
    r.step(LOUD, np.zeros(1, f32))
    after_first = r.num_connections()
    r.step(LOUD, np.zeros(1, f32))

    assert r.num_connections() == after_first


def test_created_connections_carry_the_set_named_by_creates() -> None:
    """`creates` is a frozen set, so the skips are not trained though the layers are."""
    network = SkipNetwork(trained=True)
    r = network.runner
    r.step(LOUD, np.zeros(1, f32))  # the skips are made
    w_out = r.weights(network.out)
    r.step(LOUD, np.zeros(1, f32))  # a training step with the skips in place

    assert not np.allclose(r.weights(network.out), w_out)
    hidden = r.weights(network.hidden) @ LOUD
    expected = r.weights(network.out) @ hidden + SKIP_WEIGHT * LOUD.sum()
    support.close(r.predict(LOUD).outputs, expected, rtol=1e-4)


@pytest.mark.open_question(
    "A proposal into a unit outside the rule's `into` set is not created and is "
    "emitted "
    "as a non-fatal error. How the error queue is held and read is undecided; "
    "`result.errors.refused_growth` is a guess."
)
def test_proposal_into_a_unit_outside_into_is_refused_and_reported() -> None:
    r = SkipNetwork(target="lone").runner  # `lone` is not of the `into` set
    before = r.num_connections()
    result = r.step(LOUD, np.zeros(1, f32))

    assert r.num_connections() == before
    assert int(result.errors.refused_growth) == 3


@pytest.mark.open_question(
    "A connection grown between units that a dense block does not cover is stored in a "
    "sparse format beside it. How that is laid out is undecided; the result must not "
    "depend on it."
)
def test_growth_outside_a_dense_block_gives_the_same_result() -> None:
    r = SkipNetwork(fmt="DENSE").runner
    r.step(LOUD, np.zeros(1, f32))

    support.close(r.predict(LOUD).outputs, with_skips())
