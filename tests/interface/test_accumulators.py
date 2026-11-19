"""Accumulators: what passes from a connection's `map` to a unit's `apply`.

Design document section 3.6. An accumulator may be a struct of monoids, and a network
whose policy sets have different accumulators reduces into their product. The error for
a connection attached to a unit with a different accumulator is in test_build_errors.py.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from jaxtyping import Array, Float

import plastax as px

from . import support

Scalar = Float[Array, ""]
Pair = tuple[Scalar, Scalar]
f32 = np.float32

X = np.array([1.0, -2.0, 3.0, 0.5], f32)
W1 = (np.arange(12, dtype=f32).reshape(3, 4) - 5.0) / 10.0
W2 = np.array([[0.5, -1.0, 2.0], [1.5, 0.25, -0.75]], f32)


@px.policy
class ValueAndCount(px.ForwardConnPolicy[Pair, px.Globals]):
    """Contributes (weighted value, 1)."""

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
        return (c[px.WEIGHT, cid] * a, a * 0.0 + 1.0)


@px.policy
class Mean(px.ForwardUnitPolicy[Pair, px.Globals]):
    """Outputs the mean of its inputs' weighted values; keeps both sums."""

    combine = (px.monoid.sum_, px.monoid.sum_)
    total: px.UnitField[f32] = 0.0
    count: px.UnitField[f32] = 0.0

    def apply(self, u: px.UnitView, i: px.UnitIdx, g: px.Globals, acc: Pair) -> Any:
        total, count = acc
        return px.UnitWrite.of(
            (px.ACTIVATION, total / count), (self.total, total), (self.count, count)
        )


@px.policy
class Largest(px.ForwardUnitPolicy[Scalar, px.Globals]):
    """Outputs the largest of its inputs' weighted values."""

    combine = px.monoid.max_

    def apply(self, u: px.UnitView, i: px.UnitIdx, g: px.Globals, acc: Scalar) -> Any:
        return px.UnitWrite.of((px.ACTIVATION, acc))


@px.policy
class WeightAndReader(px.BackwardConnPolicy[Pair, px.Globals]):
    """Sends back (this connection's weight, 1) to its source."""

    def map(
        self,
        u: px.UnitView,
        src: px.UnitIdx,
        dst: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: px.Globals,
    ) -> Pair:
        w = c[px.WEIGHT, cid]
        return (w, w * 0.0 + 1.0)


@px.policy
class Mirror(px.BackwardUnitPolicy[Pair, px.Globals]):
    """A unit learns the sum of its outgoing weights and how many connections read
    it.
    """

    combine = (px.monoid.sum_, px.monoid.sum_)
    weight_out: px.UnitField[f32] = 0.0
    readers: px.UnitField[f32] = 0.0

    def apply(self, u: px.UnitView, i: px.UnitIdx, g: px.Globals, acc: Pair) -> Any:
        return px.UnitWrite.of((self.weight_out, acc[0]), (self.readers, acc[1]))


def frozen(forward: Any = None, backward: Any = None) -> Any:
    p = px.policies
    return p.Conn(
        forward=forward or p.WeightedSum(),
        backward=backward or p.WeightedSumBackward(),
        update=None,
        prune=None,
        add=None,
    )


def unit(forward: Any, backward: Any = None) -> Any:
    return px.policies.Unit(
        forward=forward, backward=backward or px.policies.PassThroughBackward()
    )


def test_accumulator_may_be_a_struct_of_monoids() -> None:
    mean = Mean()
    out = px.UnitsRef("out")
    fc = px.layers.Dense(
        3,
        bias=False,
        init=px.init.array(W1),
        conn_policy=frozen(forward=ValueAndCount()),
        unit_policy=unit(mean),
    )
    model = support.topology(lambda m, x: m.fc(x).bind(out), fc=fc)
    r = support.Runner(support.network(model, input_shape=(4,)))
    result = r.predict(X, keep=True)

    support.close(result.outputs, (W1 @ X) / 4.0)
    support.close(r.field(mean.total, out), W1 @ X)
    support.close(r.field(mean.count, out), np.full(3, 4.0, f32))


def test_policy_sets_with_different_accumulators_share_a_network() -> None:
    """A mean layer (pair of sums), a summing layer, a max layer and product units."""
    w3 = np.array([[1.0, -2.0], [0.5, 3.0]], f32)

    def call(m: Any, x: Any) -> Any:
        largest = m.largest(m.summed(m.mean(x)))
        return largest * largest

    model = support.topology(
        call,
        mean=px.layers.Dense(
            3,
            bias=False,
            init=px.init.array(W1),
            conn_policy=frozen(forward=ValueAndCount()),
            unit_policy=unit(Mean()),
        ),
        summed=px.layers.Dense(2, bias=False, init=px.init.array(W2)),
        largest=px.layers.Dense(
            2,
            bias=False,
            init=px.init.array(w3),
            conn_policy=frozen(),
            unit_policy=unit(Largest()),
        ),
    )
    out = support.Runner(support.network(model, input_shape=(4,))).predict(X).outputs

    summed = W2 @ ((W1 @ X) / 4.0)
    largest = (w3 * summed[None, :]).max(axis=1)
    support.close(out, largest * largest)


def test_backward_accumulator_is_reduced_at_the_source() -> None:
    """Each hidden unit collects (weight, 1) from the connections that read it."""
    mirror = Mirror()
    hidden = px.UnitsRef("hidden")
    fc1 = px.layers.Dense(
        3,
        bias=False,
        init=px.init.array(W1),
        unit_policy=unit(px.policies.SumThenActivate(), backward=mirror),
    )
    fc2 = px.layers.Dense(
        2,
        bias=False,
        init=px.init.array(W2),
        conn_policy=frozen(backward=WeightAndReader()),
    )
    model = support.topology(
        lambda m, x: m.fc2(m.fc1(x).bind(hidden)), fc1=fc1, fc2=fc2
    )
    r = support.Runner(support.network(model, input_shape=(4,), conn_policy=frozen()))
    r.step(X, np.zeros(2, f32))

    support.close(r.field(mirror.weight_out, hidden), W2.sum(axis=0))
    support.close(r.field(mirror.readers, hidden), np.full(3, 2.0, f32))


@pytest.mark.open_question(
    "What a unit with no incoming connections is handed. Assumed: the identity of its "
    "monoid, so a sum gives 0, a max gives -inf and a product unit gives 1."
) # Accepted
@pytest.mark.parametrize(
    ("policy", "expected"),
    [("sum", 0.0), ("max", -np.inf)],
)
def test_unit_with_no_incoming_connections_gets_the_identity(
    policy: str, expected: float
) -> None:
    wiring = px.layers.pairs(src=np.array([0, 1]), dst=np.array([0, 0]))
    forward = px.policies.SumThenActivate() if policy == "sum" else Largest()
    fc = px.layers.Dense(
        2,
        connectivity=wiring,
        bias=False,
        init=px.init.constant(1.0),
        conn_policy=frozen(),
        unit_policy=unit(forward),
    )
    model = support.topology(lambda m, x: m.fc(x), fc=fc)
    out = support.Runner(support.network(model, input_shape=(4,))).predict(X).outputs

    assert float(out[1]) == expected
