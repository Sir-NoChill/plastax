"""Random streams: `self.rng(g, element)`.

Design document section 3.5. A policy's stream is its own for an element and a step, it
is independent of every other policy's, and draws replay exactly. Two methods of one
policy called for the same element in the same step get the same stream from the same
start.
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

UNITS = 512
X = np.ones(4, f32)


@px.policy
class Draws(px.UpdateUnitPolicy[px.Globals]):
    """Each step, writes two uniform draws and one normal draw for its unit."""

    first: px.UnitField[f32] = -1.0
    second: px.UnitField[f32] = -1.0
    normal: px.UnitField[f32] = 0.0

    def update(self, u: px.UnitView, i: px.UnitIdx, g: px.Globals) -> Any:
        rng = self.rng(g, i)
        return px.UnitWrite.of(
            (self.first, rng.uniform()),
            (self.second, rng.uniform()),
            (self.normal, rng.normal()),
        )


@px.policy
class DrawsInForward(px.ForwardUnitPolicy[Scalar, px.Globals]):
    """A second policy on the same units: writes one uniform draw in the forward
    pass.
    """

    combine = px.monoid.sum_
    first: px.UnitField[f32] = -1.0

    def apply(self, u: px.UnitView, i: px.UnitIdx, g: px.Globals, acc: Scalar) -> Any:
        return px.UnitWrite.of(
            (px.ACTIVATION, acc), (self.first, self.rng(g, i).uniform())
        )


@px.policy
class DrawsTwice(px.UpdateConnPolicy[Scalar, px.Globals]):
    """`contribution` and `step` each take the first draw of this policy's stream."""

    combine = px.monoid.sum_
    in_contribution: px.ConnField[f32] = -1.0
    in_step: px.ConnField[f32] = -1.0

    def contribution(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: px.Globals,
    ) -> Scalar:
        return self.rng(g, cid).uniform()

    def step(
        self, c: px.ConnView, cid: px.ConnIdx, g: px.Globals, total: Scalar
    ) -> Any:
        return px.ConnWrite.of(
            (self.in_contribution, total), (self.in_step, self.rng(g, cid).uniform())
        )


def drawing_network(
    update: Draws, *, forward: Any = None, seed: int = 0
) -> tuple[support.Runner, Any]:
    """x -> fc (UNITS units), whose units run `update` and, if given, `forward`."""
    out = px.UnitsRef("out")
    p = px.policies
    unit = p.Unit(
        forward=forward or p.SumThenActivate(),
        backward=p.PassThroughBackward(),
        update=update,
    )
    model = support.topology(
        lambda m, x: m.fc(x).with_policy(unit).bind(out),
        fc=px.layers.Dense(UNITS, bias=False),
    )
    net = support.network(
        model,
        input_shape=(4,),
        config=px.Config(propagation=px.Propagation.TOPOLOGICAL, seed=seed),
    )
    return support.Runner(net), out


def draws(policy: Draws, *, seed: int = 0, steps: int = 1) -> np.ndarray:
    r, out = drawing_network(policy, seed=seed)
    for _ in range(steps):
        r.step(X, np.zeros(UNITS, f32))
    return r.field(policy.first, out)


def test_draws_replay_exactly_for_the_same_seed() -> None:
    np.testing.assert_array_equal(draws(Draws(), seed=3), draws(Draws(), seed=3))


def test_draws_differ_between_seeds() -> None:
    assert not np.array_equal(draws(Draws(), seed=3), draws(Draws(), seed=4))


def test_draws_differ_between_steps() -> None:
    assert not np.array_equal(draws(Draws(), steps=1), draws(Draws(), steps=2))


def test_draws_differ_between_elements() -> None:
    first = draws(Draws())

    assert len(np.unique(first)) > 0.99 * UNITS


def test_successive_draws_from_one_stream_differ() -> None:
    policy = Draws()
    r, out = drawing_network(policy)
    r.step(X, np.zeros(UNITS, f32))

    assert not np.array_equal(r.field(policy.first, out), r.field(policy.second, out))


def test_uniform_draws_are_in_the_unit_interval_and_spread_over_it() -> None:
    first = draws(Draws())

    assert first.min() >= 0.0
    assert first.max() < 1.0
    assert abs(first.mean() - 0.5) < 0.1
    assert abs(first.std() - np.sqrt(1 / 12)) < 0.05


def test_normal_draws_have_zero_mean_and_unit_variance() -> None:
    policy = Draws()
    r, out = drawing_network(policy)
    r.step(X, np.zeros(UNITS, f32))
    normal = r.field(policy.normal, out)

    assert abs(normal.mean()) < 0.2
    assert abs(normal.std() - 1.0) < 0.2


def test_two_policies_on_the_same_units_have_independent_streams() -> None:
    in_update, in_forward = Draws(), DrawsInForward()
    r, out = drawing_network(in_update, forward=in_forward)
    r.step(X, np.zeros(UNITS, f32))

    a, b = r.field(in_update.first, out), r.field(in_forward.first, out)
    assert not np.array_equal(a, b)
    assert abs(np.corrcoef(a, b)[0, 1]) < 0.2


@pytest.mark.open_question(
    "How a policy is given its stream is not specified beyond 'when the network is "
    "built'. This assumes it follows the policy's place in the network."
)
def test_stream_follows_the_policys_place_in_the_network_not_the_python_object() -> (
    None
):
    a, b = Draws(), Draws()
    ra, out_a = drawing_network(a)
    rb, out_b = drawing_network(b)
    ra.step(X, np.zeros(UNITS, f32))
    rb.step(X, np.zeros(UNITS, f32))

    # Two networks with the same seed and layout, built from two separate instances.
    np.testing.assert_array_equal(ra.field(a.first, out_a), rb.field(b.first, out_b))


def test_two_methods_of_one_policy_get_the_same_stream() -> None:
    """For a private weight, `step` sees in `total` the draw `contribution` made, and
    its own first draw from the same stream is the same number.
    """
    update = DrawsTwice()
    conn = px.policies.Conn(
        forward=px.policies.WeightedSum(),
        backward=px.policies.WeightedSumBackward(),
        update=update,
    )
    fc = px.layers.Dense(8, bias=False)
    model = support.topology(lambda m, x: m.fc(x), fc=fc)
    r = support.Runner(support.network(model, input_shape=(4,), conn_policy=conn))
    r.step(X, np.zeros(8, f32))

    in_contribution = r.conn_field(fc, update.in_contribution)
    assert len(np.unique(in_contribution)) == 8 * 4
    np.testing.assert_array_equal(in_contribution, r.conn_field(fc, update.in_step))
