"""Fields: namespaced to a policy, shared between policies, read by another policy.

Design document section 3.4.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from jaxtyping import Array, Float

import plastax as px

from . import support

Scalar = Float[Array, ""]
f32 = np.float32

X = np.array([1.0, -2.0, 3.0, 0.5], np.float32)
W1 = (np.arange(12, dtype=np.float32).reshape(3, 4) - 5.0) / 10.0
W2 = np.array([[0.5, -1.0, 2.0], [1.5, 0.25, -0.75]], np.float32)

# One module-level object, used by the policy that writes it and the one that reads it.
TRACE = px.UnitField[f32]("trace", default=0.0)


@px.policy
class Keeps(px.ForwardUnitPolicy[Scalar, px.Globals]):
    """Outputs its sum and keeps `sign * sum` in a field of its own called `pre`."""

    combine = px.monoid.sum_
    sign: float = 1.0
    pre: px.UnitField[f32] = -7.0
    count: px.UnitField[np.int32] = 0
    seen: px.UnitField[np.bool_] = False

    def apply(self, u: px.UnitView, i: px.UnitIdx, g: px.Globals, acc: Scalar) -> Any:
        return px.UnitWrite.of(
            (px.ACTIVATION, acc),
            (self.pre, self.sign * acc),
            (self.count, u[self.count, i] + 1),
            (self.seen, True),
        )


@px.policy
class WritesTrace(px.UpdateUnitPolicy[px.Globals]):
    """Writes the shared field."""

    def update(self, u: px.UnitView, i: px.UnitIdx, g: px.Globals) -> Any:
        return px.UnitWrite.of((TRACE, 10.0 * u[px.ACTIVATION, i]))


@px.policy
class CopiesAnothersField(px.UpdateUnitPolicy[px.Globals]):
    """Reads a field that belongs to another policy, handed to it as an attribute."""

    source: px.UnitField[f32]
    copy: px.UnitField[f32] = 0.0

    def update(self, u: px.UnitView, i: px.UnitIdx, g: px.Globals) -> Any:
        return px.UnitWrite.of((self.copy, u[self.source, i]))


def unit_set(forward: Any, **more: Any) -> Any:
    return px.policies.Unit(
        forward=forward, backward=px.policies.PassThroughBackward(), **more
    )


def two_groups(set_a: Any, set_b: Any) -> tuple[support.Runner, Any, Any]:
    """x -> fc1 (units with set_a) -> fc2 (units with set_b), after one forward pass."""
    a, b = px.UnitsRef("a"), px.UnitsRef("b")

    def call(m: Any, x: Any) -> Any:
        h = m.fc1(x).with_policy(set_a).bind(a)
        return m.fc2(h).with_policy(set_b).bind(b)

    model = support.topology(
        call,
        fc1=px.layers.Dense(3, bias=False, init=px.init.array(W1)),
        fc2=px.layers.Dense(2, bias=False, init=px.init.array(W2)),
    )
    r = support.Runner(support.network(model, input_shape=(4,)))
    r.predict(X, keep=True)
    return r, a, b


def test_two_policies_may_each_declare_a_field_of_the_same_name() -> None:
    plus, minus = Keeps(sign=1.0), Keeps(sign=-1.0)
    r, a, b = two_groups(unit_set(plus), unit_set(minus))

    assert plus.pre is not minus.pre
    support.close(r.field(plus.pre, a), W1 @ X)
    support.close(r.field(minus.pre, b), -(W2 @ W1 @ X))


def test_field_keeps_its_default_where_its_policy_does_not_run() -> None:
    plus, minus = Keeps(sign=1.0), Keeps(sign=-1.0)
    r, a, b = two_groups(unit_set(plus), unit_set(minus))

    # `plus` runs on group a only, so its field is untouched on group b.
    support.close(r.field(plus.pre, b), np.full(2, -7.0, f32))
    support.close(r.field(minus.pre, a), np.full(3, -7.0, f32))


def test_fields_keep_their_declared_type() -> None:
    keeps = Keeps()
    r, a, _ = two_groups(unit_set(keeps), unit_set(Keeps()))
    r.predict(X, keep=True)

    count, seen = r.field(keeps.count, a), r.field(keeps.seen, a)
    assert count.dtype == np.int32
    assert seen.dtype == np.bool_
    np.testing.assert_array_equal(count, [2, 2, 2])  # two forward passes were kept
    assert seen.all()


def test_shared_field_is_one_column_for_every_policy_that_uses_it() -> None:
    reader = CopiesAnothersField(source=TRACE)
    with_trace = px.policies.Unit(
        forward=px.policies.SumThenActivate(),
        backward=px.policies.PassThroughBackward(),
        update=WritesTrace(),
    )
    a = px.UnitsRef("a")
    fc = px.layers.Dense(3, bias=False, init=px.init.array(W1))
    model = support.topology(
        lambda m, x: m.fc(x).with_policy(with_trace).bind(a), fc=fc
    )
    r = support.Runner(support.network(model, input_shape=(4,)))
    r.step(X, np.zeros(3, f32))

    assert reader.source is TRACE
    support.close(r.field(TRACE, a), 10.0 * (W1 @ X))


def test_policy_reads_another_policys_field_passed_to_it() -> None:
    """Pass the field when only the other policy's state is needed."""
    keeps = Keeps(sign=3.0)
    copier = CopiesAnothersField(source=keeps.pre)
    a = px.UnitsRef("a")
    fc = px.layers.Dense(3, bias=False, init=px.init.array(W1))
    model = support.topology(
        lambda m, x: m.fc(x).with_policy(unit_set(keeps, update=copier)).bind(a),
        fc=fc,
    )
    r = support.Runner(support.network(model, input_shape=(4,)))
    r.step(X, np.zeros(3, f32))

    support.close(r.field(copier.copy, a), 3.0 * (W1 @ X))


def test_policy_uses_another_policys_behaviour_passed_to_it() -> None:
    """Pass the policy when its behaviour is needed: the backward policy calls the
    forward policy's activation derivative.
    """
    p = px.policies
    forward = p.SumThenActivate(activation=px.activations.tanh)
    unit = p.Unit(forward=forward, backward=p.ActivationBackward(forward=forward))
    hidden = px.UnitsRef("hidden")
    fc1 = px.layers.Dense(3, bias=False, init=px.init.array(W1))
    fc2 = px.layers.Dense(2, bias=False, init=px.init.array(W2))
    model = support.topology(
        lambda m, x: m.fc2(m.fc1(x).with_policy(unit).bind(hidden)), fc1=fc1, fc2=fc2
    )
    r = support.Runner(support.network(model, input_shape=(4,)))
    target = np.array([0.5, -1.0], f32)
    r.step(X, target)

    z = W1 @ X
    seed = W2 @ np.tanh(z) - target
    support.close(r.field(forward.pre, hidden), z)
    support.close(r.field(p.GRAD, hidden), (W2.T @ seed) * (1 - np.tanh(z) ** 2))
