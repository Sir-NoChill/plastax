"""Network.batch_reduction: how a batched step combines per-sample unit state.

A batched step runs forward, loss, backward and the unit update once per
sample; the policy says how each written unit column becomes one value before
the once-per-batch phases. The net below writes two extra columns per sample
(a float VALUE and an int32 COUNT, both from the unit update), so the checks
can tell MEAN, SUM and FIRST apart, and that a written column the policy does
not declare -- or declares NOT_BATCHED -- is rejected.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import plastax as px

VALUE = px.FieldSpec.float32("test/value")
COUNT = px.FieldSpec.int32("test/count")
UNTOUCHED = px.FieldSpec.float32("test/untouched", 0.1)

# Inputs 0, 1 -> hidden 2 -> output 3, plus a skip edge 1 -> 3. Dyadic weights
# and inputs keep every per-sample sum exact.
_N = 4
_SRC = np.asarray([0, 1, 2, 1], np.int32)
_DST = np.asarray([2, 2, 3, 3], np.int32)
_W = np.asarray([0.5, -0.25, 0.75, 0.375], np.float32)
_X = np.asarray([[1.0, 0.5], [-0.75, 2.0], [-1.0, 0.5]], np.float32)


class Forward(px.policies.ForwardConnPolicy, px.policies.ForwardUnitPolicy):
    """activation = sum(w * activation[src]) + 1."""

    combine = px.monoid.sum_

    def map(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: None,
    ) -> jax.Array:
        del dst, g
        return c[px.WEIGHT, cid] * u[px.ACTIVATION, src]

    def apply(
        self, u: px.UnitView, i: px.UnitIdx, g: None, acc: jax.Array
    ) -> px.UnitWrite:
        del u, i, g
        return px.UnitWrite.of((px.ACTIVATION, acc + jnp.float32(1.0)))


class Tally(px.policies.UpdateUnitPolicy[None]):
    """value = 2 * activation; count = 1 + (activation > 1)."""

    def update(self, u: px.UnitView, i: px.UnitIdx, g: None) -> px.UnitWrite:
        del g
        a = u[px.ACTIVATION, i]
        return px.UnitWrite.of(
            (VALUE, a * jnp.float32(2.0)),
            (COUNT, jnp.int32(1) + (a > jnp.float32(1.0)).astype(jnp.int32)),
        )


def _net(policy: px.policies.BatchReductionPolicy | None) -> type[px.Network[None]]:
    class _Net(px.Network[None]):
        forward = Forward()
        update = px.policies.Update(unit=Tally())
        extra_unit_fields = (VALUE, COUNT, UNTOUCHED)
        propagation = px.Propagation.TOPOLOGICAL
        batch_reduction = policy

    return _Net


def _build(net: type[px.Network[None]]) -> tuple[px.NetworkStatic, Any]:
    return px.NetworkBuilder.from_edges(
        net,
        _N,
        _SRC,
        _DST,
        weights=_W,
        input_ids=(0, 1),
        output_ids=(3,),
        globals_=None,
    )


def _batched(policy: px.policies.BatchReductionPolicy) -> dict[str, np.ndarray]:
    net = _net(policy)
    static, state = _build(net)
    step = px.make_step(net, static, batch_size=len(_X), layout="edge_list")
    out = step(state, px.StepInputs(inputs=jnp.asarray(_X), targets=None)).state
    return {name: np.asarray(col) for name, col in out.units.items()}


def _per_sample() -> list[dict[str, np.ndarray]]:
    net = _net(None)
    static, state = _build(net)
    step = px.make_step(net, static)
    return [
        {
            name: np.asarray(col)
            for name, col in step(
                jax.tree.map(jnp.copy, state),
                px.StepInputs(inputs=jnp.asarray(x), targets=None),
            ).state.units.items()
        }
        for x in _X
    ]


def _policy(
    value: px.policies.Reduction, count: px.policies.Reduction
) -> px.policies.FieldReductions:
    return px.policies.FieldReductions(
        {px.ACTIVATION: px.policies.Reduction.MEAN, VALUE: value, COUNT: count}
    )


def test_sum_mean_and_first_reduce_as_declared() -> None:
    samples = _per_sample()
    values = np.stack([s[VALUE.name] for s in samples])
    counts = np.stack([s[COUNT.name] for s in samples])
    assert len({tuple(v) for v in values}) == len(_X), "samples must disagree"
    assert len({tuple(c) for c in counts}) == len(_X), "counts must disagree"

    summed = _batched(_policy(px.policies.Reduction.SUM, px.policies.Reduction.SUM))
    np.testing.assert_array_equal(summed[VALUE.name], values.sum(axis=0))
    np.testing.assert_array_equal(summed[COUNT.name], counts.sum(axis=0))

    mean = _batched(_policy(px.policies.Reduction.MEAN, px.policies.Reduction.FIRST))
    np.testing.assert_array_equal(
        mean[VALUE.name], values.sum(axis=0) / np.float32(len(_X))
    )
    np.testing.assert_array_equal(mean[COUNT.name], counts[0])

    first = _batched(_policy(px.policies.Reduction.FIRST, px.policies.Reduction.FIRST))
    np.testing.assert_array_equal(first[VALUE.name], values[0])
    np.testing.assert_array_equal(first[COUNT.name], counts[0])


def test_an_unwritten_undeclared_column_keeps_its_value() -> None:
    # (0.1 + 0.1 + 0.1) / 3 != 0.1 in float32: an implicit mean would drift it.
    got = _batched(_policy(px.policies.Reduction.MEAN, px.policies.Reduction.SUM))
    np.testing.assert_array_equal(got[UNTOUCHED.name], np.float32(0.1))
    assert got[UNTOUCHED.name].dtype == np.float32


def test_mean_float_first_rest_is_mean_of_floats_and_first_of_the_rest() -> None:
    samples = _per_sample()
    # FIRST is only observable if sample 0's int column differs from the rest.
    for other in samples[1:]:
        assert not np.array_equal(other[COUNT.name], samples[0][COUNT.name])
    got = _batched(px.policies.MeanFloatFirstRest())
    for name in (px.ACTIVATION.name, VALUE.name):
        stacked = np.stack([s[name] for s in samples])
        np.testing.assert_array_equal(
            got[name], jnp.asarray(stacked).mean(axis=0), err_msg=name
        )
    np.testing.assert_array_equal(got[COUNT.name], samples[0][COUNT.name])
    np.testing.assert_array_equal(got[px.LEVEL.name], samples[0][px.LEVEL.name])


def test_a_batched_step_requires_a_policy() -> None:
    net = _net(None)
    static, _ = _build(net)
    px.make_step(net, static)  # a streaming step needs none
    with pytest.raises(ValueError, match="batch_reduction"):
        px.make_step(net, static, batch_size=2)


def test_an_undeclared_written_column_is_rejected() -> None:
    net = _net(
        px.policies.FieldReductions(
            {
                px.ACTIVATION: px.policies.Reduction.MEAN,
                VALUE: px.policies.Reduction.MEAN,
            }
        )
    )
    static, state = _build(net)
    step = px.make_step(net, static, batch_size=len(_X))
    with pytest.raises(ValueError, match=r"'test/count'.*update_unit.*not declare"):
        step(state, px.StepInputs(inputs=jnp.asarray(_X), targets=None))


def test_a_not_batched_written_column_is_rejected() -> None:
    net = _net(_policy(px.policies.Reduction.MEAN, px.policies.Reduction.NOT_BATCHED))
    static, state = _build(net)
    step = px.make_step(net, static, batch_size=len(_X))
    with pytest.raises(ValueError, match=r"'test/count'.*NOT_BATCHED.*update_unit"):
        step(state, px.StepInputs(inputs=jnp.asarray(_X), targets=None))


def test_activation_must_be_declared_when_the_net_is_defined() -> None:
    with pytest.raises(ValueError, match=r"'activation'.*input scatter"):
        _net(px.policies.FieldReductions({VALUE: px.policies.Reduction.MEAN}))
    with pytest.raises(ValueError, match="NOT_BATCHED"):
        _net(
            px.policies.FieldReductions(
                {px.ACTIVATION: px.policies.Reduction.NOT_BATCHED}
            )
        )


def test_the_loss_seed_field_must_be_declared_when_the_net_is_defined() -> None:
    seed = px.FieldSpec.float32("test/seed")

    class _Loss(px.policies.LossPolicy):
        seed_field = seed

        def calculate_loss(
            self, u: px.UnitView, outputs: jax.Array, targets: jax.Array, g: None
        ) -> tuple[jax.Array, jax.Array]:
            del g
            diff = u.gather(px.ACTIVATION, outputs) - targets
            return jnp.sum(diff * diff), diff

    with pytest.raises(ValueError, match=r"'test/seed'.*seed_field"):

        class _Bad(px.Network[None]):
            forward = Forward()
            loss = _Loss()
            extra_unit_fields = (seed,)
            batch_reduction = px.policies.FieldReductions(
                {px.ACTIVATION: px.policies.Reduction.MEAN}
            )


@pytest.mark.parametrize(
    ("policy", "error", "match"),
    [
        (
            px.policies.FieldReductions(
                {
                    px.ACTIVATION: px.policies.Reduction.MEAN,
                    COUNT: px.policies.Reduction.MEAN,
                }
            ),
            TypeError,
            "MEAN needs a floating column",
        ),
        (
            px.policies.FieldReductions(
                {
                    px.ACTIVATION: px.policies.Reduction.MEAN,
                    px.FieldSpec.boolean("test/flag"): px.policies.Reduction.SUM,
                }
            ),
            ValueError,
            "not a unit column",
        ),
        (
            px.policies.FieldReductions(
                {
                    px.ACTIVATION: px.policies.Reduction.MEAN,
                    px.LEVEL: px.policies.Reduction.FIRST,
                }
            ),
            ValueError,
            "structural phases",
        ),
        (object(), TypeError, "must satisfy BatchReduction"),
    ],
)
def test_the_policy_is_validated_when_the_net_is_defined(
    policy: Any, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        _net(policy)


def test_sum_on_a_boolean_column_is_rejected() -> None:
    flag = px.FieldSpec.boolean("test/flag")
    with pytest.raises(TypeError, match="SUM needs a numeric"):

        class _Bad(px.Network[None]):
            forward = Forward()
            extra_unit_fields = (flag,)
            batch_reduction = px.policies.FieldReductions(
                {
                    px.ACTIVATION: px.policies.Reduction.MEAN,
                    flag: px.policies.Reduction.SUM,
                }
            )


def test_field_reductions_rejects_a_field_declared_twice() -> None:
    twin = px.FieldSpec.float32(VALUE.name, 1.0)
    with pytest.raises(ValueError, match="declared twice"):
        px.policies.FieldReductions(
            {VALUE: px.policies.Reduction.MEAN, twin: px.policies.Reduction.SUM}
        )
