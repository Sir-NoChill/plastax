"""Initializers, called directly with a hand-made `Fans`. No network is built.

Plan `plans/20261120T181444Z-initializers.md` (workspace root), steps 2 and 3. Whether a
layer hands the right `Fans` to its initializer is tested with `materialize`.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import numpy as np
import pytest

import plastax.init as init
from plastax.init import Distribution, FanMode, Fans, Initializer


def fans(*shape: int) -> Fans:
    """A block of the given shape. These initializers do not read the fans."""
    return Fans(shape, np.ones(shape, np.int64), np.ones(shape, np.int64))


def drawn(make: Initializer, *shape: int, seed: int = 0) -> np.ndarray:
    return np.asarray(make(jax.random.key(seed), fans(*shape)))


# --------------------------------------------------------------------------- #
# Given values
# --------------------------------------------------------------------------- #
def test_ones_fills_the_block_with_one() -> None:
    w = drawn(init.ones(), 3, 4)

    np.testing.assert_array_equal(w, np.ones((3, 4), np.float32))


def test_array_does_not_follow_the_array_it_was_given() -> None:
    values = np.zeros((2, 2), np.float32)
    make = init.array(values)

    values[0, 0] = 9.0

    assert drawn(make, 2, 2)[0, 0] == 0.0


def test_what_array_returns_cannot_be_written() -> None:
    w = drawn(init.array(np.zeros((2, 2))), 2, 2)

    with pytest.raises(ValueError, match="read-only"):
        w[0, 0] = 9.0


def test_array_refuses_a_block_of_another_shape() -> None:
    """The same number of weights in another arrangement is refused too."""
    make = init.array(np.zeros((3, 4)))

    with pytest.raises(ValueError, match=r"shape \(3, 4\) for a block of shape"):
        make(jax.random.key(0), fans(12))


# --------------------------------------------------------------------------- #
# Drawn values
# --------------------------------------------------------------------------- #
N = 20_000  # weights per draw in the tests of a distribution


def test_normal_has_mean_zero_and_the_spread_given() -> None:
    w = drawn(init.normal(0.3), N)

    assert w.dtype == np.float32
    np.testing.assert_allclose(w.std(), 0.3, rtol=0.03)
    assert abs(w.mean()) < 0.01
    assert np.abs(w).max() > 2 * 0.3  # not cut off


def test_truncated_normal_stays_within_two_standard_deviations() -> None:
    w = drawn(init.truncated_normal(0.3), N)

    assert w.dtype == np.float32
    assert np.abs(w).max() <= 2 * 0.3
    assert np.abs(w).max() > 1.9 * 0.3
    np.testing.assert_allclose(w.std(), 0.88 * 0.3, rtol=0.03)


def test_uniform_fills_the_range_between_its_bounds() -> None:
    w = drawn(init.uniform(-0.5, 1.5), N)

    assert w.dtype == np.float32
    assert w.min() >= -0.5
    assert w.max() < 1.5
    assert w.min() < -0.49
    assert w.max() > 1.49
    np.testing.assert_allclose(w.mean(), 0.5, atol=0.02)


def test_uniform_needs_its_lower_bound_below_its_upper() -> None:
    with pytest.raises(ValueError, match="low < high"):
        init.uniform(1.0, 1.0)


RANDOM: list[Callable[[], Initializer]] = [
    lambda: init.normal(0.3),
    lambda: init.truncated_normal(0.3),
    lambda: init.uniform(-1.0, 1.0),
    lambda: init.lecun_normal(),
]


# --------------------------------------------------------------------------- #
# Variance scaling: the fans are read per connection
# --------------------------------------------------------------------------- #
FAN_IN, FAN_OUT = 10, 90  # different, so that reading the wrong one shows


def scaled(
    make: Initializer, fan_in: int = FAN_IN, fan_out: int = FAN_OUT
) -> np.ndarray:
    """`N` weights drawn for connections that all have the given fans."""
    block = Fans((N,), np.full(N, fan_in), np.full(N, fan_out))
    return np.asarray(make(jax.random.key(0), block))


@pytest.mark.parametrize(
    ("mode", "fan"),
    [(FanMode.FAN_IN, 10), (FanMode.FAN_OUT, 90), (FanMode.FAN_AVG, 50)],
)
def test_each_mode_divides_by_its_fan(mode: FanMode, fan: int) -> None:
    w = scaled(init.variance_scaling(3.0, mode, Distribution.NORMAL))

    assert w.dtype == np.float32
    np.testing.assert_allclose(w.var(), 3.0 / fan, rtol=0.05)


@pytest.mark.parametrize(
    "distribution", [d for d in Distribution if d is not Distribution.NORMAL]
)
def test_each_distribution_has_the_variance_asked_for(
    distribution: Distribution,
) -> None:
    """The uniform bound and the correction of the truncated normal are both right."""
    w = scaled(init.variance_scaling(1.0, FanMode.FAN_IN, distribution))

    np.testing.assert_allclose(w.var(), 1.0 / FAN_IN, rtol=0.05)
    assert abs(w.mean()) < 0.02


def test_the_distributions_differ_in_how_far_a_weight_can_lie() -> None:
    stddev = np.sqrt(1.0 / FAN_IN)

    def furthest(distribution: Distribution) -> float:
        w = scaled(init.variance_scaling(1.0, FanMode.FAN_IN, distribution))
        return float(np.abs(w).max() / stddev)

    assert 1.72 < furthest(Distribution.UNIFORM) <= np.sqrt(3.0)
    assert 2.2 < furthest(Distribution.TRUNCATED_NORMAL) <= 2 / 0.8796
    assert furthest(Distribution.NORMAL) > 3.0


def test_each_connection_is_drawn_for_its_own_fan_in() -> None:
    """Half the block feeds units with 4 inputs, half units with 400."""
    fan_in = np.repeat([4, 400], N).reshape(2, N)
    block = Fans((2, N), fan_in, np.ones((2, N), np.int64))

    few, many = np.asarray(init.lecun_uniform()(jax.random.key(0), block))

    assert np.abs(few).max() <= np.sqrt(3 / 4)
    assert np.abs(few).max() > 0.99 * np.sqrt(3 / 4)
    assert np.abs(many).max() <= np.sqrt(3 / 400)
    np.testing.assert_allclose(few.var(), 1 / 4, rtol=0.05)
    np.testing.assert_allclose(many.var(), 1 / 400, rtol=0.05)


@pytest.mark.parametrize(
    ("mode", "fan_in", "fan_out"),
    [(FanMode.FAN_IN, 0, 7), (FanMode.FAN_OUT, 7, 0)],
)
def test_a_fan_of_zero_is_read_as_one(mode: FanMode, fan_in: int, fan_out: int) -> None:
    w = scaled(init.variance_scaling(1.0, mode, Distribution.NORMAL), fan_in, fan_out)

    np.testing.assert_allclose(w.var(), 1.0, rtol=0.05)


# The variance each name stands for, written from its definition and not from the
# module: LeCun 1 / fan_in, Glorot 2 / (fan_in + fan_out), He 2 / fan_in.
NAMED = [
    (init.lecun_uniform, 1 / FAN_IN, True),
    (init.lecun_normal, 1 / FAN_IN, False),
    (init.glorot_uniform, 2 / (FAN_IN + FAN_OUT), True),
    (init.glorot_normal, 2 / (FAN_IN + FAN_OUT), False),
    (init.he_uniform, 2 / FAN_IN, True),
    (init.he_normal, 2 / FAN_IN, False),
]


@pytest.mark.parametrize(
    ("factory", "variance", "is_uniform"), NAMED, ids=[n[0].__name__ for n in NAMED]
)
def test_a_named_initializer_has_the_variance_of_its_definition(
    factory: Callable[[], Initializer], variance: float, is_uniform: bool
) -> None:
    w = scaled(factory())

    np.testing.assert_allclose(w.var(), variance, rtol=0.05)
    assert (np.abs(w).max() <= np.sqrt(3 * variance)) == is_uniform


def test_variance_scaling_needs_a_positive_scale() -> None:
    with pytest.raises(ValueError, match="positive scale"):
        init.variance_scaling(0.0, FanMode.FAN_IN, Distribution.UNIFORM)


# --------------------------------------------------------------------------- #
# Declared and not built
# --------------------------------------------------------------------------- #
