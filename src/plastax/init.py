"""Initializers: how a block of connections gets its first weights.

An initializer is any function `(key, fans) -> weights`. It is stored on the block it
fills while a topology is traced and called once, when the network is materialized,
with the block's own key and the block's `Fans`. Whoever builds a network never
supplies fans: they are counted from the wiring.

Everything shipped here is a function that returns an initializer, and is called
where it is written, with or without arguments:

```python
px.layers.Dense(300, init=px.init.normal(0.1), bias_init=px.init.zeros())
```

The initializers that scale with the fans (`variance_scaling` and those named after
LeCun, Glorot and He) read them per connection, so a unit with 4 inputs is drawn for
4 inputs, however many units its layer could have reached.

Plan: `plans/20261120T181444Z-initializers.md` at the workspace root.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PRNGKeyArray


@dataclass(frozen=True, eq=False)
class Fans:
    """What an initializer is told about the block of connections it fills.

    Both fans are counted over the whole network as it is first built, not within the
    block: a unit fed by two layers has the fan-in of both. The fan-in leaves out
    connections from constant units, so a bias does not count, as in PyTorch, JAX
    and Flax. The fan-out counts every connection.

    Attributes:
        shape: The block's shape. The weights returned have this shape.
        fan_in: For each connection of the block, the number of connections into
            its destination. Of shape `shape`.
        fan_out: For each connection of the block, the number of connections out of
            its source. Of shape `shape`.
    """

    shape: tuple[int, ...]
    fan_in: np.ndarray
    fan_out: np.ndarray


@runtime_checkable
class Initializer(Protocol):
    """A function that draws the first weights of one block of connections.

    `@runtime_checkable` is required because the test run type-checks plastax's
    signatures (jaxtyping and beartype), and a plain Protocol cannot be checked with
    `isinstance`. The check it allows only asks whether the object can be called.
    """

    def __call__(self, key: PRNGKeyArray, fans: Fans) -> ArrayLike:
        """Draw the weights of one block.

        Args:
            key: The block's key. The same key gives the same weights.
            fans: The block's shape and the fans of its connections.

        Returns:
            The weights, of shape `fans.shape`.
        """
        ...


class FanMode(enum.Enum):
    """Which fan a variance-scaling initializer divides by."""

    FAN_IN = "fan_in"
    FAN_OUT = "fan_out"
    FAN_AVG = "fan_avg"


class Distribution(enum.Enum):
    """What a variance-scaling initializer draws from."""

    UNIFORM = "uniform"
    NORMAL = "normal"
    TRUNCATED_NORMAL = "truncated_normal"


# The standard deviation of a standard normal distribution cut off at -2 and 2. Dividing
# by it gives a truncated draw the variance asked for. The value is JAX's.
_TRUNCATED_STDDEV = 0.87962566103423978


# --------------------------------------------------------------------------- #
# The initializers. Each is a small class, so that a record prints what fills a
# block, and is built by the function of the same name in lower case.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Constant:
    """Every weight of a block starts at the same value.

    Built by `constant`, `zeros` and `ones`.

    Attributes:
        value: The value.
    """

    value: float | int

    def __call__(self, key: PRNGKeyArray, fans: Fans) -> ArrayLike:
        """Fill the block with the value."""
        return np.full(fans.shape, self.value, np.float32)


@dataclass(frozen=True, eq=False, repr=False)
class FromArray:
    """A block starts at the weights written out in an array.

    Built by `array`. The values are copied when this is built and cannot be written
    afterwards, so what it returns is the same array every time.

    Attributes:
        values: The weights, as float32, in the block's shape.
    """

    values: np.ndarray

    def __post_init__(self) -> None:
        """Copy the values as float32 and make the copy read-only."""
        values = np.array(self.values, np.float32)
        values.flags.writeable = False
        object.__setattr__(self, "values", values)

    def __repr__(self) -> str:
        """Show the shape and not the values."""
        return f"FromArray(shape={self.values.shape})"

    def __call__(self, key: PRNGKeyArray, fans: Fans) -> ArrayLike:
        """Give the weights, or raise a `ValueError` for a block of another shape."""
        if self.values.shape != fans.shape:
            raise ValueError(
                f"px.init.array was given weights of shape {self.values.shape} "
                f"for a block of shape {fans.shape}"
            )
        return self.values


@dataclass(frozen=True)
class Normal:
    """Weights are drawn from a normal distribution with mean 0.

    Built by `normal`.

    Attributes:
        stddev: The standard deviation.
    """

    stddev: float | int

    def __call__(self, key: PRNGKeyArray, fans: Fans) -> ArrayLike:
        """Draw the block's weights."""
        return self.stddev * jax.random.normal(key, fans.shape, jnp.float32)


@dataclass(frozen=True)
class TruncatedNormal:
    """Weights are drawn from a normal distribution cut off at 2 `stddev`.

    Built by `truncated_normal`.

    Attributes:
        stddev: The standard deviation of the normal distribution that is cut off.
    """

    stddev: float | int

    def __call__(self, key: PRNGKeyArray, fans: Fans) -> ArrayLike:
        """Draw the block's weights."""
        return self.stddev * jax.random.truncated_normal(
            key, -2.0, 2.0, fans.shape, jnp.float32
        )


@dataclass(frozen=True)
class Uniform:
    """Weights are drawn uniformly from `[low, high)`.

    Built by `uniform`.

    Attributes:
        low: The lower bound.
        high: The upper bound.
    """

    low: float | int
    high: float | int

    def __post_init__(self) -> None:
        """Refuse bounds that are not in order."""
        if not self.low < self.high:
            raise ValueError(
                f"px.init.uniform draws from [low, high) and needs low < high, "
                f"got low={self.low} and high={self.high}"
            )

    def __call__(self, key: PRNGKeyArray, fans: Fans) -> ArrayLike:
        """Draw the block's weights."""
        return jax.random.uniform(
            key, fans.shape, jnp.float32, minval=self.low, maxval=self.high
        )


@dataclass(frozen=True)
class VarianceScaling:
    """Each weight is drawn with mean 0 and variance `scale / fan`.

    Built by `variance_scaling` and by the initializers named after LeCun, Glorot and
    He. The fan is that of the connection being drawn, so the weights of one block
    may have different variances. A fan of 0 is read as 1: a unit whose only inputs
    are constants, or one that feeds nothing, is drawn as a unit with one connection.
    JAX gives such weights a variance of 0 and Keras also reads the fan as 1; in both
    a fan of 0 only arises for an empty weight matrix.

    Attributes:
        scale: The numerator of the variance.
        mode: Which fan the variance is divided by. `FAN_AVG` is the mean of the two.
        distribution: What is drawn. `UNIFORM` draws within `sqrt(3 * variance)` of
            0. `TRUNCATED_NORMAL` is cut off at two standard deviations of the normal
            before the cut and rescaled, so that it has the variance asked for.
    """

    scale: float | int
    mode: FanMode
    distribution: Distribution

    def __post_init__(self) -> None:
        """Refuse a scale that is not positive."""
        if not self.scale > 0:
            raise ValueError(
                f"px.init.variance_scaling needs a positive scale, got {self.scale}"
            )

    def __call__(self, key: PRNGKeyArray, fans: Fans) -> ArrayLike:
        """Draw the block's weights."""
        fan_in = np.maximum(fans.fan_in, 1)
        fan_out = np.maximum(fans.fan_out, 1)
        if self.mode is FanMode.FAN_IN:
            fan = fan_in
        elif self.mode is FanMode.FAN_OUT:
            fan = fan_out
        else:
            fan = (fan_in + fan_out) / 2
        stddev = np.sqrt(self.scale / fan).astype(np.float32)

        # `stddev` is a numpy array, which mypy reads as Any here, and so the
        # products below; the annotation says what they are.
        drawn: Array
        if self.distribution is Distribution.UNIFORM:
            drawn = (
                np.sqrt(np.float32(3.0))
                * stddev
                * jax.random.uniform(
                    key, fans.shape, jnp.float32, minval=-1.0, maxval=1.0
                )
            )
        elif self.distribution is Distribution.NORMAL:
            drawn = stddev * jax.random.normal(key, fans.shape, jnp.float32)
        else:
            drawn = (
                stddev / np.float32(_TRUNCATED_STDDEV)
            ) * jax.random.truncated_normal(key, -2.0, 2.0, fans.shape, jnp.float32)
        return drawn


def constant(value: float | int) -> Initializer:
    """Build an initializer that gives every weight the same value.

    Args:
        value: The value.

    Returns:
        The initializer.
    """
    return Constant(value)


def zeros() -> Initializer:
    """Build an initializer that starts every weight at 0.

    Returns:
        The initializer.
    """
    return Constant(0.0)


def ones() -> Initializer:
    """Build an initializer that starts every weight at 1.

    Returns:
        The initializer.
    """
    return Constant(1.0)


def array(values: ArrayLike) -> Initializer:
    """Build an initializer that gives a block the weights written out in an array.

    The array has the block's shape: for a `Dense` layer `lead + (features, n_in)`,
    the new unit along the second-to-last axis and the source along the last. The
    values are copied when the initializer is built.

    Args:
        values: The weights.

    Returns:
        The initializer. Calling it raises a `ValueError` if the block does not have
        the shape of `values`.
    """
    return FromArray(np.asarray(values))


def normal(stddev: float | int = 0.01) -> Initializer:
    """Build an initializer that draws from a normal distribution with mean 0.

    Args:
        stddev: The standard deviation.

    Returns:
        The initializer.
    """
    return Normal(stddev)


def truncated_normal(stddev: float | int = 0.01) -> Initializer:
    """Build an initializer that draws from a normal distribution cut off at 2 `stddev`.

    As `jax.nn.initializers.truncated_normal`: `stddev` is the standard deviation
    before the cut, and every weight lies within `2 * stddev` of 0. The weights drawn
    have a standard deviation of about `0.88 * stddev`.

    Args:
        stddev: The standard deviation of the normal distribution that is cut off.

    Returns:
        The initializer.
    """
    return TruncatedNormal(stddev)


def uniform(low: float | int, high: float | int) -> Initializer:
    """Build an initializer that draws uniformly from `[low, high)`.

    Both bounds are given. `jax.nn.initializers.uniform(scale)` is `uniform(0, scale)`
    here. A `ValueError` is raised if `low` is not below `high`.

    Args:
        low: The lower bound.
        high: The upper bound.

    Returns:
        The initializer.
    """
    return Uniform(low, high)


def variance_scaling(
    scale: float | int, mode: FanMode, distribution: Distribution
) -> Initializer:
    """Build an initializer that draws each weight with variance `scale / fan`.

    The fan is counted per connection (see `Fans`). PyTorch's initializers with a
    gain are written with this: a gain `g` is a scale of `g**2`. The default of
    `torch.nn.Linear`, uniform within `1 / sqrt(fan_in)`, is a scale of `1 / 3` with
    `FanMode.FAN_IN` and `Distribution.UNIFORM`. A `ValueError` is raised if `scale` is
    not positive.

    Args:
        scale: The numerator of the variance.
        mode: Which fan the variance is divided by.
        distribution: What is drawn.

    Returns:
        The initializer.
    """
    return VarianceScaling(scale, mode, distribution)


def lecun_uniform() -> Initializer:
    """Build LeCun's initializer, uniform: variance `1 / fan_in`.

    Returns:
        The initializer.
    """
    return VarianceScaling(1.0, FanMode.FAN_IN, Distribution.UNIFORM)


def lecun_normal() -> Initializer:
    """Build LeCun's initializer, truncated normal: variance `1 / fan_in`.

    Returns:
        The initializer.
    """
    return VarianceScaling(1.0, FanMode.FAN_IN, Distribution.TRUNCATED_NORMAL)


def glorot_uniform() -> Initializer:
    """Build Glorot's initializer, uniform: variance `2 / (fan_in + fan_out)`.

    Returns:
        The initializer.
    """
    return VarianceScaling(1.0, FanMode.FAN_AVG, Distribution.UNIFORM)


def glorot_normal() -> Initializer:
    """Build Glorot's initializer, truncated normal: variance `2 / (fan_in + fan_out)`.

    Returns:
        The initializer.
    """
    return VarianceScaling(1.0, FanMode.FAN_AVG, Distribution.TRUNCATED_NORMAL)


def he_uniform() -> Initializer:
    """Build He's initializer, uniform: variance `2 / fan_in`.

    Returns:
        The initializer.
    """
    return VarianceScaling(2.0, FanMode.FAN_IN, Distribution.UNIFORM)


def he_normal() -> Initializer:
    """Build He's initializer, truncated normal: variance `2 / fan_in`.

    Returns:
        The initializer.
    """
    return VarianceScaling(2.0, FanMode.FAN_IN, Distribution.TRUNCATED_NORMAL)


def orthogonal(scale: float | int = 1.0) -> Initializer:
    """Build an initializer that draws an orthogonal matrix. Not built yet.

    Args:
        scale: The factor the matrix is multiplied by.

    Returns:
        The initializer, once this is built.

    Raises:
        NotImplementedError: Always.
    """
    raise NotImplementedError(
        "px.init.orthogonal is not built yet: it is defined for a weight matrix, and "
        "what it means for a block that is not all-to-all is not decided"
    )


def delta_orthogonal(scale: float | int = 1.0) -> Initializer:
    """Build a delta-orthogonal initializer, for convolutions. Not built yet.

    Args:
        scale: The factor the weights are multiplied by.

    Returns:
        The initializer, once this is built.

    Raises:
        NotImplementedError: Always.
    """
    raise NotImplementedError(
        "px.init.delta_orthogonal is not built yet: it is defined for a convolution "
        "kernel, and plastax has no convolution layer in this interface yet"
    )


def from_jax(initializer: Any) -> Initializer:
    """Wrap an initializer of `jax.nn.initializers`. Not built yet.

    A JAX initializer reads the fans from an array shape and assumes which axes are
    the inputs and the outputs. A block's shape does not follow that convention, so
    the layer has to say which axis is which, and that is not designed.

    Args:
        initializer: The JAX initializer, called as `initializer(key, shape)`.

    Returns:
        The initializer, once this is built.

    Raises:
        NotImplementedError: Always.
    """
    raise NotImplementedError(
        "px.init.from_jax is not built yet: a JAX initializer needs to be told which "
        "axes of a block are its inputs and outputs. The initializers of px.init "
        "cover the common ones"
    )
