"""The dense layer: all-to-all connections along the last axis.

A helper written with the two primitives; it uses nothing a user could not.
"""

from __future__ import annotations

import numpy as np

from plastax import init as px_init
from plastax.init import Initializer
from plastax.layers._module import Module
from plastax.layers._primitives import CreateConnections, CreateUnits
from plastax.layers._units import ShapeError, Units


class Dense(Module):
    """New units, each connected to every unit along the last axis of its input.

    Called with a handle of shape `lead + (n_in,)`, the layer creates
    `lead + (features,)` units. At each position along the leading axes, every one of
    the `n_in` sources is connected to every one of the `features` new units.
    Positions are not connected to each other, and every connection has its own
    weight.

    The layer is four primitives, which give its blocks their paths:

    - `units`: the new units.
    - `weight`: the connections, in row-major order over `lead + (features, n_in)`:
      position outermost, then destination, then source. A weight array of that
      shape, flattened, is in block order.
    - `one` and `bias_weight`, with `bias=True`: one unit that is always 1, and a
      connection from it to each new unit, in the row-major order of the returned
      handle.

    Args:
        features: The length of the last axis of the units the layer creates.
        bias: Whether the layer has a bias.
        init: How the weights get their first values (see `plastax.init`), or None
            for the default.
        bias_init: The same for the bias weights. None starts them at 0.

    Raises:
        TypeError: If `features` is not an integer.
        ValueError: If `features` is less than 1.
    """

    def __init__(
        self,
        features: int,
        *,
        bias: bool = True,
        init: Initializer | None = None,
        bias_init: Initializer | None = None,
    ) -> None:
        if isinstance(features, bool) or not isinstance(features, int | np.integer):
            raise TypeError(
                f"Dense takes a number of features as an integer, "
                f"got {type(features).__name__}"
            )
        if features < 1:
            raise ValueError(
                f"Dense takes a number of features of at least 1, got {features}"
            )
        self.features = int(features)
        self.bias = bias
        self.init = init
        self.bias_init = bias_init

    def setup(self) -> None:
        """Declare the primitives the layer is made of."""
        self.units = CreateUnits()
        self.weight = CreateConnections(init=self.init)
        if self.bias:
            self.one = CreateUnits(constant=1.0)
            bias_init = px_init.zeros() if self.bias_init is None else self.bias_init
            self.bias_weight = CreateConnections(init=bias_init)

    # `Module` declares `__call__` as an attribute so that each subclass may choose
    # its own arguments; ty reads a method with narrower arguments as an invalid
    # override of that attribute.
    def __call__(self, x: Units) -> Units:  # ty: ignore[incompatible-mutable-override]
        """Create the layer's units and connect `x` to them.

        Args:
            x: The source units. The layer acts on the last axis.

        Returns:
            A handle to the new units, of shape `x.shape[:-1] + (features,)`.

        Raises:
            TypeError: If `x` is not a handle.
            ShapeError: If `x` has no axes, or its last axis has length 0.
        """
        if not isinstance(x, Units):
            raise TypeError(
                f"a Dense layer is called with a handle, got {type(x).__name__}"
            )
        if x.ndim == 0:
            raise ShapeError(
                "a Dense layer acts on the last axis and this handle has none; "
                "`x.reshape(1)` gives a single unit an axis"
            )
        if x.shape[-1] == 0:
            raise ShapeError(
                f"a Dense layer needs at least one source along the last axis, "
                f"got a handle of shape {x.shape}"
            )
        # Before anything is created: is this layer part of the trace, and is this
        # its first call in it?
        x._record._claim(self)

        y = self.units(shape=(*x.shape[:-1], self.features))
        # Broadcast to `lead + (features, n_in)`: the source varies along the last
        # axis and the destination along the one before it.
        self.weight(x[..., None, :], y[..., :, None])
        if self.bias:
            self.bias_weight(self.one(shape=()), y)
        return y
