"""The dense layer: all-to-all connections along the last axis.

Nothing here imports JAX or any other part of plastax.
"""

from __future__ import annotations

import numpy as np

from plastax.layers._init import ConstInit
from plastax.layers._module import Module
from plastax.layers._record import ConnBlock
from plastax.layers._units import ShapeError, Units


class Dense(Module):
    """New units, each connected to every unit along the last axis of its input.

    Called with a handle of shape `lead + (n_in,)`, the layer creates
    `lead + (features,)` units. At each position along the leading axes, every one of
    the `n_in` sources is connected to every one of the `features` new units.
    Positions are not connected to each other, and every connection has its own
    weight.

    The connections are written as one block named `"weight"`, in row-major order
    over `lead + (features, n_in)`: position outermost, then destination, then
    source. A weight array of that shape, flattened, is in block order.

    With `bias=True` the call also creates one unit that is always 1, and a second
    block named `"bias"` connecting it to each new unit, in the row-major order of
    the returned handle.

    The order in which a call numbers its new units and its constant is not
    specified.

    Args:
        features: The length of the last axis of the units the layer creates.
        bias: Whether the layer has a bias.
        init: How the weights get their first values, or None for the layer's
            default.
        bias_init: The same for the bias weights.

    Raises:
        TypeError: If `features` is not an integer.
        ValueError: If `features` is less than 1.
    """

    def __init__(
        self,
        features: int,
        *,
        bias: bool = True,
        init: ConstInit | None = None,
        bias_init: ConstInit | None = None,
    ) -> None:
        if isinstance(features, bool) or not isinstance(features, (int, np.integer)):
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
        n_in = x.shape[-1]
        if n_in == 0:
            raise ShapeError(
                f"a Dense layer needs at least one source along the last axis, "
                f"got a handle of shape {x.shape}"
            )
        record = x._record
        path = record._claim(self)

        out = record._new_units((*x.shape[:-1], self.features))
        # One connection per element of `lead + (features, n_in)`. The source varies
        # along the last axis and the destination along the one before it.
        weights = (*x.shape[:-1], self.features, n_in)
        src = np.broadcast_to(x.ids[..., None, :], weights).reshape(-1)
        dst = np.broadcast_to(out.ids[..., :, None], weights).reshape(-1)
        record._add_block(ConnBlock(path, "weight", src, dst, self.init))

        if self.bias:
            one = record._new_constant()
            record._add_block(
                ConnBlock(
                    path,
                    "bias",
                    np.full(out.size, one, np.int32),
                    out.ids.reshape(-1),
                    self.bias_init,
                )
            )
        return out
