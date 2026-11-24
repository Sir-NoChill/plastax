"""The dense layer: all-to-all connections along the last axis.

A helper written with the two primitives; it uses nothing a user could not.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from plastax import init as px_init
from plastax._types import Capacity
from plastax.init import Initializer
from plastax.layers._module import Module
from plastax.layers._policies import Given, split_pairs
from plastax.layers._primitives import (
    CreateConnections,
    CreateUnits,
    check_capacity,
)
from plastax.layers._units import ShapeError, Units
from plastax.policies import (
    INHERIT,
    Backward,
    BackwardConnPolicy,
    Forward,
    ForwardConnPolicy,
    Grow,
    Prune,
    Update,
)


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
        capacity: How many connection slots `weight` is given: a number, or
            `plastax.Capacity.INITIAL` (the default) for exactly as many as it
            has connections. The bias always has exactly as many as it starts
            with.
        forward: The layer's forward pass, a `plastax.policies.Forward`, or one
            object with `map`, `combine` and `apply`. Like each of the four below:
            a half the pair does not give is the network's, None says the layer
            has none, and not given takes the network's.
        backward: The layer's backward pass, a `plastax.policies.Backward`, or one
            object with all three.
        update: The layer's update, a `plastax.policies.Update`.
        prune: The layer's pruning, a `plastax.policies.Prune`.
        grow: The layer's growth, a `plastax.policies.Grow`.

    The layer creates connections and units, so it takes pairs, as a network does.
    The `conn` half of each goes to `weight` and the `unit` half to `units`. The
    bias connections take the `conn` halves of `forward`, `backward` and `update`,
    and are never pruned or regrown.

    A `BuildError` is raised if `capacity` is neither a `plastax.Capacity` nor a
    non-negative integer.

    Raises:
        TypeError: If `features` is not an integer, or a policy keyword is not a
            pair of its phase.
        ValueError: If `features` is less than 1.
    """

    def __init__(
        self,
        features: int,
        *,
        bias: bool = True,
        init: Initializer | None = None,
        bias_init: Initializer | None = None,
        capacity: int | Capacity = Capacity.INITIAL,
        forward: Given[Forward[Any] | ForwardConnPolicy[Any, Any]] = INHERIT,
        backward: Given[Backward[Any] | BackwardConnPolicy[Any, Any]] = INHERIT,
        update: Given[Update[Any]] = INHERIT,
        prune: Given[Prune[Any]] = INHERIT,
        grow: Given[Grow[Any]] = INHERIT,
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
        self.capacity = check_capacity(type(self).__name__, capacity)
        self.conn_policies, self.unit_policies = split_pairs(
            type(self).__name__,
            forward=forward,
            backward=backward,
            update=update,
            prune=prune,
            grow=grow,
        )

    def setup(self) -> None:
        """Declare the primitives the layer is made of."""
        conn, unit = self.conn_policies, self.unit_policies
        self.units = CreateUnits(
            forward=unit.forward,
            backward=unit.backward,
            update=unit.update,
            prune=unit.prune,
            grow=unit.grow,
        )
        self.weight = CreateConnections(
            init=self.init,
            capacity=self.capacity,
            forward=conn.forward,
            backward=conn.backward,
            update=conn.update,
            prune=conn.prune,
            grow=conn.grow,
        )
        if self.bias:
            # The unit that is always 1 takes the network's policies, as it did
            # before layers could be given any; `constant` is what keeps its value.
            self.one = CreateUnits(constant=1.0)
            bias_init = px_init.zeros() if self.bias_init is None else self.bias_init
            # A bias is learned as a weight is, and is never pruned or regrown.
            self.bias_weight = CreateConnections(
                init=bias_init,
                forward=conn.forward,
                backward=conn.backward,
                update=conn.update,
                prune=None,
                grow=None,
            )

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
