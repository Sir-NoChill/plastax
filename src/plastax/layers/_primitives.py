"""The two primitives a topology is written with: create units, create connections.

Each call creates a block: a run of consecutive units, or a run of connections. A
block is named by the path of the primitive that made it, and that is what policy
sets, capacities, hints and initializers attach to. Everything else in
`plastax.layers` can be written with these two. Nothing here imports JAX or any other
part of plastax.
"""

from __future__ import annotations

import numpy as np

from plastax.layers._init import ConstInit
from plastax.layers._module import BuildError, Module
from plastax.layers._record import ConnBlock
from plastax.layers._units import ShapeError, Units


class CreateUnits(Module):
    """Create a block of units. Creates no connections.

    Called with a handle, it creates as many units as the handle has, in the
    handle's shape; the handle is used for its shape and nothing is connected to
    it. Called with `shape=`, it creates a group that stands on its own. Exactly
    one of the two is given.

    ```python
    s = self.sums(a)               # as many units as `a`, in its shape
    one = self.one(shape=())       # a single unit
    ```

    Args:
        constant: A value the units always hold, or None for ordinary units. A bias
            reads a unit made with `constant=1.0`.

    Raises:
        TypeError: If `constant` is neither a number nor None.
    """

    def __init__(self, *, constant: float | int | None = None) -> None:
        if constant is not None and (
            isinstance(constant, bool)
            or not isinstance(constant, int | float | np.integer | np.floating)
        ):
            raise TypeError(
                f"a constant is a number or None, got {type(constant).__name__}"
            )
        self.constant = None if constant is None else float(constant)

    # `Module` declares `__call__` as an attribute so that each subclass may choose
    # its own arguments; ty reads a method with narrower arguments as an invalid
    # override of that attribute.
    def __call__(  # ty: ignore[incompatible-mutable-override]
        self, like: Units | None = None, *, shape: tuple[int, ...] | None = None
    ) -> Units:
        """Create the units.

        Args:
            like: A handle whose shape the new units take.
            shape: The shape of the new units, when no handle gives it.

        Returns:
            A handle to the new units.

        Raises:
            TypeError: If both or neither of `like` and `shape` are given, or
                `like` is not a handle.
            BuildError: If no trace is running that this layer is part of, or
                `like` comes from another trace.
        """
        if (like is None) == (shape is None):
            raise TypeError(
                "CreateUnits is called with a handle or with `shape=`, and not both"
            )
        if like is not None and not isinstance(like, Units):
            raise TypeError(
                f"CreateUnits is called with a handle, got {type(like).__name__}"
            )
        record = self._record
        if record is None:
            raise BuildError(
                "this CreateUnits is not part of a topology being traced: either no "
                "trace is running, or it was not assigned to an attribute in `setup`"
            )
        if like is not None and like._record is not record:
            raise BuildError("CreateUnits was given a handle from another trace")
        path = record._claim(self)
        new_shape = like.shape if like is not None else tuple(shape or ())
        return record._add_unit_block(path, new_shape, self.constant)


class CreateConnections(Module):
    """Create a block of connections between units that exist. Creates no units.

    The two handles are broadcast against each other as arrays are, and each
    element of `src` is connected to the element of `dst` at the same place. The
    block is in row-major order over the broadcast shape. Arranging the handles is
    how the wiring is chosen:

    ```python
    self.skip(a, b)                                 # one to one, same shape
    self.full(a[None, :], b[:, None])               # every unit of a to every b
    self.weight(x[..., None, :], y[..., :, None])   # the same along the last axis
    self.bias(one, y)                               # one unit to every unit of y
    ```

    Several calls into the same units all feed those units, and two calls may
    connect the same pair of units, as two connections.

    Args:
        init: How the block's weights get their first values, or None for the
            default.
    """

    def __init__(self, *, init: ConstInit | None = None) -> None:
        self.init = init

    # See the note on `CreateUnits.__call__`.
    def __call__(self, src: Units, dst: Units) -> None:  # ty: ignore[incompatible-mutable-override]
        """Create the connections.

        Args:
            src: The source units.
            dst: The destination units.

        Raises:
            TypeError: If `src` or `dst` is not a handle.
            BuildError: If the two handles come from different traces.
            ShapeError: If the two handles do not broadcast against each other.
            ValueError: If the same source and destination are paired twice.
        """
        for end in (src, dst):
            if not isinstance(end, Units):
                raise TypeError(
                    f"CreateConnections is called with two handles, "
                    f"got {type(end).__name__}"
                )
        record = src._record
        if dst._record is not record:
            raise BuildError("the two handles come from different traces")
        try:
            shape = np.broadcast_shapes(src.shape, dst.shape)
        except ValueError:
            raise ShapeError(
                f"handles of shapes {src.shape} and {dst.shape} do not broadcast "
                f"against each other"
            ) from None
        sources = np.broadcast_to(src.ids, shape).reshape(-1)
        destinations = np.broadcast_to(dst.ids, shape).reshape(-1)

        pair = sources.astype(np.int64) * record.num_units + destinations
        _, first, count = np.unique(pair, return_index=True, return_counts=True)
        if (count > 1).any():
            k = int(first[count > 1].min())
            raise ValueError(
                f"unit {int(sources[k])} is connected to unit "
                f"{int(destinations[k])} more than once in one call"
            )

        path = record._claim(self)
        record._add_block(ConnBlock(path, sources, destinations, shape, self.init))
