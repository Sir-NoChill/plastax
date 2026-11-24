"""The two primitives a topology is written with: create units, create connections.

Each call creates a block: a run of consecutive units, or a run of connections. A
block is named by the path of the primitive that made it, and that is what
policies, capacities, hints and initializers attach to. Everything else in
`plastax.layers` can be written with these two.

A primitive creates one kind of element, so each of its five policy keywords
takes that kind's policy directly, not a pair.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from plastax._types import Capacity
from plastax.init import Initializer
from plastax.layers._module import BuildError, Module, _show
from plastax.layers._policies import Given, conn_policies, unit_policies
from plastax.layers._record import ConnBlock
from plastax.layers._units import ShapeError, Units
from plastax.policies import (
    INHERIT,
    AddUnitPolicy,
    BackwardConnPolicy,
    BackwardUnitPolicy,
    ForwardConnPolicy,
    ForwardUnitPolicy,
    ProposeAddConnPolicy,
    PruneConnPolicy,
    PruneUnitPolicy,
    ScoreAddConnPolicy,
    UpdateConnPolicy,
    UpdateUnitPolicy,
)


def check_capacity(layer: str, capacity: object) -> int | Capacity:
    """Check a `capacity=` keyword where it is written.

    Args:
        layer: The layer's class name, for the message.
        capacity: What the layer was given.

    Returns:
        The capacity: a number of connection slots, or a `plastax.Capacity`.

    Raises:
        BuildError: If it is neither a `plastax.Capacity` nor a non-negative
            integer.
    """
    if isinstance(capacity, Capacity):
        return capacity
    if isinstance(capacity, bool) or not isinstance(capacity, int | np.integer):
        raise BuildError(
            f"{layer}(capacity=...) takes a number of connection slots as an "
            f"integer, or px.Capacity.INITIAL for exactly as many as the layer "
            f"starts with; got {capacity!r}"
        )
    if capacity < 0:
        raise BuildError(
            f"{layer}(capacity=...) takes a number of connection slots, which "
            f"cannot be negative; got {int(capacity)}"
        )
    return int(capacity)


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
        forward: The policy that reduces what arrives at each unit and writes it
            (a `ForwardUnitPolicy`). Like each of the four below: None for no
            policy, and not given to take the network's.
        backward: The same for what is sent back to each unit (a
            `BackwardUnitPolicy`).
        update: The unit update (an `UpdateUnitPolicy`).
        prune: The unit pruning policy (a `PruneUnitPolicy`).
        grow: The unit addition policy (an `AddUnitPolicy`).

    Raises:
        TypeError: If `constant` is neither a number nor None, or a policy keyword
            is not a unit policy of its phase.
    """

    def __init__(
        self,
        *,
        constant: float | int | None = None,
        forward: Given[ForwardUnitPolicy[Any, Any]] = INHERIT,
        backward: Given[BackwardUnitPolicy[Any, Any]] = INHERIT,
        update: Given[UpdateUnitPolicy[Any]] = INHERIT,
        prune: Given[PruneUnitPolicy[Any]] = INHERIT,
        grow: Given[AddUnitPolicy[Any]] = INHERIT,
    ) -> None:
        if constant is not None and (
            isinstance(constant, bool)
            or not isinstance(constant, int | float | np.integer | np.floating)
        ):
            raise TypeError(
                f"a constant is a number or None, got {type(constant).__name__}"
            )
        self.constant = None if constant is None else float(constant)
        self.policies = unit_policies(
            type(self).__name__,
            forward=forward,
            backward=backward,
            update=update,
            prune=prune,
            grow=grow,
        )

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
        return record._add_unit_block(path, new_shape, self.constant, self.policies)


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
        init: How the block's weights get their first values: an initializer of
            `plastax.init`, or any function `(key, fans) -> weights`. None for the
            default.
        capacity: How many connection slots the block is given: a number, or
            `plastax.Capacity.INITIAL` (the default) for exactly as many as it
            has connections. A number above that leaves room for growth.
        forward: The policy that maps each connection to a value for its
            destination (a `ForwardConnPolicy`). Like each of the four below: None
            for no policy, and not given to take the network's.
        backward: The same for its source (a `BackwardConnPolicy`).
        update: The connection update (an `UpdateConnPolicy`).
        prune: The connection pruning policy (a `PruneConnPolicy`).
        grow: The connection growth policy (a `ScoreAddConnPolicy` or a
            `ProposeAddConnPolicy`).

    A `TypeError` is raised if a policy keyword is not a connection policy of its
    phase, and a `BuildError` if `capacity` is neither a `plastax.Capacity` nor a
    non-negative integer.
    """

    def __init__(
        self,
        *,
        init: Initializer | None = None,
        capacity: int | Capacity = Capacity.INITIAL,
        forward: Given[ForwardConnPolicy[Any, Any]] = INHERIT,
        backward: Given[BackwardConnPolicy[Any, Any]] = INHERIT,
        update: Given[UpdateConnPolicy[Any]] = INHERIT,
        prune: Given[PruneConnPolicy[Any]] = INHERIT,
        grow: Given[ScoreAddConnPolicy[Any] | ProposeAddConnPolicy[Any]] = INHERIT,
    ) -> None:
        self.init = init
        self.capacity = check_capacity(type(self).__name__, capacity)
        self.policies = conn_policies(
            type(self).__name__,
            forward=forward,
            backward=backward,
            update=update,
            prune=prune,
            grow=grow,
        )

    # See the note on `CreateUnits.__call__`.
    def __call__(self, src: Units, dst: Units) -> None:  # ty: ignore[incompatible-mutable-override]
        """Create the connections.

        Args:
            src: The source units.
            dst: The destination units.

        Raises:
            TypeError: If `src` or `dst` is not a handle.
            BuildError: If the two handles come from different traces, or the
                call creates more connections than the layer's capacity.
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
        if not isinstance(self.capacity, Capacity) and self.capacity < sources.size:
            raise BuildError(
                f"{_show(path)} creates {sources.size} connections and was given "
                f"capacity={self.capacity}. A capacity is the number of slots the "
                f"layer has, so it is at least the number it starts with; "
                f"px.Capacity.INITIAL gives exactly that many"
            )
        record._add_block(
            ConnBlock(
                path,
                sources,
                destinations,
                shape,
                self.init,
                self.policies,
                self.capacity,
            )
        )
