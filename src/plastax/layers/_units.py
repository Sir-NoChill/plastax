"""The handle a topology is written with: a group of units and their arrangement.

A handle names units; it holds no activations, weights or connections. Everything it
does happens on the host, in numpy, while a topology is being traced.
"""

from __future__ import annotations

from collections.abc import Iterator
from types import EllipsisType
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from plastax.layers._record import Record

# One index along one axis, as in numpy. Integers drop the axis, slices keep it, None
# adds an axis of length 1, an Ellipsis stands for the axes not written, and an integer
# array or list picks units by position, which may name a unit more than once.
type Index = int | np.integer | slice | None | EllipsisType | np.ndarray | list[int]


class ShapeError(ValueError):
    """A handle operation was given a shape that does not fit.

    Raised by the operation itself, so the traceback ends at the line of the topology
    that wrote it.
    """


class Units:
    """A handle to a group of units, arranged in a shape.

    A handle is an integer array of unit ids, in the handle's shape, together with the
    record being traced. Two handles may name the same units in different arrangements.
    A handle is never modified: every operation returns a new one.

    Users do not construct handles. A topology's `__call__` receives one per input, and
    layers and handle operations return new ones.

    Args:
        ids: The unit ids, in the handle's shape.
        record: The record the units belong to.

    Raises:
        TypeError: If `ids` are not integers.
    """

    __slots__ = ("_ids", "_record")

    def __init__(self, ids: np.ndarray, record: Record) -> None:
        arr = np.asarray(ids)
        if not np.issubdtype(arr.dtype, np.integer):
            raise TypeError(f"unit ids must be integers, got dtype {arr.dtype}")
        arr = arr.astype(np.int32, copy=False)
        # An array the caller can still write to is copied, so nothing outside can
        # change this handle. Arrays derived from another handle are already
        # read-only and are shared, not copied.
        if arr.flags.writeable:
            arr = arr.copy()
            arr.flags.writeable = False
        self._ids = arr
        self._record = record

    @property
    def shape(self) -> tuple[int, ...]:
        """The arrangement of the units, as a tuple of axis lengths."""
        return self._ids.shape

    @property
    def ndim(self) -> int:
        """The number of axes."""
        return self._ids.ndim

    @property
    def size(self) -> int:
        """The number of units."""
        return self._ids.size

    @property
    def ids(self) -> np.ndarray:
        """The unit ids in the handle's shape, as a read-only int32 array.

        These are the record's ids, which materialization may renumber.
        """
        return self._ids

    def reshape(self, *shape: int | tuple[int, ...]) -> Units:
        """Rearrange the same units into another shape, in row-major order.

        Creates no units and no connections. Accepts `h.reshape(3, 2)` and
        `h.reshape((3, 2))`. One axis may be -1, and is then worked out from the others.

        Args:
            *shape: The new axis lengths, as separate integers or as one tuple.

        Returns:
            A handle to the same units in the new shape.

        Raises:
            ShapeError: If the new shape does not hold exactly `size` units.
        """
        axes: list[int] = []
        for n in shape:
            if isinstance(n, tuple):
                axes.extend(n)
            else:
                axes.append(n)
        new_shape = tuple(axes)
        try:
            ids = self._ids.reshape(new_shape)
        except ValueError:
            raise ShapeError(
                f"cannot reshape {self.size} units of shape {self.shape} "
                f"into shape {new_shape}"
            ) from None
        return Units(ids, self._record)

    def flatten(self) -> Units:
        """Rearrange the same units into one axis, in row-major order.

        Returns:
            A handle of shape `(size,)` to the same units.
        """
        return Units(self._ids.reshape(-1), self._record)

    def __getitem__(self, index: Index | tuple[Index, ...]) -> Units:
        """Select or rearrange units as numpy indexes an array.

        Creates no units and no connections. `h[:, None, :]` adds an axis, `h[0]` and
        `h[2:5]` select a part, `h[..., 0]` selects along the last axis, and
        `h[np.array([0, 0, 3])]` picks units by position, which may name a unit more
        than once. Explicit wiring is written with the last form.

        Args:
            index: One index, or a tuple of them.

        Returns:
            A handle to the selected units.

        Raises:
            ShapeError: If an integer is out of range or there are more indices than
                axes.
            TypeError: If an index is a boolean or a boolean mask, or anything else
                that is not an integer, a slice, None, an Ellipsis or integers in an
                array or list.
        """
        parts = index if isinstance(index, tuple) else (index,)
        for part in parts:
            if part is None or isinstance(part, slice | EllipsisType):
                continue
            if isinstance(part, np.ndarray | list):
                allowed = np.issubdtype(np.asarray(part).dtype, np.integer)
            elif isinstance(part, bool | np.bool_):
                allowed = False
            else:
                # The annotation is not enforced for an index from user code, so the
                # check stays although mypy can prove it from the annotation.
                allowed = isinstance(part, int | np.integer)
            if not allowed:
                raise TypeError(
                    f"a handle is indexed with integers, slices, None, ... and "
                    f"integer arrays, not {type(part).__name__}"
                )
        try:
            ids = self._ids[index]
        except IndexError as error:
            raise ShapeError(
                f"index {index!r} does not fit a handle of shape {self.shape}: {error}"
            ) from None
        return Units(ids, self._record)

    def __iter__(self) -> Iterator[Units]:
        """Iterate over the first axis, as an array does.

        Yields:
            Units: `self[0]`, `self[1]`, and so on.

        Raises:
            TypeError: If the handle has no axes.
        """
        if self.ndim == 0:
            raise TypeError("a handle with no axes cannot be iterated")
        for i in range(self.shape[0]):
            yield self[i]

    def __array__(self, *args: object, **kwargs: object) -> np.ndarray:
        """Refuse to become an array.

        A handle carries no data. Without this, passing one to a numpy or JAX function
        would fail far from the cause, or silently iterate over it.

        Args:
            *args: Ignored.
            **kwargs: Ignored.

        Returns:
            Never returns.

        Raises:
            TypeError: Always, saying that a handle holds no values and naming `ids`.
        """
        raise TypeError(
            "a handle names units and holds no values, so it cannot be used as an "
            "array; `.ids` gives the unit ids"
        )

    def __repr__(self) -> str:
        """Show the shape only: `Units(shape=(28, 28, 1))`.

        Returns:
            The representation.
        """
        return f"Units(shape={self.shape})"
