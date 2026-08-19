"""Core shared types.

Typing discipline for the package.

Plastax defines a network as a Struct of Arrays (SoA),
each array containing some metadata fields of either
the units (sometimes called neurons, depending on the
algorithm) or the connections between them. Some
metadata is implemented by default for all connections
and units, but the user can also specify arbitrary
metadata so long as it is of a known size at compile
time.

Implementing additional metadata fields uses the
FieldSpec type, which is generic over any numpy scalar
or fixed size type, so view access returns correctly-
typed arrays.

Indices are distinct NewTypes so they cannot be
interchanged; array shapes are expressed with jaxtyping
annotations.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import NewType

import numpy as np
from jaxtyping import Array, Bool, Int32

# All units are in unit space, all connections in connection space
UnitIdx = NewType("UnitIdx", Int32[Array, ""])
ConnIdx = NewType("ConnIdx", Int32[Array, ""])
Level = NewType("Level", int)  # for topological sort layer deps


class Propagation(enum.Enum):
    """Sweep scheduling mode."""

    TOPOLOGICAL = "topological"
    PIPELINE = "pipeline"


@dataclasses.dataclass(frozen=True)
class ShardSpec:
    """Scheme-A sharding config: connections split over a device-mesh axis.

    Units and globals are replicated on every device; the connection arenas
    are sharded on their capacity axis, and each phase's per-shard partial
    accumulators are combined with a monoid collective. Held as a static,
    hashable field on NetworkStatic so a sharding change forces a recompile.

    Attributes:
        axis_name: Name of the mesh axis the connections are sharded over.
        num_shards: Number of shards (devices) along that axis.
    """

    axis_name: str
    num_shards: int


@dataclasses.dataclass(frozen=True)
class FieldSpec[DT: np.generic]:
    """One SOA column.

    Type Args:
        DT: the numpy scalar type of the column, flowing through
            UnitView/ConnView.__getitem__.

    Attributes:
        name: Column name (the SOA tag, used as the arena dict key).
        dtype: numpy dtype of the column's values.
        default: Fill value for freshly allocated or tombstoned rows.
    """

    name: str
    dtype: np.dtype[DT]
    default: DT

    @staticmethod
    def field[T: np.generic](name: str, dtype: type[T], default: T) -> FieldSpec[T]:
        """Build a column spec for an arbitrary numpy scalar dtype.

        Type Args:
            T: the numpy scalar type of the column.

        Args:
            name: Column name.
            dtype: numpy scalar type of the column.
            default: Fill value for freshly allocated rows.

        Returns:
            A FieldSpec typed to the given scalar dtype.
        """
        return FieldSpec(name, np.dtype(dtype), default)

    # Convenience constructors for the common scalar types; each is a thin
    # typed wrapper over `field` that also coerces a plain-Python default.

    @staticmethod
    def float32(name: str, default: float = 0.0) -> FieldSpec[np.float32]:
        """Build a float32 column spec."""
        return FieldSpec.field(name, np.float32, np.float32(default))

    @staticmethod
    def int32(name: str, default: int = 0) -> FieldSpec[np.int32]:
        """Build an int32 column spec."""
        return FieldSpec.field(name, np.int32, np.int32(default))

    @staticmethod
    def boolean(name: str, default: bool = False) -> FieldSpec[np.bool_]:
        """Build a boolean column spec."""
        return FieldSpec.field(name, np.bool_, np.bool_(default))


# Built-in conn columns
FROM_ID: FieldSpec[np.int32] = FieldSpec.int32("from_id")
TO_ID: FieldSpec[np.int32] = FieldSpec.int32("to_id")
DEAD: FieldSpec[np.bool_] = FieldSpec.boolean("dead", default=True)
WEIGHT: FieldSpec[np.float32] = FieldSpec.float32("weight")

# Built-in unit columns (accumulator columns are materialized per
# monoid leaf by the sweep builder, not declared here).
ACTIVATION: FieldSpec[np.float32] = FieldSpec.float32("activation")
LEVEL: FieldSpec[np.int32] = FieldSpec.int32("level")

DeadMask = Bool[Array, " capacity"]
