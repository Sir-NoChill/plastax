"""Core shared types.

Typing discipline for the package: FieldSpec is generic over its numpy
scalar type so view access returns correctly-typed arrays; unit and conn
indices are distinct NewTypes so they cannot be interchanged; array shapes
are expressed with jaxtyping annotations.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import NewType

import numpy as np
from jaxtyping import Array, Bool

# Distinct index spaces; a UnitIdx must never index a conn column and vice
# versa. Traced values: 0-d/1-d int32 arrays wrapped at the view boundary.
# NewType's 2nd arg must be a real subclassable type (mypy valid-newtype);
# jax.Array/jaxtyping shapes resolve to Any here since pyproject pins
# follow_imports="skip" for jax.*/jaxtyping.*, so Any is rejected. `object`
# is a static-only placeholder -- NewType is an identity fn at runtime, so
# the wrapped value is still the traced int32 array described above.
UnitIdx = NewType("UnitIdx", object)
ConnIdx = NewType("ConnIdx", object)
Level = NewType("Level", int)


class Propagation(enum.Enum):
    TOPOLOGICAL = "topological"
    PIPELINE = "pipeline"


@dataclasses.dataclass(frozen=True)
class FieldSpec[DT: np.generic]:
    """One SOA column: analogue of plastix::alloc::SOAField<Tag, T>.

    The generic parameter is the numpy scalar type of the column; it flows
    through UnitView/ConnView.__getitem__ so reads are typed end to end.
    """

    name: str
    dtype: np.dtype[DT]
    default: DT

    @staticmethod
    def f32(name: str, default: float = 0.0) -> FieldSpec[np.float32]:
        return FieldSpec(name, np.dtype(np.float32), np.float32(default))

    @staticmethod
    def i32(name: str, default: int = 0) -> FieldSpec[np.int32]:
        return FieldSpec(name, np.dtype(np.int32), np.int32(default))

    @staticmethod
    def boolean(name: str, default: bool = False) -> FieldSpec[np.bool_]:
        return FieldSpec(name, np.dtype(np.bool_), np.bool_(default))


# Built-in conn columns, mirroring conn.hpp:13-37.
FROM_ID: FieldSpec[np.int32] = FieldSpec.i32("from_id")
TO_ID: FieldSpec[np.int32] = FieldSpec.i32("to_id")
DEAD: FieldSpec[np.bool_] = FieldSpec.boolean("dead", default=True)
WEIGHT: FieldSpec[np.float32] = FieldSpec.f32("weight")

# Built-in unit columns, mirroring unit_state.hpp:12-45 (accumulator columns
# are materialized per monoid leaf by the sweep builder, not declared here).
ACTIVATION: FieldSpec[np.float32] = FieldSpec.f32("activation")
LEVEL: FieldSpec[np.int32] = FieldSpec.i32("level")

DeadMask = Bool[Array, " capacity"]
