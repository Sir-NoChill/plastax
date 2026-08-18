"""Typed read views and write records over the SOA arenas.

User policies interact with the underlying data via views, rather
than directly indexing a column in user code. Writes are returned
as the associated record so that policies remain functionally
pure and 'vmap'-able.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import TypeVar

import numpy as np
from jaxtyping import Array, Shaped

from plastax._types import ConnIdx, FieldSpec, UnitIdx
from plastax.state import Columns

DT = TypeVar("DT", bound=np.generic)


@dataclasses.dataclass(frozen=True)
class UnitView:
    """A typed read view over the unit SOA columns."""

    _cols: Columns

    def __getitem__(self, key: tuple[FieldSpec[DT], UnitIdx]) -> Shaped[Array, ""]:
        """Return the scalar value for a field at a unit index.

        Args:
            key: Field spec and unit index to look up.

        Returns:
            The scalar value at that field and index.
        """
        spec, idx = key
        return self._cols[spec.name][idx]


@dataclasses.dataclass(frozen=True)
class ConnView:
    """A typed read view over the connection SOA columns."""

    _cols: Columns

    def __getitem__(self, key: tuple[FieldSpec[DT], ConnIdx]) -> Shaped[Array, ""]:
        """Return the scalar value for a field at a connection index.

        Args:
            key: Field spec and connection index to look up.

        Returns:
            The scalar value at that field and index.
        """
        spec, idx = key
        return self._cols[spec.name][idx]


@dataclasses.dataclass(frozen=True)
class UnitWrite:
    """Per-unit field writes returned by apply/update policies.

    Attributes:
        fields: Mapping from field name to the scalar value to write.
    """

    fields: Mapping[str, Shaped[Array, ""]]

    @staticmethod
    def of(*pairs: tuple[FieldSpec[DT], Shaped[Array, ""]]) -> UnitWrite:
        """Build a UnitWrite from field-spec/value pairs.

        Args:
            *pairs: Field spec and value pairs to write.

        Returns:
            A UnitWrite mapping field names to values.
        """
        return UnitWrite({spec.name: value for spec, value in pairs})


@dataclasses.dataclass(frozen=True)
class ConnWrite:
    """Per-connection field writes returned by apply/update policies.

    Attributes:
        fields: Mapping from field name to the scalar value to write.
    """

    fields: Mapping[str, Shaped[Array, ""]]

    @staticmethod
    def of(*pairs: tuple[FieldSpec[DT], Shaped[Array, ""]]) -> ConnWrite:
        """Build a ConnWrite from field-spec/value pairs.

        Args:
            *pairs: Field spec and value pairs to write.

        Returns:
            A ConnWrite mapping field names to values.
        """
        return ConnWrite({spec.name: value for spec, value in pairs})
