"""Typed read views and write records over the SOA arenas.

User policies interact with the underlying data via views, rather
than directly indexing a column in user code. Writes are returned
as the associated record so that policies remain functionally
pure and 'vmap'-able.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, TypeVar

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Int32, Shaped

from plastax._types import ConnIdx, FieldSpec, UnitIdx
from plastax.state import Columns

DT = TypeVar("DT", bound=np.generic)

type Written = Shaped[Array, ""] | bool | int | float
"""A value a policy writes to one field of one element.

A Python number takes the field's dtype, where it converts without changing kind;
`of` holds it as a weakly typed array.
"""


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

    def gather(
        self, spec: FieldSpec[DT], ids: Int32[Array, " n"]
    ) -> Shaped[Array, " n"]:
        """Return a field's values at several units, in the order of `ids`.

        Whole-output rules (`LossPolicy.calculate_loss`) read every output at once
        through this.

        Args:
            spec: Field spec to read.
            ids: Unit indices to read, any order.

        Returns:
            The field's values at `ids`.
        """
        return self._cols[spec.name][ids]


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


def _named(
    kind: str, pairs: tuple[tuple[FieldSpec[Any], Any], ...]
) -> tuple[dict[str, Any], dict[str, FieldSpec[Any]]]:
    """Key the pairs of a write by field name, refusing a name given twice.

    Args:
        kind: "UnitWrite" or "ConnWrite", for the message.
        pairs: The field spec and value pairs.

    Returns:
        The values by field name, and the field each name came from.

    Raises:
        ValueError: If two pairs name the same field.
    """
    fields: dict[str, Any] = {}
    specs: dict[str, FieldSpec[Any]] = {}
    for spec, value in pairs:
        if spec.name in fields:
            raise ValueError(
                f"{kind}.of was given the field {spec.name!r} twice; a write "
                f"holds one value for each field"
            )
        # A Python number becomes a weakly typed array: it keeps saying "I take
        # the dtype of what I am written to", and every value is an array.
        number = isinstance(value, bool | int | float)
        fields[spec.name] = jnp.asarray(value) if number else value
        specs[spec.name] = spec
    return fields, specs


@dataclasses.dataclass(frozen=True)
class UnitWrite:
    """Per-unit field writes returned by apply/update policies.

    Attributes:
        fields: Mapping from field name to the scalar value to write.
        specs: The field each name came from, when the write was built with
            `of`. The engine does not read it; the checks that run when a
            network is built do.
    """

    fields: Mapping[str, Shaped[Array, ""]]
    specs: Mapping[str, FieldSpec[Any]] = dataclasses.field(default_factory=dict)

    @staticmethod
    def of(*pairs: tuple[FieldSpec[DT], Written]) -> UnitWrite:
        """Build a UnitWrite from field-spec/value pairs.

        Args:
            *pairs: Field spec and value pairs to write.

        Returns:
            A UnitWrite mapping field names to values.
        """
        return UnitWrite(*_named("UnitWrite", pairs))


@dataclasses.dataclass(frozen=True)
class ConnWrite:
    """Per-connection field writes returned by apply/update policies.

    Attributes:
        fields: Mapping from field name to the scalar value to write.
        specs: The field each name came from, when the write was built with
            `of`. The engine does not read it; the checks that run when a
            network is built do.
    """

    fields: Mapping[str, Shaped[Array, ""]]
    specs: Mapping[str, FieldSpec[Any]] = dataclasses.field(default_factory=dict)

    @staticmethod
    def of(*pairs: tuple[FieldSpec[DT], Written]) -> ConnWrite:
        """Build a ConnWrite from field-spec/value pairs.

        Args:
            *pairs: Field spec and value pairs to write.

        Returns:
            A ConnWrite mapping field names to values.
        """
        return ConnWrite(*_named("ConnWrite", pairs))
