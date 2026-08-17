"""Typed read views and write records over the SOA arenas.

Policies never touch raw columns: reads go through views (GetField<Tag>
analogue), writes are returned as records so policy code stays pure and
vectorizable (rung0 design section 2).
"""
from __future__ import annotations

import dataclasses
from typing import Generic, Mapping, TypeVar

import numpy as np
from jaxtyping import Array, Shaped

from plastax._types import ConnIdx, FieldSpec, UnitIdx
from plastax.state import Columns

DT = TypeVar("DT", bound=np.generic)


@dataclasses.dataclass(frozen=True)
class UnitView:
    _cols: Columns

    def __getitem__(
        self, key: tuple[FieldSpec[DT], UnitIdx]
    ) -> Shaped[Array, ""]:
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class ConnView:
    _cols: Columns

    def __getitem__(
        self, key: tuple[FieldSpec[DT], ConnIdx]
    ) -> Shaped[Array, ""]:
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class UnitWrite:
    """Per-unit field writes returned by apply/update policies."""

    fields: Mapping[str, Shaped[Array, ""]]

    @staticmethod
    def of(*pairs: tuple[FieldSpec[DT], Shaped[Array, ""]]) -> "UnitWrite":
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class ConnWrite:
    fields: Mapping[str, Shaped[Array, ""]]

    @staticmethod
    def of(*pairs: tuple[FieldSpec[DT], Shaped[Array, ""]]) -> "ConnWrite":
        raise NotImplementedError
