"""How a batched step reduces per-sample unit columns."""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import numpy as np

from plastax._types import (
    LEVEL,
    PRUNED,
    FieldSpec,
)


class Reduction(enum.Enum):
    """How a batched step combines one unit column's per-sample values.

    Attributes:
        MEAN: the batch mean (sum over the samples, then divided by B); a
            floating column only.
        SUM: the batch sum; a numeric, non-boolean column only.
        FIRST: sample 0's value.
        NOT_BATCHED: the column is shared by every sample and no per-sample
            phase may write it; the step keeps its pre-step value.
    """

    MEAN = "mean"
    SUM = "sum"
    FIRST = "first"
    NOT_BATCHED = "not_batched"


@runtime_checkable
class BatchReductionPolicy(Protocol):
    """Batch-reduction policy: how a batched step combines per-sample unit state.

    In a batched step (`make_step(..., batch_size=B)`) the input scatter,
    forward, loss, backward and unit update run once per sample, so every unit
    column they write holds B values. Before the once-per-batch phases
    (connection update, prune, add, growth, reset) those B values become one,
    by the `Reduction` this policy returns for the column.

    Every column a per-sample phase writes must be declared with MEAN, SUM or
    FIRST: an undeclared written column, or one declared NOT_BATCHED, is an
    error (see `make_step`). A column no per-sample phase writes needs no
    declaration and keeps its pre-step value. `LEVEL` and `PRUNED` belong to
    the structural phases: a policy may declare them NOT_BATCHED or leave them
    undeclared. There is no default: a batched step requires the network to
    set `Network.batch_reduction`. `FieldReductions` declares columns one by
    one; `MeanFloatFirstRest` is the mean of every floating column and sample
    0 of every other one.
    """

    def reduction(self, field: FieldSpec[Any]) -> Reduction | None:
        """Return how `field` is reduced over the batch.

        Args:
            field: a unit column of the network.

        Returns:
            The column's reduction, or None when the policy does not declare
            it.
        """
        ...


@dataclasses.dataclass(frozen=True, init=False)
class FieldReductions:
    """A batch-reduction policy that declares unit columns one by one.

    ``FieldReductions({ACTIVATION: Reduction.MEAN, COUNT: Reduction.SUM})``
    declares two columns; every other column is undeclared. Each declared
    column must be a unit column of the network the policy is set on
    (checked when the `Network` subclass is defined).

    Args:
        fields: each declared unit column and its reduction.

    Attributes:
        fields: the declared columns and their reductions, as (field,
            reduction) pairs in declaration order.

    Raises:
        TypeError: if a key is not a FieldSpec or a value is not a Reduction.
        ValueError: if two declared fields share a name.
    """

    fields: tuple[tuple[FieldSpec[Any], Reduction], ...]

    def __init__(self, fields: Mapping[FieldSpec[Any], Reduction]) -> None:
        pairs = tuple(fields.items())
        names: set[str] = set()
        for spec, red in pairs:
            if not isinstance(spec, FieldSpec):
                raise TypeError(f"FieldReductions: {spec!r} is not a FieldSpec")
            if not isinstance(red, Reduction):
                raise TypeError(
                    f"FieldReductions: {spec.name!r} maps to {red!r}, not a Reduction"
                )
            if spec.name in names:
                raise ValueError(f"FieldReductions: {spec.name!r} declared twice")
            names.add(spec.name)
        object.__setattr__(self, "fields", pairs)

    def reduction(self, field: FieldSpec[Any]) -> Reduction | None:
        """Return the declared reduction of `field`, or None when undeclared.

        Args:
            field: a unit column of the network.

        Returns:
            The declared reduction, or None.
        """
        for spec, red in self.fields:
            if spec.name == field.name:
                return red
        return None


@dataclasses.dataclass(frozen=True)
class MeanFloatFirstRest:
    """The batch mean of every floating unit column, sample 0 of every other.

    A convenience `BatchReductionPolicy` that declares every column: MEAN for a
    floating one, FIRST for any other, and NOT_BATCHED for the structural
    `LEVEL` and `PRUNED` columns. This is the reduction batched steps applied
    implicitly before the policy became explicit; setting it is the one-line
    way to keep that behaviour.
    """

    def reduction(self, field: FieldSpec[Any]) -> Reduction:
        """Return MEAN for a floating column and FIRST for any other.

        Args:
            field: a unit column of the network.

        Returns:
            The column's reduction (NOT_BATCHED for `LEVEL` and `PRUNED`).
        """
        if field.name in _STRUCTURAL_UNIT_FIELDS:
            return Reduction.NOT_BATCHED
        if np.issubdtype(field.dtype, np.floating):
            return Reduction.MEAN
        return Reduction.FIRST


# Unit columns only the structural phases write: never reduced over a batch.
_STRUCTURAL_UNIT_FIELDS = frozenset({LEVEL.name, PRUNED.name})


def batch_written_reduction(
    policy: BatchReductionPolicy, field: FieldSpec[Any], *, where: str
) -> Reduction:
    """Return the reduction of a column a per-sample phase writes, or raise.

    Args:
        policy: the network's batch-reduction policy.
        field: the written unit column.
        where: what writes it, for the error message.

    Returns:
        The column's reduction: MEAN, SUM or FIRST.

    Raises:
        ValueError: if the policy leaves the column undeclared or declares it
            NOT_BATCHED.
    """
    red = policy.reduction(field)
    if red is None:
        raise ValueError(
            f"batch_reduction: unit column {field.name!r} is written per sample "
            f"({where}) but the batch-reduction policy does not declare it; "
            "declare it MEAN, SUM or FIRST (or set "
            "batch_reduction = MeanFloatFirstRest())"
        )
    if red is Reduction.NOT_BATCHED:
        raise ValueError(
            f"batch_reduction: unit column {field.name!r} is declared "
            f"NOT_BATCHED but is written per sample ({where})"
        )
    return red
