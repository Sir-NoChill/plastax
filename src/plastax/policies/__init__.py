"""The policy traits, the shipped policies, and the policy sets.

A policy is written for one element: `_conn` holds those written for a
connection, `_unit` those written for a unit. A network declares a phase as a
pair of the two (`_pairs`). The rest are network-wide: the loss, the globals,
and the reduction of a batched step.
"""

from __future__ import annotations

from plastax.policies._batch import (
    BatchReductionPolicy,
    FieldReductions,
    MeanFloatFirstRest,
    Reduction,
    batch_written_reduction,
)
from plastax.policies._conn import (
    BackwardConnPolicy,
    ForwardConnPolicy,
    Proposal,
    ProposeAddConnPolicy,
    PruneConnPolicy,
    ScoreAddConnPolicy,
    UpdateConnPolicy,
    predicate_add_conn,
)
from plastax.policies._globals import ResetGlobalPolicy
from plastax.policies._loss import LossPolicy, SoftmaxCrossEntropyLossPolicy
from plastax.policies._pairs import (
    INHERIT,
    Backward,
    Forward,
    Grow,
    Inherit,
    Prune,
    Update,
)
from plastax.policies._unit import (
    AddUnitPolicy,
    BackwardUnitPolicy,
    ForwardUnitPolicy,
    PruneUnitPolicy,
    UpdateUnitPolicy,
)

__all__ = [
    "Backward",
    "Forward",
    "Grow",
    "INHERIT",
    "Inherit",
    "Prune",
    "Update",
    "AddUnitPolicy",
    "BackwardConnPolicy",
    "BackwardUnitPolicy",
    "BatchReductionPolicy",
    "FieldReductions",
    "ForwardConnPolicy",
    "ForwardUnitPolicy",
    "LossPolicy",
    "MeanFloatFirstRest",
    "Proposal",
    "ProposeAddConnPolicy",
    "PruneConnPolicy",
    "PruneUnitPolicy",
    "Reduction",
    "ResetGlobalPolicy",
    "ScoreAddConnPolicy",
    "SoftmaxCrossEntropyLossPolicy",
    "UpdateConnPolicy",
    "UpdateUnitPolicy",
    "batch_written_reduction",
    "predicate_add_conn",
]
