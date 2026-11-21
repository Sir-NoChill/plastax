"""Reusable optimizer bundles for plastax weight updates.

An optimizer in plastax is a *bundle*, not just a weight-update rule: an
UpdateConnPolicy policy that computes the step, the extra per-connection fields its
state needs (e.g. Adam's first/second moments), and -- when the rule needs the
global step count for bias correction -- a step-counter carried in the
network's globals. Because that optimizer state lives as per-connection SOA
columns, it shards with the connections under Scheme-A sharding: a distributed
optimizer for free, with no separate optimizer-state plumbing.

This package will ship validated implementations (SGD, momentum, Adam, ...),
using optax only as a correctness reference in tests, never as a runtime
dependency.

Status: sgd, momentum, adam, adamw, and rmsprop implemented. Stateful
optimizers keep their per-connection state (moments and, for Adam, a step-count
column) in their `conn_fields`; optimizer columns use the ``opt/`` prefix.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from plastax._types import FieldSpec
from plastax.optim._adam import Adam, adam, adamw
from plastax.optim._momentum import Momentum, momentum
from plastax.optim._rmsprop import RMSProp, rmsprop
from plastax.optim._sgd import SGD, sgd
from plastax.policies import Update

__all__ = [
    "SGD",
    "Adam",
    "Momentum",
    "Optimizer",
    "RMSProp",
    "adam",
    "adamw",
    "momentum",
    "rmsprop",
    "sgd",
]


@runtime_checkable
class Optimizer(Protocol):
    """An optimizer bundle: its update policies and the state they keep.

    A network names one as `optimizer`. The optimizer's fields are added to the
    network's columns and its update fills the network's update phase; nothing
    outside the optimizer needs to name its state.

    Attributes:
        conn_fields: The per-connection FieldSpecs (any numpy dtype) the
            optimizer reads and writes for its own state (empty for a stateless
            rule like SGD). Each FieldSpec's ``default`` is also the value a
            regrown connection's state initializes to: the add_conn phase resets
            a new edge's untouched fields to their default, so a stateful
            optimizer wired with a growth policy (ScoreAddConnPolicy or
            ProposeAddConnPolicy) zeroes its moments on
            regrow automatically (RigL behaviour). Growth policies write WEIGHT
            and their own fields only, never optimizer columns.
        unit_fields: The per-unit FieldSpecs the optimizer keeps for its own
            state (empty for every shipped optimizer). An optimizer with unit
            state supplies the `unit` half of its update to write it.
        needs_step_counter: Whether the rule needs a step counter in the
            network's globals. The shipped optimizers keep any step count as a
            per-connection column instead (Adam's ``opt/t``), so this is False
            for all of them; it is reserved for a future globals-based counter.
    """

    conn_fields: tuple[FieldSpec[np.generic], ...]
    unit_fields: tuple[FieldSpec[np.generic], ...]
    needs_step_counter: bool

    def update(self) -> Update[object]:
        """Build the update policies implementing one optimizer step.

        Returns:
            The update pair. The shipped optimizers set only its `conn` half.
        """
        ...
