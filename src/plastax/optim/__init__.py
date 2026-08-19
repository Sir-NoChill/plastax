"""Reusable optimizer bundles for plastax weight updates (STUB).

An optimizer in plastax is a *bundle*, not just a weight-update rule: an
UpdateConn policy that computes the step, the extra per-connection fields its
state needs (e.g. Adam's first/second moments), and -- when the rule needs the
global step count for bias correction -- a step-counter carried in the
network's globals. Because that optimizer state lives as per-connection SOA
columns, it shards with the connections under Scheme-A sharding: a distributed
optimizer for free, with no separate optimizer-state plumbing.

This package will ship validated implementations (SGD, momentum, Adam, ...),
using optax only as a correctness reference in tests, never as a runtime
dependency.

Status: STUB. It reserves the namespace and fixes the bundle interface below.
Concrete optimizers are being prototyped as traits in
examples/parallel_mnist (AdamUpdateConn) and will be promoted here once the
field-declaration and step-counter ergonomics are settled.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from plastax._types import FieldSpec
from plastax.traits import UpdateConn

__all__ = ["Optimizer"]


@runtime_checkable
class Optimizer(Protocol):
    """A weight-optimizer bundle: an UpdateConn policy plus its state needs.

    Attributes:
        state_fields: Extra per-connection FieldSpecs the optimizer reads and
            writes for its own state (empty for a stateless rule like SGD).
            The owning Network must include these in its extra_conn_fields.
        needs_step_counter: Whether the rule requires the network's globals to
            carry an int32 step counter (e.g. Adam's bias correction). This
            field is provisional -- the globals contract is still being
            settled against the example prototype.
    """

    state_fields: tuple[FieldSpec[np.generic], ...]
    needs_step_counter: bool

    def update_conn(self) -> UpdateConn[object]:
        """Build the UpdateConn trait implementing one optimizer step.

        Returns:
            The UpdateConn policy applying this optimizer's weight update.
        """
        ...
