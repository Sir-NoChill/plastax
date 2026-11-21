"""The loss trait and the shipped losses."""

from __future__ import annotations

import dataclasses
from typing import Protocol, runtime_checkable

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int32

from plastax._types import (
    ACTIVATION,
    FieldSpec,
)
from plastax.views import UnitView


@runtime_checkable
class LossPolicy[GS](Protocol):
    """Whole-output loss policy: one call sees every output unit.

    `calculate_loss` reads the output units (any field, through the unit view)
    and the targets, and returns the scalar loss together with the gradient
    seed dL/d(output) of every output. The framework writes the seed into the
    unit column the policy declares as `seed_field`, at the output ids only;
    the loss writes nothing else. A whole-output signature is what makes losses
    that couple the outputs expressible, e.g. softmax cross-entropy
    (`SoftmaxCrossEntropyLossPolicy`).

    The backward accumulator is not writable by any policy: it reaches a rule
    only as `BackwardPass.apply`'s `acc` argument, read-only. The backward pass
    picks the seed up from `seed_field` (the output level's own `acc` is the
    identity, since no edge sources from the deepest level).

    Type Args:
        GS: the global state type threaded through the network.

    Attributes:
        seed_field: the float unit column the gradient seed is written to; one
            of the network's unit columns (normally an `extra_unit_fields`
            entry).
    """

    seed_field: FieldSpec[np.float32]

    def calculate_loss(
        self,
        u: UnitView,
        outputs: Int32[Array, " num_outputs"],
        targets: Float[Array, " num_outputs"],
        g: GS,
    ) -> tuple[Float[Array, ""], Float[Array, " num_outputs"]]:
        """Compute the scalar loss and the gradient seed over every output.

        Args:
            u: the unit view.
            outputs: the output unit ids, in the builder's output order;
                `targets[k]` is the target of unit `outputs[k]`.
            targets: the target value of every output.
            g: the global state.

        Returns:
            A (loss, seed) pair: the scalar loss and the `(num_outputs,)`
            gradient seed, aligned with `outputs`.
        """
        ...


@dataclasses.dataclass(frozen=True)
class SoftmaxCrossEntropyLossPolicy:
    """Softmax over the output activations, cross-entropy against the targets.

    The targets are a distribution over the outputs (one-hot or soft). With
    logits ``a``, ``m = max(a)``, ``z = a - m`` and ``s = sum(exp(z))``:

    - loss ``L = sum(t * (log(s) - z))``, the max-subtracted log-sum-exp form
      of ``-sum(t * log(softmax(a)))``, finite for any finite logits;
    - seed ``dL/da = exp(z) / s - t``, written to `seed_field`.

    Matches plastax-cpp's `SoftmaxCrossEntropyLossPolicy` operation for operation in
    float32 (max, then the shifted exponentials and their sum, then the
    per-output quotient and the target-weighted log-sum-exp).

    Attributes:
        seed_field: the unit column the gradient seed is written to.
    """

    seed_field: FieldSpec[np.float32]

    def calculate_loss(
        self,
        u: UnitView,
        outputs: Int32[Array, " num_outputs"],
        targets: Float[Array, " num_outputs"],
        g: object,
    ) -> tuple[Float[Array, ""], Float[Array, " num_outputs"]]:
        """Compute the cross-entropy and the softmax-minus-target seed.

        Args:
            u: the unit view.
            outputs: the output unit ids.
            targets: the target distribution over the outputs.
            g: the global state (unused).

        Returns:
            The (loss, seed) pair.
        """
        del g
        logits = u.gather(ACTIVATION, outputs)
        z = logits - jnp.max(logits)
        shifted = jnp.exp(z)
        total = jnp.sum(shifted)
        seed = shifted / total - targets
        loss = jnp.sum(targets * (jnp.log(total) - z))
        return loss, seed
