"""RMSProp as a plastax optimizer bundle.

One per-connection state column -- a running mean square of the gradient
(``opt/v``) -- carried as an SoA column, so it shards with the connections under
Scheme-A. No first moment and no bias correction (the classic, uncentered
RMSProp), matching ``optax.rmsprop(learning_rate, decay, eps)`` (eps *inside*
the sqrt):

    v <- decay*v + (1-decay)*g**2
    w <- w - lr * g / sqrt(v + eps)
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np

from plastax._types import ACTIVATION, WEIGHT, ConnIdx, FieldSpec, UnitIdx
from plastax.views import ConnView, ConnWrite, UnitView

# Per-connection running mean square of the gradient; default 0.0 (optax's
# initial_scale = 0), so a fresh or regrown edge starts unscaled.
_MEAN_SQ: FieldSpec[np.float32] = FieldSpec.float32("opt/v")


@dataclasses.dataclass(frozen=True)
class _RMSPropUpdateConn:
    """UpdateConn applying one RMSProp step per connection.

    Reads and rewrites the mean-square column alongside the weight, both from
    the destination's perspective, so ``outgoing`` is a no-op. ``g`` is typed
    ``object`` so one instance satisfies ``UpdateConn[GS]`` for every ``GS``.

    Attributes:
        lr: the learning rate.
        decay: the mean-square decay (rho).
        eps: the denominator epsilon, added inside the sqrt.
        grad_field: the unit field carrying dL/dz at the destination unit.
        mean_sq: the per-connection running-mean-square column.
    """

    lr: float
    decay: float
    eps: float
    grad_field: FieldSpec[np.float32]
    mean_sq: FieldSpec[np.float32]

    def incoming(
        self,
        u: UnitView,
        dst: UnitIdx,
        src: UnitIdx,
        c: ConnView,
        cid: ConnIdx,
        g: object,
    ) -> ConnWrite:
        """Advance the mean square and step the weight by the scaled gradient.

        Args:
            u: the unit view.
            dst: index of the destination unit.
            src: index of the source unit.
            c: the connection view.
            cid: index of the connection.
            g: the global state (unused).

        Returns:
            The ConnWrite updating this connection's weight and mean square.
        """
        del g
        grad = u[self.grad_field, dst] * u[ACTIVATION, src]
        mean_sq = jnp.float32(self.decay) * c[self.mean_sq, cid] + jnp.float32(
            1.0 - self.decay
        ) * (grad * grad)
        update = jnp.float32(self.lr) * grad / jnp.sqrt(mean_sq + jnp.float32(self.eps))
        return ConnWrite.of((WEIGHT, c[WEIGHT, cid] - update), (self.mean_sq, mean_sq))

    def outgoing(
        self,
        u: UnitView,
        src: UnitIdx,
        dst: UnitIdx,
        c: ConnView,
        cid: ConnIdx,
        g: object,
    ) -> ConnWrite:
        """Return an empty write; RMSProp does all its work in the incoming pass.

        Args:
            u: the unit view.
            src: index of the source unit.
            dst: index of the destination unit.
            c: the connection view.
            cid: index of the connection.
            g: the global state (unused).

        Returns:
            An empty ConnWrite.
        """
        del u, src, dst, c, cid, g
        return ConnWrite.of()


@dataclasses.dataclass(frozen=True)
class RMSProp:
    """RMSProp optimizer bundle with a per-connection running mean square.

    Stateful: it declares one per-connection column (``opt/v``) in
    `state_fields`, which the owning Network merges into its
    `extra_conn_fields`. No step counter (`needs_step_counter` is False).

    Attributes:
        lr: the learning rate.
        grad_field: the unit field a Network's backward pass writes dL/dz into.
        decay: the mean-square decay (rho); defaults to 0.9.
        eps: the denominator epsilon, added inside the sqrt.
        state_fields: the mean-square column.
        needs_step_counter: False -- RMSProp keeps no step counter.
    """

    lr: float
    grad_field: FieldSpec[np.float32]
    decay: float = 0.9
    eps: float = 1e-8
    state_fields: tuple[FieldSpec[np.float32], ...] = (_MEAN_SQ,)
    needs_step_counter: bool = False

    def update_conn(self) -> _RMSPropUpdateConn:
        """Build the UpdateConn that applies this RMSProp rule.

        Returns:
            An UpdateConn advancing each connection's mean square and weight.
        """
        return _RMSPropUpdateConn(
            self.lr, self.decay, self.eps, self.grad_field, _MEAN_SQ
        )


def rmsprop(
    lr: float,
    grad_field: FieldSpec[np.float32],
    *,
    decay: float = 0.9,
    eps: float = 1e-8,
) -> RMSProp:
    """Build an RMSProp optimizer.

    Args:
        lr: the learning rate.
        grad_field: the unit field carrying dL/dz at the destination unit --
            the field a Network's backward pass writes (e.g. `mlp_xor`'s
            ``grad_pre_act``).
        decay: the mean-square decay (rho); defaults to 0.9.
        eps: denominator epsilon added inside the sqrt; defaults to 1e-8.

    Returns:
        An RMSProp optimizer bundle.
    """
    return RMSProp(lr=lr, grad_field=grad_field, decay=decay, eps=eps)
