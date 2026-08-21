"""SGD with classical (heavy-ball) momentum as a plastax optimizer bundle.

The first stateful optimizer: it carries one per-connection velocity column
(``opt/v``), exercising the state_fields -> extra_conn_fields contract and the
multi-field ConnWrite (weight and velocity written together). The velocity
trace matches ``optax.sgd(momentum=...)`` (non-Nesterov):

    v <- momentum * v + dL/dw
    w <- w - lr * v

Adam reuses this shape with a second moment column and a bias-correction step
counter carried in the globals.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np

from plastax._types import ACTIVATION, WEIGHT, ConnIdx, FieldSpec, UnitIdx
from plastax.views import ConnView, ConnWrite, UnitView

# Per-connection optimizer state. The "opt/" prefix namespaces optimizer columns
# away from user fields (ECOSYSTEM_ROADMAP.md A1.1); default 0.0 so a freshly
# built or grown edge starts at rest.
_VELOCITY: FieldSpec[np.float32] = FieldSpec.float32("opt/v")


@dataclasses.dataclass(frozen=True)
class _MomentumUpdateConn:
    """UpdateConn applying one momentum-SGD step per connection.

    Reads and rewrites the velocity column alongside the weight, both from the
    destination's perspective, so ``outgoing`` is a no-op. ``g`` is typed
    ``object`` so one instance satisfies ``UpdateConn[GS]`` for every ``GS``.

    Attributes:
        lr: the learning rate.
        momentum: the velocity decay (0 recovers plain SGD).
        grad_field: the unit field carrying dL/dz at the destination unit.
        velocity: the per-connection velocity state column.
    """

    lr: float
    momentum: float
    grad_field: FieldSpec[np.float32]
    velocity: FieldSpec[np.float32]

    def incoming(
        self,
        u: UnitView,
        dst: UnitIdx,
        src: UnitIdx,
        c: ConnView,
        cid: ConnIdx,
        g: object,
    ) -> ConnWrite:
        """Advance the velocity by the gradient and step the weight along it.

        Args:
            u: the unit view.
            dst: index of the destination unit.
            src: index of the source unit.
            c: the connection view.
            cid: index of the connection.
            g: the global state (unused).

        Returns:
            The ConnWrite updating this connection's weight and velocity.
        """
        del g
        grad = u[self.grad_field, dst] * u[ACTIVATION, src]
        velocity = jnp.float32(self.momentum) * c[self.velocity, cid] + grad
        weight = c[WEIGHT, cid] - jnp.float32(self.lr) * velocity
        return ConnWrite.of((WEIGHT, weight), (self.velocity, velocity))

    def outgoing(
        self,
        u: UnitView,
        src: UnitIdx,
        dst: UnitIdx,
        c: ConnView,
        cid: ConnIdx,
        g: object,
    ) -> ConnWrite:
        """Return an empty write; momentum does all its work in the incoming pass.

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
class Momentum:
    """Heavy-ball momentum SGD bundle: ``v <- mu*v + dL/dw``, ``w <- w - lr*v``.

    Stateful: it declares one per-connection velocity column in `state_fields`,
    which the owning Network merges into its `extra_conn_fields`. No step
    counter is needed. The gradient is the delta rule ``dL/dw = grad[dst] *
    activation[src]``, read after the backward pass writes ``grad_field``.

    Attributes:
        lr: the learning rate.
        momentum: the velocity decay factor (e.g. 0.9).
        grad_field: the unit field a Network's backward pass writes dL/dz into.
        state_fields: the per-connection velocity column (``opt/v``).
        needs_step_counter: False -- momentum needs no globals step counter.
    """

    lr: float
    momentum: float
    grad_field: FieldSpec[np.float32]
    state_fields: tuple[FieldSpec[np.generic], ...] = (_VELOCITY,)
    needs_step_counter: bool = False

    def update_conn(self) -> _MomentumUpdateConn:
        """Build the UpdateConn that applies this momentum rule.

        Returns:
            An UpdateConn advancing each connection's velocity and weight.
        """
        return _MomentumUpdateConn(self.lr, self.momentum, self.grad_field, _VELOCITY)


def momentum(lr: float, momentum: float, grad_field: FieldSpec[np.float32]) -> Momentum:
    """Build a momentum-SGD optimizer.

    Args:
        lr: the learning rate.
        momentum: the velocity decay factor (e.g. 0.9); 0 recovers plain SGD.
        grad_field: the unit field carrying dL/dz at the destination unit --
            the field a Network's backward pass writes (e.g. `mlp_xor`'s
            ``grad_pre_act``).

    Returns:
        A Momentum optimizer bundle.
    """
    return Momentum(lr=lr, momentum=momentum, grad_field=grad_field)
