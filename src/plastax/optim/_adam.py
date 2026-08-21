"""Adam and AdamW as plastax optimizer bundles.

Three per-connection state columns -- first moment ``opt/m``, second moment
``opt/v``, and a step counter ``opt/t`` -- carried as SoA columns, so the whole
adaptive optimizer shards with the connections under Scheme-A. The step counter
is per-connection rather than a single global (the roadmap's globals mechanism
is unbuilt, and reset_global is episode-reset, not a per-step bump): on any
dense net every edge updates every step, so ``opt/t`` equals a global count and
the optax parity is exact; for sparse regrowth later, a per-edge counter also
restarts a regrown edge's schedule, which is the intended RigL behavior.

Matches ``optax.adam(learning_rate, b1, b2, eps)`` (eps outside the sqrt,
eps_root = 0):

    m <- b1*m + (1-b1)*g;  v <- b2*v + (1-b2)*g**2;  t <- t + 1
    m_hat = m / (1 - b1**t);  v_hat = v / (1 - b2**t)
    w <- w - lr * m_hat / (sqrt(v_hat) + eps)

`adamw` sets a non-zero decoupled `weight_decay`, matching ``optax.adamw``:

    w <- w - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * w)
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np

from plastax._types import ACTIVATION, WEIGHT, ConnIdx, FieldSpec, UnitIdx
from plastax.views import ConnView, ConnWrite, UnitView

# Per-connection optimizer state (the "opt/" prefix namespaces optimizer columns
# away from user fields); all default 0.0, so a fresh or regrown edge starts
# with zero moments and t = 0 (its first update sees t = 1).
_M: FieldSpec[np.float32] = FieldSpec.float32("opt/m")
_V: FieldSpec[np.float32] = FieldSpec.float32("opt/v")
_T: FieldSpec[np.float32] = FieldSpec.float32("opt/t")


@dataclasses.dataclass(frozen=True)
class _AdamUpdateConn:
    """UpdateConn applying one Adam(W) step per connection.

    Reads and rewrites the three state columns alongside the weight, all from
    the destination's perspective, so ``outgoing`` is a no-op. ``g`` is typed
    ``object`` so one instance satisfies ``UpdateConn[GS]`` for every ``GS``.

    Attributes:
        lr: the learning rate.
        b1: first-moment decay (beta1).
        b2: second-moment decay (beta2).
        eps: denominator epsilon, added outside the sqrt.
        weight_decay: decoupled weight decay (0 for plain Adam).
        grad_field: the unit field carrying dL/dz at the destination unit.
        m: the per-connection first-moment column.
        v: the per-connection second-moment column.
        t: the per-connection step-count column.
    """

    lr: float
    b1: float
    b2: float
    eps: float
    weight_decay: float
    grad_field: FieldSpec[np.float32]
    m: FieldSpec[np.float32]
    v: FieldSpec[np.float32]
    t: FieldSpec[np.float32]

    def incoming(
        self,
        u: UnitView,
        dst: UnitIdx,
        src: UnitIdx,
        c: ConnView,
        cid: ConnIdx,
        g: object,
    ) -> ConnWrite:
        """Advance the moments and step the weight by the bias-corrected update.

        Args:
            u: the unit view.
            dst: index of the destination unit.
            src: index of the source unit.
            c: the connection view.
            cid: index of the connection.
            g: the global state (unused).

        Returns:
            The ConnWrite updating this connection's weight and Adam state.
        """
        del g
        weight = c[WEIGHT, cid]
        grad = u[self.grad_field, dst] * u[ACTIVATION, src]
        step = c[self.t, cid] + jnp.float32(1.0)
        m = jnp.float32(self.b1) * c[self.m, cid] + jnp.float32(1.0 - self.b1) * grad
        v = jnp.float32(self.b2) * c[self.v, cid] + jnp.float32(1.0 - self.b2) * (
            grad * grad
        )
        m_hat = m / (jnp.float32(1.0) - jnp.float32(self.b1) ** step)
        v_hat = v / (jnp.float32(1.0) - jnp.float32(self.b2) ** step)
        step_dir = m_hat / (jnp.sqrt(v_hat) + jnp.float32(self.eps))
        update = jnp.float32(self.lr) * (
            step_dir + jnp.float32(self.weight_decay) * weight
        )
        return ConnWrite.of(
            (WEIGHT, weight - update), (self.m, m), (self.v, v), (self.t, step)
        )

    def outgoing(
        self,
        u: UnitView,
        src: UnitIdx,
        dst: UnitIdx,
        c: ConnView,
        cid: ConnIdx,
        g: object,
    ) -> ConnWrite:
        """Return an empty write; Adam does all its work in the incoming pass.

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
class Adam:
    """Adam(W) optimizer bundle with per-connection moments and step counter.

    Stateful: it declares three per-connection columns (``opt/m``, ``opt/v``,
    ``opt/t``) in `state_fields`, which the owning Network merges into its
    `extra_conn_fields`. No globals step counter is used (`needs_step_counter`
    is False); the step count is the per-connection ``opt/t`` column. A non-zero
    `weight_decay` gives AdamW (decoupled decay); use the `adamw` factory.

    Attributes:
        lr: the learning rate.
        grad_field: the unit field a Network's backward pass writes dL/dz into.
        b1: first-moment decay (beta1).
        b2: second-moment decay (beta2).
        eps: denominator epsilon, added outside the sqrt.
        weight_decay: decoupled weight decay (0 for plain Adam).
        state_fields: the moment and step-count columns.
        needs_step_counter: False -- the step count lives in ``opt/t``.
    """

    lr: float
    grad_field: FieldSpec[np.float32]
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    state_fields: tuple[FieldSpec[np.float32], ...] = (_M, _V, _T)
    needs_step_counter: bool = False

    def update_conn(self) -> _AdamUpdateConn:
        """Build the UpdateConn that applies this Adam(W) rule.

        Returns:
            An UpdateConn advancing each connection's moments, step, and weight.
        """
        return _AdamUpdateConn(
            self.lr,
            self.b1,
            self.b2,
            self.eps,
            self.weight_decay,
            self.grad_field,
            _M,
            _V,
            _T,
        )


def adam(
    lr: float,
    grad_field: FieldSpec[np.float32],
    *,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
) -> Adam:
    """Build an Adam optimizer.

    Args:
        lr: the learning rate.
        grad_field: the unit field carrying dL/dz at the destination unit --
            the field a Network's backward pass writes (e.g. `mlp_xor`'s
            ``grad_pre_act``).
        b1: first-moment decay (beta1); defaults to 0.9.
        b2: second-moment decay (beta2); defaults to 0.999.
        eps: denominator epsilon added outside the sqrt; defaults to 1e-8.

    Returns:
        An Adam optimizer bundle.
    """
    return Adam(lr=lr, grad_field=grad_field, b1=b1, b2=b2, eps=eps)


def adamw(
    lr: float,
    grad_field: FieldSpec[np.float32],
    *,
    weight_decay: float = 1e-4,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
) -> Adam:
    """Build an AdamW optimizer (Adam with decoupled weight decay).

    Args:
        lr: the learning rate.
        grad_field: the unit field carrying dL/dz at the destination unit.
        weight_decay: decoupled weight-decay coefficient; defaults to 1e-4.
        b1: first-moment decay (beta1); defaults to 0.9.
        b2: second-moment decay (beta2); defaults to 0.999.
        eps: denominator epsilon added outside the sqrt; defaults to 1e-8.

    Returns:
        An Adam optimizer bundle configured with decoupled weight decay.
    """
    return Adam(
        lr=lr, grad_field=grad_field, b1=b1, b2=b2, eps=eps, weight_decay=weight_decay
    )
