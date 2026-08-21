"""Stateless stochastic gradient descent as a plastax optimizer bundle.

SGD is the minimal instance of the optimizer protocol (see the package
docstring): no per-connection state, no globals step counter, the whole rule
living in the `UpdateConn` it builds. The per-edge weight gradient is formed
by the delta rule ``dL/dw = dL/dz[dst] * activation[src]`` -- the gradient of
a weighted-sum layer's loss with respect to one edge weight -- which is exact
for any such layer, dense or the unrolled convolution of `topology.conv2d`.
This generalizes `examples/mlp_xor.py`'s hand-written `SgdUpdateConn`.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np

from plastax._types import ACTIVATION, WEIGHT, ConnIdx, FieldSpec, UnitIdx
from plastax.views import ConnView, ConnWrite, UnitView


@dataclasses.dataclass(frozen=True)
class _SGDUpdateConn:
    """UpdateConn applying one SGD step to each connection.

    The whole update happens in the incoming pass, so `outgoing` is a genuine
    no-op (mirroring `examples/mlp_xor.py`'s `SgdUpdateConn`). `g` is typed
    `object` so a single instance satisfies `UpdateConn[GS]` for every `GS`.

    Attributes:
        lr: the learning rate.
        grad_field: the unit field carrying dL/dz at the destination unit.
    """

    lr: float
    grad_field: FieldSpec[np.float32]

    def incoming(
        self,
        u: UnitView,
        dst: UnitIdx,
        src: UnitIdx,
        c: ConnView,
        cid: ConnIdx,
        g: object,
    ) -> ConnWrite:
        """Subtract ``lr * dL/dz[dst] * activation[src]`` from the weight.

        Args:
            u: the unit view.
            dst: index of the destination unit.
            src: index of the source unit.
            c: the connection view.
            cid: index of the connection.
            g: the global state (unused).

        Returns:
            The ConnWrite updating this connection's weight.
        """
        del g
        grad = u[self.grad_field, dst] * u[ACTIVATION, src]
        weight = c[WEIGHT, cid] - jnp.float32(self.lr) * grad
        return ConnWrite.of((WEIGHT, weight))

    def outgoing(
        self,
        u: UnitView,
        src: UnitIdx,
        dst: UnitIdx,
        c: ConnView,
        cid: ConnIdx,
        g: object,
    ) -> ConnWrite:
        """Return an empty write; SGD does all its work in the incoming pass.

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
class SGD:
    """Stateless SGD optimizer bundle: ``w <- w - lr * dL/dw``.

    Because SGD keeps no state, `state_fields` is empty and
    `needs_step_counter` is False: a Network wiring this optimizer adds nothing
    to its `extra_conn_fields`. The gradient is read from the destination's
    back-propagated dL/dz and the source activation, so the Network's backward
    pass must write dL/dz into `grad_field` before the update phase runs (the
    phase order forward -> loss -> backward -> update_conn guarantees this).

    Attributes:
        lr: the learning rate.
        grad_field: the unit field a Network's backward pass writes dL/dz into
            (e.g. `mlp_xor`'s ``grad_pre_act``); read at the destination unit.
        state_fields: empty -- SGD is stateless.
        needs_step_counter: False -- SGD needs no globals step counter.
    """

    lr: float
    grad_field: FieldSpec[np.float32]
    state_fields: tuple[FieldSpec[np.float32], ...] = ()
    needs_step_counter: bool = False

    def update_conn(self) -> _SGDUpdateConn:
        """Build the UpdateConn that applies this SGD rule.

        Returns:
            An UpdateConn subtracting ``lr * gradient`` from each weight.
        """
        return _SGDUpdateConn(self.lr, self.grad_field)


def sgd(lr: float, grad_field: FieldSpec[np.float32]) -> SGD:
    """Build a stateless SGD optimizer.

    Args:
        lr: the learning rate.
        grad_field: the unit field carrying dL/dz at the destination unit --
            the field a Network's backward pass writes (e.g. `mlp_xor`'s
            ``grad_pre_act``).

    Returns:
        An SGD optimizer bundle.
    """
    return SGD(lr=lr, grad_field=grad_field)
