"""Utility-based Perturbed Gradient Descent (Elsayed & Mahmood, ICLR 2024).

CBP's idea at EDGE granularity instead of unit granularity, from the same lab,
and already evaluated at batch 1 with no replay and unknown task boundaries --
the setting this plan targets. UPGD is one `UpdateConn`: no churn net, no
topology change. It gates the gradient by a per-weight utility, so useful
weights are protected from being overwritten and useless ones are perturbed back
into plasticity.

Per-weight utility is the first-order loss change from zeroing that weight:

    U(w) = -(dL/dw) * w

which on the edge arena is pure per-edge local -- the delta rule factorises
`dL/dw` into `grad_pre_act[dst] * activation[src]`, two columns the backward
pass already wrote. Then, following Algorithm 1:

    U   <- beta*U + (1 - beta)*M          running average
    Uhat = U / (1 - beta^t)               bias correction, as in Adam
    Ubar = sigmoid(Uhat / eta)            scaled utility
    w    <- w - alpha*(dL/dw + xi)*(1 - Ubar),   xi ~ N(0, sigma^2)

**`eta` is where the paper needs global state, and this file builds both
answers.** Algorithm 1 sets `eta = max(Uhat)` over ALL weights, and unlike CBP's
per-churn threshold it is needed on EVERY step.

* **v0 (global).** The host reduces the utility column and broadcasts the max.
  Faithful to the paper, and the arm that can be checked against the authors'
  released code -- at the cost of a device-to-host sync per training step.
* **v1 (local).** Each destination takes the max over its OWN incoming edges,
  as a `max_` monoid reduction folded into the forward pass. Zero global state.
  It reads the utilities the PREVIOUS step stored, because the forward runs
  before the backward that produces this step's gradient; that one-step lag is
  the price of keeping it local.

Only the first-order utility is implemented. The paper's second-order term needs
a Hessian-diagonal approximation, which is a separate contribution of theirs and
is recorded here as not implemented rather than approximated.

Run:  uv run python examples/upgd.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from dst_sparse import _hash01
from mlp_xor import GradPreAct, LossGrad, MSELoss
from nonstationary import ACT_EMA, IS_OUT, ReluBackward

import plastax as px

# Per-EDGE running-average utility, and its step count for bias correction.
UPGD_UTIL = px.FieldSpec.float32("upgd/util")
UPGD_T = px.FieldSpec.float32("upgd/t")
# The scaling denominator, per destination unit: a broadcast global max in v0,
# a local max over incoming edges in v1.
UPGD_ETA = px.FieldSpec.float32("upgd/eta")

_UNIT_FIELDS = (GradPreAct, LossGrad, ACT_EMA, IS_OUT, UPGD_ETA)
_CONN_FIELDS = (UPGD_UTIL, UPGD_T)


def _gaussian(src: jax.Array, dst: jax.Array, salt: jax.Array) -> jax.Array:
    """Box-Muller standard normal from two stateless hashes of the edge."""
    u1 = jnp.maximum(_hash01(src, dst, salt), jnp.float32(1e-7))
    u2 = _hash01(dst, src, salt)
    return jnp.sqrt(-2.0 * jnp.log(u1)) * jnp.cos(2.0 * jnp.pi * u2)


class UpgdForward(px.ForwardPass):
    """ReLU forward that also reduces a local utility max into each destination.

    The product monoid carries both reductions in one sweep: `sum_` builds the
    pre-activation, `max_` builds v1's per-unit `eta`. The utilities it reduces
    were stored by the previous step's update, since this pass runs before the
    backward that produces the current gradient.
    """

    combine = (px.monoid.sum_, px.monoid.max_)

    def __init__(self, ema_decay: float = 0.05) -> None:
        """Bind the activation-EMA decay used for the dormancy diagnostic."""
        self.ema_decay = ema_decay

    def map(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: None,
    ) -> tuple[jax.Array, jax.Array]:
        """Contribute the weighted activation and this edge's stored utility."""
        del dst, g
        weighted = c[px.WEIGHT, cid] * u[px.ACTIVATION, src]
        return weighted, c[UPGD_UTIL, cid]

    def apply(
        self,
        u: px.UnitView,
        i: px.UnitIdx,
        g: None,
        acc: tuple[jax.Array, jax.Array],
    ) -> px.UnitWrite:
        """Write the activation, its EMA, and v1's local utility scale."""
        del g
        pre_activation, max_utility = acc
        is_out = u[IS_OUT, i] > jnp.float32(0.5)
        activation = jnp.where(is_out, pre_activation, jax.nn.relu(pre_activation))
        beta = jnp.float32(self.ema_decay)
        ema = (jnp.float32(1.0) - beta) * u[ACT_EMA, i] + beta * jnp.abs(activation)
        return px.UnitWrite.of(
            (px.ACTIVATION, activation),
            (ACT_EMA, ema),
            (UPGD_ETA, max_utility),
        )


class UpgdUpdate(px.UpdateConn[None]):
    """The UPGD weight update: utility-gated, perturbed gradient descent.

    A weight with scaled utility 1 is left alone by both the gradient and the
    noise; one with scaled utility 0 receives the full update. That single gate
    is what addresses forgetting and plasticity loss at the same time.
    """

    def __init__(
        self,
        *,
        lr: float = 0.01,
        beta: float = 0.999,
        sigma: float = 0.001,
        eps: float = 1e-8,
    ) -> None:
        """Bind the step size, utility decay, noise scale and denominator floor.

        Args:
            lr: step size alpha.
            beta: utility decay rate.
            sigma: standard deviation of the perturbation noise.
            eps: floor on the utility-scaling denominator.
        """
        self.lr = lr
        self.beta = beta
        self.sigma = sigma
        self.eps = eps

    def incoming(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: None,
    ) -> px.ConnWrite:
        """Advance the utility trace and take one utility-gated noisy step."""
        del g
        weight = c[px.WEIGHT, cid]
        gradient = u[GradPreAct, dst] * u[px.ACTIVATION, src]
        instantaneous = -gradient * weight

        beta = jnp.float32(self.beta)
        step = c[UPGD_T, cid] + jnp.float32(1.0)
        utility = beta * c[UPGD_UTIL, cid] + (jnp.float32(1.0) - beta) * instantaneous
        corrected = utility / jnp.maximum(
            jnp.float32(1.0) - beta**step, jnp.float32(self.eps)
        )

        scale = jnp.maximum(u[UPGD_ETA, dst], jnp.float32(self.eps))
        gated = jax.nn.sigmoid(corrected / scale)
        noise = jnp.float32(self.sigma) * _gaussian(src, dst, step.astype(jnp.int32))
        updated = weight - jnp.float32(self.lr) * (gradient + noise) * (
            jnp.float32(1.0) - gated
        )
        return px.ConnWrite.of(
            (px.WEIGHT, updated), (UPGD_UTIL, utility), (UPGD_T, step)
        )

    def outgoing(
        self,
        u: px.UnitView,
        src: px.UnitIdx,
        dst: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: None,
    ) -> px.ConnWrite:
        """Return an empty write; UPGD does all its work in the incoming pass."""
        del u, src, dst, c, cid, g
        return px.ConnWrite.of()


def set_global_eta(state: px.NetworkState[None]) -> px.NetworkState[None]:
    """v0: broadcast `max(Uhat)` over every live edge, as Algorithm 1 specifies.

    Costs a device-to-host reduction on EVERY training step, which is the price
    of the paper's global ordering over weights. v1 avoids it entirely.

    Args:
        state: the state to reduce and write into.

    Returns:
        The state with a uniform `upgd/eta` column.
    """
    best = 0.0
    for bucket in state.conns:
        dead = np.asarray(bucket[px.DEAD.name])
        utility = np.asarray(bucket[UPGD_UTIL.name])[~dead]
        if utility.size:
            best = max(best, float(np.max(utility)))
    column = jnp.full_like(state.units[UPGD_ETA.name], jnp.float32(max(best, 1e-8)))
    state.units = {**state.units, UPGD_ETA.name: column}
    return state


def make_net(
    *,
    mode: str,
    lr: float = 0.01,
    beta: float = 0.999,
    sigma: float = 0.001,
    ema_decay: float = 0.05,
) -> type[px.Network[None]]:
    """Build the train or eval net for UPGD.

    There is no churn net: UPGD changes no topology, so `prune_conn` and
    `add_conn` are absent and the whole method is one `update_conn`.

    Args:
        mode: ``"train"`` or ``"eval"``.
        lr: step size.
        beta: utility decay rate.
        sigma: perturbation noise scale.
        ema_decay: activation-EMA rate for the dormancy diagnostic.

    Returns:
        A Network subclass for the requested mode.

    Raises:
        ValueError: on an unknown mode.
    """
    if mode == "train":

        class _Train(px.Network[None]):
            forward_pass = UpgdForward(ema_decay)
            backward_pass = ReluBackward()
            loss = MSELoss()
            update_conn = UpgdUpdate(lr=lr, beta=beta, sigma=sigma)
            extra_unit_fields = _UNIT_FIELDS
            extra_conn_fields = _CONN_FIELDS
            propagation = px.Propagation.TOPOLOGICAL

        return _Train

    if mode == "eval":

        class _Eval(px.Network[None]):
            forward_pass = UpgdForward(ema_decay)
            extra_unit_fields = _UNIT_FIELDS
            extra_conn_fields = _CONN_FIELDS
            propagation = px.Propagation.TOPOLOGICAL

        return _Eval

    raise ValueError(f"make_net: unknown mode {mode!r}")
