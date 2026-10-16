"""SwiftTD (Javed, Sharifnassab & Sutton, RLC 2024) as plastax traits.

SwiftTD is True Online TD(lambda) on a linear predictor with per-feature
step sizes `alpha[i] = exp(beta[i])` learned by step-size optimization
(IDBD-style meta-gradients), an overshoot bound `eta` on the correction
ratio `tau`, and step-size decay when the bound trips. The per-feature state
(`w, z, zd, dw, h, h_old, h_temp, zbar, p, beta`) lives on the feature's
edge into the output unit; `v_old` and the TD error live on the output unit.

One conceptual point: SwiftTD's step has two loops over the features with
three *global* sums in between -- `tau = sum exp(beta) phi^2` over the just
updated step sizes, `b = sum z phi` over the just decayed traces, and
`v_delta = sum dw phi` -- and no trait slot reduces over edges between the
two halves of a connection update. The sums are instead taken one step late
by the forward pass, as one struct accumulator over the output's incoming
edges, and the connection update runs the *second* loop of the previous
step immediately followed by the *first* loop of this step, per edge. It is
the same arithmetic in a different order: no global state, every rule a pure
per-element function, and bit-for-bit the algorithm plastax-cpp runs.

The reference `SwiftTDNonSparse::Step` (github.com/kjaved0/swifttd) is
ported below as `SwiftTDReference`; `main` checks the trait version against
it on a random binary stream and on the 5-state random walk.

Run:  uv run python examples/swifttd.py
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np

import plastax as px

# --- hyperparameters -------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SwiftTDParams:
    """SwiftTD's hyperparameters (thesis Algorithm 14 defaults where it has them).

    Attributes:
        gamma: the problem's discount, the value the drivers feed as the
            per-step `gamma_t` when the stream has no termination.
        lambda_: trace decay.
        eta: the overshoot bound on the correction ratio, and the step-size
            ceiling.
        eta_min: the step-size floor.
        epsilon: step-size decay rate applied when the bound trips.
        theta: meta step size of the step-size optimization.
        alpha_init: initial step size of every feature.
    """

    gamma: float = 0.99
    lambda_: float = 0.9
    eta: float = 0.1
    eta_min: float = math.exp(-15.0)
    epsilon: float = 0.999
    theta: float = 1e-3
    alpha_init: float = 3.0**-3


# --- columns ---------------------------------------------------------------

# Unit kinds. The imprinting learner adds pattern and memory units; SwiftTD
# alone has observation units, the output, and one control unit carrying the
# per-step termination `gamma_t` (a GVF's discount signal: constant for the
# thesis's streams, 0 on the terminal step of an episodic task).
KIND_OBSERVATION, KIND_PATTERN, KIND_MEMORY, KIND_OUTPUT, KIND_CONTROL = 0, 1, 2, 3, 4

KIND = px.FieldSpec.int32("swifttd/kind", default=KIND_OBSERVATION)
# The activation at the end of the previous step, `phi_{t-1}`.
ACT_PREV = px.FieldSpec.float32("swifttd/act_prev")
# The value the output consumed in its last prediction: the SwiftTD `phi` of
# the lagged second loop. An observation unit's activation is scattered fresh
# each step, so the output saw this step's value; a generated unit's stored
# activation is overwritten by the sweep, so the output saw the previous one.
ACT_USED = px.FieldSpec.float32("swifttd/act_used")
# `act_used` of the previous step. The unit update (which rolls these) runs
# before the connection update within a step, so the fused update's `phi` of
# the lagged second loop is this column, while the forward pass (which runs
# before the roll) reads `act_used` for the same step's value.
ACT_USED_PREV = px.FieldSpec.float32("swifttd/act_used_prev")
# Output unit only.
V_OLD = px.FieldSpec.float32("swifttd/v_old")
DELTA = px.FieldSpec.float32("swifttd/delta")  # the loss seed: delta'_t
TAU_PREV = px.FieldSpec.float32("swifttd/tau_prev")  # sum exp(beta) phi_{t-1}^2
B_PREV = px.FieldSpec.float32("swifttd/b_prev")  # sum z phi_{t-1}
VD_PREV = px.FieldSpec.float32("swifttd/vd_prev")  # sum dw phi_{t-1}
# The growth-gate inputs of the imprinting learner (estimated tau_t and the
# number of active tenured sources); unused by SwiftTD alone but part of the
# one shared accumulator.
TAU_HAT = px.FieldSpec.float32("swifttd/tau_hat")
N_ELIG = px.FieldSpec.float32("swifttd/n_elig")

SWIFTTD_UNIT_FIELDS = (
    KIND,
    ACT_PREV,
    ACT_USED,
    ACT_USED_PREV,
    V_OLD,
    DELTA,
    TAU_PREV,
    B_PREV,
    VD_PREV,
    TAU_HAT,
    N_ELIG,
)

# Connection columns: the SwiftTD per-feature state, on the edge into the
# output. `px.WEIGHT` is `w`.
Z = px.FieldSpec.float32("swifttd/z")
ZD = px.FieldSpec.float32("swifttd/zd")
DW = px.FieldSpec.float32("swifttd/dw")
H = px.FieldSpec.float32("swifttd/h")
H_OLD = px.FieldSpec.float32("swifttd/h_old")
H_TEMP = px.FieldSpec.float32("swifttd/h_temp")
ZBAR = px.FieldSpec.float32("swifttd/zbar")
P = px.FieldSpec.float32("swifttd/p")
BETA = px.FieldSpec.float32("swifttd/beta")

SWIFTTD_CONN_FIELDS = (Z, ZD, DW, H, H_OLD, H_TEMP, ZBAR, P, BETA)


# --- the shared per-edge arithmetic -----------------------------------------

# Forward accumulator over the output's incoming edges, member-wise sum:
# (v, tau_hat, tau_prev, b_prev, vd_prev, n_elig).
OUTPUT_ACC = (
    px.monoid.sum_,
    px.monoid.sum_,
    px.monoid.sum_,
    px.monoid.sum_,
    px.monoid.sum_,
    px.monoid.sum_,
)


def output_edge_terms(
    u: px.UnitView,
    src: px.UnitIdx,
    c: px.ConnView,
    cid: px.ConnIdx,
    tenure_threshold: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """One output-incoming edge's contribution to the forward accumulator.

    Args:
        u: the unit view.
        src: the feature unit.
        c: the connection view.
        cid: the edge into the output.
        tenure_threshold: |w| at or above which the feature is tenured.

    Returns:
        `(w a, e^beta a^2, e^beta f^2, z f, dw f, [a > 0 and tenured])` with
        `a` the feature value the output reads now and `f` the value it read
        last step.
    """
    a = u[px.ACTIVATION, src]
    f = u[ACT_USED, src]
    w = c[px.WEIGHT, cid]
    eb = jnp.exp(c[BETA, cid])
    elig = (a > 0.0) & (jnp.abs(w) >= jnp.float32(tenure_threshold))
    return (
        w * a,
        eb * a * a,
        eb * f * f,
        c[Z, cid] * f,
        c[DW, cid] * f,
        elig.astype(jnp.float32),
    )


def output_values(
    acc: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
) -> tuple[tuple[px.FieldSpec[np.float32], jax.Array], ...]:
    """The output unit's columns from the reduced accumulator, as pairs."""
    v, tau_hat, tau_prev, b_prev, vd_prev, n_elig = acc
    return (
        (px.ACTIVATION, v),
        (TAU_HAT, tau_hat),
        (TAU_PREV, tau_prev),
        (B_PREV, b_prev),
        (VD_PREV, vd_prev),
        (N_ELIG, n_elig),
    )


def output_apply(
    acc: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
) -> px.UnitWrite:
    """The output unit's write: the prediction and the lagged reductions."""
    return px.UnitWrite.of(*output_values(acc))


EdgeValues = tuple[tuple[px.FieldSpec[np.float32], jax.Array], ...]


def swifttd_edge_values(
    hp: SwiftTDParams,
    c: px.ConnView,
    cid: px.ConnIdx,
    *,
    delta: jax.Array,
    tau: jax.Array,
    b: jax.Array,
    vd: jax.Array,
    f: jax.Array,
    gamma_t: jax.Array,
) -> EdgeValues:
    """The fused per-edge SwiftTD update, as `(column, new value)` pairs.

    Runs the second loop of the previous step (over `phi_{t-1} = f`, with that
    step's `tau`, `b`) and then the first loop of this step (with `delta'_t`
    and the previous step's `v_delta = vd`), exactly as the reference does
    them in sequence with the step boundary in between. The `tau > eta`
    decay-and-reset applies to active features only (the paper's listing and
    the reference's sparse classes; its non-sparse class resets every
    feature).

    Args:
        hp: the hyperparameters.
        c: the connection view.
        cid: the edge into the output.
        delta: `delta'_t`, this step's TD error.
        tau: the previous step's correction ratio.
        b: the previous step's `sum z phi`.
        vd: the previous step's `sum dw phi`.
        f: the feature's value in the previous step's prediction.
        gamma_t: this step's termination signal; the traces decay by
            `gamma_t lambda` (so `gamma_t = 0` ends an episode exactly).

    Returns:
        The edge's new SwiftTD state, one pair per column, for
        `px.ConnWrite.of` or for masking against another rule's write.
    """
    gl = gamma_t * jnp.float32(hp.lambda_)
    eta = jnp.float32(hp.eta)
    w = c[px.WEIGHT, cid]
    z = c[Z, cid]
    zd = c[ZD, cid]
    h = c[H, cid]
    h_old = c[H_OLD, cid]
    h_temp = c[H_TEMP, cid]
    zbar = c[ZBAR, cid]
    p = c[P, cid]
    beta = c[BETA, cid]

    # --- second loop of step t-1 (the reference's `for phi[i] != 0`).
    mult = jnp.where(tau > eta, eta / tau, jnp.float32(1.0))
    zd = mult * jnp.exp(beta) * f
    z = z + zd * (1.0 - b)
    p = p + h_old * f
    zbar = zbar + zd * (1.0 - b - zbar * f)
    h_temp = h - h_old * f * (z - zd) - h * zd * f
    decay = (tau > eta) & (f != 0.0)
    beta = jnp.where(decay, beta + f * f * jnp.float32(math.log(hp.epsilon)), beta)
    h_temp = jnp.where(decay, 0.0, h_temp)
    h = jnp.where(decay, 0.0, h)
    h_old = jnp.where(decay, 0.0, h_old)
    zbar = jnp.where(decay, 0.0, zbar)

    # --- first loop of step t (the reference's `for z[i] != 0`).
    dw = delta * z - zd * vd
    w = w + dw
    beta = beta + jnp.float32(hp.theta) / jnp.exp(beta) * (delta - vd) * p
    beta = jnp.clip(
        beta, jnp.float32(math.log(hp.eta_min)), jnp.float32(math.log(hp.eta))
    )
    h_old = h
    h = h_temp + delta * zbar - zd * vd
    h_temp = h
    zd = jnp.float32(0.0)
    z = z * gl
    p = p * gl
    zbar = zbar * gl
    return (
        (px.WEIGHT, w),
        (Z, z),
        (ZD, zd),
        (DW, dw),
        (H, h),
        (H_OLD, h_old),
        (H_TEMP, h_temp),
        (ZBAR, zbar),
        (P, p),
        (BETA, beta),
    )


def swifttd_edge_update(
    hp: SwiftTDParams,
    c: px.ConnView,
    cid: px.ConnIdx,
    *,
    delta: jax.Array,
    tau: jax.Array,
    b: jax.Array,
    vd: jax.Array,
    f: jax.Array,
    gamma_t: jax.Array,
) -> px.ConnWrite:
    """`swifttd_edge_values` as one connection write (see it for the arguments).

    Args:
        hp: the hyperparameters.
        c: the connection view.
        cid: the edge into the output.
        delta: `delta'_t`, this step's TD error.
        tau: the previous step's correction ratio.
        b: the previous step's `sum z phi`.
        vd: the previous step's `sum dw phi`.
        f: the feature's value in the previous step's prediction.
        gamma_t: this step's termination signal.

    Returns:
        The edge's new SwiftTD state.
    """
    return px.ConnWrite.of(
        *swifttd_edge_values(
            hp, c, cid, delta=delta, tau=tau, b=b, vd=vd, f=f, gamma_t=gamma_t
        )
    )


# --- the policies (static features) -----------------------------------------


class SwiftTDForward(px.ForwardPass):
    """Every edge feeds the output: the prediction plus the lagged sums."""

    combine = OUTPUT_ACC

    def __init__(self, hp: SwiftTDParams, tenure_threshold: float = 0.01) -> None:
        self.hp = hp
        self.tenure_threshold = tenure_threshold

    def map(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        del dst, g
        return output_edge_terms(u, src, c, cid, self.tenure_threshold)

    def apply(
        self,
        u: px.UnitView,
        i: px.UnitIdx,
        g: None,
        acc: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> px.UnitWrite:
        del u, i, g
        return output_apply(acc)


class TDErrorLoss(px.Loss):
    """`delta' = r + gamma_t v - v_old`, seeded into the output's DELTA column.

    `targets[0]` is the reward observed with this step's input and `gamma_t`
    is the control unit's activation (scattered with the inputs). The scalar
    loss is `0.5 delta'^2`; nothing reads it, but it is what GetLastLoss
    reports in plastax-cpp.
    """

    seed_field = DELTA

    def __init__(self, hp: SwiftTDParams, gamma_unit: int) -> None:
        self.hp = hp
        self.gamma_unit = gamma_unit

    def calculate_loss(
        self, u: px.UnitView, outputs: jnp.ndarray, targets: jnp.ndarray, g: None
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        del g
        v = u.gather(px.ACTIVATION, outputs)
        v_old = u.gather(V_OLD, outputs)
        gamma_t = u[px.ACTIVATION, px.UnitIdx(jnp.int32(self.gamma_unit))]
        delta = targets + gamma_t * v - v_old
        return jnp.sum(jnp.float32(0.5) * delta * delta), delta


class RecordActivations(px.UpdateUnit):
    """Record what the output read this step, and roll the activations.

    Runs after the loss and before the connection update, on every live
    unit: `act_used` becomes the value the output consumed in this step's
    prediction (an observation unit's fresh activation; a generated unit's
    pre-sweep activation, i.e. `act_prev`) after the old value moved to
    `act_used_prev`, `act_prev` becomes this step's activation, and the
    output's `v_old` becomes this step's prediction.
    """

    def update(self, u: px.UnitView, i: px.UnitIdx, g: None) -> px.UnitWrite:
        del g
        act = u[px.ACTIVATION, i]
        fresh = u[KIND, i] == KIND_OBSERVATION
        return px.UnitWrite.of(
            (ACT_USED_PREV, u[ACT_USED, i]),
            (ACT_USED, jnp.where(fresh, act, u[ACT_PREV, i])),
            (ACT_PREV, act),
            (V_OLD, act),
        )


class SwiftTDUpdate(px.UpdateConn):
    """The fused update on every edge into an output unit."""

    def __init__(self, hp: SwiftTDParams, gamma_unit: int) -> None:
        self.hp = hp
        self.gamma_unit = gamma_unit

    def incoming(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: None,
    ) -> px.ConnWrite:
        del g
        return swifttd_edge_update(
            self.hp,
            c,
            cid,
            delta=u[DELTA, dst],
            tau=u[TAU_PREV, dst],
            b=u[B_PREV, dst],
            vd=u[VD_PREV, dst],
            f=u[ACT_USED_PREV, src],
            gamma_t=u[px.ACTIVATION, px.UnitIdx(jnp.int32(self.gamma_unit))],
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
        del u, src, dst, c, cid, g
        return px.ConnWrite.of()


def make_net(hp: SwiftTDParams, num_features: int) -> type[px.Network[None]]:
    """The SwiftTD network class for a given set of hyperparameters.

    Args:
        hp: the hyperparameters, baked into the policies at trace time.
        num_features: the feature count; the control unit carrying
            `gamma_t` is unit `num_features` (see `build`).

    Returns:
        A pipeline-mode Network subclass: forward, loss, unit update and the
        fused connection update; no structural phases.
    """
    gamma_unit = num_features

    class SwiftTDNet(px.Network[None]):
        forward_pass = SwiftTDForward(hp)
        loss = TDErrorLoss(hp, gamma_unit)
        update_unit = RecordActivations()
        update_conn = SwiftTDUpdate(hp, gamma_unit)
        extra_unit_fields = SWIFTTD_UNIT_FIELDS
        extra_conn_fields = SWIFTTD_CONN_FIELDS
        propagation = px.Propagation.PIPELINE

    return SwiftTDNet


def build(
    net: type[px.Network[None]], num_features: int, hp: SwiftTDParams
) -> tuple[px.NetworkStatic, px.NetworkState[None], int]:
    """`num_features` observation units densely wired to one output unit.

    The inputs of a step are the `num_features` feature values followed by
    `gamma_t`: unit `num_features` is the control unit, an input with no
    edges that the loss and the update read by id.

    Args:
        net: the network class (`make_net`).
        num_features: feature count; units `0..num_features-1`.
        hp: the hyperparameters (`alpha_init` seeds every `beta`).

    Returns:
        The static config, the state, and the output unit's id.
    """
    b = px.NetworkBuilder(net, None)
    for _ in range(num_features):
        b.add_unit()
    gamma_unit = b.add_unit(**{KIND.name: KIND_CONTROL})
    out = b.add_unit(**{KIND.name: KIND_OUTPUT})
    for i in range(num_features):
        b.mark_input(i)
        b.add_conn(i, out, **{BETA.name: math.log(hp.alpha_init)})
    b.mark_input(gamma_unit)
    b.mark_output(out)
    static, state = b.finalize()
    return static, state, out


# --- the reference ------------------------------------------------------------


class SwiftTDReference:
    """`SwiftTDNonSparse::Step` of github.com/kjaved0/swifttd, in float32 NumPy.

    Operation for operation the reference's order, with one documented
    choice: the `tau > eta` decay-and-reset applies to active features only
    (the listing in the paper and the reference's sparse classes), where the
    non-sparse class applies the reset to every feature.
    """

    def __init__(self, n: int, hp: SwiftTDParams) -> None:
        self.hp = hp
        f32 = np.float32
        self.w = np.zeros(n, f32)
        self.z = np.zeros(n, f32)
        self.zd = np.zeros(n, f32)
        self.dw = np.zeros(n, f32)
        self.h = np.zeros(n, f32)
        self.h_old = np.zeros(n, f32)
        self.h_temp = np.zeros(n, f32)
        self.zbar = np.zeros(n, f32)
        self.p = np.zeros(n, f32)
        self.beta = np.full(n, math.log(hp.alpha_init), f32)
        self.v_old = f32(0.0)
        self.vd = f32(0.0)

    def predict(self, phi: np.ndarray) -> float:
        """The prediction with the current weights."""
        return float(np.dot(self.w, phi.astype(np.float32)))

    def step(
        self, phi: np.ndarray, reward: float, gamma_t: float | None = None
    ) -> float:
        """One SwiftTD step on `phi`, the reward observed with it, and `gamma_t`."""
        hp = self.hp
        f32 = np.float32
        gamma = f32(hp.gamma if gamma_t is None else gamma_t)
        phi = phi.astype(f32)
        v = f32(np.dot(self.w, phi))
        delta = f32(reward + gamma * v - self.v_old)
        self.dw = delta * self.z - self.zd * self.vd
        self.w += self.dw
        self.beta += f32(hp.theta) / np.exp(self.beta) * (delta - self.vd) * self.p
        self.beta = np.clip(self.beta, f32(math.log(hp.eta_min)), f32(math.log(hp.eta)))
        self.h_old = self.h.copy()
        self.h = self.h_temp + delta * self.zbar - self.zd * self.vd
        self.h_temp = self.h.copy()
        self.zd = np.zeros_like(self.zd)
        gl = f32(gamma * hp.lambda_)
        self.z *= gl
        self.p *= gl
        self.zbar *= gl
        self.vd = f32(0.0)
        tau = f32(np.sum(np.exp(self.beta) * phi * phi))
        b = f32(np.sum(self.z * phi))
        self.vd = f32(np.sum(self.dw * phi))
        mult = f32(hp.eta / tau) if tau > hp.eta else f32(1.0)
        self.zd = mult * np.exp(self.beta) * phi
        self.z += self.zd * (1.0 - b)
        self.p += self.h_old * phi
        self.zbar += self.zd * (1.0 - b - self.zbar * phi)
        self.h_temp = (
            self.h - self.h_old * phi * (self.z - self.zd) - self.h * self.zd * phi
        )
        if tau > hp.eta:
            active = phi != 0.0
            self.beta[active] += phi[active] ** 2 * f32(math.log(hp.epsilon))
            for arr in (self.h_temp, self.h, self.h_old, self.zbar):
                arr[active] = 0.0
        self.v_old = v
        return float(v)


# --- checks ----------------------------------------------------------------


def run_stream(
    hp: SwiftTDParams,
    phis: np.ndarray,
    rewards: np.ndarray,
    gammas: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, px.Driver[None]]:
    """Run the trait version and the reference on one stream.

    Args:
        hp: the hyperparameters.
        phis: (steps, n) binary features.
        rewards: (steps,) rewards.
        gammas: (steps,) per-step `gamma_t`, or None for `hp.gamma` throughout.

    Returns:
        The two prediction traces, `(trait, reference)`, and the trait
        driver after the run.
    """
    steps, n = phis.shape
    if gammas is None:
        gammas = np.full((steps,), hp.gamma, np.float32)
    net = make_net(hp, n)
    static, state, out = build(net, n, hp)
    driver = px.Driver(net, static, state)
    ref = SwiftTDReference(n, hp)
    got = np.zeros(steps, np.float32)
    want = np.zeros(steps, np.float32)
    for t in range(steps):
        driver.step(
            px.StepInputs(
                inputs=jnp.asarray(np.append(phis[t], gammas[t]), jnp.float32),
                targets=jnp.asarray([rewards[t]], jnp.float32),
            )
        )
        got[t] = float(driver.state.units[px.ACTIVATION.name][out])
        want[t] = ref.step(phis[t], float(rewards[t]), float(gammas[t]))
    return got, want, driver


def random_walk_stream(
    episodes: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The 5-state random walk (Sutton & Barto 6.2) as one stream.

    States 1..5 one-hot, terminals on either side, reward 1 on the right.
    Episodes are concatenated: a terminal step has the all-zero feature
    vector and `gamma_t = 0`, which ends the episode exactly (the TD target
    stops bootstrapping and the traces clear), so no reset trick is needed
    and the next episode's first step updates nothing. The true values are
    `V*(i) = i / 6`.

    Args:
        episodes: how many episodes to run.
        seed: the walk's seed.

    Returns:
        `(phis, rewards, gammas)` of the stream.
    """
    rng = np.random.default_rng(seed)
    phis: list[np.ndarray] = []
    rewards: list[float] = []
    gammas: list[float] = []
    for _ in range(episodes):
        s = 3
        while True:
            phi = np.zeros(5, np.float32)
            phi[s - 1] = 1.0
            phis.append(phi)
            rewards.append(0.0)
            gammas.append(1.0)
            s += 1 if rng.random() < 0.5 else -1
            if s == 0 or s == 6:
                phis.append(np.zeros(5, np.float32))
                rewards.append(1.0 if s == 6 else 0.0)
                gammas.append(0.0)
                break
    return (
        np.asarray(phis, np.float32),
        np.asarray(rewards, np.float32),
        np.asarray(gammas, np.float32),
    )


def output_weights(driver: px.Driver[None], num_features: int) -> np.ndarray:
    """The live weights of the feature -> output edges, by feature id."""
    conns = driver.state.conns[0]
    w = np.asarray(conns[px.WEIGHT.name])
    src = np.asarray(conns[px.FROM_ID.name])
    dead = np.asarray(conns[px.DEAD.name])
    values = np.zeros(num_features, np.float32)
    for k in range(w.shape[0]):
        if not dead[k] and src[k] < num_features:
            values[src[k]] = w[k]
    return values


def main() -> None:
    """Check the trait version against the reference, then learn the random walk."""
    hp = SwiftTDParams(gamma=0.9, lambda_=0.9, theta=1e-2, alpha_init=1e-2)
    rng = np.random.default_rng(0)
    steps, n = 1500, 12
    phis = (rng.random((steps, n)) < 0.25).astype(np.float32)
    rewards = (rng.random(steps) < 0.05).astype(np.float32) * np.sign(
        rng.standard_normal(steps)
    )
    got, want, _ = run_stream(hp, phis, rewards.astype(np.float32))
    err = float(np.max(np.abs(got - want)))
    print(
        f"random binary stream: max |trait - reference| = {err:.2e} over {steps} steps"
    )
    assert err < 1e-4, err

    walk = random_walk_stream(3000)
    hp_walk = SwiftTDParams(gamma=1.0, lambda_=0.9, theta=1e-2, alpha_init=0.05)
    got, want, driver = run_stream(hp_walk, *walk)
    err = float(np.max(np.abs(got - want)))
    values = output_weights(driver, 5)
    truth = np.arange(1, 6) / 6.0
    rmse = float(np.sqrt(np.mean((values - truth) ** 2)))
    print(
        f"random walk ({walk[0].shape[0]} steps): max |trait - reference| = {err:.2e}; "
        f"V(1..5) = {np.round(values, 4)}, RMSE vs i/6 = {rmse:.4f}"
    )
    assert err < 1e-4, err
    assert rmse < 0.05, rmse
    print("PASS")


if __name__ == "__main__":
    main()
