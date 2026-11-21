"""SKETCH -- does not run. The imprinting learner (Javed 2025, thesis chapter 9) written
against the interface we are designing. The running version is `examples/imprinting_learner.py`
on top of `examples/swifttd.py`; this follows it rule for rule.

What changes in the telling. Today the learner is one policy per phase for the whole network,
and every rule begins by asking what kind of unit it is looking at (`jnp.where(is_out, ...)`).
Here each kind of unit and each kind of connection has its own policies, so a rule only
contains its own arithmetic:

    units        observation   an input bit
                 control       carries gamma_t
                 feature       a generated unit, one policy holding both behaviours:
                                 pattern  fires when enough of what it imprinted on was active
                                 memory   relays one feature into the future (delay k2, length k1)
                 readout       the SwiftTD prediction
    connections  prediction    feature -> readout, carries the SwiftTD per-feature state
                 imprint       feature -> generated feature, no learned state

The topology is three lines: the inputs and a readout that reduces over every feature. Every
other unit and connection is created and destroyed at run time by the policies.

Where each kind gets its policies. Nearly everything in this network is created at run time by
the network's own growth, and a new element has the policies of the place whose `grow` created
it. So the network's policies are the generated kinds, `feature` and `imprint`, and the three
kinds that exist at the start (`observation`, `control`, `readout` with its `prediction`
connections) are given their own in the topology, under the same keywords.
PROPOSED: plans/20261121T191023Z-layer-policies.md, Q4.

Left out: the optional `unique_patterns` rule (no two pattern features on the same eligible
set), which is off by default in the running example. Everything else is here.

Lines marked `GAP:` are places where writing this exposed something the interface spec has not
settled.
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int32

import plastax as px


# --------------------------------------------------------------------------- #
# Hyperparameters
# --------------------------------------------------------------------------- #
@dataclasses.dataclass(frozen=True)
class Params:
    """Numeric hyperparameters. Held in global state: read-only to every policy as `g.hp`,
    and run-time values, so changing one does not recompile.
    """

    # SwiftTD
    lambda_: float = 0.95
    eta: float = 0.1  # overshoot bound on the correction ratio, and step-size ceiling
    eta_min: float = math.exp(-15.0)
    epsilon: float = 0.999  # step-size decay when the bound trips
    theta: float = 1e-3  # meta step size
    alpha_init: float = 0.01
    # Imprinting
    tenure_threshold: float = 0.01  # |w| at or above which a feature is tenured
    tenure_track_threshold: float = 1e-3  # |w| below which a generated feature is idle
    epsilon_z: float = 0.01  # a trace has faded when z < e^beta * epsilon_z
    k_mem: int = 10  # memory features generated per step, at most, in expectation
    k_pat: int = 10
    k0: tuple[float, ...] = (0.6, 0.7, 0.8, 0.9)  # pattern thresholds
    k1: tuple[int, int] = (1, 3)  # memory active length
    k2: tuple[int, int] = (0, 20)  # memory delay


# Structural values fix a shape or are used in plain Python, so they cannot be run-time
# values. They stay static.
NUM_K0 = len(Params.k0)  # length of the pattern-threshold table
SHORTLIST_SIZE = 256
UNIT_CAPACITY = 4096


@px.globals
class Globals:
    """Global state: immutable, readable by every policy. `step` and `seed` are supplied
    by the framework.
    """

    hp: Params


# --------------------------------------------------------------------------- #
# Types used in the signatures
# --------------------------------------------------------------------------- #
# Every policy method is written for ONE element (one unit, or one connection) and the
# framework vmaps it over all of them. So every index and every value below is a scalar.
#
#   u: px.UnitView     read-only view of all unit state:        u[FIELD, unit_index]
#   c: px.ConnView     read-only view of all connection state:  c[FIELD, conn_index]
#   i, src, dst, parent, child: px.UnitIdx    one unit's index (a scalar int32)
#   cid: px.ConnIdx                           one connection's index (a scalar int32)
#   g: Globals         the global state, read-only
#
# A policy never assigns into `u` or `c`. It returns a write, `px.UnitWrite` or
# `px.ConnWrite`: a set of (field, new value) pairs for the one element it was called on,
# which the framework scatters into the state after the whole sweep. Fields not named in
# the write keep their value.
Scalar = Float[Array, ""]
Flag = Bool[Array, ""]

# What flows from a connection's `map` to its destination unit's `combine` and `apply`.
# These are the type parameters of the forward and backward traits, so a connection policy
# and a unit policy that disagree fail the type check.
ReadoutAcc = tuple[Scalar, Scalar, Scalar, Scalar, Scalar, Scalar]
#            (w*a, e^b a^2, e^b f^2, z*f, dw*f, tenured and active)   -> the readout
ImprintAcc = tuple[Scalar, Scalar]  # (input's previous activation, 1)  -> pattern / memory
ReaderAcc = tuple[Scalar, Scalar]  # (prediction weight, 1)             -> back to a feature
# What `SwiftTDUpdate.contribution` hands to its `step`.
Lagged = tuple[Scalar, Scalar, Scalar, Scalar, Scalar, Scalar]
#        (delta, tau_prev, b_prev, vd_prev, feature value used last step, gamma_t)


# --------------------------------------------------------------------------- #
# Fields shared between policies (one module-level object, passed to each)
# --------------------------------------------------------------------------- #
f32, i32 = np.float32, np.int32

# On every feature unit.
ACT_PREV = px.UnitField[f32]("act_prev")  # activation at the end of the previous step
ACT_USED = px.UnitField[f32]("act_used")  # the value the readout consumed last prediction
ACT_USED_PREV = px.UnitField[f32]("act_used_prev")
W_OUT = px.UnitField[f32]("w_out")  # the feature's prediction weight, mirrored onto it
OUT_DEG = px.UnitField[f32]("out_deg")  # how many connections still read this unit
TRACK_THR = px.UnitField[f32]("track_thr")  # 0 for observation units: they are never idle
BORN = px.UnitField[i32]("born", default=-1)  # the step a generated unit was created on
MEM_SRC = px.UnitField[i32]("mem_src", default=-1)  # a memory unit's single input

# On the readout.
V_OLD = px.UnitField[f32]("v_old")
DELTA = px.UnitField[f32]("delta")  # the TD error, written by the loss
TAU_HAT = px.UnitField[f32]("tau_hat")  # estimated correction ratio this step
TAU_PREV = px.UnitField[f32]("tau_prev")  # sum e^beta phi_{t-1}^2
B_PREV = px.UnitField[f32]("b_prev")  # sum z phi_{t-1}
VD_PREV = px.UnitField[f32]("vd_prev")  # sum dw phi_{t-1}
N_ELIG = px.UnitField[f32]("n_elig")  # active tenured features this step

# Inputs. An input is declared once, as a module-level object, and that object is used in
# three places: the network gives it a shape, the step is fed data under it, and a policy
# may hold it as a reference to its unit (which requires it to be a single unit).
OBS = px.Input("obs")  # the frame
GAMMA = px.Input("gamma")  # one unit carrying this step's discount

# A unit that is not an input and that policies refer to by name. The topology binds it
# (`.bind(READOUT)`). Like an input, it resolves to the unit's index when the network is
# materialized, and the name does not exist at run time. A referenced unit must never be
# pruned, and a reference must be bound exactly once: both checked at initialization.
READOUT = px.UnitRef("readout")

# On a prediction connection: the SwiftTD per-feature state. `px.WEIGHT` is w.
Z = px.ConnField[f32]("z")
DW = px.ConnField[f32]("dw")
BETA = px.ConnField[f32]("beta")  # log step size


def eligible(u: px.UnitView, i: px.UnitIdx, g: Globals) -> Flag:
    """An active tenured feature: what a new feature may imprint on."""
    return (u[px.ACTIVATION, i] > 0.0) & (jnp.abs(u[W_OUT, i]) >= g.hp.tenure_threshold)


# --------------------------------------------------------------------------- #
# Connection policies
# --------------------------------------------------------------------------- #
@px.policy
class PredictionForward(px.policies.ForwardConnPolicy[ReadoutAcc, Globals]):
    """A feature's contribution to the readout: the prediction and SwiftTD's lagged sums.

    SwiftTD needs three sums over the features between the two halves of its update. They are
    taken here, one step late, as one accumulator over the readout's incoming connections.
    """

    def map(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: Globals,
    ) -> ReadoutAcc:
        a = u[px.ACTIVATION, src]  # what the readout reads now
        f = u[ACT_USED, src]  # what it read last step
        w = c[px.WEIGHT, cid]
        eb = jnp.exp(c[BETA, cid])
        tenured_active = (a > 0.0) & (jnp.abs(w) >= g.hp.tenure_threshold)
        return (
            w * a,
            eb * a * a,
            eb * f * f,
            c[Z, cid] * f,
            c[DW, cid] * f,
            tenured_active.astype(f32),  # summed into a count
        )


@px.policy
class PredictionBackward(px.policies.BackwardConnPolicy[ReaderAcc, Globals]):
    """Mirror the prediction weight onto the feature, and count this reader."""

    def map(
        self,
        u: px.UnitView,
        src: px.UnitIdx,
        dst: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: Globals,
    ) -> ReaderAcc:
        return (c[px.WEIGHT, cid], jnp.float32(1.0))


@px.policy
class SwiftTDUpdate(px.policies.UpdateConnPolicy[Lagged, Globals]):
    """The fused SwiftTD update: the second loop of step t-1, then the first loop of step t.

    On top of it, the sparse algorithm's eligible set: a feature enters when active and leaves,
    with its trace state zeroed and `dropped` raised, once its trace has faded.
    """

    gamma: px.UnitRef  # the control unit; a single-unit `px.Input` is a `px.UnitRef`
    # State owned by this policy, so namespaced to it.
    zd: px.ConnField[f32] = 0.0
    h: px.ConnField[f32] = 0.0
    h_old: px.ConnField[f32] = 0.0
    h_temp: px.ConnField[f32] = 0.0
    zbar: px.ConnField[f32] = 0.0
    p: px.ConnField[f32] = 0.0
    in_set: px.ConnField[np.bool_] = False
    dropped: px.ConnField[np.bool_] = False  # read by IdleFeaturePrune below

    def contribution(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: Globals,
    ) -> Lagged:
        # Everything this connection needs from the units at its two ends. A prediction
        # weight has one use, so there is nothing to combine.
        return (
            u[DELTA, dst],
            u[TAU_PREV, dst],
            u[B_PREV, dst],
            u[VD_PREV, dst],
            u[ACT_USED_PREV, src],
            u[px.ACTIVATION, self.gamma],
        )

    def step(self, c: px.ConnView, cid: px.ConnIdx, g: Globals, total: Lagged) -> px.ConnWrite:
        # `step` takes the connection view and the globals, and returns a `px.ConnWrite`: the
        # new values of the twelve fields this rule changes, for connection `cid`.
        delta, tau, b, vd, f, gamma_t = total
        hp = g.hp
        gl = gamma_t * hp.lambda_
        w, z, beta = c[px.WEIGHT, cid], c[Z, cid], c[BETA, cid]
        h, h_old, zbar, p = (
            c[self.h, cid],
            c[self.h_old, cid],
            c[self.zbar, cid],
            c[self.p, cid],
        )

        # Second loop of step t-1.
        mult = jnp.where(tau > hp.eta, hp.eta / tau, 1.0)
        zd = mult * jnp.exp(beta) * f
        z = z + zd * (1.0 - b)
        p = p + h_old * f
        zbar = zbar + zd * (1.0 - b - zbar * f)
        h_temp = h - h_old * f * (z - zd) - h * zd * f
        decay = (tau > hp.eta) & (f != 0.0)
        beta = jnp.where(decay, beta + f * f * jnp.log(hp.epsilon), beta)
        h_temp, h, h_old, zbar = (jnp.where(decay, 0.0, x) for x in (h_temp, h, h_old, zbar))

        # First loop of step t.
        dw = delta * z - zd * vd
        w = w + dw
        beta = beta + hp.theta / jnp.exp(beta) * (delta - vd) * p
        beta = jnp.clip(beta, jnp.log(hp.eta_min), jnp.log(hp.eta))
        h_old, h = h, h_temp + delta * zbar - zd * vd
        z, p, zbar = z * gl, p * gl, zbar * gl

        # The eligible set.
        in_set = c[self.in_set, cid] | (f != 0.0)
        dropped = in_set & (z < jnp.exp(beta) * hp.epsilon_z)
        z, p, zbar, dw = (jnp.where(dropped, 0.0, x) for x in (z, p, zbar, dw))

        return px.ConnWrite.of(
            (px.WEIGHT, w), (Z, z), (DW, dw), (BETA, beta),
            (self.zd, 0.0), (self.h, h), (self.h_old, h_old), (self.h_temp, h),
            (self.zbar, zbar), (self.p, p),
            (self.in_set, in_set & ~dropped), (self.dropped, dropped),
        )  # fmt: skip


@px.policy
class IdleFeaturePrune(px.policies.PruneConnPolicy):
    """Remove a feature from the prediction when it is idle as its trace fades.

    Only the prediction connection goes. If other features still read the unit it stays, as a
    ghost: no prediction weight, still part of the state update.
    """

    # Another policy's field, passed in. The rule: pass the field when only its state is
    # needed, pass the policy when its behaviour is needed.
    dropped: px.ConnField[np.bool_]

    def predicate(self, u: px.UnitView, c: px.ConnView, cid: px.ConnIdx, g: Globals) -> Flag:
        src = c[px.FROM_ID, cid]
        idle = jnp.abs(c[px.WEIGHT, cid]) < u[TRACK_THR, src]  # never true for observations
        return idle & c[self.dropped, cid]


@px.policy
class ImprintForward(px.policies.ForwardConnPolicy[ImprintAcc, Globals]):
    """A generated feature reads its inputs' previous activations, and counts its inputs."""

    def map(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: Globals,
    ) -> ImprintAcc:
        return (u[ACT_PREV, src], jnp.float32(1.0))


@px.policy
class ImprintBackward(px.policies.BackwardConnPolicy[ReaderAcc, Globals]):
    """No prediction weight to mirror; still one reader."""

    def map(
        self,
        u: px.UnitView,
        src: px.UnitIdx,
        dst: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: Globals,
    ) -> ReaderAcc:
        return (jnp.float32(0.0), jnp.float32(1.0))


def prediction_init(g: Globals, w: Scalar) -> dict[px.FieldSpec, Scalar]:
    """Initial values of a new prediction connection's other fields: the initial step size.

    The weight itself is set by the topology (the readout layer's initializer, zero) and
    handed in as `w`.
    """
    return {BETA: jnp.log(g.hp.alpha_init)}


_swifttd = SwiftTDUpdate(gamma=GAMMA)
# The two kinds of connection are not objects. `prediction` is what the readout layer is given
# in the topology below; `imprint` is the network's default, declared on `Learner`.


# --------------------------------------------------------------------------- #
# Unit policies
# --------------------------------------------------------------------------- #
@px.policy
class MirrorReaders(px.policies.BackwardUnitPolicy[ReaderAcc, Globals]):
    """Every feature learns its own prediction weight and how many connections read it."""

    combine = (px.monoid.sum_, px.monoid.sum_)  # why do we need two of these?

    def apply(self, u: px.UnitView, i: px.UnitIdx, g: Globals, acc: ReaderAcc) -> px.UnitWrite:
        return px.UnitWrite.of((W_OUT, acc[0]), (OUT_DEG, acc[1]))


@px.policy
class RollFresh(px.policies.UpdateUnitPolicy):
    """An observation is scattered fresh each step, so the readout saw this step's value."""

    def update(self, u: px.UnitView, i: px.UnitIdx, g: Globals) -> px.UnitWrite:
        act = u[px.ACTIVATION, i]
        return px.UnitWrite.of((ACT_USED_PREV, u[ACT_USED, i]), (ACT_USED, act), (ACT_PREV, act))


@px.policy
class RollLagged(px.policies.UpdateUnitPolicy):
    """A generated feature is recomputed by the sweep, so the readout saw the previous value."""

    def update(self, u: px.UnitView, i: px.UnitIdx, g: Globals) -> px.UnitWrite:
        return px.UnitWrite.of(
            (ACT_USED_PREV, u[ACT_USED, i]),
            (ACT_USED, u[ACT_PREV, i]),
            (ACT_PREV, u[px.ACTIVATION, i]),
        )


@px.policy
class FeatureForward(px.policies.ForwardUnitPolicy[ImprintAcc, Globals]):
    """A generated feature: a pattern or a memory, chosen once when the unit is spawned.

    One policy holds both behaviours and selects by the unit's `is_mem` field. Unit fields
    are columns over every unit and the selection is computed either way, so this costs
    nothing over two separate policies.

    pattern  fires when at least `thresh` of the features it imprinted on were active.
    memory   a shift register: bit 0 is 'active now'; an active input reloads the pulse.
    """

    combine = (px.monoid.sum_, px.monoid.sum_)  # (inputs active last step, inputs)
    is_mem: px.UnitField[np.bool_] = False
    thresh: px.UnitField[f32] = 0.0  # pattern
    pulse: px.UnitField[np.uint32] = 0  # memory
    pulse_pattern: px.UnitField[np.uint32] = 0  # memory: ((1 << k1) - 1) << k2

    def apply(self, u: px.UnitView, i: px.UnitIdx, g: Globals, acc: ImprintAcc) -> px.UnitWrite:
        active, count = acc
        is_mem = u[self.is_mem, i]

        fires = (count > 0.0) & (active >= u[self.thresh, i] * count)

        pulse = u[self.pulse, i]
        reload = jnp.where(active > 0.0, u[self.pulse_pattern, i], np.uint32(0))

        return px.UnitWrite.of(
            (
                px.ACTIVATION,
                jnp.where(is_mem, (pulse & 1).astype(f32), fires.astype(f32)),
            ),
            (self.pulse, jnp.where(is_mem, (pulse >> 1) | reload, pulse)),
        )


@px.policy
class OrphanPrune(px.policies.PruneUnitPolicy):
    """Delete a generated unit nothing reads any more (but not one created this step)."""

    def predicate(self, u: px.UnitView, i: px.UnitIdx, g: Globals) -> Flag:
        return (u[OUT_DEG, i] == 0.0) & (u[BORN, i] < g.step)


@px.policy
class ReadoutForward(px.policies.ForwardUnitPolicy[ReadoutAcc, Globals]):
    """The prediction, and the sums the next step's update needs."""

    combine = (px.monoid.sum_,) * 6

    def apply(self, u: px.UnitView, i: px.UnitIdx, g: Globals, acc: ReadoutAcc) -> px.UnitWrite:
        v, tau_hat, tau_prev, b_prev, vd_prev, n_elig = acc
        return px.UnitWrite.of(
            (px.ACTIVATION, v), (TAU_HAT, tau_hat), (TAU_PREV, tau_prev),
            (B_PREV, b_prev), (VD_PREV, vd_prev), (N_ELIG, n_elig),
        )  # fmt: skip


@px.policy
class RememberPrediction(px.policies.UpdateUnitPolicy):
    def update(self, u: px.UnitView, i: px.UnitIdx, g: Globals) -> px.UnitWrite:
        return px.UnitWrite.of((V_OLD, u[px.ACTIVATION, i]))


@px.policy
class TDErrorLoss(px.policies.LossPolicy):
    """delta' = r + gamma_t v - v_old, written to the readout for the connection update."""

    gamma: px.UnitRef

    # NOTE (owner): the running example has one readout. If every prediction is a GVF, the
    # tenure and spawn-budget rules below need revisiting. To be tested once this runs.
    def calculate_loss(
        self,
        u: px.UnitView,
        outputs: Int32[Array, " num_outputs"],  # the output unit ids: here, the one readout
        targets: Float[Array, " num_outputs"],  # here, the reward
        g: Globals,
    ) -> tuple[Scalar, Float[Array, " num_outputs"]]:
        # The one policy that sees several units at once, so these are vectors.
        v = u.gather(px.ACTIVATION, outputs)
        delta = targets + u[px.ACTIVATION, self.gamma] * v - u.gather(V_OLD, outputs)
        return jnp.sum(0.5 * delta * delta), delta  # second value is written to DELTA


_feature = FeatureForward()

# A connection's `map` must return what the unit at its other end combines. Going forward that
# unit is its destination: six sums into the readout, two into a feature. Going backward it is
# its source: every connection that leaves a feature or an observation returns a `ReaderAcc`,
# whether it is a prediction or an imprint connection, because `MirrorReaders` reduces both.
# Each policy states its accumulator as a type parameter (`ReadoutAcc`, `ImprintAcc`,
# `ReaderAcc`). The framework combines the different accumulators into one, their product, in
# which a connection fills its own part.
#
# A mismatch is always an error, never something the framework lets through (owner). It is
# raised when the network is built, for every block of connections against the units it ends in
# and the units it starts from. A connection the network's `grow` creates has the network's
# connection policies, so a proposal into a unit that reduces something else (the readout) is
# not created and is reported.


# --------------------------------------------------------------------------- #
# Growth: creating units and wiring them
# --------------------------------------------------------------------------- #
@px.policy
class ImprintSpawn(px.policies.AddUnitPolicy):
    """Feature generation: an active tenured feature spawns a pattern or memory child.

    The step's budget is at most `k_mem + k_pat` features, and only while the correction ratio
    leaves room under the overshoot bound. Each eligible parent spawns with probability
    `budget / n_eligible`, so the expected count is the budget.
    """

    out: px.UnitRef  # the readout
    # Every unit this rule creates has the network's unit policies, because this rule is the
    # network's `grow`. A grown unit does not inherit from its parent: the parent may be an
    # observation unit.
    features: FeatureForward  # the child's forward half, for its fields

    # A policy that needs randomness asks for it: `self.rng(g, parent)` is this policy's
    # stream for this parent and this step. `spawn` and `init` are methods of one policy,
    # so for the same parent they get the same stream, starting from the same place.
    def spawn(self, u: px.UnitView, parent: px.UnitIdx, g: Globals) -> Flag:
        hp = g.hp
        room = jnp.floor((hp.eta - u[TAU_HAT, self.out]) / hp.alpha_init)
        budget = jnp.minimum(hp.k_mem + hp.k_pat, room)
        n = u[N_ELIG, self.out]
        p = jnp.where(n > 0.0, budget / jnp.maximum(n, 1.0), 0.0)
        return eligible(u, parent, g) & (budget >= 1.0) & (self.rng(g, parent).uniform() < p)

    def init(
        self, u: px.UnitView, child: px.UnitIdx, parent: px.UnitIdx, g: Globals
    ) -> px.UnitWrite:
        hp = g.hp
        rng = self.rng(g, parent)
        rng.uniform()  # the draw `spawn` made from this same stream
        is_mem = rng.uniform() < hp.k_mem / (hp.k_mem + hp.k_pat)
        k0 = hp.k0[rng.uniform_int(NUM_K0)]
        k1 = hp.k1[0] + rng.uniform_int(hp.k1[1] - hp.k1[0] + 1)
        k2 = hp.k2[0] + rng.uniform_int(hp.k2[1] - hp.k2[0] + 1)
        pulse = ((np.uint32(1) << k1) - 1) << k2
        # A pattern feature is active on creation; a memory feature is triggered on creation
        # and becomes active once its delay has passed.
        act = jnp.where(is_mem, 0.0, 1.0)
        # Pattern or memory is a field of the one feature policy, not a choice of policy.
        ft = self.features
        return px.UnitWrite.of(
            (px.ACTIVATION, act), (ACT_PREV, act), (ACT_USED, act),
            (BORN, g.step),
            (TRACK_THR, hp.tenure_track_threshold),
            (MEM_SRC, jnp.where(is_mem, parent, -1)),
            (ft.is_mem, is_mem), (ft.thresh, k0), (ft.pulse, pulse), (ft.pulse_pattern, pulse),
        )  # fmt: skip


@px.policy
class ImprintWiring(px.policies.ScoreAddConnPolicy):
    """Wire this step's new units to what they imprint on.

    Every eligible feature into a new pattern unit; the chosen input into a new memory unit.
    """

    candidates = px.Candidates.SHORTLIST
    selection = px.Selection.ALL
    trigger = px.Trigger.ON_UNITS_ADDED
    shortlist_size = SHORTLIST_SIZE
    is_mem: px.UnitField[np.bool_]  # the feature policy's field, passed in
    # Every connection this rule creates has the network's connection policies (imprint). It
    # may connect any two units; one into a unit that does not reduce an `ImprintAcc` is
    # refused at run time and reported.

    def importance(self, u: px.UnitView, i: px.UnitIdx, g: Globals) -> Scalar:
        new = u[BORN, i] == g.step
        return jnp.where(new, 2.0, jnp.where(eligible(u, i, g), 1.0, 0.0))

    def score(self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: Globals) -> Scalar:
        new = u[BORN, dst] == g.step
        is_mem = u[self.is_mem, dst]
        to_pattern = new & ~is_mem & eligible(u, src, g)
        to_memory = new & is_mem & (src == u[MEM_SRC, dst])
        return jnp.where(to_pattern | to_memory, 0.0, -jnp.inf)

    def init(self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: Globals) -> px.ConnWrite:
        # Its identity weight is set here.
        return px.ConnWrite.of((px.WEIGHT, 1.0))


# --------------------------------------------------------------------------- #
# Topology: what exists before the first step
# --------------------------------------------------------------------------- #
class Imprinting(px.layers.Module):
    def setup(self) -> None:
        # A connection from every feature unit to the readout, now and whenever a feature is
        # created. Pruning one of those connections leaves a ghost and is never undone, as
        # the thesis specifies.
        #
        # The layer is given its policies under the network's keywords. Its `conn` halves go
        # to the prediction connections and its `unit` halves to the readout. `None` is said
        # where the readout must not take the network's default: nothing leaves the readout,
        # so it has no backward half, and it must never be pruned as an orphan feature.
        self.readout = px.layers.ReduceAll(
            1,
            init=px.init.zeros(),
            forward=px.policies.Forward(conn=PredictionForward(), unit=ReadoutForward()),
            backward=px.policies.Backward(conn=PredictionBackward(), unit=None),
            update=px.policies.Update(conn=_swifttd, unit=RememberPrediction()),
            prune=px.policies.Prune(conn=IdleFeaturePrune(dropped=_swifttd.dropped), unit=None),
            grow=None,
            # GAP: the other fields of a prediction connection the layer creates inside the
            # compiled step (one per new feature). A function of the globals, because the
            # initial step size is a hyperparameter. It was `init=` on the connection set;
            # where it is written now is not settled.
            field_init=prediction_init,
        )

    def __call__(self, obs: px.layers.Units, gamma: px.layers.Units) -> px.layers.Units:
        # One argument per declared input, matched by name: `px.Input("obs")` arrives as
        # `obs`. Each is a handle to that input's units. gamma is a unit and not a global
        # because global state is not to be mutable.
        #
        # The match between the declared inputs and this signature is checked when the
        # network is built (names and count). The type checker sees only `px.layers.Units`
        # for each; it does not know the names or the shapes. Accepted as a limit of
        # Python's type system.
        #
        # The network's unit policies are a generated feature's, so the inputs say where they
        # differ. An input runs no forward half. An observation keeps the network's backward
        # half (`MirrorReaders`), rolls its own way and is never pruned. The control unit is
        # an input and nothing else.
        obs = obs.with_policies(update=RollFresh(), prune=None)
        gamma.with_policies(backward=None, update=None, prune=None)
        return self.readout(obs).bind(READOUT)


# --------------------------------------------------------------------------- #
# The network
# --------------------------------------------------------------------------- #
def make_learner(num_obs: int) -> type[px.Network]:
    class Learner(px.Network):
        topology = Imprinting()
        inputs = {OBS: (num_obs,), GAMMA: ()}  # the shape of each declared input

        # The network's policies are those of what its growth creates: a generated feature
        # and an imprint connection.
        forward = px.policies.Forward(conn=ImprintForward(), unit=_feature)
        backward = px.policies.Backward(conn=ImprintBackward(), unit=MirrorReaders())
        # An imprint connection is frozen: no `conn` half here or in `prune`.
        update = px.policies.Update(unit=RollLagged())
        prune = px.policies.Prune(unit=OrphanPrune())
        # Network-level growth, free to connect any two units. What it creates has the
        # policies above.
        grow = px.policies.Grow(
            unit=ImprintSpawn(out=READOUT, features=_feature),
            conn=ImprintWiring(is_mem=_feature.is_mem),
        )
        loss = TDErrorLoss(gamma=GAMMA)

        config = px.Config(
            propagation=px.Propagation.PIPELINE,  # every feature reads the previous step
            format=px.Format.COO,  # the network grows every step
            unit_capacity=UNIT_CAPACITY,
            seed=0,
        )

    # The topology and the policies both refer to `GAMMA` and `READOUT`; neither is
    # parameterized on the other. The references are resolved when `init` materializes the
    # network and flattens topology, policies and constants together.
    return Learner


# --------------------------------------------------------------------------- #
# Running it
# --------------------------------------------------------------------------- #
def main(steps: int = 20_000) -> None:
    frames, rewards = load_stream(steps)  # (steps, num_obs) bits, (steps,) rewards
    learner = make_learner(num_obs=frames.shape[1])

    # The numeric hyperparameters enter here, as global state.
    state = learner.init(jax.random.PRNGKey(0), globals=Globals(hp=Params()))
    step = px.make_step(learner)
    for t in range(steps):
        inputs = px.StepInputs(inputs={OBS: frames[t], GAMMA: 0.99}, targets=rewards[t])
        result = step(state, inputs)
        state = result.state
        # The readout's activation is the prediction made at this step.


def load_stream(steps: int) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError("see examples/imprinting_stream.py")


if __name__ == "__main__":
    main()
