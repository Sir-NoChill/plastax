"""Continual Backprop (Dohare et al., Nature 2024) as per-unit-local traits.

CBP continually reinitializes the lowest-**contribution-utility** mature hidden
units, where a unit's utility is

    utility(i) = |activation(i)| * sum_{j in out(i)} |w_ij|

running-averaged and gated by an age threshold so fresh units are not recycled.

**Why this is the sharpest demonstration in the RL plan.** That sum is a
reduction over unit `i`'s *outgoing* edges. The literature writes it as a dense
column norm of the weight matrix because that is the operation a dense framework
offers; on the edge arena it is one `BackwardPass` monoid reduction -- the native
operation, at O(E) rather than O(N^2), with no global state.

CBP changes no topology: `prune_conn` and `add_conn` are both absent. It
reallocates *through* the existing wiring, which is why it needs `update_conn`
and nothing else.

**Two thresholds, both built.**

* **v0 (oracle).** The host reads `cbp/util`, takes a per-level quantile at the
  replacement rate, and broadcasts it into `cbp/thresh` -- an O(N) round-trip per
  churn. Obviously correct, and the reference v1 is scored against.
* **v1 (local, two-hop).** Each destination reduces the utilities of its sources
  over its *incoming* edges into a fan-in mean; each source then reduces those
  destination means over its *outgoing* edges and compares its own utility
  against that neighbourhood average. Pure local, two monoid reductions, zero
  global state -- the same move that turned SET's global top-k into a per-unit
  half-normal quantile.

Deviations from the paper, recorded rather than hidden:

* The paper's utility uses batch activation statistics; at batch 1 it is an EMA
  over steps, so `eta` is a real hyperparameter and is swept.
* An edge whose BOTH endpoints reset is reinitialized by the incoming pass and
  then zeroed by the outgoing pass, because `update_conn` runs incoming first.
  Zero wins, which is the paper's intent -- a reset unit must not perturb the
  network's current function through its outputs.
* The forward pass reads `cbp/util` values written by the PREVIOUS churn's
  backward pass, so the two-hop threshold carries a one-churn lag.

Run:  uv run python examples/cbp.py
"""

from __future__ import annotations

import math
import time

import jax
import jax.numpy as jnp
import numpy as np
from dst_sparse import _hash01
from mlp_xor import GradPreAct, LossGrad, MSELoss
from nonstationary import (
    ACT_EMA,
    IS_OUT,
    CycleRecord,
    DriftingTask,
    ReluBackward,
    ReluForward,
    build_dense_mlp,
    growth_slope,
    mark_outputs,
    recovery_times,
)

import plastax as px

# Running average of contribution utility.
CBP_UTIL = px.FieldSpec.float32("cbp/util")
# Churns since last reset; gates replacement so a fresh unit can earn utility.
CBP_AGE = px.FieldSpec.int32("cbp/age")
# 1.0 for units selected for reinitialization this churn.
CBP_RESET = px.FieldSpec.float32("cbp/reset")
# The bar a utility must clear: host-written in v0, two-hop local in v1.
# Always written back so the host can score one against the other.
CBP_THRESH = px.FieldSpec.float32("cbp/thresh")
# Mean utility of a unit's fan-in sources -- the first hop of the local rule.
CBP_FANIN_UTIL = px.FieldSpec.float32("cbp/fanin_util")
# Running average of activation. The utility is mean-corrected (|h - f_hat|),
# because a large but CONSTANT activation carries no information downstream.
CBP_ACT_AVG = px.FieldSpec.float32("cbp/act_avg")
# sum |w_in|, the paper's adaptation denominator: heavy incoming weights make a
# unit harder to repurpose. An INCOMING reduction, where sum|w_out| is outgoing.
CBP_IN_ABS = px.FieldSpec.float32("cbp/in_abs")
# Bias-corrected utility: written by the trace so host v0 selection and the
# in-trace reset test rank by the same quantity.
CBP_UTIL_HAT = px.FieldSpec.float32("cbp/util_hat")
# Live incoming-edge count, used to scale reinitialized weights.
CBP_FANIN = px.FieldSpec.float32("cbp/fanin")
# Monotonic churn counter, the salt for reinit draws. CBP_AGE cannot serve: it
# is zeroed on every reset, so every reset would hash identically.
CBP_CURSOR = px.FieldSpec.int32("cbp/cursor")

_UNIT_FIELDS = (
    GradPreAct,
    LossGrad,
    ACT_EMA,
    IS_OUT,
    CBP_UTIL,
    CBP_AGE,
    CBP_RESET,
    CBP_THRESH,
    CBP_FANIN_UTIL,
    CBP_ACT_AVG,
    CBP_IN_ABS,
    CBP_UTIL_HAT,
    CBP_FANIN,
    CBP_CURSOR,
)


class CbpFanInUtility(px.ForwardPass):
    """Hop one of the local rule: mean utility of each unit's fan-in sources.

    Reduces `(utility(src), 1)` over a unit's incoming edges with a product
    monoid and writes the mean, plus the live fan-in count that scales a
    reinitialized weight. Reads utilities written by the previous churn.
    """

    combine = (px.monoid.sum_, px.monoid.sum_, px.monoid.sum_)

    def map(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Contribute (source utility, |w|, 1) for one live incoming edge."""
        del dst, g
        return u[CBP_UTIL_HAT, src], jnp.abs(c[px.WEIGHT, cid]), jnp.float32(1.0)

    def apply(
        self,
        u: px.UnitView,
        i: px.UnitIdx,
        g: None,
        acc: tuple[jax.Array, jax.Array, jax.Array],
    ) -> px.UnitWrite:
        """Write the fan-in mean utility, the adaptation sum, and the fan-in."""
        del u, i, g
        total_util, total_abs_w, count = acc
        return px.UnitWrite.of(
            (CBP_FANIN_UTIL, total_util / jnp.maximum(count, jnp.float32(1.0))),
            (CBP_IN_ABS, total_abs_w),
            (CBP_FANIN, count),
        )


class CbpUtility(px.BackwardPass):
    """Contribution utility over OUTGOING edges, and the reset decision.

    `sum_{j in out(i)} |w_ij|` is a reduction into the SOURCE unit, which is
    exactly what a BackwardPass does: `src` is the accumulator target and `dst`
    is the edge's destination, whose weight this map contributes.
    """

    combine = (px.monoid.sum_, px.monoid.sum_, px.monoid.sum_)

    def __init__(
        self,
        *,
        decay: float = 0.99,
        maturity: int = 5,
        local: bool = False,
        local_scale: float = 0.5,
        eps: float = 1e-8,
    ) -> None:
        """Bind the running-average decay, age gate and threshold mode.

        Args:
            decay: the paper's eta, the DECAY of both running averages (0.99).
                Note this is the retention weight, not the update weight.
            maturity: churns a unit must survive before it may be reset.
            local: use the two-hop local threshold (v1) rather than the
                host-written oracle threshold (v0).
            local_scale: fraction of the neighbourhood mean utility that sets
                the v1 bar.
            eps: floor on the adaptation denominator.
        """
        self.decay = decay
        self.maturity = maturity
        self.local = local
        self.local_scale = local_scale
        self.eps = eps

    def map(
        self,
        u: px.UnitView,
        src: px.UnitIdx,
        dst: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Contribute (|w|, destination's fan-in mean utility, 1) per out-edge."""
        del src, g
        return (
            jnp.abs(c[px.WEIGHT, cid]),
            u[CBP_FANIN_UTIL, dst],
            jnp.float32(1.0),
        )

    def apply(
        self,
        u: px.UnitView,
        i: px.UnitIdx,
        g: None,
        acc: tuple[jax.Array, jax.Array, jax.Array],
    ) -> px.UnitWrite:
        """Update the utility average, age the unit, and decide replacement.

        Implements the paper's full utility rather than the |h| * sum|w_out|
        sketch: the contribution is MEAN-CORRECTED, both running averages are
        bias-corrected against the unit's age the way Adam corrects against its
        step count, and the whole thing is divided by the adaptation term
        sum|w_in| that the forward pass reduced.
        """
        del g
        sum_abs_w_out, sum_neighbour_util, out_count = acc
        decay = jnp.float32(self.decay)
        age = u[CBP_AGE, i]
        next_age = age + jnp.int32(1)
        # Adam-style: an average seeded at 0 stays biased low for ~1/(1-decay)
        # updates, which would rank young units below old ones spuriously.
        correction = jnp.maximum(
            jnp.float32(1.0) - decay ** jnp.maximum(age.astype(jnp.float32), 1.0),
            jnp.float32(self.eps),
        )

        activation = u[px.ACTIVATION, i]
        act_avg_prev = u[CBP_ACT_AVG, i]
        act_avg = decay * act_avg_prev + (jnp.float32(1.0) - decay) * activation
        act_hat = act_avg_prev / correction
        contribution = jnp.abs(activation - act_hat) * sum_abs_w_out
        adaptation = jnp.maximum(u[CBP_IN_ABS, i], jnp.float32(self.eps))
        instantaneous = contribution / adaptation

        utility_prev = u[CBP_UTIL, i]
        utility = decay * utility_prev + (jnp.float32(1.0) - decay) * instantaneous
        utility_hat = utility_prev / correction

        if self.local:
            # Hop two: compare against the downstream neighbourhood average.
            # No outgoing edges (an output) => bar of -1, so never reset.
            threshold = jnp.float32(self.local_scale) * (
                sum_neighbour_util / jnp.maximum(out_count, jnp.float32(1.0))
            )
            threshold = jnp.where(
                out_count > jnp.float32(0.0), threshold, jnp.float32(-1.0)
            )
        else:
            threshold = u[CBP_THRESH, i]

        mature = age > jnp.int32(self.maturity)
        reset = (utility_hat < threshold) & mature
        zero = jnp.float32(0.0)
        return px.UnitWrite.of(
            # The paper resets utility, the activation average and age to zero.
            (CBP_UTIL, jnp.where(reset, zero, utility)),
            (CBP_UTIL_HAT, jnp.where(reset, zero, utility_hat)),
            (CBP_ACT_AVG, jnp.where(reset, zero, act_avg)),
            (CBP_AGE, jnp.where(reset, jnp.int32(0), next_age)),
            (CBP_RESET, jnp.where(reset, jnp.float32(1.0), zero)),
            (CBP_THRESH, threshold),
            (CBP_CURSOR, u[CBP_CURSOR, i] + jnp.int32(1)),
        )


class CbpReinit(px.UpdateConn[None]):
    """Reinitialize a reset unit's incoming edges; zero its outgoing edges.

    Zeroing the outputs is the paper's rule: a replaced unit must not perturb
    the network's current function until it has relearned something. Both passes
    write unconditionally and select with `where`, because a vmapped policy must
    produce the same write structure for every edge.
    """

    def __init__(
        self,
        state_fields: tuple[px.FieldSpec[np.generic], ...],
        *,
        init_scale: float = 1.0,
    ) -> None:
        """Bind the optimizer state columns to clear and the init scale.

        Args:
            state_fields: the optimizer's per-connection state columns. Taken
                from the bundle rather than hardcoded so any optimizer works.
            init_scale: multiplies the 1/sqrt(fan-in) reinit scale.
        """
        self.state_fields = state_fields
        self.init_scale = init_scale

    def incoming(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: None,
    ) -> px.ConnWrite:
        """Draw a fresh weight into a reset destination and clear its state."""
        del g
        reset = u[CBP_RESET, dst] > jnp.float32(0.5)
        scale = jnp.float32(self.init_scale) / jnp.sqrt(
            jnp.maximum(u[CBP_FANIN, dst], jnp.float32(1.0))
        )
        fresh = (_hash01(src, dst, u[CBP_CURSOR, dst]) * 2.0 - 1.0) * scale
        writes: list[tuple[px.FieldSpec[np.generic], jax.Array]] = [
            (px.WEIGHT, jnp.where(reset, fresh, c[px.WEIGHT, cid]))
        ]
        for spec in self.state_fields:
            zero = jnp.zeros((), dtype=spec.dtype)
            writes.append((spec, jnp.where(reset, zero, c[spec, cid])))
        return px.ConnWrite.of(*writes)

    def outgoing(
        self,
        u: px.UnitView,
        src: px.UnitIdx,
        dst: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: None,
    ) -> px.ConnWrite:
        """Zero the outgoing weights of a reset source."""
        del dst, g
        reset = u[CBP_RESET, src] > jnp.float32(0.5)
        return px.ConnWrite.of(
            (px.WEIGHT, jnp.where(reset, jnp.float32(0.0), c[px.WEIGHT, cid]))
        )


def make_net(
    optimizer: px.optim.Optimizer,
    *,
    mode: str,
    decay: float = 0.99,
    maturity: int = 5,
    local: bool = False,
    local_scale: float = 0.5,
    init_scale: float = 1.0,
    ema_decay: float = 0.05,
) -> type[px.Network[None]]:
    """Build the train / churn / eval net for CBP.

    Args:
        optimizer: the plastax.optim bundle.
        mode: ``"train"``, ``"churn"`` or ``"eval"``.
        decay: retention weight of the running averages (the paper's eta,
            0.99); NOT the update weight.
        maturity: churns before a unit may be reset (churn).
        local: use the v1 two-hop threshold instead of the v0 oracle (churn).
        local_scale: fraction of the neighbourhood mean setting the v1 bar.
        init_scale: multiplies the 1/sqrt(fan-in) reinit scale (churn).
        ema_decay: activation-EMA rate for the dormancy diagnostic.

    Returns:
        A Network subclass for the requested mode.

    Raises:
        ValueError: on an unknown mode.
    """
    if mode == "train":

        class _Train(px.Network[None]):
            forward_pass = ReluForward(ema_decay)
            backward_pass = ReluBackward()
            loss = MSELoss()
            update_conn = optimizer.update_conn()
            extra_unit_fields = _UNIT_FIELDS
            extra_conn_fields = optimizer.state_fields
            propagation = px.Propagation.TOPOLOGICAL

        return _Train

    if mode == "eval":

        class _Eval(px.Network[None]):
            forward_pass = ReluForward(ema_decay)
            extra_unit_fields = _UNIT_FIELDS
            extra_conn_fields = optimizer.state_fields
            propagation = px.Propagation.TOPOLOGICAL

        return _Eval

    if mode == "churn":

        class _Churn(px.Network[None]):
            forward_pass = CbpFanInUtility()
            backward_pass = CbpUtility(
                decay=decay,
                maturity=maturity,
                local=local,
                local_scale=local_scale,
            )
            update_conn = CbpReinit(optimizer.state_fields, init_scale=init_scale)
            extra_unit_fields = _UNIT_FIELDS
            extra_conn_fields = optimizer.state_fields
            propagation = px.Propagation.TOPOLOGICAL

        return _Churn

    raise ValueError(f"make_net: unknown mode {mode!r}")


def hidden_ids(static: px.NetworkStatic) -> np.ndarray:
    """Unit ids that CBP may replace: neither inputs nor outputs."""
    excluded = set(static.input_ids) | set(static.output_ids)
    return np.array(
        [i for i in range(static.num_units) if i not in excluded], dtype=np.int32
    )


def set_oracle_threshold(
    static: px.NetworkStatic,
    state: px.NetworkState[None],
    rho: float,
    *,
    maturity: int = 5,
) -> px.NetworkState[None]:
    """v0: mark the lowest-utility mature units per level, at rate `rho`.

    Selection is by RANK, not by a utility value. A value threshold cannot
    express "replace the lowest rho" once the utility distribution has mass at a
    point, and here it always does: every ReLU-dormant unit has |activation| = 0
    and therefore utility exactly 0, so the rho-quantile IS 0 and the strict
    `utility < threshold` test excludes every one of them -- exactly the units
    CBP exists to replace. Measured, that collapsed the replacement rate from
    the requested 6.4 units/churn to 1.3.

    The decision is encoded in `cbp/thresh` rather than in a new column: a
    selected unit gets an unreachable bar (+inf) and everything else gets -1, so
    the traced rule `reset = (utility < threshold) & mature` is unchanged and
    the in-trace maturity gate stays a safety net over the host's own filter.

    Args:
        static: the network's static config.
        state: the state to read utilities and ages from.
        rho: replacement rate -- the fraction of each level's units to reset.
        maturity: churns a unit must survive before it may be selected.

    Returns:
        The state with `cbp/thresh` written.
    """
    utility = np.asarray(state.units[CBP_UTIL_HAT.name])
    ages = np.asarray(state.units[CBP_AGE.name])
    levels = np.asarray(state.units[px.LEVEL.name])
    threshold = np.full(static.num_units, -1.0, dtype=np.float32)
    replaceable = hidden_ids(static)
    for level in np.unique(levels[replaceable]):
        members = replaceable[levels[replaceable] == level]
        mature = members[ages[members] > maturity]
        if mature.size == 0:
            continue
        # rho of the ELIGIBLE (mature) units, per the authors' released code
        # (gnt.py:145). Algorithm 1 in the paper says n_l * rho -- rho of the
        # LAYER -- and the two differ sharply while units are still maturing.
        count = min(mature.size, max(1, int(round(rho * mature.size))))
        chosen = mature[np.argsort(utility[mature], kind="stable")[:count]]
        threshold[chosen] = np.inf
    state.units = {**state.units, CBP_THRESH.name: jnp.asarray(threshold)}
    return state


def reset_ids(state: px.NetworkState[None]) -> set[int]:
    """Unit ids flagged for reinitialization by the most recent churn."""
    flags = np.asarray(state.units[CBP_RESET.name])
    return set(np.flatnonzero(flags > 0.5).tolist())


def jaccard(a: set[int], b: set[int]) -> float | None:
    """Jaccard similarity of two reset sets, or None when BOTH are empty.

    Returning None rather than 1.0 for the empty-empty case matters: churns
    where neither rule fires carry no evidence that the rules agree, and scoring
    them as perfect agreement inflates the gate. Measured here, that alone moved
    the reported similarity from 0.10 to 0.72.
    """
    if not a and not b:
        return None
    return len(a & b) / len(a | b)


def build(
    optimizer: px.optim.Optimizer,
    layers: tuple[int, ...],
    seed: int,
    **net_kwargs: object,
) -> tuple[px.NetworkStatic, px.NetworkState[None], type[px.Network[None]]]:
    """Build a dense CBP net and its marked state."""
    train_net = make_net(optimizer, mode="train")
    static, state = build_dense_mlp(train_net, layers, seed)
    del net_kwargs
    return static, mark_outputs(static, state), train_net


def run(
    *,
    threshold: str = "v0",
    theta: float = math.pi / 4,
    switch_period: int | None = 30,
    d: int = 32,
    hidden_layers: tuple[int, ...] = (128, 128),
    classes: int = 8,
    lr: float = 0.001,
    rho: float = 0.05,
    decay: float = 0.99,
    maturity: int = 5,
    local_scale: float = 0.5,
    churn_period: int = 1,
    steps_per_cycle: int = 100,
    num_cycles: int = 330,
    seed: int = 0,
) -> list[CycleRecord]:
    """Train through teacher rotations with CBP replacing low-utility units.

    Args:
        threshold: ``"v0"`` (host quantile oracle), ``"v1"`` (two-hop local) or
            ``"off"`` (no churn -- the dense control).
        theta: rotation angle per switch, in radians.
        switch_period: cycles between switches, or None for stationary.
        d: input dimensionality.
        hidden_layers: width of each hidden layer; shared by every arm so the
            comparison stays matched (NE needs an interior transition).
        classes: output classes.
        lr: adam learning rate.
        rho: replacement rate for the v0 quantile.
        decay: retention weight of the running averages (the paper's eta).
        maturity: churns before a unit may be reset.
        local_scale: fraction of the neighbourhood mean setting the v1 bar.
        churn_period: cycles between replacement events.
        steps_per_cycle: training examples per cycle.
        num_cycles: cycles to run.
        seed: PRNG seed.

    Returns:
        One CycleRecord per cycle, reusing the Stage 0 record so CBP and the
        dense baseline are scored by the same `recovery_times`.

    Raises:
        ValueError: on an unknown threshold mode.
    """
    if threshold not in ("v0", "v1", "off"):
        raise ValueError(f"run: unknown threshold {threshold!r}")
    layers = (d + 1, *hidden_layers, classes)
    optimizer = px.optim.adam(lr, GradPreAct)
    static, state, train_net = build(optimizer, layers, seed)
    train_step = px.make_step(train_net, static)
    churn_step = None
    if threshold != "off":
        churn_net = make_net(
            optimizer,
            mode="churn",
            decay=decay,
            maturity=maturity,
            local=(threshold == "v1"),
            local_scale=local_scale,
        )
        churn_step = px.make_step(churn_net, static)

    task = DriftingTask(d, classes, theta=theta, switch_period=switch_period, seed=seed)
    eye = np.eye(classes, dtype=np.float32)
    hidden_units = np.arange(layers[0], sum(layers) - classes)
    out_ids = np.asarray(static.output_ids)
    dense_edges = sum(a * b for a, b in zip(layers[:-1], layers[1:], strict=True))
    records: list[CycleRecord] = []
    last_inputs = jnp.zeros((layers[0],), dtype=jnp.float32)

    for cycle in range(num_cycles):
        task.advance(cycle)
        started = time.perf_counter()
        total_loss, correct = 0.0, 0
        for _ in range(steps_per_cycle):
            inputs, label = task.sample()
            result = train_step(
                state, px.StepInputs(inputs=inputs, targets=jnp.asarray(eye[label]))
            )
            state = result.state
            total_loss += float(result.loss)
            preds = np.asarray(state.units[px.ACTIVATION.name])[out_ids]
            correct += int(np.argmax(preds) == label)
            last_inputs = inputs
        if churn_step is not None and cycle % churn_period == 0:
            if threshold == "v0":
                state = set_oracle_threshold(static, state, rho)
            state = churn_step(
                state, px.StepInputs(inputs=last_inputs, targets=None)
            ).state
        ema = np.asarray(state.units[ACT_EMA.name])[hidden_units]
        live = int(px.state.live_conn_count(state))
        records.append(
            CycleRecord(
                cycle=cycle,
                loss=total_loss / steps_per_cycle,
                accuracy=correct / steps_per_cycle,
                live_edges=live,
                density=live / dense_edges,
                dormant=float(np.mean(ema < 0.01)),
                mean_abs_w=0.0,
                switched=task.switched,
                seconds=time.perf_counter() - started,
            )
        )
    return records


def jaccard_gate(
    *,
    d: int = 32,
    hidden_layers: tuple[int, ...] = (128, 128),
    classes: int = 8,
    rho: float = 0.05,
    decay: float = 0.99,
    maturity: int = 5,
    local_scale: float = 0.5,
    theta: float = math.pi / 4,
    switch_period: int = 30,
    steps_per_cycle: int = 100,
    num_cycles: int = 120,
    seed: int = 0,
) -> tuple[list[float], list[tuple[int, int]]]:
    """Score the v1 local threshold against the v0 oracle, churn by churn.

    Both thresholds are applied to the SAME state at every churn -- the
    trajectory is advanced by v0 so the comparison never drifts into two
    different networks. Returns one Jaccard similarity per churn; the plan's
    gate is a mean of at least 0.9.

    Args:
        d: input dimensionality.
        hidden_layers: width of each hidden layer.
        classes: output classes.
        rho: replacement rate for the v0 quantile.
        decay: retention weight of the running averages (the paper's eta).
        maturity: churns before a unit may be reset.
        local_scale: fraction of the neighbourhood mean setting the v1 bar.
        theta: rotation angle per switch.
        switch_period: cycles between switches.
        steps_per_cycle: training examples per cycle.
        num_cycles: cycles to run.
        seed: PRNG seed.

    Returns:
        The per-churn Jaccard similarities (churns where neither rule fired are
        omitted, not scored as agreement) and the per-churn (v0, v1) reset-set
        sizes, which distinguish a rate mismatch from a ranking disagreement.
    """
    layers = (d + 1, *hidden_layers, classes)
    optimizer = px.optim.adam(0.001, GradPreAct)
    static, state, train_net = build(optimizer, layers, seed)
    train_step = px.make_step(train_net, static)
    common = {"decay": decay, "maturity": maturity, "local_scale": local_scale}
    step_v0 = px.make_step(
        make_net(optimizer, mode="churn", local=False, **common), static
    )
    step_v1 = px.make_step(
        make_net(optimizer, mode="churn", local=True, **common), static
    )

    task = DriftingTask(d, classes, theta=theta, switch_period=switch_period, seed=seed)
    eye = np.eye(classes, dtype=np.float32)
    scores: list[float] = []
    sizes: list[tuple[int, int]] = []
    last_inputs = jnp.zeros((layers[0],), dtype=jnp.float32)

    for cycle in range(num_cycles):
        task.advance(cycle)
        for _ in range(steps_per_cycle):
            inputs, label = task.sample()
            state = train_step(
                state, px.StepInputs(inputs=inputs, targets=jnp.asarray(eye[label]))
            ).state
            last_inputs = inputs
        churn_inputs = px.StepInputs(inputs=last_inputs, targets=None)
        # make_step donates its state, so each branch needs its own copy.
        oracle = set_oracle_threshold(static, jax.tree.map(jnp.copy, state), rho)
        oracle = step_v0(oracle, churn_inputs).state
        local = step_v1(jax.tree.map(jnp.copy, state), churn_inputs).state
        ids0, ids1 = reset_ids(oracle), reset_ids(local)
        score = jaccard(ids0, ids1)
        if score is not None:
            scores.append(score)
        sizes.append((len(ids0), len(ids1)))
        state = oracle
    return scores, sizes


def main() -> None:
    """Report the v0/v1 gate and CBP's effect on recovery-time growth."""
    print("CBP -- contribution utility as an outgoing-edge monoid reduction")
    print("=" * 70)
    scores, sizes = jaccard_gate(num_cycles=90)
    settled = scores[10:]
    n0 = float(np.mean([a for a, _ in sizes[10:]]))
    n1 = float(np.mean([b for _, b in sizes[10:]]))
    mean_score = float(np.mean(settled)) if settled else float("nan")
    print(
        f"v0/v1 reset-set Jaccard over {len(settled)} scored churns: "
        f"mean {mean_score:.3f}  "
        f"{'PASS' if mean_score >= 0.9 else 'FAILS the 0.9 gate'}"
    )
    print(
        f"  resets/churn: v0 {n0:.1f} (rank-selected, rate-controlled)  "
        f"v1 {n1:.1f} (uncontrolled)"
    )
    print()
    print("recovery-time growth at theta=pi/4 (lower slope = more plastic):")
    for label, mode in (("dense (no CBP)", "off"), ("CBP v0", "v0"), ("CBP v1", "v1")):
        records = run(threshold=mode, theta=math.pi / 4, num_cycles=330)
        times = recovery_times(records)
        final = float(np.mean([r.accuracy for r in records[-5:]]))
        print(
            f"  {label:15s} n={len(times):2d}  mean {np.mean(times):5.1f}"
            f"  slope {growth_slope(times):+6.2f}/switch  acc {final:.3f}"
        )


if __name__ == "__main__":
    main()
