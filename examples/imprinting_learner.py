"""The imprinting learner (Javed 2025, thesis chapter 9) as plastax traits.

A SwiftTD linear predictor (see `swifttd.py`) over a feature vector that
grows and shrinks: *pattern* features that recognize the configuration of
tenured features they imprinted on, and *memory* features that relay one
tenured feature into the future for `k1` steps after a delay of `k2`.
Features are generated every step the correction ratio leaves room under
the overshoot bound, and removed once their prediction weight is idle and
their trace has faded. A removed feature that still feeds other features
stays as a *ghost*: no prediction weight, still part of the state update.

One conceptual point: everything the thesis describes as operations on a
feature vector is a unit- or edge-local rule here. The feature vector is the
set of live edges into the output (their SwiftTD state lives on the edge);
tenure is the magnitude of that edge's weight, mirrored onto the unit by the
backward pass; generation is `add_unit` (an active tenured unit spawns a
child, thinned to the step's budget) plus one shortlist growth step that
wires the children; removal is a `prune_conn` on the output edge and a
`prune_unit` for units nothing reads any more, which is exactly the ghost
rule. No global state beyond a step counter; every draw is a counter-based
site keyed by `(seed, step, parent)`, so plastax-cpp generates the same
features.

Run:  uv run python examples/imprinting_learner.py --steps 20000
      uv run python examples/imprinting_learner.py --stream stream.npz --csv run.csv
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import plastax as px
from plastax.rng import Rng

sys.path.insert(0, str(Path(__file__).resolve().parent))
import imprinting_stream as ims  # noqa: E402
from swifttd import (  # noqa: E402
    ACT_PREV,
    ACT_USED,
    ACT_USED_PREV,
    B_PREV,
    BETA,
    DELTA,
    DW,
    KIND,
    KIND_CONTROL,
    KIND_MEMORY,
    KIND_OBSERVATION,
    KIND_OUTPUT,
    KIND_PATTERN,
    N_ELIG,
    OUTPUT_ACC,
    SWIFTTD_CONN_FIELDS,
    SWIFTTD_UNIT_FIELDS,
    TAU_HAT,
    TAU_PREV,
    VD_PREV,
    ZBAR,
    P,
    RecordActivations,
    SwiftTDParams,
    TDErrorLoss,
    Z,
    output_edge_terms,
    output_values,
    swifttd_edge_values,
)

# --- hyperparameters -------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ImprintingParams:
    """The imprinting learner's hyperparameters (thesis section 9.5 values).

    Attributes:
        swifttd: the predictor's hyperparameters. The defaults differ from the
            thesis's audio values (`alpha_init = 3^-3`, which with 8 active
            bits keeps `tau` above `eta` from the first step, so generation
            only starts after tens of thousands of steps of step-size decay)
            and use `lambda = 0.95`, whose trace reaches across the stream's
            reward delay.
        tenure_threshold: |w| at or above which a feature is tenured.
        tenure_track_threshold: |w| below which a generated feature is idle
            (observation features never are). The thesis prints 3^-4, above
            its tenure threshold; 1e-3 keeps the tenure-track band.
        epsilon_z: a feature is removed when idle and `z < e^beta epsilon_z`.
        k_mem: memory features generated per step, in expectation, at most.
        k_pat: pattern features generated per step, in expectation, at most.
        k0: the pattern thresholds a generation draws from, as fractions.
        k1: inclusive range of a memory feature's active length.
        k2: inclusive range of a memory feature's delay.
        shortlist_size: the growth shortlist; must cover the step's new
            units, its active tenured units and the output.
        unique_patterns: the thesis's "no identical feature" rule for
            pattern features, approximated locally: only the lowest-id
            eligible unit spawns pattern children (one per step), and a child
            is vetoed when a live pattern unit already imprinted on the same
            eligible set (a signature kept per unit) with the same `k0`.
        unit_capacity: unit slots.
        seed: the network seed keying every draw.
    """

    swifttd: SwiftTDParams = SwiftTDParams(lambda_=0.95, alpha_init=0.01)
    tenure_threshold: float = 0.01
    tenure_track_threshold: float = 1e-3
    epsilon_z: float = 0.01
    k_mem: int = 10
    k_pat: int = 10
    k0: tuple[float, ...] = (0.6, 0.7, 0.8, 0.9)
    k1: tuple[int, int] = (1, 3)
    k2: tuple[int, int] = (0, 20)
    shortlist_size: int = 256
    unit_capacity: int = 4096
    seed: int = 0
    unique_patterns: bool = False


# --- columns ---------------------------------------------------------------

# Mirrors written by the backward pass (one step stale): the unit's
# prediction weight and its live out-degree.
W_OUT = px.FieldSpec.float32("imprint/w_out")
OUT_DEG = px.FieldSpec.float32("imprint/out_deg")
# Pattern: the fraction of inputs that must be active.
THRESH = px.FieldSpec.float32("imprint/thresh")
# Memory: the shift register (bit 0 = active now) and the trigger pattern
# `((1 << k1) - 1) << k2`.
PULSE = px.FieldSpec.field("imprint/pulse", np.uint32, np.uint32(0))
PULSE_PATTERN = px.FieldSpec.field("imprint/pulse_pattern", np.uint32, np.uint32(0))
# Memory: the single input unit; -1 otherwise.
MEM_SRC = px.FieldSpec.int32("imprint/mem_src", default=-1)
# This unit's tenure-track threshold (0 for observation units).
TRACK_THR = px.FieldSpec.float32("imprint/track_thr")
# The step the unit was generated on; -1 for built units.
BORN = px.FieldSpec.int32("imprint/born", default=-1)
# Output only: the deepest level among this step's eligible (active tenured)
# features. A child spawns one level below it, so every edge growth commits
# runs deeper -- no edge is ever stale, and no resort runs. Levels carry no
# other meaning here: generated features read their inputs' previous
# activations whatever their level, and the output sits at `OUTPUT_LEVEL`.
MAX_ELIG_LEVEL = px.FieldSpec.int32("imprint/max_elig_level")

MAX_LEVELS = 1 << 20
OUTPUT_LEVEL = MAX_LEVELS - 1

# The uniqueness rule (`unique_patterns`). Output only: the lowest eligible
# unit id, a signature of the eligible set (sum of a hash of its ids), that
# signature one and two steps back, and four flags: a live pattern unit
# with the previous step's signature exists for each `k0`. Pattern units:
# the signature at birth and the index of their `k0`.
MIN_ELIG_ID = px.FieldSpec.int32("imprint/min_elig_id")
ELIG_SIG = px.FieldSpec.int32("imprint/elig_sig")
ELIG_SIG_PREV = px.FieldSpec.int32("imprint/elig_sig_prev")
ELIG_SIG_PREV2 = px.FieldSpec.int32("imprint/elig_sig_prev2")
DUP_FLAGS = tuple(px.FieldSpec.int32(f"imprint/dup{k}") for k in range(4))
SIG = px.FieldSpec.int32("imprint/sig")
K0_IDX = px.FieldSpec.int32("imprint/k0_idx", default=-1)
_NO_UNIT = 1 << 30


def _id_hash(i: jax.Array) -> jax.Array:
    """A 32-bit multiplicative hash of a unit id, as int32 (sums wrap)."""
    word = i.astype(jnp.uint32) * jnp.uint32(2654435761)
    return jax.lax.bitcast_convert_type(word, jnp.int32)


# Edge into the output: whether the feature is in SwiftTD's eligible set (its
# trace is live: it was active since the trace last faded), and whether this
# step dropped it from that set. The thesis's learner runs on the sparse
# SwiftTD, which only visits features with a live trace: a feature's trace
# fades when `z < e^beta epsilon_z`, at which point the trace state is zeroed
# and, if the feature is idle, the feature is removed. A feature that was
# never active has no trace and is never a removal candidate.
ELIGIBLE = px.FieldSpec.float32("imprint/eligible")
DROPPED = px.FieldSpec.float32("imprint/dropped")
IMPRINT_CONN_FIELDS = (*SWIFTTD_CONN_FIELDS, ELIGIBLE, DROPPED)

IMPRINT_UNIT_FIELDS = (
    *SWIFTTD_UNIT_FIELDS,
    W_OUT,
    OUT_DEG,
    THRESH,
    PULSE,
    PULSE_PATTERN,
    MEM_SRC,
    TRACK_THR,
    BORN,
    MAX_ELIG_LEVEL,
    MIN_ELIG_ID,
    ELIG_SIG,
    ELIG_SIG_PREV,
    ELIG_SIG_PREV2,
    *DUP_FLAGS,
    SIG,
    K0_IDX,
)

# Draw streams (stream 1 is the framework's growth stream).
_STREAM_SPAWN = 2

Globals = dict[str, jax.Array]


def _generated(u: px.UnitView, i: px.UnitIdx) -> jax.Array:
    kind = u[KIND, i]
    return (kind == KIND_PATTERN) | (kind == KIND_MEMORY)


def _eligible(u: px.UnitView, i: px.UnitIdx, tenure_threshold: float) -> jax.Array:
    """An active tenured feature: what a new feature may imprint on."""
    kind = u[KIND, i]
    feature = (kind == KIND_OBSERVATION) | _generated(u, i)
    return (
        feature
        & (u[px.ACTIVATION, i] > 0.0)
        & (jnp.abs(u[W_OUT, i]) >= jnp.float32(tenure_threshold))
    )


# --- the policies ------------------------------------------------------------


class ImprintingForward(px.ForwardPass):
    """One sweep: the output's prediction and lagged sums; the features' values.

    Into the output, each edge contributes the SwiftTD terms. Into a
    generated unit, each edge contributes its source's *previous* activation
    and a count: a pattern unit fires when at least `thresh` of its inputs
    were active, a memory unit shifts its register and re-triggers it when
    its input was active. Generated features are therefore functions of the
    previous feature vector, as the thesis defines them.
    """

    combine = (
        *OUTPUT_ACC,
        px.monoid.sum_,
        px.monoid.sum_,
        px.monoid.max_,
        px.monoid.min_,
        px.monoid.sum_,
        *(px.monoid.max_ for _ in range(4)),
    )

    def __init__(self, hp: ImprintingParams) -> None:
        self.hp = hp

    def map(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: Globals,
    ) -> tuple[jax.Array, ...]:
        del g
        is_out = (u[KIND, dst] == KIND_OUTPUT).astype(jnp.float32)
        terms = output_edge_terms(u, src, c, cid, self.hp.tenure_threshold)
        elig = terms[5] * is_out > 0.0
        elig_level = jnp.where(elig, u[px.LEVEL, src], 0)
        same_sig = (
            (is_out > 0.0)
            & (u[KIND, src] == KIND_PATTERN)
            & (u[SIG, src] == u[ELIG_SIG_PREV, dst])
        )
        return (
            *(t * is_out for t in terms),
            (1.0 - is_out) * u[ACT_PREV, src],
            1.0 - is_out,
            elig_level.astype(jnp.int32),
            jnp.where(elig, src, _NO_UNIT).astype(jnp.int32),
            jnp.where(elig, _id_hash(src), 0).astype(jnp.int32),
            *((same_sig & (u[K0_IDX, src] == k)).astype(jnp.int32) for k in range(4)),
        )

    def apply(
        self, u: px.UnitView, i: px.UnitIdx, g: Globals, acc: tuple[jax.Array, ...]
    ) -> px.UnitWrite:
        del g
        kind = u[KIND, i]
        is_out = kind == KIND_OUTPUT
        is_pat = kind == KIND_PATTERN
        is_mem = kind == KIND_MEMORY
        summed, count = acc[6], acc[7]
        pattern_act = ((count > 0.0) & (summed >= u[THRESH, i] * count)).astype(
            jnp.float32
        )
        pulse = u[PULSE, i]
        memory_act = (pulse & jnp.uint32(1)).astype(jnp.float32)
        triggered = summed > 0.0
        pulse_next = (pulse >> jnp.uint32(1)) | jnp.where(
            triggered, u[PULSE_PATTERN, i], jnp.uint32(0)
        )
        # Every unit writes every column it might own; a column that is not
        # this unit's keeps its value.
        written = output_values(acc[:6])
        activation = jnp.where(
            is_out,
            written[0][1],
            jnp.where(
                is_pat, pattern_act, jnp.where(is_mem, memory_act, u[px.ACTIVATION, i])
            ),
        )
        return px.UnitWrite.of(
            (px.ACTIVATION, activation),
            (PULSE, jnp.where(is_mem, pulse_next, pulse)),
            (MAX_ELIG_LEVEL, jnp.where(is_out, acc[8], u[MAX_ELIG_LEVEL, i])),
            (MIN_ELIG_ID, jnp.where(is_out, acc[9], u[MIN_ELIG_ID, i])),
            (ELIG_SIG, jnp.where(is_out, acc[10], u[ELIG_SIG, i])),
            *(
                (DUP_FLAGS[k], jnp.where(is_out, acc[11 + k], u[DUP_FLAGS[k], i]))
                for k in range(4)
            ),
            *(
                (spec, jnp.where(is_out, value, u[spec, i]))
                for spec, value in written[1:]
            ),
        )


class OutDegreeBackward(px.BackwardPass):
    """Mirror each unit's prediction weight and live out-degree onto the unit."""

    combine = (px.monoid.sum_, px.monoid.sum_)

    def map(
        self,
        u: px.UnitView,
        src: px.UnitIdx,
        dst: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: Globals,
    ) -> tuple[jax.Array, jax.Array]:
        del src, g
        is_out = u[KIND, dst] == KIND_OUTPUT
        return jnp.where(is_out, c[px.WEIGHT, cid], 0.0), jnp.float32(1.0)

    def apply(
        self,
        u: px.UnitView,
        i: px.UnitIdx,
        g: Globals,
        acc: tuple[jax.Array, jax.Array],
    ) -> px.UnitWrite:
        del u, i, g
        return px.UnitWrite.of((W_OUT, acc[0]), (OUT_DEG, acc[1]))


class RollSignatures(px.UpdateUnit):
    """`RecordActivations` plus the eligible-set signature history roll."""

    def __init__(self) -> None:
        self.base = RecordActivations()

    def update(self, u: px.UnitView, i: px.UnitIdx, g: Globals) -> px.UnitWrite:
        base = self.base.update(u, i, g)
        return px.UnitWrite(
            {
                **base.fields,
                ELIG_SIG_PREV2.name: u[ELIG_SIG_PREV, i],
                ELIG_SIG_PREV.name: u[ELIG_SIG, i],
            }
        )


class ImprintingUpdate(px.UpdateConn):
    """The fused SwiftTD update on the edges into the output; others untouched.

    On top of the fused update, the sparse version's eligible set: a feature
    enters it when active, leaves it (trace state zeroed, `dropped` raised
    for this step's prune) once its decayed `z` falls under
    `e^beta epsilon_z`.
    """

    def __init__(self, hp: ImprintingParams, gamma_unit: int) -> None:
        self.hp = hp
        self.gamma_unit = gamma_unit

    def incoming(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: Globals,
    ) -> px.ConnWrite:
        del g
        is_out = u[KIND, dst] == KIND_OUTPUT
        f = u[ACT_USED_PREV, src]
        values = dict(
            swifttd_edge_values(
                self.hp.swifttd,
                c,
                cid,
                delta=u[DELTA, dst],
                tau=u[TAU_PREV, dst],
                b=u[B_PREV, dst],
                vd=u[VD_PREV, dst],
                f=f,
                gamma_t=u[px.ACTIVATION, px.UnitIdx(jnp.int32(self.gamma_unit))],
            )
        )
        eligible = (c[ELIGIBLE, cid] > 0.0) | (f != 0.0)
        faded = values[Z] < jnp.exp(values[BETA]) * jnp.float32(self.hp.epsilon_z)
        dropped = eligible & faded
        for spec in (Z, P, ZBAR, DW):
            values[spec] = jnp.where(dropped, 0.0, values[spec])
        values[ELIGIBLE] = (eligible & ~dropped).astype(jnp.float32)
        values[DROPPED] = dropped.astype(jnp.float32)
        return px.ConnWrite.of(
            *(
                (spec, jnp.where(is_out, value, c[spec, cid]))
                for spec, value in values.items()
            )
        )

    def outgoing(
        self,
        u: px.UnitView,
        src: px.UnitIdx,
        dst: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: Globals,
    ) -> px.ConnWrite:
        del u, src, dst, c, cid, g
        return px.ConnWrite.of()


class IdleFeaturePrune(px.PruneConn):
    """Remove a generated feature (its output edge) when idle as its trace fades."""

    def predicate(
        self, u: px.UnitView, c: px.ConnView, cid: px.ConnIdx, g: Globals
    ) -> jax.Array:
        del g
        src = px.UnitIdx(c[px.FROM_ID, cid])
        dst = px.UnitIdx(c[px.TO_ID, cid])
        is_out = u[KIND, dst] == KIND_OUTPUT
        idle = jnp.abs(c[px.WEIGHT, cid]) < u[TRACK_THR, src]
        return is_out & _generated(u, src) & idle & (c[DROPPED, cid] > 0.0)


class OrphanPrune(px.PruneUnit):
    """Delete a generated unit nothing reads any more (not a fresh one)."""

    def predicate(self, u: px.UnitView, i: px.UnitIdx, g: Globals) -> jax.Array:
        return _generated(u, i) & (u[OUT_DEG, i] == 0.0) & (u[BORN, i] < g["step"])


class ImprintSpawn(px.AddUnit):
    """Feature generation: active tenured units spawn pattern or memory children.

    The step's budget is the thesis's: at most `k_mem + k_pat` features while
    `tau + alpha_init <= eta`, only if an active tenured feature exists. Each
    eligible parent spawns with probability `budget / n_elig`, so the
    expected count is the budget. The child's kind and parameters come from
    the same draw stream `init` replays.
    """

    def __init__(
        self, hp: ImprintingParams, output_unit: int, network_seed: int
    ) -> None:
        self.hp = hp
        self.output_unit = output_unit
        self.network_seed = network_seed

    def _site(self, parent: px.UnitIdx, g: Globals) -> Rng:
        return Rng.for_site(
            self.network_seed,
            g["step"],
            _STREAM_SPAWN,
            parent.astype(jnp.uint32),
            jnp.uint32(0),
        )

    def _budget(self, u: px.UnitView) -> tuple[jax.Array, jax.Array]:
        out = px.UnitIdx(jnp.int32(self.output_unit))
        hp = self.hp
        room = jnp.floor((hp.swifttd.eta - u[TAU_HAT, out]) / hp.swifttd.alpha_init)
        budget = jnp.minimum(jnp.float32(hp.k_mem + hp.k_pat), room)
        return budget, u[N_ELIG, out]

    def spawn(
        self, u: px.UnitView, parent: px.UnitIdx, g: Globals
    ) -> tuple[jax.Array, jax.Array]:
        budget, n_elig = self._budget(u)
        rng = self._site(parent, g)
        p = jnp.where(n_elig > 0.0, budget / jnp.maximum(n_elig, 1.0), 0.0)
        fire = _eligible(u, parent, self.hp.tenure_threshold) & (budget >= 1.0)
        fire = fire & (rng.uniform() < p)
        out = px.UnitIdx(jnp.int32(self.output_unit))
        if self.hp.unique_patterns:
            is_mem, k0_idx, _, _ = self._child_draws(rng)
            lowest = parent == u[MIN_ELIG_ID, out]
            same_set = u[ELIG_SIG, out] == u[ELIG_SIG_PREV2, out]
            dup = jnp.zeros((), jnp.bool_)
            for k in range(4):
                dup = dup | ((k0_idx == k) & (u[DUP_FLAGS[k], out] > 0))
            fire = fire & (is_mem | (lowest & ~(same_set & dup)))
        # One level below the deepest feature it may imprint on (see
        # MAX_ELIG_LEVEL), so every edge it receives runs deeper.
        child_level = u[MAX_ELIG_LEVEL, out] + jnp.int32(1)
        return fire, child_level - u[px.LEVEL, parent]

    def _child_draws(
        self, rng: Rng
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """The child's draws after the spawn draw: kind, k0 index, k1, k2."""
        hp = self.hp
        p_mem = hp.k_mem / max(hp.k_mem + hp.k_pat, 1)
        is_mem = rng.uniform() < jnp.float32(p_mem)
        k0_idx = rng.uniform_int(len(hp.k0)).astype(jnp.int32)
        k1 = jnp.uint32(hp.k1[0]) + rng.uniform_int(hp.k1[1] - hp.k1[0] + 1)
        k2 = jnp.uint32(hp.k2[0]) + rng.uniform_int(hp.k2[1] - hp.k2[0] + 1)
        return is_mem, k0_idx, k1, k2

    def init(
        self, u: px.UnitView, child: px.UnitIdx, parent: px.UnitIdx, g: Globals
    ) -> px.UnitWrite:
        del child
        hp = self.hp
        rng = self._site(parent, g)
        rng.uniform()  # the spawn draw
        is_mem, k0_idx, k1, k2 = self._child_draws(rng)
        k0 = jnp.asarray(hp.k0, jnp.float32)[k0_idx]
        pattern = ((jnp.uint32(1) << k1) - jnp.uint32(1)) << k2
        out = px.UnitIdx(jnp.int32(self.output_unit))
        kind = jnp.where(is_mem, KIND_MEMORY, KIND_PATTERN).astype(jnp.int32)
        # A pattern feature is active on creation; a memory feature is
        # triggered on creation and active once its delay has passed.
        act = jnp.where(is_mem, 0.0, 1.0).astype(jnp.float32)
        return px.UnitWrite.of(
            (KIND, kind),
            (BORN, g["step"].astype(jnp.int32)),
            (TRACK_THR, jnp.float32(hp.tenure_track_threshold)),
            (THRESH, jnp.where(is_mem, 0.0, k0).astype(jnp.float32)),
            (PULSE_PATTERN, jnp.where(is_mem, pattern, jnp.uint32(0))),
            (PULSE, jnp.where(is_mem, pattern, jnp.uint32(0))),
            (MEM_SRC, jnp.where(is_mem, parent, -1).astype(jnp.int32)),
            (px.ACTIVATION, act),
            (ACT_PREV, act),
            (ACT_USED, act),
            (SIG, jnp.where(is_mem, 0, u[ELIG_SIG, out]).astype(jnp.int32)),
            (K0_IDX, jnp.where(is_mem, -1, k0_idx).astype(jnp.int32)),
        )


class ImprintWiring(px.ScoreAddConn):
    """Wire the step's new units: imprint inputs plus one edge into the output.

    A global shortlist over the units that matter this step -- the output,
    the new units, the active tenured units -- and a score that admits
    exactly the thesis's edges: every eligible unit into a new pattern unit,
    the chosen input into a new memory unit, every new unit into the output.
    """

    candidates = "shortlist"
    selection = "all"
    trigger = "on_units_added"
    # Levels are bookkeeping only (see MAX_ELIG_LEVEL): never veto on the gap.
    max_level_gap = MAX_LEVELS

    def __init__(self, hp: ImprintingParams) -> None:
        self.hp = hp
        self.shortlist_size = hp.shortlist_size

    def importance(self, u: px.UnitView, i: px.UnitIdx, g: Globals) -> jax.Array:
        kind = u[KIND, i]
        new = u[BORN, i] == g["step"]
        return jnp.where(
            kind == KIND_OUTPUT,
            3.0,
            jnp.where(
                new, 2.0, jnp.where(_eligible(u, i, self.hp.tenure_threshold), 1.0, 0.0)
            ),
        ).astype(jnp.float32)

    def score(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: Globals
    ) -> jax.Array:
        step = g["step"]
        new_dst = u[BORN, dst] == step
        kind_dst = u[KIND, dst]
        to_pattern = (
            new_dst
            & (kind_dst == KIND_PATTERN)
            & _eligible(u, src, self.hp.tenure_threshold)
        )
        to_memory = new_dst & (kind_dst == KIND_MEMORY) & (src == u[MEM_SRC, dst])
        to_output = (u[BORN, src] == step) & (kind_dst == KIND_OUTPUT)
        return jnp.where(to_pattern | to_memory | to_output, 0.0, -jnp.inf).astype(
            jnp.float32
        )

    def init(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: Globals
    ) -> px.ConnWrite:
        del src, g
        to_output = u[KIND, dst] == KIND_OUTPUT
        return px.ConnWrite.of(
            (px.WEIGHT, jnp.where(to_output, 0.0, 1.0).astype(jnp.float32)),
            (
                BETA,
                jnp.where(to_output, math.log(self.hp.swifttd.alpha_init), 0.0).astype(
                    jnp.float32
                ),
            ),
        )


class Tick(px.ResetGlobal):
    """Advance the step counter the draw sites key on."""

    def reset(self, g: Globals) -> Globals:
        return {"step": g["step"] + 1}


# --- the network ------------------------------------------------------------


def make_net(hp: ImprintingParams, num_obs: int) -> type[px.Network[Globals]]:
    """The imprinting-learner network class.

    Units: observation `0..num_obs-1`, the control unit `num_obs` (carries
    `gamma_t`), the output `num_obs + 1`; generated units take the free slots
    above.

    Args:
        hp: the hyperparameters, baked into the policies.
        num_obs: observation bits per frame.

    Returns:
        A pipeline-mode Network subclass with every phase of the learner.
    """
    gamma_unit = num_obs
    output_unit = num_obs + 1

    class ImprintingNet(px.Network[Globals]):
        forward_pass = ImprintingForward(hp)
        loss = TDErrorLoss(hp.swifttd, gamma_unit)
        backward_pass = OutDegreeBackward()
        update_unit = RollSignatures()
        update_conn = ImprintingUpdate(hp, gamma_unit)
        prune_unit = OrphanPrune()
        prune_conn = IdleFeaturePrune()
        add_unit = ImprintSpawn(hp, output_unit, hp.seed)
        add_conn = ImprintWiring(hp)
        reset_global = Tick()
        extra_unit_fields = IMPRINT_UNIT_FIELDS
        extra_conn_fields = IMPRINT_CONN_FIELDS
        propagation = px.Propagation.PIPELINE
        unit_capacity = hp.unit_capacity
        max_levels = MAX_LEVELS
        seed = hp.seed

    return ImprintingNet


def build(
    net: type[px.Network[Globals]], num_obs: int, hp: ImprintingParams
) -> tuple[px.NetworkStatic, px.NetworkState[Globals], int]:
    """Build the initial network: observation units wired to the output.

    Args:
        net: the class from `make_net`.
        num_obs: observation bits per frame.
        hp: the hyperparameters (`alpha_init` seeds every `beta`).

    Returns:
        The static config, the state, and the output unit's id.
    """
    b = px.NetworkBuilder(net, {"step": jnp.int32(0)})
    for _ in range(num_obs):
        b.add_unit(**{TRACK_THR.name: 0.0})
    gamma_unit = b.add_unit(**{KIND.name: KIND_CONTROL})
    out = b.add_unit(**{KIND.name: KIND_OUTPUT})
    for i in range(num_obs):
        b.mark_input(i)
        b.add_conn(i, out, **{BETA.name: math.log(hp.swifttd.alpha_init)})
    b.mark_input(gamma_unit)
    b.mark_output(out)
    static, state = b.finalize()
    return static, place_output(state, out), out


def place_output(state: px.NetworkState[Globals], out: int) -> px.NetworkState[Globals]:
    """Move the output to `OUTPUT_LEVEL` so every feature -> output edge runs deeper.

    The builder levels the output at 1 from the observation edges; levels
    are bookkeeping in pipeline mode (see MAX_ELIG_LEVEL).

    Args:
        state: the built state.
        out: the output unit's id.

    Returns:
        The state with the output's level set.
    """
    levels = state.units[px.LEVEL.name].at[out].set(OUTPUT_LEVEL)
    return dataclasses.replace(state, units={**state.units, px.LEVEL.name: levels})


# --- the driver --------------------------------------------------------------


@dataclasses.dataclass
class Counts:
    """Feature counts at one step (every live unit, by kind and status)."""

    step: int
    units: int
    pattern: int
    memory: int
    tenured: int
    tenure_track: int
    edges: int
    seconds: float


def counts(
    driver: px.Driver[Globals], hp: ImprintingParams, step: int, seconds: float
) -> Counts:
    """Read the live feature counts off the state (a host sync)."""
    units = driver.state.units
    live = ~np.asarray(units[px.PRUNED.name])
    kind = np.asarray(units[KIND.name])
    w_out = np.abs(np.asarray(units[W_OUT.name]))
    feature = live & (kind != KIND_OUTPUT) & (kind != KIND_CONTROL)
    tenured = feature & (w_out >= hp.tenure_threshold)
    track = feature & (w_out >= hp.tenure_track_threshold) & ~tenured
    dead = np.asarray(driver.state.conns[0][px.DEAD.name])
    return Counts(
        step=step,
        units=int(live.sum()),
        pattern=int((live & (kind == KIND_PATTERN)).sum()),
        memory=int((live & (kind == KIND_MEMORY)).sum()),
        tenured=int(tenured.sum()),
        tenure_track=int(track.sum()),
        edges=int((~dead).sum()),
        seconds=seconds,
    )


def run(
    hp: ImprintingParams,
    stream: ims.Stream,
    *,
    log_every: int = 500,
    csv_path: Path | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, list[Counts]]:
    """Run the learner over a stream.

    Args:
        hp: the hyperparameters (`hp.swifttd.gamma` is overridden by the
            stream's discount).
        stream: the stream.
        log_every: steps between count read-backs.
        csv_path: where to write `step, reward, prediction` per step and the
            counts (or None).
        verbose: print the counts as they are read.

    Returns:
        The prediction trace and the count log.
    """
    hp = dataclasses.replace(
        hp, swifttd=dataclasses.replace(hp.swifttd, gamma=stream.gamma)
    )
    steps, num_obs = stream.obs.shape
    net = make_net(hp, num_obs)
    static, state, out = build(net, num_obs, hp)
    driver = px.Driver(net, static, state)
    prediction = np.zeros(steps, np.float32)
    gamma = np.float32(stream.gamma)
    log: list[Counts] = []
    t0 = time.perf_counter()
    for t in range(steps):
        driver.step(
            px.StepInputs(
                inputs=jnp.asarray(np.append(stream.obs[t].astype(np.float32), gamma)),
                targets=jnp.asarray([stream.reward[t]], jnp.float32),
            )
        )
        prediction[t] = float(driver.state.units[px.ACTIVATION.name][out])
        if (t + 1) % log_every == 0 or t + 1 == steps:
            c = counts(driver, hp, t + 1, time.perf_counter() - t0)
            log.append(c)
            if verbose:
                print(
                    f"  step {c.step:>6d}  units {c.units:>5d}  "
                    f"pattern {c.pattern:>5d}  memory {c.memory:>5d}  "
                    f"tenured {c.tenured:>5d}  track {c.tenure_track:>5d}  "
                    f"edges {c.edges:>7d}  {c.seconds:7.1f}s"
                )
            if bool(driver.state.unit_overflow):
                print("  unit capacity exhausted: spawns are being dropped")
    if csv_path is not None:
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "reward", "prediction"])
            for t in range(steps):
                w.writerow([t, float(stream.reward[t]), float(prediction[t])])
        with open(csv_path.with_suffix(".counts.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([field.name for field in dataclasses.fields(Counts)])
            for c in log:
                w.writerow(dataclasses.astuple(c))
    return prediction, log


_HP0 = ImprintingParams().swifttd


def _parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--stream", type=Path, default=None, help="a stream .npz; else generated"
    )
    p.add_argument("--steps", type=int, default=20_000, help="generated stream length")
    p.add_argument("--stream-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0, help="network seed")
    p.add_argument("--csv", type=Path, default=None, help="prediction log")
    p.add_argument("--alpha-init", type=float, default=_HP0.alpha_init)
    p.add_argument("--eta", type=float, default=_HP0.eta)
    p.add_argument("--theta", type=float, default=_HP0.theta)
    p.add_argument("--lambda", dest="lambda_", type=float, default=_HP0.lambda_)
    p.add_argument("--tenure", type=float, default=ImprintingParams.tenure_threshold)
    p.add_argument(
        "--track", type=float, default=ImprintingParams.tenure_track_threshold
    )
    p.add_argument("--k-mem", type=int, default=ImprintingParams.k_mem)
    p.add_argument("--k-pat", type=int, default=ImprintingParams.k_pat)
    p.add_argument("--unit-capacity", type=int, default=ImprintingParams.unit_capacity)
    p.add_argument("--unique", action="store_true", help="the pattern uniqueness rule")
    p.add_argument("--shortlist", type=int, default=ImprintingParams.shortlist_size)
    p.add_argument("--log-every", type=int, default=1000)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the learner on a stream and score the GVF shape of its predictions."""
    a = _parse(argv)
    if a.stream is not None:
        stream = ims.load_npz(a.stream)
    else:
        stream = ims.generate(ims.StreamConfig(steps=a.steps, seed=a.stream_seed))
    hp = ImprintingParams(
        swifttd=SwiftTDParams(
            alpha_init=a.alpha_init, eta=a.eta, theta=a.theta, lambda_=a.lambda_
        ),
        tenure_threshold=a.tenure,
        tenure_track_threshold=a.track,
        k_mem=a.k_mem,
        k_pat=a.k_pat,
        unit_capacity=a.unit_capacity,
        shortlist_size=a.shortlist,
        seed=a.seed,
        unique_patterns=a.unique,
    )
    print(
        f"imprinting learner: {stream.obs.shape[0]} steps, {stream.obs.shape[1]} bits, "
        f"{stream.events.shape[0]} events, k_mem {hp.k_mem}, k_pat {hp.k_pat}, "
        f"alpha_init {hp.swifttd.alpha_init:.4g}, eta {hp.swifttd.eta:.3g}, "
        f"theta {hp.swifttd.theta:.3g}, lambda {hp.swifttd.lambda_:.3g}, "
        f"tenure {hp.tenure_threshold:.3g}, track {hp.tenure_track_threshold:.3g}"
    )
    prediction, _ = run(hp, stream, log_every=a.log_every, csv_path=a.csv)
    target = ims.learnable_return(stream)
    err = ims.lifetime_error(prediction, target)
    print(f"lifetime error vs the learnable target: {err:.4f}")
    _, summary = ims.shape_metrics(prediction, stream)
    print("GVF shape (last quarter of the stream):\n" + ims.format_summary(summary))


if __name__ == "__main__":
    main()
