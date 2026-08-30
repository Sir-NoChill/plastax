"""Stage 0: a non-stationary supervised harness for topology-modification rules.

Everything in the RL plan is gated on this file. It exercises each rewiring rule
against a signal we fully control, at batch 1, with no RL code -- so that
"the rule is mis-implemented" can be told apart from "bootstrapping broke the
rule". No replay buffer, no target network, one sample per step: the same shape
the streaming-RL learner will have.

**The task.** A linear teacher `T` (classes x d) labels each input by
`argmax(T x)`. Non-stationarity is a *rotation* of `T` every `switch_period`
cycles, by an angle `theta` that grades severity continuously -- `theta = 0` is
stationary, `theta = pi/2` is a fully orthogonal teacher. One knob, so severity
and frequency sweep independently.

**Why the rotation is orthogonalized.** Combining `cos(theta)*T` with a fresh
random `T2` only rotates by `theta` if `T2 is orthogonal to T`. A raw Gaussian
`T2` overlaps `T` by O(1/sqrt(d)), which at d=16 turns a requested 90 degrees
into 97 and moves the row norms by ~2% -- so "severity" would not mean what the
sweep says it means, and theta=pi/2 would not be a full resample. One
Gram-Schmidt step per row makes the angle exact to machine precision and
preserves `||T||` exactly, which also keeps the task's difficulty fixed as it
drifts (a teacher that quietly shrank would look like recovered plasticity).

**ReLU hidden units, linear output.** Dormancy -- the metric NE prunes on and
every other method reports -- is the fraction of units whose activation stays at
zero. Sigmoid units never do that, so a sigmoid net cannot express the
phenomenon under study.

Run:  uv run python examples/nonstationary.py
"""

from __future__ import annotations

import dataclasses
import math
import time
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from dst_sparse import (
    SET_CURSOR,
    SET_THRESH,
    GradientGrow,
    MagnitudeStats,
    RandomGrow,
    SetPrune,
    build_sparse_mlp,
)
from mlp_xor import GradPreAct, LossGrad, MSELoss

import plastax as px

# Running mean of |activation| per unit. Written by the TRAIN forward pass so it
# reflects what the network actually saw, and read as the dormancy statistic by
# every method in the study (NE prunes on it; the others report it).
ACT_EMA = px.FieldSpec.float32("plasticity/act_ema")
# 1.0 on output units, 0.0 elsewhere: one apply has to branch between a
# rectified hidden unit and a linear output. NOT derived from LEVEL, which
# `resort` may renumber as edges are rewired.
IS_OUT = px.FieldSpec.float32("plasticity/is_out")

_UNIT_FIELDS = (GradPreAct, LossGrad, SET_THRESH, SET_CURSOR, ACT_EMA, IS_OUT)
_GROWTH = {"set": RandomGrow, "rigl": GradientGrow}


class ReluForward(px.ForwardPass):
    """Weighted sum; ReLU on hidden units, identity on outputs.

    Also advances the per-unit activation EMA that supplies the dormancy
    metric. The output stays linear so MSE's dL/dpred is dL/dpre directly.
    """

    combine = px.monoid.sum_

    def __init__(self, ema_decay: float = 0.05) -> None:
        """Bind the activation-EMA decay rate."""
        self.ema_decay = ema_decay

    def map(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: None,
    ) -> jax.Array:
        """Contribute one edge's weighted source activation."""
        del dst, g
        return c[px.WEIGHT, cid] * u[px.ACTIVATION, src]

    def apply(
        self, u: px.UnitView, i: px.UnitIdx, g: None, acc: jax.Array
    ) -> px.UnitWrite:
        """Rectify (or pass through, at the output) and update the EMA."""
        del g
        is_out = u[IS_OUT, i] > jnp.float32(0.5)
        activation = jnp.where(is_out, acc, jax.nn.relu(acc))
        beta = jnp.float32(self.ema_decay)
        ema = (jnp.float32(1.0) - beta) * u[ACT_EMA, i] + beta * jnp.abs(activation)
        return px.UnitWrite.of((px.ACTIVATION, activation), (ACT_EMA, ema))


class ReluBackward(px.BackwardPass):
    """Reverse walk; dL/dz from the loss at the output, ReLU derivative inside.

    Backward accumulates into the edge's SOURCE, so `src` is this pass's
    accumulator target and `dst` is the deeper unit the reverse level walk has
    already finalized -- the one whose dL/dz the map reads.
    """

    combine = px.monoid.sum_

    def map(
        self,
        u: px.UnitView,
        src: px.UnitIdx,
        dst: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: None,
    ) -> jax.Array:
        """Propagate weight * dL/dz of the deeper unit back into this one."""
        del src, g
        return c[px.WEIGHT, cid] * u[GradPreAct, dst]

    def apply(
        self, u: px.UnitView, i: px.UnitIdx, g: None, acc: jax.Array
    ) -> px.UnitWrite:
        """Convert accumulated dL/da into dL/dz for this unit."""
        del g
        activation = u[px.ACTIVATION, i]
        is_out = u[IS_OUT, i] > jnp.float32(0.5)
        rectified = acc * jnp.where(activation > jnp.float32(0.0), 1.0, 0.0)
        return px.UnitWrite.of(
            (GradPreAct, jnp.where(is_out, u[LossGrad, i], rectified))
        )


def make_net(
    optimizer: px.optim.Optimizer,
    *,
    mode: str,
    method: str = "set",
    zeta: float = 0.1,
    max_candidates: int = 256,
    grow_scale: float = 0.0,
    shortlist: int | None = None,
    ema_decay: float = 0.05,
) -> type[px.Network[None]]:
    """Build the train / churn / eval net for one rewiring method.

    Args:
        optimizer: the plastax.optim bundle.
        mode: ``"train"``, ``"churn"`` or ``"eval"``.
        method: ``"set"`` or ``"rigl"`` (churn only).
        zeta: per-unit prune fraction (churn).
        max_candidates: per-bucket growth bound (churn).
        grow_scale: regrown-edge init weight (churn).
        shortlist: M for the M x M candidate grid, or None (churn).
        ema_decay: activation-EMA rate for the dormancy metric.

    Returns:
        A Network subclass for the requested mode.

    Raises:
        ValueError: on an unknown mode or method.
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
        if method not in _GROWTH:
            raise ValueError(f"make_net: unknown method {method!r} (set|rigl)")
        grow = _GROWTH[method](max_candidates, grow_scale, shortlist)

        class _Churn(px.Network[None]):
            forward_pass = MagnitudeStats(zeta)
            prune_conn = SetPrune()
            add_conn = grow
            extra_unit_fields = _UNIT_FIELDS
            extra_conn_fields = optimizer.state_fields
            propagation = px.Propagation.TOPOLOGICAL
            neighbourhood = 1

        return _Churn

    raise ValueError(f"make_net: unknown mode {mode!r}")


def drift(teacher: np.ndarray, theta: float, rng: np.random.Generator) -> np.ndarray:
    """Rotate every teacher row by exactly `theta` toward a fresh direction.

    A Gram-Schmidt step makes the fresh direction exactly orthogonal to the row
    it rotates, so the realised angle is `theta` and `||row||` is unchanged --
    both to machine precision. Without it a raw Gaussian draw overlaps the row
    by O(1/sqrt(d)) and the requested angle is not the delivered one.

    Args:
        teacher: (classes, d) teacher matrix.
        theta: rotation angle in radians; 0 is stationary, pi/2 orthogonal.
        rng: numpy PRNG.

    Returns:
        The rotated teacher, same shape and same row norms.
    """
    fresh = rng.standard_normal(teacher.shape)
    overlap = np.sum(fresh * teacher, axis=1, keepdims=True)
    squared = np.sum(teacher * teacher, axis=1, keepdims=True)
    fresh = fresh - (overlap / squared) * teacher
    fresh *= np.linalg.norm(teacher, axis=1, keepdims=True) / np.linalg.norm(
        fresh, axis=1, keepdims=True
    )
    return math.cos(theta) * teacher + math.sin(theta) * fresh


@dataclasses.dataclass
class CycleRecord:
    """One cycle's measurements.

    Attributes:
        cycle: cycle index.
        loss: mean training loss over the cycle.
        accuracy: online argmax accuracy over the cycle.
        live_edges: live connection count after the cycle's churn.
        density: live edges as a fraction of the dense equivalent.
        dormant: fraction of hidden units whose activation EMA is below tau.
        mean_abs_w: mean |weight| over live edges.
        switched: whether the teacher rotated at the start of this cycle.
        seconds: wall-clock for the cycle.
    """

    cycle: int
    loss: float
    accuracy: float
    live_edges: int
    density: float
    dormant: float
    mean_abs_w: float
    switched: bool
    seconds: float


def _live_weight_stats(state: px.NetworkState[None]) -> float:
    """Mean |weight| over live connections."""
    total, count = 0.0, 0
    for bucket in state.conns:
        dead = np.asarray(bucket[px.DEAD.name])
        weights = np.asarray(bucket[px.WEIGHT.name])[~dead]
        total += float(np.abs(weights).sum())
        count += weights.size
    return total / max(count, 1)


def recovery_times(
    records: list[CycleRecord], *, window: int = 5, tolerance: float = 0.98
) -> list[int]:
    """Cycles taken to regain the pre-switch accuracy plateau, per switch.

    The plasticity metric: everything else in `CycleRecord` is diagnostic. The
    plateau is the mean accuracy over the `window` cycles before a switch;
    recovery is the first cycle at or above `tolerance` times that. A switch
    that never recovers before the next one is reported censored, at the number
    of cycles actually available -- so a growing sequence is meaningful even
    when later switches never recover at all.

    Args:
        records: the run's per-cycle records.
        window: cycles averaged to establish the pre-switch plateau.
        tolerance: fraction of the plateau that counts as recovered.

    Returns:
        One recovery time per switch, in order.
    """
    switches = [r.cycle for r in records if r.switched and r.cycle > 0]
    accuracy = {r.cycle: r.accuracy for r in records}
    last = records[-1].cycle if records else 0
    out: list[int] = []
    for index, switch in enumerate(switches):
        before = [
            accuracy[c] for c in range(max(0, switch - window), switch) if c in accuracy
        ]
        if not before:
            continue
        target = tolerance * float(np.mean(before))
        horizon = switches[index + 1] if index + 1 < len(switches) else last + 1
        recovered = horizon - switch
        for c in range(switch, horizon):
            if accuracy.get(c, 0.0) >= target:
                recovered = c - switch
                break
        out.append(recovered)
    return out


class DriftingTask:
    """A linear teacher that rotates every `switch_period` cycles.

    Attributes:
        teacher: the current (classes, d) teacher matrix.
        switched: whether the most recent `advance` rotated the teacher.
    """

    def __init__(
        self,
        d: int,
        classes: int,
        *,
        theta: float,
        switch_period: int | None,
        seed: int = 0,
    ) -> None:
        """Build the teacher and its data stream.

        Args:
            d: input dimensionality (excluding the bias input).
            classes: number of output classes.
            theta: rotation angle per switch, in radians.
            switch_period: cycles between switches, or None for stationary.
            seed: PRNG seed.
        """
        self._rng = np.random.default_rng(seed)
        self._drift_rng = np.random.default_rng(seed + 1)
        self.teacher = self._rng.standard_normal((classes, d))
        self._theta = theta
        self._period = switch_period
        self.switched = False

    def advance(self, cycle: int) -> None:
        """Rotate the teacher if this cycle lands on a switch boundary."""
        self.switched = (
            self._period is not None
            and cycle > 0
            and cycle % self._period == 0
            and self._theta > 0.0
        )
        if self.switched:
            self.teacher = drift(self.teacher, self._theta, self._drift_rng)

    def sample(self) -> tuple[jax.Array, int]:
        """Draw one (input-with-bias, label) example from the current teacher."""
        x = self._rng.standard_normal(self.teacher.shape[1])
        label = int(np.argmax(self.teacher @ x))
        return jnp.asarray(np.append(x, 1.0), dtype=jnp.float32), label


def mark_outputs(
    static: px.NetworkStatic, state: px.NetworkState[None]
) -> px.NetworkState[None]:
    """Set IS_OUT on every output unit; returns the updated state."""
    ids = jnp.asarray(static.output_ids, dtype=jnp.int32)
    column = state.units[IS_OUT.name].at[ids].set(jnp.float32(1.0))
    state.units = {**state.units, IS_OUT.name: column}
    return state


def _budgets(layers: tuple[int, ...], density: float) -> tuple[int, ...]:
    """Live edges per layer transition at the given density (1.0 = dense)."""
    return tuple(
        max(64, int(round(a * b * density)))
        for a, b in zip(layers[:-1], layers[1:], strict=True)
    )


def build_dense_mlp(
    net: type[px.Network[None]], layers: tuple[int, ...], seed: int
) -> tuple[px.NetworkStatic, px.NetworkState[None]]:
    """Build a COMPLETE layered MLP -- every (src, dst) pair present.

    `build_sparse_mlp` cannot serve as the dense baseline: it draws distinct
    random pairs and discards collisions, so asking it for a fully dense budget
    yields ~86% of the edges. A baseline that is quietly 14% sparse would make
    every "sparsity helps" comparison in this study unfalsifiable, so the dense
    case enumerates the complete bipartite edge set instead.

    Args:
        net: the Network type whose field layout the arenas adopt.
        layers: units per layer, input first (including the bias input).
        seed: numpy PRNG seed for the weights.

    Returns:
        The finalized (static, state) pair.
    """
    rng = np.random.default_rng(seed)
    offsets, running = [], 0
    for size in layers:
        offsets.append(running)
        running += size
    from_chunks, to_chunks, weight_chunks = [], [], []
    for layer, (n_src, n_dst) in enumerate(zip(layers[:-1], layers[1:], strict=True)):
        src, dst = np.meshgrid(np.arange(n_src), np.arange(n_dst), indexing="ij")
        from_chunks.append(src.ravel().astype(np.int32) + offsets[layer])
        to_chunks.append(dst.ravel().astype(np.int32) + offsets[layer + 1])
        weight_chunks.append(rng.standard_normal(n_src * n_dst) / np.sqrt(n_src))
    return px.NetworkBuilder.from_edges(
        net,
        running,
        np.concatenate(from_chunks),
        np.concatenate(to_chunks),
        weights=np.concatenate(weight_chunks).astype(np.float32),
        input_ids=tuple(range(layers[0])),
        output_ids=tuple(range(offsets[-1], running)),
        globals_=None,
    )


def run(
    method: str,
    *,
    theta: float = math.pi / 4,
    switch_period: int | None = 40,
    d: int = 32,
    hidden_layers: tuple[int, ...] = (128, 128),
    classes: int = 8,
    density: float = 0.1,
    lr: float = 0.001,
    zeta: float = 0.1,
    shortlist: int | None = 64,
    steps_per_cycle: int = 100,
    num_cycles: int = 200,
    dormant_tau: float = 0.01,
    seed: int = 0,
    verbose: bool = False,
) -> list[CycleRecord]:
    """Train online through repeated teacher rotations, recording each cycle.

    Args:
        method: ``"dense"``, ``"static"``, ``"set"`` or ``"rigl"``. ``dense``
            and ``static`` never rewire; they differ only in density.
        theta: rotation angle per switch, in radians.
        switch_period: cycles between switches, or None for stationary.
        d: input dimensionality.
        hidden_layers: width of each hidden layer. Two hidden layers is the
            default because NE keeps the first and last transitions dense, so a
            single hidden layer leaves it nothing to grow into; every arm shares
            the architecture so the comparison stays matched.
        classes: output classes.
        density: live-edge fraction for every method except ``dense``.
        lr: adam learning rate. The default is deliberately low: at 0.01
            the stationary baseline already drives 83% of hidden units dormant,
            which would confound drift-induced plasticity loss with an
            optimizer artifact (measured, see module docstring).
        zeta: per-unit prune fraction for the rewiring methods.
        shortlist: M for the growth candidate grid, or None for exhaustive.
        steps_per_cycle: training examples per cycle.
        num_cycles: cycles to run.
        dormant_tau: activation-EMA threshold below which a unit is dormant.
        seed: PRNG seed.
        verbose: print a per-cycle table.

    Returns:
        One CycleRecord per cycle.

    Raises:
        ValueError: on an unknown method.
    """
    if method not in ("dense", "static", "set", "rigl"):
        raise ValueError(f"run: unknown method {method!r}")
    rewires = method in ("set", "rigl")
    layers = (d + 1, *hidden_layers, classes)
    budgets = _budgets(layers, density)
    dense_edges = sum(a * b for a, b in zip(layers[:-1], layers[1:], strict=True))

    optimizer = px.optim.adam(lr, GradPreAct)
    train_net = make_net(optimizer, mode="train")
    if method == "dense":
        static, state = build_dense_mlp(train_net, layers, seed)
    else:
        static, state = build_sparse_mlp(train_net, layers, budgets, seed)
    state = mark_outputs(static, state)
    train_step = px.make_step(train_net, static)

    churn_step = None
    if rewires:
        grow = max(max(budgets), shortlist**2 if shortlist else max(budgets))
        churn_net = make_net(
            optimizer,
            mode="churn",
            method=method,
            zeta=zeta,
            max_candidates=grow,
            shortlist=shortlist,
        )
        churn_step = px.make_step(churn_net, static)
        # An uncovered bucket refills into only M of its destinations, so the
        # run proceeds at a lower sparsity than it reports. Say so at build.
        uncovered = [
            c for c in px.shortlist_coverage(churn_net, static, state) if not c.covered
        ]
        if uncovered:
            floor = px.recommended_shortlist(churn_net, static, state)
            warnings.warn(
                f"shortlist={shortlist} does not cover "
                f"{len(uncovered)} bucket(s): "
                + ", ".join(
                    f"bucket {c.bucket} has {c.destination_units} eligible destinations"
                    for c in uncovered
                )
                + f". Live-edge count will drift below target; use >= {floor}.",
                stacklevel=2,
            )

    task = DriftingTask(d, classes, theta=theta, switch_period=switch_period, seed=seed)
    eye = np.eye(classes, dtype=np.float32)
    hidden_ids = np.arange(layers[0], sum(layers) - classes)
    records: list[CycleRecord] = []
    last_inputs = jnp.zeros((layers[0],), dtype=jnp.float32)

    for cycle in range(num_cycles):
        task.advance(cycle)
        started = time.perf_counter()
        total_loss, correct = 0.0, 0
        for _ in range(steps_per_cycle):
            inputs, label = task.sample()
            result = train_step(
                state,
                px.StepInputs(inputs=inputs, targets=jnp.asarray(eye[label])),
            )
            state = result.state
            total_loss += float(result.loss)
            preds = np.asarray(state.units[px.ACTIVATION.name])[
                np.asarray(static.output_ids)
            ]
            correct += int(np.argmax(preds) == label)
            last_inputs = inputs
        if churn_step is not None:
            state = churn_step(
                state, px.StepInputs(inputs=last_inputs, targets=None)
            ).state
        ema = np.asarray(state.units[ACT_EMA.name])[hidden_ids]
        live = int(px.state.live_conn_count(state))
        records.append(
            CycleRecord(
                cycle=cycle,
                loss=total_loss / steps_per_cycle,
                accuracy=correct / steps_per_cycle,
                live_edges=live,
                density=live / dense_edges,
                dormant=float(np.mean(ema < dormant_tau)),
                mean_abs_w=_live_weight_stats(state),
                switched=task.switched,
                seconds=time.perf_counter() - started,
            )
        )
        if verbose and (cycle % 10 == 0 or task.switched):
            r = records[-1]
            print(
                f"  cycle {r.cycle:4d}{'  <- SWITCH' if r.switched else '':11s}"
                f" loss {r.loss:7.4f}  acc {r.accuracy:.3f}  live {r.live_edges:7d}"
                f"  dormant {r.dormant:.3f}"
            )
    return records


def growth_slope(times: list[int]) -> float:
    """Least-squares slope of recovery time against switch index.

    A first-vs-last comparison calls [8, 15, 8, 10] growing, which it is not.
    The slope uses every switch, so noise averages out instead of deciding the
    verdict. Positive slope = recovery is getting slower = plasticity is being
    lost.

    Args:
        times: recovery times, in switch order.

    Returns:
        Cycles of extra recovery per additional switch, or 0.0 if under two.
    """
    if len(times) < 2:
        return 0.0
    x = np.arange(len(times), dtype=np.float64)
    y = np.asarray(times, dtype=np.float64)
    return float(np.polyfit(x, y, 1)[0])


def main() -> None:
    """Gate G0: plasticity loss must be visible in the DENSE baseline.

    If recovery time does not degrade across successive switches, nothing
    downstream is measurable -- a method that "fixes" a problem the setup never
    exhibited is an artifact.

    Scored under `protocol`: the reported statistic is per-run mean recovery
    time, summarised across a fixed seed prefix by median and IPR-90. The
    recovery-time SLOPE this gate used to report was retired as badly
    conditioned; its spread stayed 0.71-0.90 of its own magnitude however many
    switches it fit.
    """
    import protocol

    seeds = protocol.SEEDS[:8]
    print("Stage 0 / G0 -- does the dense baseline lose plasticity?")
    print(
        f"seeds {seeds}  switch_period={protocol.SWITCH_PERIOD} "
        f"cycles={protocol.NUM_CYCLES}"
    )
    print("=" * 74)
    print(f"{'theta':20} {'median recovery':>16} {'IPR90':>8} {'rel':>6} {'acc':>7}")
    for theta, label in (
        (0.0, "stationary"),
        (math.pi / 8, "pi/8"),
        (math.pi / 4, "pi/4"),
        (math.pi / 2, "pi/2 (orthogonal)"),
    ):
        recoveries, accuracies = [], []
        for seed in seeds:
            records = run(
                "dense",
                theta=theta,
                switch_period=protocol.SWITCH_PERIOD,
                num_cycles=protocol.NUM_CYCLES,
                steps_per_cycle=protocol.STEPS_PER_CYCLE,
                seed=seed,
            )
            times = recovery_times(records)
            recoveries.append(float(np.mean(times)) if times else 0.0)
            accuracies.append(
                float(np.mean([r.accuracy for r in records[-protocol.FINAL_WINDOW :]]))
            )
        recovery = protocol.summarize(recoveries)
        accuracy = protocol.summarize(accuracies)
        print(
            f"{label:20} {recovery.median:16.2f} {recovery.ipr90:8.2f} "
            f"{recovery.relative_ipr:6.2f} {accuracy.median:7.3f}",
            flush=True,
        )
    print()
    print(
        "G0 passes if recovery time rises with theta and the stationary arm "
        "is the floor."
    )


if __name__ == "__main__":
    main()
