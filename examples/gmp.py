"""Gradual magnitude pruning (Zhu & Gupta 2017; Obando-Ceron et al., ICML 2024).

Dense to sparse on a cubic schedule, with no growth: GMP is the one method in
this study that only ever removes edges, and a pruned edge never returns. It is
almost entirely a scheduling change on machinery that already exists --
`AnnealedMagnitudeStats` plus `SetPrune`, with `add_conn` absent -- which is the
point: if a published method reduces to a schedule over shipped traits, the
trait vocabulary is the right one.

**The schedule.** Zhu & Gupta's cubic, in the form Obando-Ceron et al. use:

    s_t = s_f * (1 - (1 - (t - t_start)/(t_end - t_start))^3)

Pruning is confined to a window; the RL paper begins 20% into training and stops
at 80%, leaving the network dense while it finds a solution and fixed while it
consolidates one.

**Absolute sparsity vs per-churn fraction.** The schedule names a CUMULATIVE
sparsity, while `AnnealedMagnitudeStats` prunes a fraction of the *currently
live* edges. Converting is one line and is where an implementation silently
drifts if it treats `s_t` as the per-step rate.

Deviation to record: the paper prunes globally by magnitude; `SetPrune` compares
each edge against its destination's own half-normal quantile, so the realized
sparsity tracks the schedule only to within that quantile's error. Local rules
are the library's premise, so this is a deliberate substitution rather than an
approximation of convenience.

Run:  uv run python examples/gmp.py
"""

from __future__ import annotations

import math
import time

import jax.numpy as jnp
import numpy as np
from dst_sparse import (
    SET_CURSOR,
    SET_THRESH,
    ZETA,
    AnnealedMagnitudeStats,
    SetPrune,
    set_zeta,
)
from mlp_xor import GradPreAct, LossGrad, MSELoss
from nonstationary import (
    ACT_EMA,
    IS_OUT,
    CycleRecord,
    DriftingTask,
    ReluBackward,
    ReluForward,
    build_dense_mlp,
    mark_outputs,
)

import plastax as px

# SET_THRESH carries the per-unit prune bar; SET_CURSOR is bumped by the shared
# stats pass and unused here, since GMP never regrows.
_UNIT_FIELDS = (GradPreAct, LossGrad, ACT_EMA, IS_OUT, ZETA, SET_THRESH, SET_CURSOR)


def cubic_sparsity(step: int, *, start: int, end: int, final_sparsity: float) -> float:
    """Zhu & Gupta's cubic cumulative-sparsity schedule.

    Zero before `start`, `final_sparsity` from `end` onward, cubic in between --
    fast early, tapering as it approaches the target.

    Args:
        step: current step.
        start: step at which pruning begins.
        end: step at which the target sparsity is reached.
        final_sparsity: target cumulative sparsity in [0, 1).

    Returns:
        The cumulative sparsity for this step.
    """
    # `end` is tested first so a zero-width window (start == end) resolves to
    # the target rather than staying at zero forever.
    if step >= end:
        return final_sparsity
    if step <= start:
        return 0.0
    progress = (step - start) / (end - start)
    return final_sparsity * (1.0 - (1.0 - progress) ** 3)


def zeta_for_step(current: float, previous: float) -> float:
    """Fraction of the LIVE edges to remove to move sparsity `previous`->`current`.

    The schedule is cumulative but the prune policy takes a fraction of what is
    still live, so the two are not interchangeable: passing `current` straight
    through would over-prune every cycle after the first.

    Args:
        current: the schedule's cumulative sparsity for this step.
        previous: the cumulative sparsity already realized.

    Returns:
        The per-churn prune fraction, clamped to [0, 1).
    """
    live = 1.0 - previous
    if live <= 1e-8:
        return 0.0
    return float(min(max((current - previous) / live, 0.0), 0.999))


def make_net(
    optimizer: px.optim.Optimizer, *, mode: str, ema_decay: float = 0.05
) -> type[px.Network[None]]:
    """Build the train / churn / eval net for GMP.

    The churn net declares `prune_conn` and no `add_conn`: GMP never regrows.

    Args:
        optimizer: the plastax.optim bundle.
        mode: ``"train"``, ``"churn"`` or ``"eval"``.
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
            forward_pass = AnnealedMagnitudeStats()
            prune_conn = SetPrune()
            extra_unit_fields = _UNIT_FIELDS
            extra_conn_fields = optimizer.state_fields
            propagation = px.Propagation.TOPOLOGICAL
            neighbourhood = 1

        return _Churn

    raise ValueError(f"make_net: unknown mode {mode!r}")


def run(
    *,
    final_sparsity: float = 0.95,
    start_fraction: float = 0.2,
    end_fraction: float = 0.8,
    theta: float = math.pi / 4,
    switch_period: int | None = 15,
    d: int = 32,
    hidden_layers: tuple[int, ...] = (128, 128),
    classes: int = 8,
    lr: float = 0.001,
    steps_per_cycle: int = 100,
    num_cycles: int = 600,
    seed: int = 0,
) -> list[CycleRecord]:
    """Train dense-to-sparse on the drifting teacher under the cubic schedule.

    Args:
        final_sparsity: target cumulative sparsity.
        start_fraction: fraction of the run at which pruning begins.
        end_fraction: fraction of the run at which the target is reached.
        theta: rotation angle per switch.
        switch_period: cycles between switches, or None for stationary.
        d: input dimensionality.
        hidden_layers: width of each hidden layer.
        classes: output classes.
        lr: adam learning rate.
        steps_per_cycle: training examples per cycle.
        num_cycles: cycles to run.
        seed: PRNG seed.

    Returns:
        One CycleRecord per cycle, sharing Stage 0's record so GMP is scored by
        the same `recovery_times`.
    """
    layers = (d + 1, *hidden_layers, classes)
    optimizer = px.optim.adam(lr, GradPreAct)
    train_net = make_net(optimizer, mode="train")
    churn_net = make_net(optimizer, mode="churn")
    static, state = build_dense_mlp(train_net, layers, seed)
    state = mark_outputs(static, state)
    train_step = px.make_step(train_net, static)
    churn_step = px.make_step(churn_net, static)

    task = DriftingTask(d, classes, theta=theta, switch_period=switch_period, seed=seed)
    eye = np.eye(classes, dtype=np.float32)
    hidden_units = np.arange(layers[0], sum(layers) - classes)
    out_ids = np.asarray(static.output_ids)
    dense_edges = sum(a * b for a, b in zip(layers[:-1], layers[1:], strict=True))
    start = int(start_fraction * num_cycles)
    end = int(end_fraction * num_cycles)

    records: list[CycleRecord] = []
    realized = 0.0
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
        target = cubic_sparsity(
            cycle, start=start, end=end, final_sparsity=final_sparsity
        )
        zeta = zeta_for_step(target, realized)
        if zeta > 0.0:
            state = set_zeta(state, zeta)
            state = churn_step(
                state, px.StepInputs(inputs=last_inputs, targets=None)
            ).state
        live = int(px.state.live_conn_count(state))
        realized = 1.0 - live / dense_edges
        ema = np.asarray(state.units[ACT_EMA.name])[hidden_units]
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


def main() -> None:
    """Report how closely the realized sparsity tracks the cubic schedule."""
    num_cycles = 200
    for final_sparsity in (0.75, 0.90, 0.95):
        records = run(final_sparsity=final_sparsity, num_cycles=num_cycles, seed=0)
        start = int(0.2 * num_cycles)
        end = int(0.8 * num_cycles)
        target = cubic_sparsity(
            num_cycles - 1, start=start, end=end, final_sparsity=final_sparsity
        )
        realized = 1.0 - records[-1].density
        accuracy = float(np.mean([r.accuracy for r in records[-5:]]))
        print(
            f"s_f={final_sparsity:.2f}  target {target:.3f}  realized "
            f"{realized:.3f}  live {records[-1].live_edges:6d}  acc {accuracy:.3f}"
        )


if __name__ == "__main__":
    main()
