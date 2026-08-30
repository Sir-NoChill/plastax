"""Extreme multi-label classification on the O(E) edge arena.

XMC is the sharpest form of the native-sparse thesis: with 10^5-10^6 labels the
output layer is intrinsically billions of edges, a dense classifier is
infeasible at any batch size, and *discovering* which hidden units connect to
which labels is exactly what SET/RigL rewiring does. Nothing here is a
mask over a dense weight matrix -- the label edges that exist are the only
ones stored.

Differences from `dst_sparse.py`, which supplies the DST policies unchanged
(magnitude pruning, `RandomGrow`/`GradientGrow` regrowth, the shortlist):

* **ReLU hidden, linear output.** Sigmoid hidden units saturate on BoW inputs
  with 10^5 features; the output unit holds the raw logit so the loss can use
  the numerically stable BCE-with-logits form.
* **BCE-with-logits, not MSE.** Each label is an independent binary decision.
  With a handful of positives among 10^6 labels the negatives dominate, so
  `pos_weight` upweights the positive term.
* **`IS_OUT` per-unit flag.** One field distinguishes "output unit" from
  "hidden unit" inside the *same* apply, so the forward can be linear at the
  output and ReLU everywhere else, and the backward can take dL/dz straight
  from the loss at the output instead of routing it through an activation
  derivative. Deriving this from LEVEL would be wrong: `resort` can renumber
  levels as edges are rewired.

Run:  uv run python examples/xmc.py            # synthetic fallback
      XMC_TRAIN=/path/amazon670k_train.txt uv run python examples/xmc.py
"""

from __future__ import annotations

import math
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from dst_sparse import (
    SET_CURSOR,
    SET_THRESH,
    GradientGrow,
    RandomGrow,
    SetPrune,
    build_sparse_mlp,
)
from mlp_xor import GradPreAct, LossGrad
from xmc_data import XmcSplit, load_xmc, precision_at_k

import plastax as px

# 1.0 for a label (output) unit, 0.0 for every other unit. Set once after
# construction (see `mark_outputs`) and never written by a phase.
IS_OUT = px.FieldSpec.float32("xmc/is_out")

# The per-unit prune fraction. A COLUMN rather than a trace-time constant so the
# host can anneal it between cycles without retracing the churn step: both SET
# (Mocanu 2018) and RigL (Evci 2020) decay the rewiring fraction to zero over
# training, and a constant fraction right up to the final cycle leaves the net
# freshly disrupted at the point it is measured.
ZETA = px.FieldSpec.float32("xmc/zeta")

_UNIT_FIELDS = (GradPreAct, LossGrad, SET_THRESH, SET_CURSOR, IS_OUT, ZETA)
_GROWTH = {"set": RandomGrow, "rigl": GradientGrow}


class XmcForward(px.ForwardPass):
    """map = weight*activation[src]; apply = ReLU, except linear at the output.

    The output unit stores the raw logit rather than sigmoid(logit): the loss
    needs the logit for its stable form, and ranking labels by logit is
    identical to ranking them by sigmoid(logit), so precision@k is unaffected.
    """

    combine = px.monoid.sum_

    def map(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: None,
    ) -> jax.Array:
        """Contribute one incoming edge's weighted activation."""
        del dst, g
        return c[px.WEIGHT, cid] * u[px.ACTIVATION, src]

    def apply(
        self, u: px.UnitView, i: px.UnitIdx, g: None, acc: jax.Array
    ) -> px.UnitWrite:
        """Write ReLU(acc) for a hidden unit, acc itself for a label unit."""
        del g
        is_out = u[IS_OUT, i] > jnp.float32(0.5)
        return px.UnitWrite.of(
            (px.ACTIVATION, jnp.where(is_out, acc, jax.nn.relu(acc)))
        )


class XmcBackward(px.BackwardPass):
    """Direction-reversed accumulate; dL/dz comes straight from the loss at
    the output level and through the ReLU derivative everywhere else.

    A label unit's dL/dz is `sigmoid(z) - t`, which `BCEWithLogitsLoss` already
    staged into `LossGrad`; multiplying it by an activation derivative (as the
    sigmoid-MLP backward does) would be wrong for a linear output *and*
    numerically fatal, since the analytic cancellation it relies on divides by
    p(1-p), which underflows for the overwhelmingly negative labels of XMC.
    """

    combine = px.monoid.sum_

    def map(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: None,
    ) -> jax.Array:
        """Propagate weight * dL/dz of the deeper unit back into this one.

        Note the inverted naming: `build_backward_accumulate` passes the
        accumulator target (FROM_ID, the shallower unit being written) as the
        `dst` parameter and the already-finalized deeper unit (TO_ID) as `src`,
        because map_fn's first unit-id argument is always the accumulator
        target in both directions. So the finalized dL/dz to read is `src`.
        """
        del dst, g
        return c[px.WEIGHT, cid] * u[GradPreAct, src]

    def apply(
        self, u: px.UnitView, i: px.UnitIdx, g: None, acc: jax.Array
    ) -> px.UnitWrite:
        """Convert the accumulated dL/da into this unit's dL/dz."""
        del g
        a = u[px.ACTIVATION, i]
        is_out = u[IS_OUT, i] > jnp.float32(0.5)
        relu_grad = acc * jnp.where(a > jnp.float32(0.0), 1.0, 0.0)
        return px.UnitWrite.of(
            (GradPreAct, jnp.where(is_out, u[LossGrad, i], relu_grad))
        )


class BCEWithLogitsLoss(px.Loss):
    """Per-label binary cross-entropy on logits, with a positive-class weight.

    Uses the stable form `max(z,0) - z*t + log1p(exp(-|z|))`, never
    `log(sigmoid(z))`: an XMC point has a handful of positives among 10^5-10^6
    labels, so almost every logit sits far negative where `sigmoid` underflows
    to exactly 0 and its log is -inf. `pos_weight` scales the positive term,
    countering an imbalance of order 10^5:1 that would otherwise be minimised
    by predicting every label absent.
    """

    def __init__(self, pos_weight: float = 1.0) -> None:
        """Bind the positive-class weight."""
        self.pos_weight = pos_weight

    def per_output(
        self, u: px.UnitView, i: px.UnitIdx, target: jax.Array, g: None
    ) -> tuple[jax.Array, px.UnitWrite]:
        """Return this label's loss and stage dL/dz = w'*sigmoid(z) - w*t."""
        del g
        z = u[px.ACTIVATION, i]
        w = jnp.float32(self.pos_weight)
        # weight on the shared log-partition term: 1 for a negative label,
        # pos_weight for a positive one.
        scale = jnp.float32(1.0) + target * (w - jnp.float32(1.0))
        loss = (
            scale * (jnp.maximum(z, jnp.float32(0.0)) + jnp.log1p(jnp.exp(-jnp.abs(z))))
            - w * target * z
        )
        grad = scale * jax.nn.sigmoid(z) - w * target
        return loss, px.UnitWrite.of((LossGrad, grad))


class AnnealedMagnitudeStats(px.ForwardPass):
    """dst_sparse.MagnitudeStats with the prune fraction read from a column.

    Identical statistics -- reduce `(count, sum|w|)` over each unit's live
    incoming edges and write the half-normal zeta-quantile threshold
    `tau = sqrt(pi)*erfinv(zeta)*mean|w|` -- but `zeta` comes from the ZETA unit
    column instead of being bound at construction, so `set_zeta` can change it
    between cycles against the same compiled step. At zeta = 0 the threshold is
    0, nothing is pruned, and the churn step becomes a no-op: that is what lets
    an annealing schedule end training with the learned wiring intact.
    """

    combine = (px.monoid.sum_, px.monoid.sum_)

    def map(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: None,
    ) -> tuple[jax.Array, jax.Array]:
        """Contribute (1, |weight|) for one live incoming edge."""
        del u, dst, src, g
        return jnp.float32(1.0), jnp.abs(c[px.WEIGHT, cid])

    def apply(
        self,
        u: px.UnitView,
        i: px.UnitIdx,
        g: None,
        acc: tuple[jax.Array, jax.Array],
    ) -> px.UnitWrite:
        """Write this unit's prune threshold and advance its rewiring cursor."""
        del g
        count, sum_abs = acc
        mean_abs = sum_abs / jnp.maximum(count, jnp.float32(1.0))
        alpha = jnp.sqrt(jnp.pi) * jax.scipy.special.erfinv(u[ZETA, i])
        return px.UnitWrite.of(
            (SET_THRESH, alpha * mean_abs),
            (SET_CURSOR, u[SET_CURSOR, i] + jnp.int32(1)),
        )


def set_zeta(state: px.NetworkState[None], zeta: float) -> px.NetworkState[None]:
    """Set the prune fraction on every unit; returns the updated state."""
    column = jnp.full_like(state.units[ZETA.name], jnp.float32(zeta))
    state.units = {**state.units, ZETA.name: column}
    return state


def cosine_zeta(zeta0: float, cycle: int, num_cycles: int) -> float:
    """RigL's cosine-annealed rewiring fraction for one cycle.

    `f_decay(t) = zeta0/2 * (1 + cos(pi*t/T_end))` (Evci 2020 eq. 3), which
    starts at zeta0 and reaches exactly 0 on the final cycle.

    Args:
        zeta0: the initial prune fraction.
        cycle: 0-based index of the current cycle.
        num_cycles: total cycles in the run.

    Returns:
        The prune fraction for this cycle.
    """
    if num_cycles <= 1:
        return 0.0
    return 0.5 * zeta0 * (1.0 + math.cos(math.pi * cycle / (num_cycles - 1)))


def make_net(
    optimizer: px.optim.Optimizer,
    *,
    method: str = "rigl",
    mode: str,
    pos_weight: float = 1.0,
    zeta: float = 0.3,
    max_candidates: int = 256,
    grow_scale: float = 0.0,
    shortlist: int | None = None,
) -> type[px.Network[None]]:
    """Build the XMC net for one mode; churn selects SET vs RigL growth.

    Args:
        optimizer: the plastax.optim bundle.
        method: ``"set"`` (random growth) or ``"rigl"`` (gradient growth).
        mode: ``"train"``, ``"churn"``, or ``"eval"``.
        pos_weight: positive-class weight for the loss (train).
        zeta: target per-unit prune fraction (churn).
        max_candidates: per-bucket growth bound (churn).
        grow_scale: regrown-edge init weight (churn).
        shortlist: M for the M x M candidate grid, or None for exhaustive
            (churn).

    Returns:
        A Network subclass for the requested mode.

    Raises:
        ValueError: on an unknown mode or method.
    """
    if method not in _GROWTH:
        raise ValueError(f"make_net: unknown method {method!r} (set|rigl)")
    if mode == "train":

        class _Train(px.Network[None]):
            forward_pass = XmcForward()
            backward_pass = XmcBackward()
            loss = BCEWithLogitsLoss(pos_weight)
            update_conn = optimizer.update_conn()
            extra_unit_fields = _UNIT_FIELDS
            extra_conn_fields = optimizer.state_fields
            propagation = px.Propagation.TOPOLOGICAL

        return _Train

    if mode == "eval":

        class _Eval(px.Network[None]):
            forward_pass = XmcForward()
            extra_unit_fields = _UNIT_FIELDS
            extra_conn_fields = optimizer.state_fields
            propagation = px.Propagation.TOPOLOGICAL

        return _Eval

    if mode == "churn":
        del zeta  # now a runtime column (ZETA); see AnnealedMagnitudeStats
        grow = _GROWTH[method](max_candidates, grow_scale, shortlist)

        class _Churn(px.Network[None]):
            forward_pass = AnnealedMagnitudeStats()
            prune_conn = SetPrune()
            add_conn = grow
            extra_unit_fields = _UNIT_FIELDS
            extra_conn_fields = optimizer.state_fields
            propagation = px.Propagation.TOPOLOGICAL
            neighbourhood = 1

        return _Churn

    raise ValueError(f"make_net: unknown mode {mode!r}")


def mark_outputs(
    static: px.NetworkStatic, state: px.NetworkState[None]
) -> px.NetworkState[None]:
    """Set the IS_OUT flag on every output unit; returns the updated state."""
    ids = jnp.asarray(static.output_ids, dtype=jnp.int32)
    column = state.units[IS_OUT.name].at[ids].set(jnp.float32(1.0))
    state.units = {**state.units, IS_OUT.name: column}
    return state


def build_xmc_net(
    net: type[px.Network[None]],
    num_features: int,
    num_labels: int,
    hidden: int,
    *,
    hidden_fan_in: int = 32,
    label_fan_in: int = 8,
    seed: int = 0,
    sharding: px.ShardSpec | None = None,
) -> tuple[px.NetworkStatic, px.NetworkState[None]]:
    """Build the sparse (features -> hidden -> labels) net at a fan-in budget.

    Args:
        net: the Network type whose field layout the arenas adopt.
        num_features: BoW/TF-IDF feature count (one bias input is appended).
        num_labels: size of the label set.
        hidden: hidden layer width.
        hidden_fan_in: average live incoming edges per hidden unit.
        label_fan_in: average live incoming edges per label unit.
        seed: numpy PRNG seed.
        sharding: Scheme-A ShardSpec for per-shard construction, or None.

    Returns:
        The finalized (static, state) pair, with IS_OUT already marked.
    """
    layer_sizes = (num_features + 1, hidden, num_labels)
    budgets = (hidden * hidden_fan_in, num_labels * label_fan_in)
    static, state = build_sparse_mlp(net, layer_sizes, budgets, seed, sharding=sharding)
    return static, mark_outputs(static, state)


def synthetic_split(
    num_points: int,
    num_features: int,
    num_labels: int,
    *,
    nnz: int = 16,
    labels_per_point: int = 4,
    rank: int = 32,
    seed: int = 0,
) -> XmcSplit:
    """A learnable synthetic stand-in for a repository split.

    The feature->label teacher is deliberately LOW RANK -- `features @ A @ B`
    with an inner dimension of `rank` -- so a hidden layer of at least `rank`
    units can represent it exactly. A full-rank random teacher would be
    linear-but-incompressible: no bottleneck narrower than `num_features`
    could fit it, and the net would score at chance no matter how correct the
    rewiring was, which makes it useless as a sanity task.

    Each point takes the `labels_per_point` labels its features score highest,
    so the structure to discover is genuine but bounded.

    Args:
        num_points: points to generate.
        num_features: feature-space size.
        num_labels: label-set size.
        nnz: nonzero features per point.
        labels_per_point: labels assigned per point.
        rank: inner dimension of the teacher, the width a hidden layer needs.
        seed: PRNG seed.

    Returns:
        The generated split.
    """
    rng = np.random.default_rng(seed)
    left = rng.standard_normal((num_features, rank)).astype(np.float32)
    right = rng.standard_normal((rank, num_labels)).astype(np.float32)
    x_indices = np.empty((num_points * nnz,), dtype=np.int32)
    x_values = np.empty((num_points * nnz,), dtype=np.float32)
    y_indices = np.empty((num_points * labels_per_point,), dtype=np.int32)
    for p in range(num_points):
        cols = rng.choice(num_features, size=nnz, replace=False).astype(np.int32)
        vals = np.abs(rng.standard_normal(nnz)).astype(np.float32)
        scores = (vals @ left[cols]) @ right
        top = np.argpartition(-scores, labels_per_point)[:labels_per_point]
        x_indices[p * nnz : (p + 1) * nnz] = cols
        x_values[p * nnz : (p + 1) * nnz] = vals
        y_indices[p * labels_per_point : (p + 1) * labels_per_point] = top
    return XmcSplit(
        num_features=num_features,
        num_labels=num_labels,
        x_indptr=np.arange(num_points + 1, dtype=np.int64) * nnz,
        x_indices=x_indices,
        x_values=x_values,
        y_indptr=np.arange(num_points + 1, dtype=np.int64) * labels_per_point,
        y_indices=y_indices,
    )


class TargetBuffer:
    """Reusable dense multi-hot target vector.

    `StepInputs.targets` is dense over the label set, so a fresh
    `np.zeros(num_labels)` per step would memset 2.7 MB (Amazon-670K) or 12 MB
    (Amazon-3M) every example. Clearing only the previous point's positives
    makes the host-side cost O(labels per point) instead of O(num_labels).
    """

    def __init__(self, num_labels: int) -> None:
        """Allocate the buffer for a label set of ``num_labels``."""
        self._buf = np.zeros((num_labels,), dtype=np.float32)
        self._live: np.ndarray = np.zeros((0,), dtype=np.int32)

    def set(self, labels: np.ndarray) -> np.ndarray:
        """Return the buffer holding the multi-hot encoding of ``labels``."""
        self._buf[self._live] = 0.0
        self._buf[labels] = 1.0
        self._live = labels
        return self._buf


def evaluate(
    eval_step: object,
    static: px.NetworkStatic,
    state: px.NetworkState[None],
    split: XmcSplit,
    indices: np.ndarray,
    ks: tuple[int, ...] = (1, 3, 5),
) -> tuple[tuple[float, ...], px.NetworkState[None]]:
    """Mean precision@k over ``indices`` of ``split``; threads state back.

    Args:
        eval_step: the forward-only step function.
        static: the network static config.
        state: the network state.
        split: the dataset split to score.
        indices: point indices to evaluate.
        ks: precision cutoffs.

    Returns:
        The mean precision per cutoff and the threaded state.
    """
    output_ids = np.asarray(static.output_ids)
    totals = np.zeros((len(ks),), dtype=np.float64)
    for i in indices:
        inputs = jnp.asarray(split.dense_input(int(i)))
        result = eval_step(state, px.StepInputs(inputs=inputs, targets=None))  # type: ignore[operator]
        state = result.state
        scores = np.asarray(state.units[px.ACTIVATION.name])[output_ids]
        totals += precision_at_k(scores, split.labels(int(i)), ks)
    return tuple(totals / max(len(indices), 1)), state


def run(
    split: XmcSplit,
    *,
    method: str = "rigl",
    hidden: int = 512,
    hidden_fan_in: int = 32,
    label_fan_in: int = 8,
    shortlist: int | None = 512,
    lr: float = 0.01,
    zeta: float = 0.3,
    anneal: bool = True,
    pos_weight: float = 64.0,
    steps_per_cycle: int = 512,
    num_cycles: int = 20,
    eval_points: int = 256,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[px.NetworkState[None], list[int], tuple[float, ...]]:
    """Train the sparse XMC net online, rewiring between cycles.

    Args:
        split: the training split.
        method: ``"set"`` or ``"rigl"``.
        hidden: hidden layer width.
        hidden_fan_in: average live incoming edges per hidden unit.
        label_fan_in: average live incoming edges per label unit.
        shortlist: M for the shortlisted growth grid, or None for exhaustive.
        lr: adam learning rate.
        zeta: initial per-unit prune fraction.
        anneal: cosine-decay the prune fraction to 0 over the run (RigL eq. 3);
            False holds it at ``zeta`` for every cycle.
        pos_weight: positive-class weight in the loss.
        steps_per_cycle: training examples between rewiring events.
        num_cycles: number of (train-then-churn) cycles.
        eval_points: points scored for precision@k.
        seed: PRNG seed.
        verbose: print a progress table.

    Returns:
        The final state, per-cycle post-churn live-edge counts, and the final
        precision@(1,3,5).
    """
    budget = max(hidden * hidden_fan_in, split.num_labels * label_fan_in)
    grow = budget if shortlist is None else max(budget, shortlist**2)
    optimizer = px.optim.adam(lr, GradPreAct)
    train_net = make_net(optimizer, method=method, mode="train", pos_weight=pos_weight)
    churn_net = make_net(
        optimizer,
        method=method,
        mode="churn",
        zeta=zeta,
        max_candidates=grow,
        shortlist=shortlist,
    )
    eval_net = make_net(optimizer, method=method, mode="eval")

    static, state = build_xmc_net(
        train_net,
        split.num_features,
        split.num_labels,
        hidden,
        hidden_fan_in=hidden_fan_in,
        label_fan_in=label_fan_in,
        seed=seed,
    )
    train_step = px.make_step(train_net, static)
    churn_step = px.make_step(churn_net, static)
    eval_step = px.make_step(eval_net, static)

    rng = np.random.default_rng(seed + 1)
    targets = TargetBuffer(split.num_labels)
    eval_indices = rng.choice(
        split.num_points, size=min(eval_points, split.num_points), replace=False
    )
    live_history: list[int] = []
    precision: tuple[float, ...] = (0.0, 0.0, 0.0)
    live = hidden * hidden_fan_in + split.num_labels * label_fan_in
    if verbose:
        print(
            f"XMC[{method}] features={split.num_features} "
            f"labels={split.num_labels} hidden={hidden} live={live}"
        )
    last_inputs = jnp.zeros((split.num_features + 1,), dtype=jnp.float32)
    for cycle in range(num_cycles):
        cycle_loss = 0.0
        started = time.perf_counter()
        for _ in range(steps_per_cycle):
            point = int(rng.integers(split.num_points))
            last_inputs = jnp.asarray(split.dense_input(point))
            result = train_step(
                state,
                px.StepInputs(
                    inputs=last_inputs,
                    targets=jnp.asarray(targets.set(split.labels(point))),
                ),
            )
            state = result.state
            cycle_loss += float(result.loss)
        # Anneal the rewiring fraction toward 0 (RigL eq. 3) so the final
        # cycles refine the wiring the earlier ones found instead of tearing it
        # up right before the net is measured. Writing the ZETA column reuses
        # the already-compiled churn step -- no retrace.
        cycle_zeta = cosine_zeta(zeta, cycle, num_cycles) if anneal else zeta
        state = set_zeta(state, cycle_zeta)
        # Re-scatter the last training input so the activations RigL's growth
        # score reads stay consistent with the persisted grad_pre_act.
        state = churn_step(state, px.StepInputs(inputs=last_inputs, targets=None)).state
        live_history.append(int(px.state.live_conn_count(state)))
        if verbose:
            precision, state = evaluate(eval_step, static, state, split, eval_indices)
            elapsed = time.perf_counter() - started
            print(
                f"  cycle {cycle:3d}  loss {cycle_loss / steps_per_cycle:9.4f}"
                f"  zeta {cycle_zeta:.3f}  live {live_history[-1]:9d}"
                f"  P@1 {precision[0]:.3f}  P@3 {precision[1]:.3f}"
                f"  P@5 {precision[2]:.3f}  {elapsed:6.1f}s"
            )
    precision, state = evaluate(eval_step, static, state, split, eval_indices)
    return state, live_history, precision


def main() -> None:
    """Train on $XMC_TRAIN if set, else on the synthetic fallback split."""
    path = os.environ.get("XMC_TRAIN")
    if path:
        print(f"loading {path}")
        split = load_xmc(path)
    else:
        print("XMC_TRAIN unset -- using the synthetic fallback split")
        split = synthetic_split(2048, 4096, 8192)
    _, live_history, precision = run(split, method="rigl")
    settled = live_history[1:]
    print(
        f"live edges settled: {sorted(set(settled))}  "
        f"P@1 {precision[0]:.3f}  P@3 {precision[1]:.3f}  P@5 {precision[2]:.3f}"
    )


if __name__ == "__main__":
    main()
