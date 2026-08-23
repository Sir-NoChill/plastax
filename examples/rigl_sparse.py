"""RigL (Evci 2020) as plastax policies: SET's per-unit-local magnitude prune,
but *gradient-informed* regrowth -- regrow the absent edges with the largest
|dL/dw| instead of at random.

SET and RigL differ in exactly one place, the growth score, and in plastax's
delta-rule world that difference is a single expression. For a weighted-sum
layer the loss gradient of ANY edge -- live or absent -- factors into two
per-unit quantities:

    dL/dw_{s->d} = (dL/dz_d) * a_s = grad_pre_act[d] * activation[s]

Both are unit columns the ordinary forward/backward already compute, so the
gradient an *absent* edge would receive is a purely local read of its two
endpoints -- no dense gradient over missing edges, no O(N^2) materialization.
`RigLRegrow.score` is exactly |grad_pre_act[dst] * activation[src]| (restricted
to deeper candidates), and the framework top_k selects the largest-gradient
absent edges. Everything else is reused from `set_sparse`: the per-unit
magnitude threshold (`MagnitudeStats`), the connection-local magnitude prune
(`SetPrune`), the count-conserving fill-to-capacity growth, the host-driven
two-net cadence, and the synthetic task.

Gradient freshness: this uses the *single* most recent training sample's
gradient -- the churn step re-scatters that sample's input and reads the
grad_pre_act/activation the last train step left in the unit columns. A true
batch-averaged RigL score is the average of the product (not the product of the
averages), an O(N^2) outer product that must be restricted to a candidate
shortlist -- the refinement SPARSE_PLAN.md's Risks section calls out. The
single-sample score keeps growth O(E) and connection-local.

Run:  uv run python examples/rigl_sparse.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import set_sparse as S
from mlp_xor import GradPreAct

import plastax as px


class RigLRegrow(px.AddConn[None]):
    """Gradient-informed regrowth: score an absent edge by its |dL/dw|.

    The score reads the two endpoint unit columns the delta rule factors the
    edge gradient into -- ``grad_pre_act[dst] * activation[src]`` -- so a
    candidate that does not yet exist is still scored by the gradient it would
    receive. Non-deeper candidates score ``-inf`` (a hard veto since the phases
    fix), keeping growth level-preserving. Grown weights start at ``grow_scale``
    (0.0 = RigL's zero-init; the edge then moves under the very gradient that
    selected it), optimizer moments auto-zero (S0).
    """

    def __init__(self, max_candidates: int, grow_scale: float = 0.0) -> None:
        """Bind the growth budget and the new-edge init weight.

        Args:
            max_candidates: per-bucket top-k growth bound; >= bucket capacity
                to refill every freed slot each churn.
            grow_scale: initial weight of a regrown edge (0.0 for RigL).
        """
        self.max_candidates = max_candidates
        self.grow_scale = grow_scale

    def score(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: None
    ) -> jax.Array:
        """Score a candidate edge by |dL/dw|, or ``-inf`` if not deeper.

        Args:
            u: the unit view (read for levels, grad_pre_act, activation).
            src: index of the candidate source unit.
            dst: index of the candidate destination unit.
            g: the global state (unused).

        Returns:
            The absolute edge gradient for a deeper candidate, else ``-inf``.
        """
        del g
        deeper = u[px.LEVEL, dst] > u[px.LEVEL, src]
        grad = u[GradPreAct, dst] * u[px.ACTIVATION, src]
        return jnp.where(deeper, jnp.abs(grad), -jnp.inf)

    def init(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: None
    ) -> px.ConnWrite:
        """Initialize a regrown edge at ``grow_scale`` (0.0 for RigL).

        Args:
            u: the unit view (unused).
            src: index of the source unit (unused).
            dst: index of the destination unit (unused).
            g: the global state (unused).

        Returns:
            A ConnWrite setting only ``weight`` (optimizer state auto-zeroes).
        """
        del u, src, dst, g
        return px.ConnWrite.of((px.WEIGHT, jnp.float32(self.grow_scale)))


def make_net(
    optimizer: px.optim.Optimizer,
    *,
    mode: str,
    zeta: float = 0.3,
    max_candidates: int = 256,
    grow_scale: float = 0.0,
) -> type[px.Network[None]]:
    """Build a RigL net; train/eval reuse set_sparse, churn swaps in RigLRegrow.

    Args:
        optimizer: the plastax.optim bundle.
        mode: one of ``"train"``, ``"churn"``, ``"eval"``.
        zeta: target per-unit prune fraction (churn mode).
        max_candidates: per-bucket growth bound (churn mode).
        grow_scale: regrown-edge init weight (churn mode).

    Returns:
        A Network subclass for the requested mode.

    Raises:
        ValueError: if ``mode`` is not one of the three known modes.
    """
    if mode in ("train", "eval"):
        return S.make_net(optimizer, mode=mode)
    if mode == "churn":

        class _RigLChurn(px.Network[None]):
            forward_pass = S.MagnitudeStats(zeta)
            prune_conn = S.SetPrune()
            add_conn = RigLRegrow(max_candidates, grow_scale)
            extra_unit_fields = S._EXTRA_UNIT_FIELDS
            extra_conn_fields = optimizer.state_fields
            propagation = px.Propagation.TOPOLOGICAL
            neighbourhood = 1

        return _RigLChurn
    raise ValueError(f"make_net: unknown mode {mode!r}")


def run(
    *,
    lr: float = 0.05,
    zeta: float = 0.3,
    steps_per_cycle: int = 50,
    num_cycles: int = 120,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[px.NetworkState[None], list[int], float]:
    """Train the sparse MLP online with RigL rewiring every ``steps_per_cycle``.

    Args:
        lr: adam learning rate.
        zeta: target per-unit prune fraction.
        steps_per_cycle: differentiable steps between rewiring events.
        num_cycles: number of (train-then-churn) cycles.
        seed: seed for teacher, structure, and the data stream.
        verbose: whether to print a progress table.

    Returns:
        The final state, the post-churn live-edge count per cycle, and the
        final accuracy.
    """
    optimizer = px.optim.adam(lr, GradPreAct)
    train_net = make_net(optimizer, mode="train")
    churn_net = make_net(
        optimizer, mode="churn", zeta=zeta, max_candidates=max(S._BUDGETS)
    )
    eval_net = make_net(optimizer, mode="eval")

    static, state = S.build_sparse_mlp(train_net, S._LAYER_SIZES, S._BUDGETS, seed)
    train_step = px.make_step(train_net, static)
    churn_step = px.make_step(churn_net, static)
    eval_step = px.make_step(eval_net, static)

    teacher = S._teacher(seed)
    rng = np.random.default_rng(seed + 1)
    last_inputs = jnp.zeros((S._LAYER_SIZES[0],), dtype=jnp.float32)

    live_history: list[int] = []
    accuracy = 0.0
    if verbose:
        print(
            f"RigL sparse MLP  (zeta={zeta}, sparsity~{S._sparsity():.0%}, "
            f"live={sum(S._BUDGETS)})"
        )
        print(f"{'cycle':>5s} {'loss':>10s} {'live':>6s} {'acc':>6s}")
    for cycle in range(num_cycles):
        cycle_loss = jnp.float32(0.0)
        for _ in range(steps_per_cycle):
            inputs, label = S._sample(teacher, rng)
            result = train_step(
                state, px.StepInputs(inputs=inputs, targets=S._one_hot(label))
            )
            state = result.state
            cycle_loss = cycle_loss + result.loss
            last_inputs = inputs
        # Reuse the last train sample's gradient: re-scatter its input so the
        # input activations stay consistent with the grad_pre_act/activation the
        # last train step left in the unit columns, which RigLRegrow reads.
        result = churn_step(state, px.StepInputs(inputs=last_inputs, targets=None))
        state = result.state
        live = int(px.state.live_conn_count(state))
        live_history.append(live)
        if verbose and (cycle % 10 == 0 or cycle == num_cycles - 1):
            accuracy, state = S.evaluate(eval_step, static, state, teacher, rng, 256)
            mean_loss = float(cycle_loss) / steps_per_cycle
            print(f"{cycle:5d} {mean_loss:10.5f} {live:6d} {accuracy:6.2f}")
    accuracy, state = S.evaluate(eval_step, static, state, teacher, rng, 512)
    return state, live_history, accuracy


def main() -> None:
    """Train the RigL sparse MLP and report constant sparsity + accuracy."""
    _, live_history, accuracy = run()
    settled = live_history[1:]
    held = len(set(settled)) == 1
    print("=" * 34)
    print(
        f"live-edge count constant across churns: {held} "
        f"({settled[0] if held else sorted(set(settled))})"
    )
    print(f"final accuracy: {accuracy:.3f}  (chance = {1 / S._CLASSES:.3f})")
    assert held, f"live-edge count drifted: {sorted(set(settled))}"
    assert accuracy > 0.7, f"did not learn (accuracy {accuracy:.3f})"


if __name__ == "__main__":
    main()
