"""SET (Mocanu 2018) as plastax policies: per-unit-local magnitude pruning +
count-conserving random regrowth, at O(E) on the edge arena, with *zero* global
state.

SET is normally a *global* algorithm: each rewiring step prunes the fraction
``zeta`` of all weights closest to zero, then regrows the same number at random.
The prune fraction is a global order statistic -- a quantile over every weight --
which a mask-based implementation gets from one reduction over the dense tensor.
plastax has no global reduction to lean on (a connection-local
``PruneConn.predicate`` sees one edge, its endpoints, and the opaque globals
``g``), and the whole point of the arena is to avoid an O(N^2) / all-reduce
global coupling. So this file *estimates the global prune threshold locally*:

  * A dedicated churn-step forward pass (`MagnitudeStats`) reduces, per
    destination unit and over that unit's *live incoming* edges, the count and
    the mean absolute weight -- one map-reduce with a product monoid
    ``(sum_, sum_)``, exactly the machinery the forward pass already is. From the
    per-unit ``mean|w|`` it writes a per-unit pruning threshold ``set/thresh``.
  * `SetPrune` (connection-local) tombstones an edge when ``|w|`` is below its
    destination unit's threshold. The global zeta-quantile becomes a *per-unit*
    zeta-quantile -- estimated, under a mean-zero half-normal model of each
    unit's incoming weights, as ``tau = sqrt(pi) * erfinv(zeta) * mean|w|``.
  * `SetRegrow` refills the freed slots with random *deeper* candidate edges.
    Growth needs no order statistic at all (SET regrows at random); the only
    globally-conserved quantity is the *count*, which the arena conserves
    structurally -- regrow fills each bucket back to its fixed capacity, so the
    live-edge count is pinned. Randomness that varies per rewiring event -- with
    no PRNG key threaded through the state and no global step counter -- comes
    from a per-unit rewiring cursor (`set/cursor`), incremented once per churn in
    the stats pass and mixed into a stateless integer hash of ``(src, dst)``.

Nothing here writes to ``g``: the thresholds live in unit columns, populated by
the ordinary sweep, and every decision is a connection- or unit-local read. It
therefore shards edge-wise under Scheme A for free (the stats monoid has a
collective; ``sum_`` -> ``psum``).

The cadence is host-driven (as in tests/test_optim_sparse.py): train `N` steps
with the differentiable net, then run one churn step with the stats+prune+grow
net. Both nets share one (static, state) -- they declare the same field layout --
so the churn step reads the weights the train steps left. A regrown edge's adam
moments auto-zero (S0: the add_conn phase resets untouched fields to their
default), restarting that edge's schedule the way RigL/SET intend.

Run:  uv run python examples/set_sparse.py
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from mlp_xor import (
    GradPreAct,
    LossGrad,
    MSELoss,
    SigmoidBackward,
    SigmoidForward,
)

import plastax as px

# Per-unit SET state columns (the "set/" prefix namespaces them away from user
# and optimizer fields, as "opt/" does for the optimizer). thresh is the
# locally-estimated per-unit magnitude-prune threshold, refreshed every churn;
# cursor counts rewiring events for that unit, the deterministic salt that makes
# random regrowth explore new edges from one churn to the next.
SET_THRESH = px.FieldSpec.float32("set/thresh")
SET_CURSOR = px.FieldSpec.int32("set/cursor")


def _hash01(a: jax.Array, b: jax.Array, c: jax.Array) -> jax.Array:
    """Stateless integer hash of three int scalars to a float in [0, 1).

    A plain arithmetic mix (no PRNG state, vmap-safe): the source of the
    pseudo-randomness SET's regrowth needs, with no key threaded through the
    network state. uint32 multiply/xor wrap mod 2^32 by construction.

    Args:
        a: first mixed integer (e.g. source unit id).
        b: second mixed integer (e.g. destination unit id).
        c: third mixed integer (e.g. the destination's rewiring cursor).

    Returns:
        A float32 scalar in [0, 1), deterministic in (a, b, c).
    """
    h = (a.astype(jnp.uint32) + jnp.uint32(0x9E3779B1)) * jnp.uint32(0x85EBCA77)
    h = (h ^ b.astype(jnp.uint32)) * jnp.uint32(0xC2B2AE3D)
    h = (h ^ c.astype(jnp.uint32)) * jnp.uint32(0x27D4EB2F)
    h = h ^ (h >> 15)
    h = h * jnp.uint32(0x2C1B3C6D)
    h = h ^ (h >> 13)
    return (h >> jnp.uint32(8)).astype(jnp.float32) / jnp.float32(1 << 24)


class MagnitudeStats(px.ForwardPass):
    """Churn-step forward pass: per-unit incoming-weight stats -> prune threshold.

    Not a signal-propagating forward pass -- it ignores activations and reduces
    only the weights. The product-monoid accumulator carries ``(count, sum|w|)``
    over each destination unit's live incoming edges; `apply` turns the per-unit
    ``mean|w|`` into the local prune threshold ``tau = alpha * mean|w|`` and bumps
    that unit's rewiring cursor. ``alpha = sqrt(pi) * erfinv(zeta)`` is the ratio
    ``tau / E|w|`` for which a mean-zero half-normal puts fraction ``zeta`` of its
    mass below ``tau`` -- so the local threshold estimates the per-unit
    zeta-quantile in one pass, no sort.
    """

    combine = (px.monoid.sum_, px.monoid.sum_)

    def __init__(self, zeta: float) -> None:
        """Bind the target prune fraction and precompute the half-normal ratio.

        Args:
            zeta: target fraction of each unit's incoming edges to prune.
        """
        self.zeta = zeta
        # erfinv is evaluated eagerly here (outside any jit), so alpha is a
        # baked Python float in the traced apply below, not a runtime special
        # function on the hot path.
        self.alpha = float(
            jnp.sqrt(jnp.pi) * jax.scipy.special.erfinv(jnp.float32(zeta))
        )

    def map(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: None,
    ) -> tuple[jax.Array, jax.Array]:
        """Contribute (1, |weight|) for one live incoming edge.

        Args:
            u: the unit view (unused; stats depend only on the weight).
            dst: index of the destination unit (the accumulator target).
            src: index of the source unit (unused).
            c: the connection view.
            cid: index of the connection.
            g: the global state (unused).

        Returns:
            The per-edge ``(count, abs-weight)`` accumulator contribution.
        """
        del u, dst, src, g
        w = c[px.WEIGHT, cid]
        return jnp.float32(1.0), jnp.abs(w)

    def apply(
        self,
        u: px.UnitView,
        i: px.UnitIdx,
        g: None,
        acc: tuple[jax.Array, jax.Array],
    ) -> px.UnitWrite:
        """Write this unit's prune threshold and advance its rewiring cursor.

        Args:
            u: the unit view (read for the current cursor value).
            i: index of the unit being finalized.
            g: the global state (unused).
            acc: the reduced ``(count, sum|w|)`` over live incoming edges.

        Returns:
            A UnitWrite setting ``set/thresh`` and incrementing ``set/cursor``.
        """
        del g
        count, sum_abs = acc
        mean_abs = sum_abs / jnp.maximum(count, jnp.float32(1.0))
        thresh = jnp.float32(self.alpha) * mean_abs
        cursor = u[SET_CURSOR, i] + jnp.int32(1)
        return px.UnitWrite.of((SET_THRESH, thresh), (SET_CURSOR, cursor))


class SetPrune(px.PruneConn):
    """Connection-local magnitude pruning against the destination's threshold.

    Tombstones an edge whose ``|weight|`` falls below its destination unit's
    locally-estimated per-unit zeta-quantile (`MagnitudeStats` wrote it into
    ``set/thresh`` earlier this same churn step). No global reduction, no ``g``.
    """

    def predicate(
        self, u: px.UnitView, c: px.ConnView, cid: px.ConnIdx, g: None
    ) -> jax.Array:
        """Return True to prune when ``|weight|`` is below the local threshold.

        Args:
            u: the unit view (read for the destination's threshold).
            c: the connection view.
            cid: index of the connection.
            g: the global state (unused).

        Returns:
            A scalar bool; True tombstones this connection.
        """
        del g
        dst = px.UnitIdx(c[px.TO_ID, cid])
        return jnp.abs(c[px.WEIGHT, cid]) < u[SET_THRESH, dst]


class SetRegrow(px.AddConn[None]):
    """Random regrowth into freed slots, restricted to strictly-deeper edges.

    Candidates are scored by a stateless hash of ``(src, dst, cursor)`` so each
    churn (cursor advanced by `MagnitudeStats`) draws a fresh random subset;
    non-deeper candidates score ``-inf`` so growth never violates the leveling
    invariant (no resort/retrace). ``max_candidates`` is set at least as large as
    a bucket's capacity, so the phase refills every freed slot -- pinning the
    live-edge count and holding sparsity constant. New weights are small and
    random; the optimizer's moment columns are left untouched, so the add_conn
    phase zeroes them (S0).
    """

    def __init__(self, max_candidates: int, grow_scale: float) -> None:
        """Bind the growth budget and the new-edge weight scale.

        Args:
            max_candidates: per-bucket top-k growth bound; >= bucket capacity
                to refill every freed slot each churn.
            grow_scale: half-width of a regrown edge's small uniform init.
        """
        self.max_candidates = max_candidates
        self.grow_scale = grow_scale

    def score(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: None
    ) -> jax.Array:
        """Score a candidate edge: random for deeper edges, ``-inf`` otherwise.

        Args:
            u: the unit view (read for levels and the destination's cursor).
            src: index of the candidate source unit.
            dst: index of the candidate destination unit.
            g: the global state (unused).

        Returns:
            A float score; ``-inf`` disqualifies non-deeper candidates.
        """
        del g
        deeper = u[px.LEVEL, dst] > u[px.LEVEL, src]
        r = _hash01(src, dst, u[SET_CURSOR, dst])
        return jnp.where(deeper, r, -jnp.inf)

    def init(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: None
    ) -> px.ConnWrite:
        """Initialize a regrown edge with a small random weight.

        Args:
            u: the unit view (read for the destination's cursor).
            src: index of the source unit.
            dst: index of the destination unit.
            g: the global state (unused).

        Returns:
            A ConnWrite setting only ``weight`` (optimizer state auto-zeroes).
        """
        del g
        # A salt distinct from the score hash decorrelates the init weight from
        # the selection score for the same (src, dst, cursor).
        r = _hash01(src ^ jnp.int32(0x5BD1E995), dst, u[SET_CURSOR, dst])
        w = jnp.float32(self.grow_scale) * (r - jnp.float32(0.5))
        return px.ConnWrite.of((px.WEIGHT, w))


_EXTRA_UNIT_FIELDS = (GradPreAct, LossGrad, SET_THRESH, SET_CURSOR)


def make_net(
    optimizer: px.optim.Optimizer,
    *,
    mode: str,
    zeta: float = 0.3,
    max_candidates: int = 256,
    grow_scale: float = 0.1,
) -> type[px.Network[None]]:
    """Build one of the three SET nets over a shared field layout.

    All three declare the same extra unit/conn fields, so one (static, state)
    threads through every mode:

      * ``"train"``: sigmoid forward + backward + MSE loss + the optimizer.
      * ``"churn"``: the magnitude-stats forward + prune + random regrow.
      * ``"eval"``: sigmoid forward only (read predictions, no training).

    Args:
        optimizer: the plastax.optim bundle (supplies update_conn/state_fields).
        mode: one of ``"train"``, ``"churn"``, ``"eval"``.
        zeta: target per-unit prune fraction (churn mode).
        max_candidates: per-bucket growth bound (churn mode).
        grow_scale: regrown-edge init half-width (churn mode).

    Returns:
        A Network subclass for the requested mode.

    Raises:
        ValueError: if ``mode`` is not one of the three known modes.
    """
    if mode == "train":

        class _Train(px.Network[None]):
            forward_pass = SigmoidForward()
            backward_pass = SigmoidBackward()
            loss = MSELoss()
            update_conn = optimizer.update_conn()
            extra_unit_fields = _EXTRA_UNIT_FIELDS
            extra_conn_fields = optimizer.state_fields
            propagation = px.Propagation.TOPOLOGICAL

        return _Train

    if mode == "churn":

        class _Churn(px.Network[None]):
            forward_pass = MagnitudeStats(zeta)
            prune_conn = SetPrune()
            add_conn = SetRegrow(max_candidates, grow_scale)
            extra_unit_fields = _EXTRA_UNIT_FIELDS
            extra_conn_fields = optimizer.state_fields
            propagation = px.Propagation.TOPOLOGICAL
            neighbourhood = 1  # grow only between adjacent levels (layered MLP)

        return _Churn

    if mode == "eval":

        class _Eval(px.Network[None]):
            forward_pass = SigmoidForward()
            extra_unit_fields = _EXTRA_UNIT_FIELDS
            extra_conn_fields = optimizer.state_fields
            propagation = px.Propagation.TOPOLOGICAL

        return _Eval

    raise ValueError(f"make_net: unknown mode {mode!r}")


def _choose_pairs(
    n_src: int, n_dst: int, budget: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Pick ``budget`` distinct (src, dst) local index pairs without replacement.

    Args:
        n_src: number of source units in the layer.
        n_dst: number of destination units in the layer.
        budget: number of distinct edges to select (clamped to n_src*n_dst).
        rng: the numpy PRNG.

    Returns:
        Parallel (src_local, dst_local) int arrays of length ``min(budget,
        n_src*n_dst)``.
    """
    total = n_src * n_dst
    k = min(budget, total)
    flat = rng.choice(total, size=k, replace=False)
    return (flat // n_dst).astype(np.int32), (flat % n_dst).astype(np.int32)


def build_sparse_mlp(
    net: type[px.Network[None]],
    layer_sizes: tuple[int, ...],
    budgets: tuple[int, ...],
    seed: int,
) -> tuple[px.NetworkStatic, px.NetworkState[None]]:
    """Build a layered MLP wired sparsely at a fixed per-layer edge budget.

    Each consecutive layer pair gets ``budget`` random edges. With a power-of-two
    budget (>= 64) the bucket packs exactly -- ``capacity_policy`` leaves no
    headroom -- so the arena starts full at the target sparsity and regrowth
    holds it there. Incoming weights are normal, scaled by the average sparse
    fan-in so sigmoid pre-activations start near unit scale.

    Args:
        net: the Network type whose field layout the arenas adopt.
        layer_sizes: units per layer, input first (include a bias input).
        budgets: live-edge count per consecutive layer pair; ``len ==
            len(layer_sizes) - 1``.
        seed: numpy PRNG seed for structure and initial weights.

    Returns:
        The finalized (static, state) pair.
    """
    rng = np.random.default_rng(seed)
    builder: px.NetworkBuilder[None] = px.NetworkBuilder(net, None)
    offsets: list[int] = []
    running = 0
    for size in layer_sizes:
        offsets.append(running)
        running += size
    for _ in range(running):
        builder.add_unit()
    for i in range(layer_sizes[0]):
        builder.mark_input(i)
    for i in range(offsets[-1], running):
        builder.mark_output(i)

    for layer, budget in enumerate(budgets):
        n_src, n_dst = layer_sizes[layer], layer_sizes[layer + 1]
        src_local, dst_local = _choose_pairs(n_src, n_dst, budget, rng)
        fan_in = max(1.0, len(src_local) / n_dst)
        scale = 1.0 / np.sqrt(fan_in)
        weights = rng.standard_normal(len(src_local)) * scale
        for s, d, w in zip(src_local, dst_local, weights, strict=True):
            builder.add_conn(
                offsets[layer] + int(s),
                offsets[layer + 1] + int(d),
                weight=float(w),
            )
    return builder.finalize()


# --------------------------------------------------------------------------- #
# Synthetic task: argmax of a fixed random linear teacher (learnable, offline).
# --------------------------------------------------------------------------- #
_D = 16  # feature dimension (a constant bias input is appended -> _D + 1 inputs)
_HIDDEN = 64
_CLASSES = 4
_LAYER_SIZES = (_D + 1, _HIDDEN, _CLASSES)
_BUDGETS = (256, 64)  # packed powers of two -> capacities (256, 64), sparsity ~24%


def _teacher(seed: int) -> np.ndarray:
    """Return the fixed (_CLASSES, _D) linear teacher weight matrix."""
    return np.random.default_rng(seed).standard_normal((_CLASSES, _D))


def _sample(teacher: np.ndarray, rng: np.random.Generator) -> tuple[jax.Array, int]:
    """Draw one (input-with-bias, label) example from the teacher.

    Args:
        teacher: the (_CLASSES, _D) teacher matrix.
        rng: the numpy PRNG.

    Returns:
        A ``(_D + 1,)`` float32 input (bias 1.0 appended) and its class label.
    """
    x = rng.standard_normal(_D)
    label = int(np.argmax(teacher @ x))
    inputs = jnp.asarray(np.append(x, 1.0), dtype=jnp.float32)
    return inputs, label


def _one_hot(label: int) -> jax.Array:
    """One-hot ``(_CLASSES,)`` float32 target for a class label."""
    return jnp.asarray(np.eye(_CLASSES, dtype=np.float32)[label])


def evaluate(
    eval_step: Callable[[px.NetworkState[None], px.StepInputs], px.StepResult[None]],
    static: px.NetworkStatic,
    state: px.NetworkState[None],
    teacher: np.ndarray,
    rng: np.random.Generator,
    n: int,
) -> tuple[float, px.NetworkState[None]]:
    """Online argmax accuracy over ``n`` fresh teacher samples.

    The forward-only step still donates its state buffer (make_step donates
    arg 0), so the threaded state is returned for the caller to continue from --
    eval writes only ACTIVATION, leaving weights/optimizer/SET columns intact.

    Args:
        eval_step: the forward-only step function.
        static: the network static config (for the output ids).
        state: the current network state.
        teacher: the teacher matrix.
        rng: the numpy PRNG.
        n: number of evaluation samples.

    Returns:
        The fraction of samples whose argmax prediction matches the label, and
        the state threaded through the evaluation passes.
    """
    output_ids = np.asarray(static.output_ids)
    correct = 0
    for _ in range(n):
        inputs, label = _sample(teacher, rng)
        result = eval_step(state, px.StepInputs(inputs=inputs, targets=None))
        state = result.state
        preds = np.asarray(state.units[px.ACTIVATION.name])[output_ids]
        correct += int(np.argmax(preds) == label)
    return correct / n, state


def run(
    *,
    lr: float = 0.05,
    zeta: float = 0.3,
    steps_per_cycle: int = 50,
    num_cycles: int = 120,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[px.NetworkState[None], list[int], float]:
    """Train the sparse MLP online with SET rewiring every ``steps_per_cycle``.

    Args:
        lr: adam learning rate.
        zeta: target per-unit prune fraction.
        steps_per_cycle: differentiable steps between rewiring events.
        num_cycles: number of (train-then-churn) cycles.
        seed: seed for teacher, structure, and the data stream.
        verbose: whether to print a progress table.

    Returns:
        The final state, the post-churn live-edge count after each cycle, and
        the final accuracy.
    """
    optimizer = px.optim.adam(lr, GradPreAct)
    train_net = make_net(optimizer, mode="train")
    churn_net = make_net(
        optimizer, mode="churn", zeta=zeta, max_candidates=max(_BUDGETS)
    )
    eval_net = make_net(optimizer, mode="eval")

    static, state = build_sparse_mlp(train_net, _LAYER_SIZES, _BUDGETS, seed)
    train_step = px.make_step(train_net, static)
    churn_step = px.make_step(churn_net, static)
    eval_step = px.make_step(eval_net, static)

    teacher = _teacher(seed)
    rng = np.random.default_rng(seed + 1)
    dummy = jnp.zeros((_LAYER_SIZES[0],), dtype=jnp.float32)

    live_history: list[int] = []
    accuracy = 0.0
    if verbose:
        print(
            f"SET sparse MLP  (zeta={zeta}, sparsity~{_sparsity():.0%}, "
            f"live={sum(_BUDGETS)})"
        )
        print(f"{'cycle':>5s} {'loss':>10s} {'live':>6s} {'acc':>6s}")
    for cycle in range(num_cycles):
        cycle_loss = jnp.float32(0.0)
        for _ in range(steps_per_cycle):
            inputs, label = _sample(teacher, rng)
            result = train_step(
                state, px.StepInputs(inputs=inputs, targets=_one_hot(label))
            )
            state = result.state
            cycle_loss = cycle_loss + result.loss
        result = churn_step(state, px.StepInputs(inputs=dummy, targets=None))
        state = result.state
        live = int(px.state.live_conn_count(state))
        live_history.append(live)
        if verbose and (cycle % 10 == 0 or cycle == num_cycles - 1):
            accuracy, state = evaluate(eval_step, static, state, teacher, rng, 256)
            mean_loss = float(cycle_loss) / steps_per_cycle
            print(f"{cycle:5d} {mean_loss:10.5f} {live:6d} {accuracy:6.2f}")
    accuracy, state = evaluate(eval_step, static, state, teacher, rng, 512)
    return state, live_history, accuracy


def _sparsity() -> float:
    """Return the arena's live-edge fraction over all possible layered edges."""
    possible = sum(_LAYER_SIZES[i] * _LAYER_SIZES[i + 1] for i in range(len(_BUDGETS)))
    return sum(_BUDGETS) / possible


def main() -> None:
    """Train the SET sparse MLP and report constant sparsity + learned accuracy."""
    _, live_history, accuracy = run()
    settled = live_history[1:]
    held = len(set(settled)) == 1
    print("=" * 34)
    print(
        f"live-edge count constant across churns: {held} "
        f"({settled[0] if held else sorted(set(settled))})"
    )
    print(f"final accuracy: {accuracy:.3f}  (chance = {1 / _CLASSES:.3f})")
    assert held, f"live-edge count drifted: {sorted(set(settled))}"
    assert accuracy > 0.7, f"did not learn (accuracy {accuracy:.3f})"


if __name__ == "__main__":
    main()
