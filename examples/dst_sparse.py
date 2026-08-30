"""Dynamic sparse training on the O(E) edge arena: SET and RigL in one file,
a switch between them, and a candidate shortlist so growth scales.

SET (Mocanu 2018) and RigL (Evci 2020) share everything -- per-unit-local
magnitude pruning, count-conserving fill-to-capacity regrowth, zero global
state -- and differ in exactly one expression, the growth score:

    method="set"   score = random_hash(src, dst, cursor)         # explore at random
    method="rigl"  score = |grad_pre_act[dst] * activation[src]|  # the edge's |dL/dw|

The RigL score is the delta-rule factorization of the loss gradient of an *absent*
edge into two per-unit columns the forward/backward already compute -- a local
read, no dense gradient over missing edges.

Pruning is identical for both: `MagnitudeStats` reduces each unit's incoming
`(count, sum|w|)` with a product monoid and writes a per-unit threshold
`tau = sqrt(pi)*erfinv(zeta)*mean|w|` (the mean-zero half-normal zeta-quantile);
`SetPrune` tombstones `|w| < tau_dst`. Nothing writes to globals `g`.

Scaling: the growth phase's exhaustive candidate grid is O(num_units^2), the one
term that is not O(E) -- it is the wall (empirically ~5k units before a churn
costs seconds and >1GB). Passing `shortlist=M` makes each growth policy declare
`max_candidate_units`, so the phase draws candidates from the M x M grid of the
M most *important* units (here |grad_pre_act| + |activation|) -- O(num_units + M^2)
-- and num_units scales far past the exhaustive wall. Size M >= sqrt(zeta * E):
a churn frees ~zeta*E slots and can refill only from the ~M^2 shortlisted
candidates, so too small an M leaves the arena unable to refill and sparsity
drifts *down* (still sparse, just below target). It is also a global top-M, so
on a deep MLP it can under-serve a layer transition; a per-level shortlist is
the clean fix and is future work.

Run:  uv run python examples/dst_sparse.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from mlp_xor import GradPreAct, LossGrad, MSELoss, SigmoidBackward, SigmoidForward

import plastax as px

# Per-unit DST state columns. thresh = the locally-estimated per-unit magnitude
# prune threshold (refreshed every churn); cursor = a per-unit rewiring counter,
# the deterministic salt that makes SET's random growth explore new edges each
# churn (RigL ignores it).
SET_THRESH = px.FieldSpec.float32("set/thresh")
SET_CURSOR = px.FieldSpec.int32("set/cursor")
_EXTRA_UNIT_FIELDS = (GradPreAct, LossGrad, SET_THRESH, SET_CURSOR)


def _hash01(a: jax.Array, b: jax.Array, c: jax.Array) -> jax.Array:
    """Stateless integer hash of three int scalars to a float in [0, 1)."""
    h = (a.astype(jnp.uint32) + jnp.uint32(0x9E3779B1)) * jnp.uint32(0x85EBCA77)
    h = (h ^ b.astype(jnp.uint32)) * jnp.uint32(0xC2B2AE3D)
    h = (h ^ c.astype(jnp.uint32)) * jnp.uint32(0x27D4EB2F)
    h = h ^ (h >> 15)
    h = h * jnp.uint32(0x2C1B3C6D)
    h = h ^ (h >> 13)
    return (h >> jnp.uint32(8)).astype(jnp.float32) / jnp.float32(1 << 24)


class MagnitudeStats(px.ForwardPass):
    """Churn-step forward: per-unit incoming-weight stats -> prune threshold.

    Ignores activations; reduces `(count, sum|w|)` over each destination's live
    incoming edges with a product monoid, writes `tau = alpha * mean|w|` (the
    half-normal zeta-quantile, `alpha = sqrt(pi)*erfinv(zeta)`), and bumps the
    per-unit rewiring cursor.
    """

    combine = (px.monoid.sum_, px.monoid.sum_)

    def __init__(self, zeta: float) -> None:
        """Bind the prune fraction and precompute the half-normal ratio."""
        self.zeta = zeta
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
        thresh = jnp.float32(self.alpha) * mean_abs
        return px.UnitWrite.of(
            (SET_THRESH, thresh), (SET_CURSOR, u[SET_CURSOR, i] + jnp.int32(1))
        )


class SetPrune(px.PruneConn):
    """Connection-local magnitude pruning against the destination's threshold."""

    def predicate(
        self, u: px.UnitView, c: px.ConnView, cid: px.ConnIdx, g: None
    ) -> jax.Array:
        """Return True to prune when |weight| is below the local threshold."""
        del g
        dst = px.UnitIdx(c[px.TO_ID, cid])
        return jnp.abs(c[px.WEIGHT, cid]) < u[SET_THRESH, dst]


class _Growth(px.AddConn[None]):
    """Shared growth: deeper-only (non-deeper vetoed with -inf), fill-to-capacity,
    small weight init, and -- when `shortlist` is set -- an `importance`-driven
    M x M candidate grid instead of the exhaustive num_units^2 one.

    `score` is the only method-specific piece (subclasses below).
    """

    def __init__(
        self, max_candidates: int, grow_scale: float, shortlist: int | None = None
    ) -> None:
        """Bind the growth budget, init weight, and optional shortlist size."""
        self.max_candidates = max_candidates
        self.grow_scale = grow_scale
        if shortlist is not None:
            # Read structurally by build_add_conn_phase (getattr): the phase
            # draws each bucket its own M x M grid (top-M sources at that source
            # level x top-M deeper destinations), so every layer transition of
            # the MLP is served -- a global top-M would concentrate on one level
            # and let sparsity drift down.
            self.max_candidate_units = shortlist
            self.shortlist_per_level = True

    def importance(self, u: px.UnitView, i: px.UnitIdx, g: None) -> jax.Array:
        """Per-unit shortlist score: activity + gradient magnitude.

        Favors units carrying signal (|activation|) or wanting more input
        (|grad_pre_act|), so the M x M grid concentrates on the edges most worth
        forming. Only consulted when `max_candidate_units` is set.
        """
        del g
        return jnp.abs(u[px.ACTIVATION, i]) + jnp.abs(u[GradPreAct, i])

    def init(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: None
    ) -> px.ConnWrite:
        """Initialize a regrown edge at `grow_scale` (optimizer state auto-zeroes)."""
        del u, src, dst, g
        return px.ConnWrite.of((px.WEIGHT, jnp.float32(self.grow_scale)))


class RandomGrow(_Growth):
    """SET: random growth -- score deeper candidates by a per-churn hash."""

    def score(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: None
    ) -> jax.Array:
        """Random score for deeper candidates, -inf (veto) otherwise."""
        del g
        deeper = u[px.LEVEL, dst] > u[px.LEVEL, src]
        return jnp.where(deeper, _hash01(src, dst, u[SET_CURSOR, dst]), -jnp.inf)


class GradientGrow(_Growth):
    """RigL: gradient growth -- score deeper candidates by the absent edge's |dL/dw|."""

    def score(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: None
    ) -> jax.Array:
        """|grad_pre_act[dst] * activation[src]| for deeper candidates, else -inf."""
        del g
        deeper = u[px.LEVEL, dst] > u[px.LEVEL, src]
        grad = u[GradPreAct, dst] * u[px.ACTIVATION, src]
        return jnp.where(deeper, jnp.abs(grad), -jnp.inf)


_GROWTH = {"set": RandomGrow, "rigl": GradientGrow}


def make_net(
    optimizer: px.optim.Optimizer,
    *,
    method: str = "set",
    mode: str,
    zeta: float = 0.3,
    max_candidates: int = 256,
    grow_scale: float = 0.0,
    shortlist: int | None = None,
) -> type[px.Network[None]]:
    """Build one of the three DST nets; churn selects SET vs RigL growth.

    Args:
        optimizer: the plastax.optim bundle.
        method: ``"set"`` (random growth) or ``"rigl"`` (gradient growth).
        mode: ``"train"``, ``"churn"``, or ``"eval"``.
        zeta: target per-unit prune fraction (churn).
        max_candidates: per-bucket growth bound (churn).
        grow_scale: regrown-edge init weight (churn).
        shortlist: M for the M x M candidate grid, or None for the exhaustive
            num_units^2 grid (churn).

    Returns:
        A Network subclass for the requested mode.

    Raises:
        ValueError: on an unknown mode or method.
    """
    if method not in _GROWTH:
        raise ValueError(f"make_net: unknown method {method!r} (set|rigl)")
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

    if mode == "eval":

        class _Eval(px.Network[None]):
            forward_pass = SigmoidForward()
            extra_unit_fields = _EXTRA_UNIT_FIELDS
            extra_conn_fields = optimizer.state_fields
            propagation = px.Propagation.TOPOLOGICAL

        return _Eval

    if mode == "churn":
        grow = _GROWTH[method](max_candidates, grow_scale, shortlist)

        class _Churn(px.Network[None]):
            forward_pass = MagnitudeStats(zeta)
            prune_conn = SetPrune()
            add_conn = grow
            extra_unit_fields = _EXTRA_UNIT_FIELDS
            extra_conn_fields = optimizer.state_fields
            propagation = px.Propagation.TOPOLOGICAL
            neighbourhood = 1

        return _Churn

    raise ValueError(f"make_net: unknown mode {mode!r}")


def _choose_pairs(
    n_src: int, n_dst: int, budget: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Pick ``budget`` distinct (src, dst) pairs, one incoming edge per dst.

    Every destination is seeded with one random source (no orphan units, which
    would fall to level 0 and scramble the bucketing), then the rest of the
    budget is filled with distinct random pairs.

    Args:
        n_src: number of source units.
        n_dst: number of destination units.
        budget: number of distinct edges (clamped to n_src*n_dst).
        rng: the numpy PRNG.

    Returns:
        Parallel (src_local, dst_local) int arrays.
    """
    total = n_src * n_dst
    k = min(budget, total)
    if k < n_dst:
        flat = rng.choice(total, size=k, replace=False)
        return (flat // n_dst).astype(np.int32), (flat % n_dst).astype(np.int32)
    seed_src = rng.integers(0, n_src, size=n_dst)
    seeds = seed_src.astype(np.int64) * n_dst + np.arange(n_dst)  # distinct per dst
    # Fill the remainder from a random stream, dropping any that collide.
    extra = rng.integers(0, total, size=2 * (k - n_dst) + 64, dtype=np.int64)
    extra = extra[~np.isin(extra, seeds)]
    flat = np.concatenate([seeds, np.unique(extra)])[:k]
    return (flat // n_dst).astype(np.int32), (flat % n_dst).astype(np.int32)


def build_sparse_mlp(
    net: type[px.Network[None]],
    layer_sizes: tuple[int, ...],
    budgets: tuple[int, ...],
    seed: int,
    *,
    capacity_headroom: float = 0.0,
) -> tuple[px.NetworkStatic, px.NetworkState[None]]:
    """Build a layered MLP wired sparsely at a fixed per-layer edge budget.

    A power-of-two budget (>= 64) packs its bucket exactly, so the arena starts
    full at the target sparsity and regrowth holds it there. Incoming weights are
    normal, scaled by the average sparse fan-in.

    Args:
        net: the Network type whose field layout the arenas adopt.
        layer_sizes: units per layer, input first (include a bias input).
        budgets: live edges per consecutive layer pair.
        seed: numpy PRNG seed.
        capacity_headroom: extra dead-slot fraction to pre-allocate per bucket,
            forwarded to NetworkBuilder.from_edges; 0.0 packs to the budget.

    Returns:
        The finalized (static, state) pair.
    """
    rng = np.random.default_rng(seed)
    offsets: list[int] = []
    running = 0
    for size in layer_sizes:
        offsets.append(running)
        running += size

    # Assemble every layer's edges as whole (E,) arrays and hand them to the
    # vectorized builder in one pass -- the per-edge add_conn loop this
    # replaced was O(E) Python and the wall for large sparse nets.
    from_chunks: list[np.ndarray] = []
    to_chunks: list[np.ndarray] = []
    weight_chunks: list[np.ndarray] = []
    for layer, budget in enumerate(budgets):
        n_src, n_dst = layer_sizes[layer], layer_sizes[layer + 1]
        src_local, dst_local = _choose_pairs(n_src, n_dst, budget, rng)
        scale = 1.0 / np.sqrt(max(1.0, len(src_local) / n_dst))
        from_chunks.append(src_local.astype(np.int32) + offsets[layer])
        to_chunks.append(dst_local.astype(np.int32) + offsets[layer + 1])
        weight_chunks.append(rng.standard_normal(len(src_local)) * scale)

    empty_i = np.zeros((0,), dtype=np.int32)
    return px.NetworkBuilder.from_edges(
        net,
        running,
        np.concatenate(from_chunks) if from_chunks else empty_i,
        np.concatenate(to_chunks) if to_chunks else empty_i,
        weights=(
            np.concatenate(weight_chunks)
            if weight_chunks
            else np.zeros((0,), dtype=np.float32)
        ),
        input_ids=tuple(range(layer_sizes[0])),
        output_ids=tuple(range(offsets[-1], running)),
        globals_=None,
        capacity_headroom=capacity_headroom,
    )


def teacher_task(
    d: int, classes: int, seed: int
) -> tuple[np.ndarray, np.random.Generator]:
    """A fixed random linear teacher (learnable, offline) and a data stream RNG."""
    return (
        np.random.default_rng(seed).standard_normal((classes, d)),
        np.random.default_rng(seed + 1),
    )


def _sample(teacher: np.ndarray, rng: np.random.Generator) -> tuple[jax.Array, int]:
    """Draw one (input-with-bias, label) example from the teacher."""
    classes, d = teacher.shape
    x = rng.standard_normal(d)
    label = int(np.argmax(teacher @ x))
    return jnp.asarray(np.append(x, 1.0), dtype=jnp.float32), label


def _one_hot(label: int, classes: int) -> jax.Array:
    """One-hot (classes,) float32 target."""
    return jnp.asarray(np.eye(classes, dtype=np.float32)[label])


def evaluate(
    eval_step: object,
    static: px.NetworkStatic,
    state: px.NetworkState[None],
    teacher: np.ndarray,
    rng: np.random.Generator,
    n: int,
) -> tuple[float, px.NetworkState[None]]:
    """Online argmax accuracy over ``n`` fresh samples; threads state back."""
    output_ids = np.asarray(static.output_ids)
    correct = 0
    for _ in range(n):
        inputs, label = _sample(teacher, rng)
        result = eval_step(state, px.StepInputs(inputs=inputs, targets=None))  # type: ignore[operator]
        state = result.state
        preds = np.asarray(state.units[px.ACTIVATION.name])[output_ids]
        correct += int(np.argmax(preds) == label)
    return correct / n, state


def run(
    method: str,
    layer_sizes: tuple[int, ...],
    budgets: tuple[int, ...],
    *,
    shortlist: int | None = None,
    lr: float = 0.05,
    zeta: float = 0.3,
    steps_per_cycle: int = 50,
    num_cycles: int = 120,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[px.NetworkState[None], list[int], float]:
    """Train a sparse MLP online with SET-or-RigL rewiring each cycle.

    Args:
        method: ``"set"`` or ``"rigl"``.
        layer_sizes: units per layer (input first, includes a bias input).
        budgets: live edges per consecutive layer pair.
        shortlist: M for the shortlisted growth grid, or None for exhaustive.
        lr: adam learning rate.
        zeta: target per-unit prune fraction.
        steps_per_cycle: differentiable steps between rewiring events.
        num_cycles: number of (train-then-churn) cycles.
        seed: PRNG seed.
        verbose: print a progress table.

    Returns:
        The final state, per-cycle post-churn live-edge counts, and accuracy.
    """
    grow = max(budgets) if shortlist is None else max(max(budgets), shortlist**2)
    optimizer = px.optim.adam(lr, GradPreAct)
    train_net = make_net(optimizer, method=method, mode="train")
    churn_net = make_net(
        optimizer,
        method=method,
        mode="churn",
        zeta=zeta,
        max_candidates=grow,
        shortlist=shortlist,
    )
    eval_net = make_net(optimizer, method=method, mode="eval")

    static, state = build_sparse_mlp(train_net, layer_sizes, budgets, seed)
    train_step = px.make_step(train_net, static)
    churn_step = px.make_step(churn_net, static)
    eval_step = px.make_step(eval_net, static)

    classes = layer_sizes[-1]
    teacher, rng = teacher_task(layer_sizes[0] - 1, classes, seed)
    last_inputs = jnp.zeros((layer_sizes[0],), dtype=jnp.float32)

    live_history: list[int] = []
    accuracy = 0.0
    if verbose:
        print(
            f"DST[{method}] {layer_sizes} budgets={budgets} "
            f"shortlist={shortlist} live={sum(budgets)}"
        )
    for cycle in range(num_cycles):
        cycle_loss = jnp.float32(0.0)
        for _ in range(steps_per_cycle):
            inputs, label = _sample(teacher, rng)
            result = train_step(
                state, px.StepInputs(inputs=inputs, targets=_one_hot(label, classes))
            )
            state = result.state
            cycle_loss = cycle_loss + result.loss
            last_inputs = inputs
        # Re-scatter the last train input so activations stay consistent with the
        # persisted grad_pre_act RigL reads (harmless for SET).
        state = churn_step(state, px.StepInputs(inputs=last_inputs, targets=None)).state
        live_history.append(int(px.state.live_conn_count(state)))
        if verbose and (cycle % 10 == 0 or cycle == num_cycles - 1):
            accuracy, state = evaluate(eval_step, static, state, teacher, rng, 256)
            print(
                f"  cycle {cycle:4d}  loss {float(cycle_loss) / steps_per_cycle:8.4f}"
                f"  live {live_history[-1]:7d}  acc {accuracy:.3f}"
            )
    accuracy, state = evaluate(eval_step, static, state, teacher, rng, 512)
    return state, live_history, accuracy


# Small synthetic demo config (both methods learn it; used by main + the test).
_DEMO_LAYERS = (17, 64, 4)
_DEMO_BUDGETS = (256, 64)


def main() -> None:
    """Train both SET and RigL on the small synthetic task and report."""
    for method in ("set", "rigl"):
        _, live_history, accuracy = run(
            method, _DEMO_LAYERS, _DEMO_BUDGETS, num_cycles=120, verbose=True
        )
        settled = live_history[1:]
        held = len(set(settled)) == 1
        print(f"[{method}] sparsity held: {held} ({settled[0]})  acc {accuracy:.3f}")
        assert held, f"{method}: live-edge count drifted: {sorted(set(settled))}"
        assert accuracy > 0.7, f"{method}: did not learn (acc {accuracy:.3f})"


if __name__ == "__main__":
    main()
