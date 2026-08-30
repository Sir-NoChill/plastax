"""Neuroplastic Expansion (Liu et al., ICLR 2025) on the O(E) edge arena.

NE grows a network from a small initial capacity toward its full size, pruning
dormant units as it goes, so the agent keeps free capacity to allocate as the
task changes. Three components in the paper; two are implemented here and the
third is dropped for a stated reason.

**Elastic topology generation.** `I_grow = ArgTopk_{i not in theta}(|grad_theta L|)`
-- the k ABSENT connections with the largest loss-gradient magnitude, per layer.
That is exactly `dst_sparse.GradientGrow`'s score, which is RigL's, unchanged:
the gradient of an absent edge factorises as `grad_pre_act[dst] * activation[src]`,
two per-unit columns the forward and backward already wrote, so it is a local
read rather than a dense gradient over missing edges.

What is new is the *rate*. The paper anneals k by `(alpha/2)(1 + cos(t*pi/T_end))`.
`max_candidates` is a static trace-time bound and cannot be a traced value, so
the count cannot be scheduled directly. Instead each candidate is thinned
stochastically against a per-unit rate column the host rewrites each cycle: the
expected number grown follows the schedule exactly, the decision stays local,
and no new machinery is needed.

**Dormant neuron pruning.** The paper sets the dormancy threshold to **exactly
zero** -- `I_prune = {Index(theta_i) | f(theta_i) = 0}`, only *fully* dormant
units -- and caps how many may go per layer at `omega * |I_grow|`, truncating
the excess at random. plastax's `prune_conn` is a conn-local predicate with no
global count, so the cap is computed host-side each cycle and broadcast as a
per-unit prune *probability*, the same pattern `set_zeta` and CBP's
`set_oracle_threshold` already use. Random truncation to a budget and an
independent coin per unit agree in expectation.

**First and last transitions stay dense.** The paper keeps the encoder and
decoder dense (following Tan et al. 2022) for stability, growing only the
interior. That is why the harness runs two hidden layers: with one, every
transition is an end and NE would have nothing to grow into.

**Neuron consolidation via experience review is DROPPED.** It replays sampled
transitions from a buffer, and streaming RL has no buffer. The paper introduces
it because NE alone destabilises late in training, so if that instability shows
up here it is a result about streaming + NE, not a bug to hide.

Run:  uv run python examples/ne.py
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from dst_sparse import SET_CURSOR, _hash01
from mlp_xor import GradPreAct, LossGrad, MSELoss
from nonstationary import ACT_EMA, IS_OUT, ReluBackward, ReluForward, mark_outputs

import plastax as px

# Per-unit growth rate: the cosine-annealed fraction of scored candidates that
# survive thinning. Host-written each cycle, read in the growth score.
NE_RATE = px.FieldSpec.float32("ne/rate")
# Per-unit probability that a dormant unit's incoming edges are pruned this
# cycle -- the omega cap, expressed locally.
NE_PRUNE_P = px.FieldSpec.float32("ne/prune_p")
# 1.0 on units whose INCOMING transition NE is allowed to modify: the interior
# only, never the encoder or decoder.
NE_ELASTIC = px.FieldSpec.float32("ne/elastic")

_UNIT_FIELDS = (
    GradPreAct,
    LossGrad,
    ACT_EMA,
    IS_OUT,
    SET_CURSOR,
    NE_RATE,
    NE_PRUNE_P,
    NE_ELASTIC,
)


class NeStats(px.ForwardPass):
    """Churn-step forward: advance the per-unit rewiring cursor only.

    NE's dormancy statistic is the activation EMA that the TRAIN forward pass
    already maintains, so the churn forward has nothing to reduce. It exists to
    bump the cursor that salts the stochastic thinning, which is what makes
    successive cycles explore different candidates.
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
        """Contribute nothing; the cursor bump happens in apply."""
        del u, dst, src, c, cid, g
        return jnp.float32(0.0)

    def apply(
        self, u: px.UnitView, i: px.UnitIdx, g: None, acc: jax.Array
    ) -> px.UnitWrite:
        """Advance this unit's rewiring cursor."""
        del g, acc
        return px.UnitWrite.of((SET_CURSOR, u[SET_CURSOR, i] + jnp.int32(1)))


class NeGrow(px.AddConn[None]):
    """Gradient-scored growth, thinned to a cosine-annealed rate, interior only.

    The score is RigL's absent-edge gradient. Two gates sit on top: the edge's
    destination must be in the elastic interior, and the candidate must survive
    a per-unit coin whose bias is the annealed growth rate. `-inf` is the
    framework's hard veto, so a vetoed candidate is never committed however many
    free slots the bucket has.
    """

    def __init__(
        self, max_candidates: int, grow_scale: float, shortlist: int | None = None
    ) -> None:
        """Bind the growth budget, init weight and optional shortlist size."""
        self.max_candidates = max_candidates
        self.grow_scale = grow_scale
        if shortlist is not None:
            self.max_candidate_units = shortlist
            self.shortlist_per_level = True

    def importance(self, u: px.UnitView, i: px.UnitIdx, g: None) -> jax.Array:
        """Per-unit shortlist score: activity plus gradient magnitude."""
        del g
        return jnp.abs(u[px.ACTIVATION, i]) + jnp.abs(u[GradPreAct, i])

    def score(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: None
    ) -> jax.Array:
        """|grad| for a deeper, elastic, coin-surviving candidate; -inf else."""
        del g
        deeper = u[px.LEVEL, dst] > u[px.LEVEL, src]
        elastic = u[NE_ELASTIC, dst] > jnp.float32(0.5)
        keep = _hash01(src, dst, u[SET_CURSOR, dst]) < u[NE_RATE, dst]
        gradient = jnp.abs(u[GradPreAct, dst] * u[px.ACTIVATION, src])
        return jnp.where(deeper & elastic & keep, gradient, -jnp.inf)

    def init(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: None
    ) -> px.ConnWrite:
        """Initialize a grown edge at `grow_scale`."""
        del u, src, dst, g
        return px.ConnWrite.of((px.WEIGHT, jnp.float32(self.grow_scale)))


class NePrune(px.PruneConn):
    """Tombstone the incoming edges of a fully dormant interior unit.

    The paper's threshold is exactly zero -- only units whose activation has
    genuinely collapsed, not merely small ones. `tau` is exposed because at
    batch 1 the statistic is an EMA over steps rather than a batch average, so
    an exact-zero test is reachable but brittle; the default keeps the paper's
    intent while tolerating float noise.

    The `omega` cap arrives as a per-unit probability, so the expected number
    pruned matches the paper's random truncation to a per-layer budget.
    """

    def __init__(self, tau: float = 1e-6) -> None:
        """Bind the dormancy threshold."""
        self.tau = tau

    def predicate(
        self, u: px.UnitView, c: px.ConnView, cid: px.ConnIdx, g: None
    ) -> jax.Array:
        """Prune when the destination is dormant, elastic and wins its coin."""
        del g
        dst = px.UnitIdx(c[px.TO_ID, cid])
        dormant = u[ACT_EMA, dst] <= jnp.float32(self.tau)
        elastic = u[NE_ELASTIC, dst] > jnp.float32(0.5)
        src = px.UnitIdx(c[px.FROM_ID, cid])
        keep = _hash01(src, dst, u[SET_CURSOR, dst]) < u[NE_PRUNE_P, dst]
        return dormant & elastic & keep


def make_net(
    optimizer: px.optim.Optimizer,
    *,
    mode: str,
    tau: float = 1e-6,
    max_candidates: int = 4096,
    grow_scale: float = 0.0,
    shortlist: int | None = None,
    ema_decay: float = 0.05,
) -> type[px.Network[None]]:
    """Build the train / churn / eval net for NE.

    Args:
        optimizer: the plastax.optim bundle.
        mode: ``"train"``, ``"churn"`` or ``"eval"``.
        tau: dormancy threshold (churn); the paper's value is 0.
        max_candidates: per-bucket growth bound (churn).
        grow_scale: grown-edge init weight (churn).
        shortlist: M for the candidate grid, or None for exhaustive (churn).
        ema_decay: activation-EMA rate feeding the dormancy statistic.

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
            forward_pass = NeStats()
            prune_conn = NePrune(tau)
            add_conn = NeGrow(max_candidates, grow_scale, shortlist)
            extra_unit_fields = _UNIT_FIELDS
            extra_conn_fields = optimizer.state_fields
            propagation = px.Propagation.TOPOLOGICAL
            neighbourhood = 1

        return _Churn

    raise ValueError(f"make_net: unknown mode {mode!r}")


def erdos_renyi_budget(n_src: int, n_dst: int, density: float) -> int:
    """Live-edge count for an Erdos-Renyi layer at the given overall density.

    The paper uses ER rather than uniform initialisation because the connection
    count then scales with the SUM of input and output channels rather than
    their product, which keeps narrow layers from being starved. `density` is
    interpreted as the fraction of the dense layer to allocate, and the ER shape
    determines how it is spread.

    Args:
        n_src: source-layer width.
        n_dst: destination-layer width.
        density: target fraction of the dense layer.

    Returns:
        The live-edge budget, at least one edge per destination.
    """
    return max(n_dst, min(n_src * n_dst, int(round(density * n_src * n_dst))))


def build_ne_net(
    net: type[px.Network[None]],
    layers: tuple[int, ...],
    *,
    initial_density: float = 0.2,
    terminal_density: float = 1.0,
    seed: int = 0,
) -> tuple[px.NetworkStatic, px.NetworkState[None]]:
    """Build a net with DENSE ends and a sparse, growable interior.

    The first and last transitions are fully connected, as the paper requires;
    only interior transitions start sparse. Interior buckets are given enough
    dead-slot headroom to reach `terminal_density` without a host-side
    `grow_bucket` rebuild mid-run, since an overflow forces a retrace.

    Args:
        net: the Network type whose field layout the arenas adopt.
        layers: units per layer, input first (including the bias input). At
            least four entries, or there is no interior to grow.
        initial_density: interior density at the start of training.
        terminal_density: interior density the headroom must accommodate.
        seed: numpy PRNG seed.

    Returns:
        The finalized (static, state) pair, with IS_OUT and NE_ELASTIC marked.

    Raises:
        ValueError: if `layers` has no interior transition.
    """
    if len(layers) < 4:
        raise ValueError(
            f"build_ne_net: {len(layers)} layers leaves no interior transition; "
            "NE keeps the first and last dense, so it would have nothing to grow"
        )
    rng = np.random.default_rng(seed)
    offsets, running = [], 0
    for size in layers:
        offsets.append(running)
        running += size

    from_chunks, to_chunks, weight_chunks = [], [], []
    interior = range(1, len(layers) - 2)
    for layer, (n_src, n_dst) in enumerate(zip(layers[:-1], layers[1:], strict=True)):
        if layer in interior:
            budget = erdos_renyi_budget(n_src, n_dst, initial_density)
            flat = rng.choice(n_src * n_dst, size=budget, replace=False)
            src_local = (flat // n_dst).astype(np.int32)
            dst_local = (flat % n_dst).astype(np.int32)
        else:
            grid_src, grid_dst = np.meshgrid(
                np.arange(n_src), np.arange(n_dst), indexing="ij"
            )
            src_local = grid_src.ravel().astype(np.int32)
            dst_local = grid_dst.ravel().astype(np.int32)
        from_chunks.append(src_local + offsets[layer])
        to_chunks.append(dst_local + offsets[layer + 1])
        fan_in = max(1.0, len(src_local) / n_dst)
        weight_chunks.append(rng.standard_normal(len(src_local)) / np.sqrt(fan_in))

    headroom = max(0.0, terminal_density / max(initial_density, 1e-9) - 1.0)
    static, state = px.NetworkBuilder.from_edges(
        net,
        running,
        np.concatenate(from_chunks),
        np.concatenate(to_chunks),
        weights=np.concatenate(weight_chunks).astype(np.float32),
        input_ids=tuple(range(layers[0])),
        output_ids=tuple(range(offsets[-1], running)),
        globals_=None,
        capacity_headroom=headroom,
    )
    return static, mark_elastic(static, mark_outputs(static, state), layers)


def mark_elastic(
    static: px.NetworkStatic,
    state: px.NetworkState[None],
    layers: tuple[int, ...],
) -> px.NetworkState[None]:
    """Flag the units whose incoming transition NE may modify.

    Those are the destinations of interior transitions: never the first hidden
    layer (its incoming transition is the encoder) and never the outputs (theirs
    is the decoder).

    Args:
        static: the network's static config.
        state: the state to mark.
        layers: the layer widths used to build it.

    Returns:
        The state with `ne/elastic` written.
    """
    offsets, running = [], 0
    for size in layers:
        offsets.append(running)
        running += size
    flags = np.zeros(static.num_units, dtype=np.float32)
    for layer in range(1, len(layers) - 2):
        flags[offsets[layer + 1] : offsets[layer + 1] + layers[layer + 1]] = 1.0
    state.units = {**state.units, NE_ELASTIC.name: jnp.asarray(flags)}
    return state


def cosine_growth_rate(cycle: int, num_cycles: int, alpha: float) -> float:
    """The paper's annealed growth rate `(alpha/2)(1 + cos(t*pi/T_end))`.

    Args:
        cycle: current cycle index.
        num_cycles: T_end, the cycle at which growth shuts down.
        alpha: initial growth rate.

    Returns:
        The rate for this cycle; exactly 0 at `num_cycles`.
    """
    if num_cycles <= 0:
        return 0.0
    t = min(cycle, num_cycles)
    return 0.5 * alpha * (1.0 + math.cos(math.pi * t / num_cycles))


def set_growth_rate(state: px.NetworkState[None], rate: float) -> px.NetworkState[None]:
    """Broadcast this cycle's growth rate to every elastic unit."""
    elastic = state.units[NE_ELASTIC.name]
    column = jnp.where(elastic > 0.5, jnp.float32(rate), jnp.float32(0.0))
    state.units = {**state.units, NE_RATE.name: column}
    return state


def set_prune_probability(
    state: px.NetworkState[None],
    *,
    omega: float,
    expected_grown: float,
    tau: float = 1e-6,
) -> px.NetworkState[None]:
    """Express the paper's `omega * |I_grow|` per-layer cap as a per-unit coin.

    The paper truncates the prune set at random when it exceeds the cap. With no
    global count available inside the trace, the host computes how many dormant
    units there are and converts the budget into an independent per-unit
    probability, which matches the truncation in expectation.

    Args:
        state: the state to read dormancy from and write the probability to.
        omega: the paper's discount factor in [0, 1].
        expected_grown: this cycle's expected growth count, `|I_grow|`.
        tau: dormancy threshold.

    Returns:
        The state with `ne/prune_p` written.
    """
    elastic = np.asarray(state.units[NE_ELASTIC.name]) > 0.5
    dormant = (np.asarray(state.units[ACT_EMA.name]) <= tau) & elastic
    count = int(dormant.sum())
    budget = omega * expected_grown
    probability = 1.0 if count == 0 else min(1.0, budget / count)
    column = np.where(elastic, np.float32(probability), np.float32(0.0))
    state.units = {**state.units, NE_PRUNE_P.name: jnp.asarray(column)}
    return state
