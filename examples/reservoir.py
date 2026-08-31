"""A very wide, very sparse echo state network on the O(E) edge arena.

This is the regime the arena is actually built for. A reservoir's recurrent
matrix is the classic O(N^2) object: at N = 10^6 units a dense W is 4 TB and
simply cannot be instantiated, while the SAME reservoir at fan-in 8 is 8x10^6
edges -- a few hundred MB. The interesting claim here is not "faster than
dense", it is that dense does not exist at this size.

Three things this example does that `echo_state_network.py` (120 units, 10%
density, an O(N^2) Python `add_conn` loop) cannot:

* **Vectorized construction.** Edges are built as whole (E,) numpy columns and
  handed to `NetworkBuilder.from_edges`, so a 10^6-unit reservoir builds in one
  pass instead of 10^12 Python calls.
* **Analytic spectral scaling.** Fixing the echo state property normally means
  an eigendecomposition of W, which is impossible at this size. For a random
  matrix with exactly `fan_in` nonzeros per row drawn from N(0, sigma^2), the
  circular law puts the spectral radius at ~sigma*sqrt(fan_in), so
  `sigma = rho / sqrt(fan_in)` sets it directly -- O(1) instead of O(N^3).
* **Verification through the network itself.** That prediction is then checked
  by power iteration driven by plastax: the reservoir run with a LINEAR apply
  and zero input IS the operator `x -> Wx`, so the growth ratio of ||x|| over a
  few steps estimates the spectral radius using only O(E) work.

Cycles are why this is PIPELINE mode: the reservoir's recurrent connectivity is
exactly what topological propagation rejects, and pipeline's single synchronous
sweep over the previous step's activations is the `W x(t-1)` feedback term.

Only the linear readout is trained (reservoir computing's defining trade), so
there is no backward pass, no optimizer state, and one float per edge.

Run:  uv run python examples/reservoir.py
      PLX_UNITS=1000000 PLX_FAN_IN=8 uv run python examples/reservoir.py
"""

from __future__ import annotations

import os
import time

import jax
import jax.numpy as jnp
import numpy as np

import plastax as px

WASHOUT = 400  # steps discarded before the readout sees the state;
#            also the largest delay the memory-capacity fit can score


class TanhReservoir(px.ForwardPass):
    """Reservoir update: weighted sum of the PREVIOUS step's activations, tanh.

    In pipeline mode every edge lives in one flat bucket and the sweep reads
    the previous step's activations, so a reservoir->reservoir edge realizes
    the `W x(t-1)` recurrence.
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
        """Contribute one edge's weighted source activation."""
        del dst, g
        return c[px.WEIGHT, cid] * u[px.ACTIVATION, src]

    def apply(
        self, u: px.UnitView, i: px.UnitIdx, g: None, acc: jax.Array
    ) -> px.UnitWrite:
        """Squash the accumulated pre-activation."""
        del u, i, g
        return px.UnitWrite.of((px.ACTIVATION, jnp.tanh(acc)))


class LinearReservoir(px.ForwardPass):
    """The same sweep with no squashing: the raw linear operator `x -> Wx`.

    Used only by `spectral_radius`, where the point is to iterate the actual
    linear map the tanh reservoir is built around.
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
        """Contribute one edge's weighted source activation."""
        del dst, g
        return c[px.WEIGHT, cid] * u[px.ACTIVATION, src]

    def apply(
        self, u: px.UnitView, i: px.UnitIdx, g: None, acc: jax.Array
    ) -> px.UnitWrite:
        """Pass the accumulated pre-activation through unchanged."""
        del u, i, g
        return px.UnitWrite.of((px.ACTIVATION, acc))


class Reservoir(px.Network[None]):
    """Sparse recurrent tanh reservoir, propagated in pipeline mode."""

    forward_pass = TanhReservoir()
    propagation = px.Propagation.PIPELINE


class LinearProbe(px.Network[None]):
    """Field-identical sibling of `Reservoir` with the squashing removed."""

    forward_pass = LinearReservoir()
    propagation = px.Propagation.PIPELINE


def build_reservoir(
    net: type[px.Network[None]],
    num_units: int,
    *,
    num_inputs: int = 1,
    fan_in: int = 8,
    spectral_radius: float = 0.95,
    input_scaling: float = 1.0,
    seed: int = 0,
    sharding: px.ShardSpec | None = None,
) -> tuple[px.NetworkStatic, px.NetworkState[None]]:
    """Build a sparse recurrent reservoir with `fan_in` incoming edges per unit.

    Recurrent weights are drawn N(0, sigma^2) with `sigma = spectral_radius /
    sqrt(fan_in)`, which places the circular-law spectral radius at
    `spectral_radius` without ever forming W (see the module docstring).

    Sources are drawn with replacement, so a unit may receive two edges from the
    same source; those act as one edge of the summed weight and leave the
    spectral scaling unchanged in expectation.

    Args:
        net: the Network type whose field layout the arenas adopt.
        num_units: reservoir size.
        num_inputs: number of input units (driven each step).
        fan_in: recurrent incoming edges per reservoir unit.
        spectral_radius: target spectral radius; < 1 for the echo state
            property.
        input_scaling: magnitude of the input->reservoir weights.
        seed: numpy PRNG seed.
        sharding: Scheme-A ShardSpec for per-shard construction, or None.

    Returns:
        The finalized (static, state) pair. Unit ids [0, num_inputs) are the
        inputs; the reservoir occupies the rest.
    """
    rng = np.random.default_rng(seed)
    total = num_inputs + num_units
    reservoir = np.arange(num_units, dtype=np.int64) + num_inputs

    # recurrent: fan_in random sources per reservoir unit.
    rec_dst = np.repeat(reservoir, fan_in)
    rec_src = rng.integers(0, num_units, size=num_units * fan_in) + num_inputs
    sigma = spectral_radius / np.sqrt(fan_in)
    rec_w = rng.standard_normal(rec_dst.size) * sigma

    # input -> reservoir: every reservoir unit sees every input.
    in_dst = np.repeat(reservoir, num_inputs)
    in_src = np.tile(np.arange(num_inputs, dtype=np.int64), num_units)
    in_w = rng.uniform(-input_scaling, input_scaling, size=in_dst.size)

    return px.NetworkBuilder.from_edges(
        net,
        total,
        np.concatenate([rec_src, in_src]).astype(np.int32),
        np.concatenate([rec_dst, in_dst]).astype(np.int32),
        weights=np.concatenate([rec_w, in_w]).astype(np.float32),
        input_ids=tuple(range(num_inputs)),
        output_ids=(),
        globals_=None,
        sharding=sharding,
    )


def spectral_radius(
    static: px.NetworkStatic,
    state: px.NetworkState[None],
    num_inputs: int,
    *,
    iterations: int = 24,
    seed: int = 0,
) -> float:
    """Estimate the recurrent operator's spectral radius by power iteration.

    Runs the reservoir with a LINEAR apply and zero drive, which is exactly
    `x -> Wx`, renormalising between steps. Costs O(E) per iteration, so it
    stays available at sizes where an eigendecomposition is not.

    Args:
        static: the reservoir's static config.
        state: a reservoir state (not consumed; its activations are overwritten).
        num_inputs: number of leading input units to hold at zero.
        iterations: power-iteration steps.
        seed: PRNG seed for the starting vector.

    Returns:
        The estimated spectral radius.
    """
    step = px.make_step(LinearProbe, static)
    rng = np.random.default_rng(seed)
    vector = rng.standard_normal(static.num_units).astype(np.float32)
    vector[:num_inputs] = 0.0
    vector /= np.linalg.norm(vector)
    state.units = {**state.units, px.ACTIVATION.name: jnp.asarray(vector)}
    drive = jnp.zeros((num_inputs,), dtype=jnp.float32)

    ratio = 0.0
    for _ in range(iterations):
        state = step(state, px.StepInputs(inputs=drive, targets=None)).state
        activations = np.array(state.units[px.ACTIVATION.name])
        activations[:num_inputs] = 0.0
        norm = float(np.linalg.norm(activations))
        if norm == 0.0 or not np.isfinite(norm):
            return norm
        ratio = norm
        state.units = {
            **state.units,
            px.ACTIVATION.name: jnp.asarray(activations / norm),
        }
    return ratio


def drive(
    static: px.NetworkStatic,
    state: px.NetworkState[None],
    signal: np.ndarray,
    probe_ids: np.ndarray,
) -> tuple[np.ndarray, px.NetworkState[None], float]:
    """Run the reservoir over `signal`, recording the probed units each step.

    Args:
        static: the reservoir's static config.
        state: the reservoir state to advance.
        signal: (T, num_inputs) drive sequence.
        probe_ids: unit ids to record (the readout's feature set).

    Returns:
        A (T, len(probe_ids)) state matrix, the final state, and the median
        per-step latency in seconds.
    """
    step = px.make_step(Reservoir, static)
    rows = np.empty((signal.shape[0], probe_ids.size), dtype=np.float32)
    probe = jnp.asarray(probe_ids)
    latencies: list[float] = []
    for t in range(signal.shape[0]):
        started = time.perf_counter()
        result = step(
            state,
            px.StepInputs(
                inputs=jnp.asarray(signal[t], dtype=jnp.float32), targets=None
            ),
        )
        state = result.state
        recorded = state.units[px.ACTIVATION.name][probe]
        rows[t] = np.asarray(recorded)
        latencies.append(time.perf_counter() - started)
    return rows, state, float(np.median(latencies))


def memory_capacity(
    states: np.ndarray,
    signal: np.ndarray,
    max_delay: int,
    *,
    ridge: float = 1e-6,
    train_fraction: float = 0.6,
) -> np.ndarray:
    """Score how well a linear readout recovers `u(t-d)` for d = 1..max_delay.

    The standard short-term memory metric: r^2 between a ridge readout's
    prediction and the delayed input. Summing over a CONTIGUOUS delay range
    gives the memory capacity MC, bounded above by the number of readout
    features -- so a probed subsample measures a LOWER BOUND on the true
    capacity.

    r^2 is computed OUT OF SAMPLE, on a held-out tail of the run. This is not
    a refinement: with F readout features and only a few times F training rows,
    an in-sample fit reconstructs the delayed input partly from noise, and the
    forgetting curve then flattens onto a spurious floor (~0.3 here) instead of
    decaying to zero. That floor silently inflates MC by a large factor.

    One Cholesky factorization of the training Gram matrix is shared across
    every delay; only the right-hand side changes.

    Args:
        states: (T, F) recorded reservoir states.
        signal: (T, num_inputs) drive that produced them.
        max_delay: largest delay to score; must be <= WASHOUT.
        ridge: Tikhonov regularisation on the normal equations.
        train_fraction: fraction of post-washout rows used to fit; the rest
            score.

    Returns:
        A (max_delay,) array of held-out r^2 values, index d-1 holding delay d.
    """
    import scipy.linalg as sla

    design = np.concatenate(
        [np.ones((states.shape[0], 1), dtype=np.float64), states.astype(np.float64)],
        axis=1,
    )
    predictors = design[WASHOUT:]
    split = int(predictors.shape[0] * train_fraction)
    train, test = predictors[:split], predictors[split:]
    gram = train.T @ train
    gram[np.diag_indices_from(gram)] += ridge * float(np.trace(gram)) / gram.shape[0]
    factor = sla.cho_factor(gram)

    curve = np.zeros((max_delay,), dtype=np.float64)
    for d in range(1, max_delay + 1):
        target = signal[WASHOUT - d : states.shape[0] - d, 0].astype(np.float64)
        weights = sla.cho_solve(factor, train.T @ target[:split])
        prediction = test @ weights
        held_out = target[split:]
        if np.std(prediction) < 1e-12 or np.std(held_out) < 1e-12:
            continue
        correlation = float(np.corrcoef(prediction, held_out)[0, 1])
        curve[d - 1] = correlation * correlation
    return curve


def main() -> None:
    """Build a wide sparse reservoir, verify it, and measure its memory."""
    num_units = int(os.environ.get("PLX_UNITS", 200_000))
    fan_in = int(os.environ.get("PLX_FAN_IN", 8))
    probe_dim = int(os.environ.get("PLX_PROBE", 1024))
    steps = int(os.environ.get("PLX_STEPS", 1200))
    rho = float(os.environ.get("PLX_RHO", 0.95))
    num_inputs = 1

    started = time.perf_counter()
    static, state = build_reservoir(
        Reservoir,
        num_units,
        num_inputs=num_inputs,
        fan_in=fan_in,
        spectral_radius=rho,
        seed=0,
    )
    build_s = time.perf_counter() - started
    live = int(px.state.live_conn_count(state))
    density = live / float(num_units) ** 2
    dense_gb = (float(num_units) ** 2) * 4 / 1e9
    sparse_mb = live * 12 / 1e6  # weight + from/to ids, no optimizer state
    print(f"device: {jax.devices()[0]}")
    print(
        f"reservoir: {num_units:,} units  fan_in={fan_in}  {live:,} live edges\n"
        f"  density {density:.3e} ({100 * (1 - density):.4f}% sparse)  "
        f"built in {build_s:.1f}s"
    )
    print(
        f"  plastax edge state ~{sparse_mb:,.0f} MB   "
        f"dense W would be {dense_gb:,.1f} GB"
        + ("  <- does not fit any GPU" if dense_gb > 40 else "")
    )

    estimated = spectral_radius(static, state, num_inputs)
    print(f"  spectral radius: target {rho:.3f}, power-iteration {estimated:.3f}")

    rng = np.random.default_rng(1)
    probe_ids = (
        np.sort(
            rng.choice(num_units, size=min(probe_dim, num_units), replace=False)
        ).astype(np.int32)
        + num_inputs
    )

    impulse = np.zeros((40, num_inputs), dtype=np.float32)
    impulse[0, :] = 1.0
    # make_step donates the state pytree, so each drive run needs its own.
    _, impulse_state = build_reservoir(
        Reservoir,
        num_units,
        num_inputs=num_inputs,
        fan_in=fan_in,
        spectral_radius=rho,
        seed=0,
    )
    decay, _, _ = drive(static, impulse_state, impulse, probe_ids)
    norms = np.linalg.norm(decay, axis=1)
    print(
        "  impulse response (echo state property) at steps 0/5/10/20/39: "
        f"{norms[0]:.3f} / {norms[5]:.3f} / {norms[10]:.3f} / "
        f"{norms[20]:.3f} / {norms[39]:.4f}"
    )

    signal = rng.uniform(-1.0, 1.0, size=(steps, num_inputs)).astype(np.float32)
    _, drive_state = build_reservoir(
        Reservoir,
        num_units,
        num_inputs=num_inputs,
        fan_in=fan_in,
        spectral_radius=rho,
        seed=0,
    )
    started = time.perf_counter()
    states, _, per_step = drive(static, drive_state, signal, probe_ids)
    total_s = time.perf_counter() - started
    print(
        f"  drove {steps} steps in {total_s:.1f}s  "
        f"({per_step * 1e3:.2f} ms/step, {live / per_step / 1e6:,.0f}M edges/s)"
    )

    max_delay = int(os.environ.get("PLX_MAX_DELAY", WASHOUT))
    curve = memory_capacity(states, signal, max_delay)
    print(f"  forgetting curve (ridge readout over {probe_ids.size} probed units):")
    for d in (1, 2, 4, 8, 16, 32, 64, 128, 256, max_delay):
        if d <= max_delay:
            print(f"    u(t-{d:4d}): r^2 {curve[d - 1]:.3f}")
    capacity = float(curve.sum())
    print(
        f"  memory capacity MC = {capacity:.1f} "
        f"(sum of r^2 over delays 1..{max_delay}; "
        f"bounded by the {probe_ids.size} readout features)"
    )


if __name__ == "__main__":
    main()
