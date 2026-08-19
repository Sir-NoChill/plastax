"""Recurrent Echo State Network in pipeline propagation mode.

An Echo State Network is a fixed random recurrent reservoir read out by a
trained linear layer. The reservoir's recurrent (cyclic) connectivity is
exactly what topological propagation forbids -- the builder levels the graph
with Kahn's algorithm and rejects cycles -- so the reservoir is built in
PIPELINE mode instead. There every connection lives in one flat bucket and
each step is a single synchronous sweep reading the *previous* step's
activations; that one-step feedback IS the recurrence,

    x(t) = tanh(W_in u(t) + W x(t-1)),

with reservoir->reservoir edges (including self-loops and cycles) supplying
the W x(t-1) term. Reservoir weights are scaled to a spectral radius below 1
so past inputs fade -- the echo state property.

Only the linear readout is trained, here by least squares over the collected
reservoir states. main() drives the reservoir with white noise and measures
its short-term memory: how well u(t-d) can be reconstructed from x(t) for a
range of delays d (the forgetting curve). Run it directly:
`python examples/echo_state_network.py`. It runs on CPU anywhere.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

import plastax as px
from plastax.views import UnitWrite

N_INPUTS = 1
N_RESERVOIR = 120
DENSITY = 0.1  # fraction of nonzero reservoir->reservoir edges
SPECTRAL_RADIUS = 0.9  # < 1 for the echo state property
INPUT_SCALING = 1.0
WASHOUT = 50  # steps discarded before the readout sees the state
N_STEPS = 600
SEED = 0


class _TanhReservoir(px.ForwardPass):
    """Reservoir update: weighted-sum pre-activation, tanh apply.

    In pipeline mode the sum reads the previous step's activations, so a
    reservoir->reservoir edge realizes the W x(t-1) feedback term.
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
    ) -> jnp.ndarray:
        return c[px.WEIGHT, cid] * u[px.ACTIVATION, src]

    def apply(
        self, u: px.UnitView, i: px.UnitIdx, g: None, acc: jnp.ndarray
    ) -> UnitWrite:
        return UnitWrite.of((px.ACTIVATION, jnp.tanh(acc)))


class EchoStateNetwork(px.Network[None]):
    """Sparse recurrent tanh reservoir, propagated in pipeline mode."""

    forward_pass = _TanhReservoir()
    propagation = px.Propagation.PIPELINE


def _reservoir_weights(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw input weights and a spectral-radius-scaled recurrent matrix.

    Args:
        rng: Source of randomness for both weight matrices.

    Returns:
        A (w_in, w_res) pair: w_in is (N_RESERVOIR, N_INPUTS); w_res is the
        (N_RESERVOIR, N_RESERVOIR) recurrent matrix (row=dst, col=src),
        sparsified to DENSITY and rescaled to SPECTRAL_RADIUS.
    """
    w_in = INPUT_SCALING * rng.uniform(-1.0, 1.0, size=(N_RESERVOIR, N_INPUTS))
    w_res = rng.standard_normal((N_RESERVOIR, N_RESERVOIR))
    mask = rng.random((N_RESERVOIR, N_RESERVOIR)) < DENSITY
    w_res *= mask
    radius = float(np.max(np.abs(np.linalg.eigvals(w_res))))
    if radius > 0.0:
        w_res *= SPECTRAL_RADIUS / radius
    return w_in, w_res


def _build() -> tuple[px.NetworkStatic, px.NetworkState[None]]:
    """Build the recurrent reservoir: input units, then reservoir units.

    Returns:
        The finalized (NetworkStatic, NetworkState) pair. Building only
        succeeds because pipeline propagation admits the reservoir's cycles.
    """
    rng = np.random.default_rng(SEED)
    w_in, w_res = _reservoir_weights(rng)
    b = px.NetworkBuilder(EchoStateNetwork, None)
    for _ in range(N_INPUTS + N_RESERVOIR):
        b.add_unit()
    for i in range(N_INPUTS):
        b.mark_input(i)
    # input -> reservoir (dense)
    for r in range(N_RESERVOIR):
        for i in range(N_INPUTS):
            b.add_conn(i, N_INPUTS + r, weight=float(w_in[r, i]))
    # reservoir -> reservoir (sparse, recurrent: this is what needs cycles)
    for dst in range(N_RESERVOIR):
        for src in range(N_RESERVOIR):
            weight = float(w_res[dst, src])
            if weight != 0.0:
                b.add_conn(N_INPUTS + src, N_INPUTS + dst, weight=weight)
    return b.finalize()


def _collect_states(signal: np.ndarray) -> np.ndarray:
    """Drive a fresh reservoir with `signal` and collect its states each step.

    Builds its own (static, state) so each run gets independently-allocated
    arrays: make_step donates the state pytree (donate_argnums=0), so a
    state cannot be fed to two separate drive runs.

    Args:
        signal: (T, N_INPUTS) float drive sequence.

    Returns:
        A (T, N_RESERVOIR) array of reservoir activations, one row per step.
    """
    static, state = _build()
    step = px.make_step(EchoStateNetwork, static)
    states: list[np.ndarray] = []
    for t in range(signal.shape[0]):
        drive = jnp.asarray(signal[t], dtype=jnp.float32)
        result = step(state, px.StepInputs(inputs=drive, targets=None))
        state = result.state
        states.append(np.asarray(state.units[px.ACTIVATION.name][N_INPUTS:]))
    return np.stack(states)


def _forgetting_curve(
    states: np.ndarray, signal: np.ndarray, delays: tuple[int, ...]
) -> dict[int, float]:
    """Fit a linear readout per delay and score its recall of u(t-d).

    Trains W_out by least squares to reconstruct the delayed input from the
    (bias-augmented) reservoir state, then scores each delay by the squared
    correlation between prediction and target -- the standard short-term
    memory-capacity metric, ~1 while the reservoir still remembers u(t-d)
    and decaying toward 0 as d exceeds its fading-memory horizon.

    Args:
        states: (T, N_RESERVOIR) collected reservoir activations.
        signal: (T, N_INPUTS) drive sequence used to produce `states`.
        delays: Delays d to score (each must be <= WASHOUT).

    Returns:
        A mapping from delay to squared correlation r^2 in [0, 1].
    """
    design = np.concatenate([np.ones((states.shape[0], 1)), states], axis=1)
    predictors = design[WASHOUT:]
    curve: dict[int, float] = {}
    for d in delays:
        target = signal[WASHOUT - d : states.shape[0] - d, 0]
        w_out, *_ = np.linalg.lstsq(predictors, target, rcond=None)
        pred = predictors @ w_out
        corr = np.corrcoef(pred, target)[0, 1]
        curve[d] = float(corr * corr)
    return curve


def main() -> None:
    """Build the reservoir, check echo-state decay, and score its memory."""
    _, state = _build()
    # Pipeline mode keeps a single flat conn bucket (state.conns[0]).
    n_conns = int(jnp.sum(~state.conns[0][px.DEAD.name]))
    print(
        f"built recurrent reservoir: {N_RESERVOIR} units, {n_conns} live conns, "
        f"spectral radius {SPECTRAL_RADIUS} (pipeline mode)"
    )

    # Echo state property: a single input impulse should fade, not blow up.
    impulse = np.zeros((30, N_INPUTS), dtype=np.float32)
    impulse[0, :] = 1.0
    decay = _collect_states(impulse)
    norms = np.linalg.norm(decay, axis=1)
    print(
        "impulse response norm at steps 0/5/10/20: "
        f"{norms[0]:.3f} / {norms[5]:.3f} / {norms[10]:.3f} / {norms[20]:.3f}"
    )

    # Short-term memory: how far back can a linear readout recover the input?
    rng = np.random.default_rng(SEED + 1)
    signal = rng.uniform(-1.0, 1.0, size=(N_STEPS, N_INPUTS)).astype(np.float32)
    states = _collect_states(signal)
    delays = (1, 2, 4, 8, 16)
    curve = _forgetting_curve(states, signal, delays)
    print("forgetting curve (delay -> recall r^2):")
    for d in delays:
        print(f"  u(t-{d:2d}): {curve[d]:.3f}")
    print(f"memory capacity (sum over delays): {sum(curve.values()):.3f}")


if __name__ == "__main__":
    main()
