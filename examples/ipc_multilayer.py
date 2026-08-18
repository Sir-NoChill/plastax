"""Port of examples/ipc-multilayer/ipc_multilayer.cpp: streaming iPC, the
flagship PIPELINE example and M4 oracle target.

Streaming incremental Predictive Coding on a 3 -> 16 -> 1 regression net.
Layers (generative / top-down convention): input = top, output = bottom,
one persistent hidden layer between. theta^(l) predicts layer l from
f(x^(l+1)); per observation the net runs StepsPerObs simultaneous iPC steps
and value nodes are NEVER reset between observations (the "streaming"
property).

Mapping to plastax. The three framework kernels port to traits:
  * IpcForward   -- mu = sum W * f(x_src); apply writes eps = ValueNode - mu.
  * IpcBackward  -- accumulates W * eps(downstream) into each source
                    (BackwardPass reverses direction, traits.py); apply
                    writes the hidden bottom-up signal f'(x) * (theta^T eps).
  * IpcUpdateConn -- Hebbian theta += alpha * eps_dst * f(x_src).
The C++ iPCStep wraps those kernels in two HOST-side per-unit loops -- a
pre-step `Activation = f(ValueNode)` copy and the value-node update
`x += gamma*(-eps + bottom_up)` -- neither of which is a plastax phase.
Both stay host-side here too (faithful: they are host loops in C++ as well):
the pre-step copy is dropped entirely by reading f(ValueNode[src]) directly
in the forward map, and the value-node update runs between step() calls in
`run` (state.units surgery, hidden units only). Boundary value nodes
(inputs, output) are clamped host-side once per observation.

f = identity for input sources (id < NumInputs), tanh for hidden sources --
exactly the oracle's `(SrcId < NumInputs) ? XSrc : Act(XSrc)`.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

import plastax as px

ValueNode = px.FieldSpec.float32("value_node")  # persistent predictive-coding state x
Error = px.FieldSpec.float32("error")  # eps = ValueNode - mu, written by forward apply
BottomUp = px.FieldSpec.float32(
    "bottom_up"
)  # f'(x) * theta^T eps, written by backward apply

NUM_INPUTS = 3
HIDDEN_SIZE = 16
OUTPUT_SIZE = 1
HIDDEN_BEGIN = NUM_INPUTS  # unit ids: [0,3) input, [3,19) hidden, {19} output
HIDDEN_END = HIDDEN_BEGIN + HIDDEN_SIZE
OUTPUT_BEGIN = HIDDEN_END
NUM_LAYERS = 3
STEPS_PER_OBS = 2 * NUM_LAYERS

ALPHA = 0.005  # weight learning rate
GAMMA = 0.1  # value-node inference rate

TOTAL_STEPS = 2000
PRINT_INTERVAL = 100
SEED = 42


def _f(x: jax.Array, src: px.UnitIdx) -> jax.Array:
    """Source activation: identity for input units, tanh for hidden units
    (output units are never a connection source, so no third case)."""
    return jnp.where(src < NUM_INPUTS, x, jnp.tanh(x))


class IpcForward(px.ForwardPass):
    """mu = sum_src W * f(ValueNode[src]); apply writes eps = ValueNode - mu
    into Error for every unit (ipc_multilayer.cpp's iPCForwardPass, minus
    the pre-step copy -- f(ValueNode) is read inline here)."""

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
        del dst, g
        return c[px.WEIGHT, cid] * _f(u[ValueNode, src], src)

    def apply(
        self, u: px.UnitView, i: px.UnitIdx, g: None, acc: jax.Array
    ) -> px.UnitWrite:
        del g
        return px.UnitWrite.of((Error, u[ValueNode, i] - acc))


class IpcBackward(px.BackwardPass):
    """Accumulates W * Error(downstream) into each source unit; apply stores
    the hidden bottom-up signal f'(x) * (theta^T eps) (iPCBackwardPass).

    In the backward map the framework binds `src` to the edge's ToId (the
    downstream/destination unit) and accumulates into `dst` = FromId (the
    source), so `u[Error, src]` reads eps at the destination -- matching the
    oracle's `GetField<ErrorTag>(U, ToId)` (sweep.py: map's first unit-id
    arg is always the accumulator target)."""

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
        del dst, g
        return c[px.WEIGHT, cid] * u[Error, src]

    def apply(
        self, u: px.UnitView, i: px.UnitIdx, g: None, acc: jax.Array
    ) -> px.UnitWrite:
        del g
        x = u[ValueNode, i]
        deriv = jnp.float32(1.0) - jnp.tanh(x) ** 2
        is_hidden = (i >= HIDDEN_BEGIN) & (i < HIDDEN_END)
        # Input/output units are clamped: the oracle leaves their BottomUp
        # untouched; 0.0 is equivalent since only hidden BottomUp is read.
        return px.UnitWrite.of((BottomUp, jnp.where(is_hidden, deriv * acc, 0.0)))


class IpcUpdateConn(px.UpdateConn):
    """Local Hebbian rule theta += alpha * eps_dst * f(x_src), entirely in
    the incoming pass; outgoing is a genuine no-op (iPCUpdateConn)."""

    def incoming(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: None,
    ) -> px.ConnWrite:
        del g
        eps = u[Error, dst]
        fx = _f(u[ValueNode, src], src)
        return px.ConnWrite.of(
            (px.WEIGHT, c[px.WEIGHT, cid] + jnp.float32(ALPHA) * eps * fx)
        )

    def outgoing(
        self,
        u: px.UnitView,
        src: px.UnitIdx,
        dst: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: None,
    ) -> px.ConnWrite:
        del u, src, dst, c, cid, g
        return px.ConnWrite.of()


class IpcNet(px.Network[None]):
    forward_pass = IpcForward()
    backward_pass = IpcBackward()
    update_conn = IpcUpdateConn()
    extra_unit_fields = (ValueNode, Error, BottomUp)
    propagation = px.Propagation.PIPELINE


def build_net(key: jax.Array) -> tuple[px.NetworkStatic, px.NetworkState[None]]:
    """3 inputs -> 16 tanh hidden -> 1 output, fully connected, LeCun-uniform
    init (U(-sqrt(3/fan_in), sqrt(3/fan_in)) -- the oracle's per-destination
    BoundHidden/BoundOutput, since fan_in is 3 for the hidden layer and 16
    for the output layer)."""
    lecun = jax.nn.initializers.lecun_uniform()
    topology_fn = px.topology.sequential(
        px.topology.input_units(NUM_INPUTS),
        px.topology.dense(NUM_INPUTS, HIDDEN_SIZE, init=lecun),
        px.topology.dense(HIDDEN_SIZE, OUTPUT_SIZE, init=lecun),
    )
    static, state = px.NetworkBuilder.from_topology(
        IpcNet, topology_fn, key, globals_=None
    )
    # The unit-id layout the module constants assume (inputs, then hidden,
    # then the single output) -- assert it so a topology change fails loudly
    # rather than silently miswiring the hidden-only value-node update.
    assert state.units[ValueNode.name].shape == (OUTPUT_BEGIN + OUTPUT_SIZE,)
    assert int(static.output_ids[0]) == OUTPUT_BEGIN
    return static, state


def _clamp_boundaries(
    state: px.NetworkState[None], inputs: np.ndarray, y: float
) -> px.NetworkState[None]:
    """Clamp the boundary value nodes once per observation: ValueNode[input]
    = x_in, ValueNode[output] = y (the oracle's per-observation clamp)."""
    vn = state.units[ValueNode.name]
    vn = vn.at[jnp.arange(NUM_INPUTS)].set(jnp.asarray(inputs, dtype=jnp.float32))
    vn = vn.at[OUTPUT_BEGIN].set(jnp.float32(y))
    return dataclasses.replace(state, units={**state.units, ValueNode.name: vn})


def _value_node_update(state: px.NetworkState[None]) -> px.NetworkState[None]:
    """Hidden-unit inference step x += gamma*(-eps + bottom_up), host-side
    (the oracle's post-kernel value-node loop). Boundary units are excluded
    by the mask, so their clamped values persist across iPC steps."""
    vn = state.units[ValueNode.name]
    eps = state.units[Error.name]
    bottom_up = state.units[BottomUp.name]
    ids = jnp.arange(vn.shape[0])
    is_hidden = (ids >= HIDDEN_BEGIN) & (ids < HIDDEN_END)
    delta = jnp.float32(GAMMA) * (-eps + bottom_up)
    vn = jnp.where(is_hidden, vn + delta, vn)
    return dataclasses.replace(state, units={**state.units, ValueNode.name: vn})


def run(
    static: px.NetworkStatic,
    state: px.NetworkState[None],
    *,
    total_steps: int = TOTAL_STEPS,
    seed: int = SEED,
    verbose: bool = True,
) -> tuple[float, float]:
    """Stream `total_steps` iPC steps over y = 2*x1 - x2 + 0.5*x3. Returns
    the final print-window (avg iPC error^2, avg predict-previous baseline);
    iPC has learned iff the first < the second."""
    step = px.make_step(IpcNet, static)
    rng = np.random.default_rng(seed)

    inputs = np.zeros(NUM_INPUTS, dtype=np.float32)
    y = 0.0
    prev_y = 0.0
    avg_loss = jnp.float32(0.0)
    avg_baseline = 0.0
    last_avg_loss = 0.0
    last_avg_baseline = 0.0

    for t in range(total_steps):
        if t % STEPS_PER_OBS == 0:
            inputs = rng.uniform(-1.0, 1.0, size=NUM_INPUTS).astype(np.float32)
            prev_y = y
            y = float(2.0 * inputs[0] - inputs[1] + 0.5 * inputs[2])
            state = _clamp_boundaries(state, inputs, y)

        step_inputs = px.StepInputs(
            inputs=jnp.asarray(inputs, dtype=jnp.float32), targets=None
        )
        result = step(state, step_inputs)
        state = _value_node_update(result.state)

        # Prediction mu = ValueNode[out] - eps[out]; both live on device.
        pred = (
            state.units[ValueNode.name][OUTPUT_BEGIN]
            - state.units[Error.name][OUTPUT_BEGIN]
        )
        avg_loss = avg_loss + (pred - jnp.float32(y)) ** 2
        avg_baseline += (prev_y - y) ** 2

        if (t + 1) % PRINT_INTERVAL == 0:
            last_avg_loss = float(avg_loss) / PRINT_INTERVAL
            last_avg_baseline = avg_baseline / PRINT_INTERVAL
            if verbose:
                print(
                    f"Step {t + 1:6d}  avg error^2: {last_avg_loss:8.4f}"
                    f"  baseline: {last_avg_baseline:8.4f}"
                )
            avg_loss = jnp.float32(0.0)
            avg_baseline = 0.0

    return last_avg_loss, last_avg_baseline


def main() -> None:
    print("Plastax iPC Multi-Layer Regression")
    print("===================================")
    print("Target: y = 2*x1 - x2 + 0.5*x3")
    print(f"Steps per observation: {STEPS_PER_OBS}\n")
    key = jax.random.PRNGKey(SEED)
    static, state = build_net(key)
    ipc, baseline = run(static, state)
    print(f"\nFinal window  iPC: {ipc:.4f}  baseline: {baseline:.4f}")
    ok = ipc < baseline
    print("\n" + ("PASS (beats baseline)" if ok else "FAIL (worse than baseline)"))
    assert ok, "ipc_multilayer failed to beat the predict-previous baseline"


if __name__ == "__main__":
    main()
