"""Port of examples/mlp-xor/mlp_xor.cpp: sigmoid MLP, MSE loss, SGD.

Canonical backprop trait example and the first oracle target: topological
mode, all four differentiable phases, no dynamics.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp

import plastax as px

GradPreAct = px.FieldSpec.f32("grad_pre_act")
# dL/dActivation, staged by MSELoss for output units only and consumed by
# SigmoidBackward.apply at the output level. dispatch_cpu.hpp stages this
# into BackwardAcc, a framework-internal per-unit column that is always
# fresh (zeroed right after the Apply that consumes it) -- plastax's
# backward accumulator instead lives as a value local to backward_phase's
# own trace closure (phases.py's build_phases), with no channel for an
# earlier, separate phase function to write into it. LossGrad is the
# Deviation that bridges the gap: written only for output_ids (every step,
# always fresh), permanently 0.0 for every other unit since nothing else
# ever touches it, so -- unlike reusing GradPreAct itself -- it can never
# carry a stale value from a previous step into a hidden unit's gradient.
LossGrad = px.FieldSpec.f32("loss_grad")

LEARNING_RATE = 0.5
NUM_EPOCHS = 5000
NUM_HIDDEN = 4
SEED = 0
PRINT_EVERY = 500

# (x1, x2, bias) -- the constant 1.0 "bias" is the third input unit, exactly
# mlp_xor.cpp's Inputs[.][2]: it lets the hidden/output units learn a
# per-unit bias through their own incoming weights with no extra plumbing.
_BIAS = 1.0
_XOR_INPUTS: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (0.0, 1.0),
    (1.0, 0.0),
    (1.0, 1.0),
)
_XOR_TARGETS: tuple[float, ...] = (0.0, 1.0, 1.0, 0.0)


class SigmoidForward(px.ForwardPass):
    """map = weight*activation[src]; apply = sigmoid(acc) (mlp_xor.cpp's
    SigmoidForwardPass)."""

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
    ) -> px.UnitWrite:
        return px.UnitWrite.of((px.ACTIVATION, jax.nn.sigmoid(acc)))


class SigmoidBackward(px.BackwardPass):
    """Direction-reversed accumulate (mlp_xor.cpp's SigmoidBackwardPass):
    map reads the destination's ALREADY-finalized dL/dz (grad_pre_act,
    the reverse level walk guarantees it is written before an earlier
    level's map reads it), combine sums weight*grad_pre_act[dst] into the
    source's dL/da, apply converts dL/da -> dL/dz via the sigmoid
    derivative a*(1-a) and stores into grad_pre_act for a still-earlier
    level to read next.

    `acc` is the real backward-accumulated dL/da for every unit except the
    output level (dispatch_cpu.hpp:328-333: no edge sources from the
    deepest level, so the output unit's own `acc` here is always the fresh
    identity 0.0); `u[LossGrad, i]` is 0.0 for every unit except the
    output_ids MSELoss.per_output wrote this step. Summing them reproduces
    the oracle's single BackwardAcc value (module docstring, LossGrad).
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
        return c[px.WEIGHT, cid] * u[GradPreAct, src]

    def apply(
        self, u: px.UnitView, i: px.UnitIdx, g: None, acc: jnp.ndarray
    ) -> px.UnitWrite:
        a = u[px.ACTIVATION, i]
        grad_pre_act = (acc + u[LossGrad, i]) * a * (jnp.float32(1.0) - a)
        return px.UnitWrite.of((GradPreAct, grad_pre_act))


class MSELoss(px.Loss):
    """0.5*(pred-target)**2 per output unit (mlp_xor.cpp's
    plastix::MSELoss); stages dL/dActivation = pred-target into LossGrad
    for SigmoidBackward.apply to pick up at the output level."""

    def per_output(
        self, u: px.UnitView, i: px.UnitIdx, target: jnp.ndarray, g: None
    ) -> tuple[jnp.ndarray, px.UnitWrite]:
        pred = u[px.ACTIVATION, i]
        diff = pred - target
        loss = jnp.float32(0.5) * diff * diff
        return loss, px.UnitWrite.of((LossGrad, diff))


class SgdUpdateConn(px.UpdateConn):
    """w -= lr * dL/dz_dst * a_src, entirely in the incoming pass; outgoing
    is a genuine no-op (mlp_xor.cpp's GradientDescentConn)."""

    def incoming(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: None,
    ) -> px.ConnWrite:
        grad = u[GradPreAct, dst]
        activation_src = u[px.ACTIVATION, src]
        delta = jnp.float32(LEARNING_RATE) * grad * activation_src
        return px.ConnWrite.of((px.WEIGHT, c[px.WEIGHT, cid] - delta))

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


class XorNet(px.Network[None]):
    forward_pass = SigmoidForward()
    backward_pass = SigmoidBackward()
    loss = MSELoss()
    update_conn = SgdUpdateConn()
    extra_unit_fields = (GradPreAct, LossGrad)
    propagation = px.Propagation.TOPOLOGICAL


class XorNetEval(px.Network[None]):
    """Forward-only sibling sharing XorNet's exact unit/conn field sets, so
    the same (static, state) drives either's make_step (rung0 design
    section 2 phase elision): mirrors mlp_xor.cpp's separate
    Net.DoForwardPass(...) used for final, read-only inference, as opposed
    to Net.DoStep(...), which also back-propagates and updates weights."""

    forward_pass = SigmoidForward()
    extra_unit_fields = (GradPreAct, LossGrad)
    propagation = px.Propagation.TOPOLOGICAL


def build_net(key: jax.Array) -> tuple[px.NetworkStatic, px.NetworkState[None]]:
    """3 inputs (x1, x2, bias=1.0) -> NUM_HIDDEN sigmoid hidden -> 1 sigmoid
    output (mlp_xor.cpp's MlpNetwork construction)."""
    topology_fn = px.topology.sequential(
        px.topology.input_units(3),
        px.topology.dense(3, NUM_HIDDEN),
        px.topology.dense(NUM_HIDDEN, 1),
    )
    return px.NetworkBuilder.from_topology(XorNet, topology_fn, key, globals_=None)


def _step_inputs(x1: float, x2: float, target: float | None) -> px.StepInputs:
    targets = None if target is None else jnp.asarray([target], dtype=jnp.float32)
    return px.StepInputs(
        inputs=jnp.asarray([x1, x2, _BIAS], dtype=jnp.float32), targets=targets
    )


def train(
    static: px.NetworkStatic,
    state: px.NetworkState[None],
    *,
    num_epochs: int = NUM_EPOCHS,
    verbose: bool = True,
) -> tuple[px.NetworkState[None], float]:
    """Trains on the 4 XOR patterns each epoch (mlp_xor.cpp's main loop:
    one Net.DoStep per pattern, patterns fixed order every epoch). Returns
    the trained state and the final epoch's total loss."""
    step = px.make_step(XorNet, static)
    epoch_loss = jnp.float32(0.0)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error", message=".*Some donated buffers were not usable.*"
        )
        for epoch in range(num_epochs):
            epoch_loss = jnp.float32(0.0)
            for (x1, x2), target in zip(_XOR_INPUTS, _XOR_TARGETS, strict=True):
                result = step(state, _step_inputs(x1, x2, target))
                state = result.state
                epoch_loss = epoch_loss + result.loss
            if verbose and epoch % PRINT_EVERY == 0:
                print(f"Epoch {epoch:5d}  loss: {float(epoch_loss):.4f}")
    return state, float(epoch_loss)


def evaluate(
    static: px.NetworkStatic, state: px.NetworkState[None]
) -> tuple[px.NetworkState[None], bool, tuple[float, ...]]:
    """Forward-only pass over the 4 patterns (mlp_xor.cpp's final
    Net.DoForwardPass loop). Returns the state, whether every pattern is
    classified correctly (threshold 0.5), and the raw predictions."""
    eval_step = px.make_step(XorNetEval, static)
    output_id = static.output_ids[0]
    ok = True
    predictions: list[float] = []
    print("\nFinal predictions:")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error", message=".*Some donated buffers were not usable.*"
        )
        for (x1, x2), target in zip(_XOR_INPUTS, _XOR_TARGETS, strict=True):
            result = eval_step(state, _step_inputs(x1, x2, None))
            state = result.state
            pred = float(state.units[px.ACTIVATION.name][output_id])
            predictions.append(pred)
            predicted_class = 1 if pred > 0.5 else 0
            want = int(target)
            print(
                f"  [{x1:.1f}, {x2:.1f}] -> {pred:.4f} "
                f"(class {predicted_class}, target {want})"
            )
            if predicted_class != want:
                ok = False
    return state, ok, tuple(predictions)


def main() -> None:
    print("Plastax MLP / XOR Example")
    print("=========================")
    key = jax.random.PRNGKey(SEED)
    static, state = build_net(key)
    state, final_loss = train(static, state)
    _, ok, _ = evaluate(static, state)
    print(f"\nFinal epoch loss: {final_loss:.6f}")
    print("PASS" if ok else "FAIL")
    assert ok, "mlp_xor failed to learn XOR"


if __name__ == "__main__":
    main()
