"""Port of examples/mlp-xor/mlp_xor.cpp: sigmoid MLP, MSE loss, backprop.

Canonical backprop trait example and the first oracle target: topological mode,
all four differentiable phases, no dynamics. The forward/backward/loss traits
are fixed; only the optimizer changes. `main` showcases that every optimizer in
`plastax.optim` -- sgd, momentum, adam, adamw, rmsprop -- learns XOR through the
same traits, wired in via `update_conn` + `extra_conn_fields`.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp

import plastax as px

GradPreAct = px.FieldSpec.float32("grad_pre_act")
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
LossGrad = px.FieldSpec.float32("loss_grad")

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
        src: px.UnitIdx,
        dst: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: None,
    ) -> jnp.ndarray:
        return c[px.WEIGHT, cid] * u[GradPreAct, dst]

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


def make_net(optimizer: px.optim.Optimizer, *, train: bool) -> type[px.Network[None]]:
    """Build the XOR MLP Network class wired to `optimizer`.

    The forward/backward/loss traits are identical across optimizers; only
    `update_conn` and the `extra_conn_fields` it needs differ. `train=False`
    returns a forward-only sibling sharing the same field layout, so it drives
    the same (static, state) for read-only inference (rung0 design section 2
    phase elision; mirrors mlp_xor.cpp's Net.DoForwardPass vs Net.DoStep).

    Args:
        optimizer: the plastax.optim bundle supplying update_conn/state_fields.
        train: whether to include the backward/loss/update phases.

    Returns:
        A Network subclass for the given optimizer.
    """
    if train:

        class _XorNet(px.Network[None]):
            forward_pass = SigmoidForward()
            backward_pass = SigmoidBackward()
            loss = MSELoss()
            update_conn = optimizer.update_conn()
            extra_unit_fields = (GradPreAct, LossGrad)
            extra_conn_fields = optimizer.state_fields
            propagation = px.Propagation.TOPOLOGICAL

        return _XorNet

    class _XorNetEval(px.Network[None]):
        forward_pass = SigmoidForward()
        extra_unit_fields = (GradPreAct, LossGrad)
        extra_conn_fields = optimizer.state_fields
        propagation = px.Propagation.TOPOLOGICAL

    return _XorNetEval


def build_net(
    optimizer: px.optim.Optimizer, key: jax.Array
) -> tuple[type[px.Network[None]], px.NetworkStatic, px.NetworkState[None]]:
    """3 inputs (x1, x2, bias=1.0) -> NUM_HIDDEN sigmoid hidden -> 1 sigmoid
    output (mlp_xor.cpp's MlpNetwork construction), wired to `optimizer`."""
    net = make_net(optimizer, train=True)
    topology_fn = px.topology.sequential(
        px.topology.input_units(3),
        px.topology.dense(3, NUM_HIDDEN),
        px.topology.dense(NUM_HIDDEN, 1),
    )
    static, state = px.NetworkBuilder.from_topology(
        net, topology_fn, key, globals_=None
    )
    return net, static, state


def _step_inputs(x1: float, x2: float, target: float | None) -> px.StepInputs:
    targets = None if target is None else jnp.asarray([target], dtype=jnp.float32)
    return px.StepInputs(
        inputs=jnp.asarray([x1, x2, _BIAS], dtype=jnp.float32), targets=targets
    )


def train(
    net: type[px.Network[None]],
    static: px.NetworkStatic,
    state: px.NetworkState[None],
    *,
    num_epochs: int,
    verbose: bool = True,
) -> tuple[px.NetworkState[None], float]:
    """Trains on the 4 XOR patterns each epoch (mlp_xor.cpp's main loop:
    one Net.DoStep per pattern, patterns fixed order every epoch). Returns
    the trained state and the final epoch's total loss."""
    step = px.make_step(net, static)
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
    optimizer: px.optim.Optimizer,
    static: px.NetworkStatic,
    state: px.NetworkState[None],
) -> tuple[px.NetworkState[None], bool, tuple[float, ...]]:
    """Forward-only pass over the 4 patterns (mlp_xor.cpp's final
    Net.DoForwardPass loop). Returns the state, whether every pattern is
    classified correctly (threshold 0.5), and the raw predictions."""
    eval_step = px.make_step(make_net(optimizer, train=False), static)
    output_id = static.output_ids[0]
    ok = True
    predictions: list[float] = []
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error", message=".*Some donated buffers were not usable.*"
        )
        for (x1, x2), target in zip(_XOR_INPUTS, _XOR_TARGETS, strict=True):
            result = eval_step(state, _step_inputs(x1, x2, None))
            state = result.state
            pred = float(state.units[px.ACTIVATION.name][output_id])
            predictions.append(pred)
            if (1 if pred > 0.5 else 0) != int(target):
                ok = False
    return state, ok, tuple(predictions)


# Each optimizer plus the (lr, epochs) that take XOR to convergence through the
# shared traits. optax-parity is checked in tests/test_optim.py; here the point
# is that swapping the optimizer is a one-line change and every one learns.
def showcase() -> list[tuple[str, px.optim.Optimizer, int]]:
    """Return the (name, optimizer, epochs) rows main() trains and checks."""
    return [
        ("sgd", px.optim.sgd(0.5, GradPreAct), 5000),
        ("momentum", px.optim.momentum(0.2, 0.9, GradPreAct), 3000),
        ("adam", px.optim.adam(0.05, GradPreAct), 2000),
        ("adamw", px.optim.adamw(0.05, GradPreAct), 2000),
        ("rmsprop", px.optim.rmsprop(0.01, GradPreAct), 3000),
    ]


def main() -> None:
    print("Plastax MLP / XOR -- one trait bundle, every optimizer")
    print("=" * 54)
    print(f"{'optimizer':9s} {'epochs':>6s} {'loss':>10s}  result   predictions")
    for name, optimizer, epochs in showcase():
        net, static, state = build_net(optimizer, jax.random.PRNGKey(SEED))
        state, final_loss = train(net, static, state, num_epochs=epochs, verbose=False)
        _, ok, preds = evaluate(optimizer, static, state)
        preds_str = ", ".join(f"{p:.3f}" for p in preds)
        result = "LEARNED" if ok else "FAILED "
        print(f"{name:9s} {epochs:6d} {final_loss:10.6f}  {result}  [{preds_str}]")
        assert ok, f"{name} failed to learn XOR (loss {final_loss:.4f})"


if __name__ == "__main__":
    main()
