"""Reference networks for the plastax-cpp conformance vectors.

plastax is the oracle: each network here defines what a step *should* produce,
and plastax-cpp's `tests/test_parity_plastax.cpp` checks that the C++ implementation
agrees within tolerance. Every net in this file has a counterpart traits struct
in plastax-cpp's `tests/parity/parity_fixtures.hpp` under the same `traits` name --
that pairing is the contract, and the two must be edited together.

These are deliberately self-contained rather than imported from `examples/`.
Example code changes for algorithmic reasons; a conformance vector has to stay
pinned to the algorithm it was generated for, or a regenerated golden silently
redefines the thing it was supposed to be testing.

Weight initialisation goes through `tests/_plastax_cpp_rng`, which reproduces
`plastax::UniformReal` bit-for-bit, so both implementations start from
identical weights. That is what makes a multi-step trajectory comparison
meaningful: any divergence a vector reports was accumulated by the algorithm,
not inherited from different initial conditions.

See `notes/parity/00-parity-harness.md` in plastax-cpp.
"""

from __future__ import annotations

import dataclasses
import pathlib
import sys
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import plastax as px

# The RNG port lives under tests/ because it is test-support code, not part
# of the shipped library -- so reach it by path rather than by package.
# `ty` cannot resolve a runtime sys.path insert; that is expected here and
# costs nothing, since the type gates (ty, mypy) are scoped to src/ only.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tests"))
from _plastax_cpp_rng import fully_connected_weights  # noqa: E402

# Per-unit columns. `grad_pre_act` carries dL/dz between backward levels;
# plastax-cpp persists the same quantity in a user field of the same role, because
# its framework BackwardAcc is cleared right after each per-level Apply.
GRAD_PRE_ACT = px.FieldSpec.float32("grad_pre_act")
# dL/dActivation, staged by the loss for output units only. plastax-cpp's MSELoss
# stages this into the framework's BackwardAcc column, which its backward pass
# then accumulates into rather than resetting; plastax's backward accumulator
# is local to the phase's trace, so the handoff needs an explicit column.
# Either way the arithmetic is identical -- see the fixtures header.
LOSS_GRAD = px.FieldSpec.float32("loss_grad")


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


class _WeightedSumMap:
    """Shared `map`: the per-edge product that every forward pass here uses."""

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
        """Return weight * activation[src] for one edge."""
        del dst, g
        return c[px.WEIGHT, cid] * u[px.ACTIVATION, src]


class TanhForward(_WeightedSumMap, px.ForwardPass):
    """apply = tanh(acc). Mirrors the fixture's TanhForward."""

    def apply(
        self, u: px.UnitView, i: px.UnitIdx, g: None, acc: jax.Array
    ) -> px.UnitWrite:
        """Write tanh of the accumulated input."""
        del u, i, g
        return px.UnitWrite.of((px.ACTIVATION, jnp.tanh(acc)))


class SigmoidForward(_WeightedSumMap, px.ForwardPass):
    """apply = sigmoid(acc). Mirrors the fixture's SigmoidForward."""

    def apply(
        self, u: px.UnitView, i: px.UnitIdx, g: None, acc: jax.Array
    ) -> px.UnitWrite:
        """Write the logistic sigmoid of the accumulated input."""
        del u, i, g
        return px.UnitWrite.of((px.ACTIVATION, jax.nn.sigmoid(acc)))


class LinearForward(_WeightedSumMap, px.ForwardPass):
    """apply = acc (identity activation). Mirrors the fixture's LinearForward."""

    def apply(
        self, u: px.UnitView, i: px.UnitIdx, g: None, acc: jax.Array
    ) -> px.UnitWrite:
        """Write the accumulated input unchanged."""
        del u, i, g
        return px.UnitWrite.of((px.ACTIVATION, acc))


class SigmoidBackward(px.BackwardPass):
    """Reverse walk: dL/da -> dL/dz through the sigmoid derivative.

    `map`'s first unit-id argument is the accumulator target, which for the
    backward direction is the *source* unit (sweep.py's calling convention);
    the second is the destination whose dL/dz has already been finalized by a
    deeper level. Mirrors the fixture's SigmoidBackward.
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
    ) -> jax.Array:
        """Return weight * dL/dz[dst] for one outgoing edge."""
        del src, g
        return c[px.WEIGHT, cid] * u[GRAD_PRE_ACT, dst]

    def apply(
        self, u: px.UnitView, i: px.UnitIdx, g: None, acc: jax.Array
    ) -> px.UnitWrite:
        """Convert dL/da to dL/dz and stage it for the next level down."""
        del g
        a = u[px.ACTIVATION, i]
        # `acc` is the backward-accumulated dL/da for hidden units and the
        # identity 0.0 for output units (no edge sources from the deepest
        # level); `loss_grad` is non-zero only for outputs. Their sum is
        # exactly the single value plastax-cpp keeps in BackwardAcc.
        grad = (acc + u[LOSS_GRAD, i]) * a * (jnp.float32(1.0) - a)
        return px.UnitWrite.of((GRAD_PRE_ACT, grad))


class MSELoss(px.Loss):
    """L = 0.5*sum((pred - target)^2); seed dL/dpred = pred - target.

    Matches plastax::MSELoss, which seeds the same gradient into BackwardAcc.
    """

    seed_field = LOSS_GRAD

    def calculate_loss(
        self, u: px.UnitView, outputs: jax.Array, targets: jax.Array, g: None
    ) -> tuple[jax.Array, jax.Array]:
        """Return the loss and the dL/dActivation seed of every output."""
        del g
        diff = u.gather(px.ACTIVATION, outputs) - targets
        return jnp.sum(jnp.float32(0.5) * diff * diff), diff


# ---------------------------------------------------------------------------
# Vector specification
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ConstantInit:
    """Every weight in the layer set to one value."""

    value: float

    def as_json(self) -> dict[str, Any]:
        """Return the JSON form the C++ runner parses."""
        return {"kind": "constant", "value": self.value}

    def weights(self, n_src: int, n_dst: int, base_conn_id: int) -> np.ndarray:
        """Return the (n_src, n_dst) weight matrix."""
        del base_conn_id
        return np.full((n_src, n_dst), self.value, dtype=np.float32)


@dataclasses.dataclass(frozen=True)
class UniformInit:
    """`plastax::RandomUniformWeight`, reproduced exactly.

    The range width must be a power of two. plastax-cpp maps its uniform sample
    into the range as ``min + (max - min) * u``, and nvcc contracts that into a
    single fused multiply-add on device while the host evaluates a multiply
    then an add. When the width is a power of two the product is exact and the
    rounding order cannot matter; otherwise host and device differ by up to
    1 ULP (see plastax-cpp's ``tests/test_parity_rng_cuda.cpp``).

    A vector's ``expect_initial_weights`` is compared *exactly*, since identical
    starting weights are the precondition that makes the whole trajectory
    comparison interpretable -- so a non-power-of-two width would pass on CPU
    and fail on GPU for reasons having nothing to do with the algorithm under
    test. Rejecting it here turns that into a clear error at authoring time.

    Attributes:
        seed: The initialiser seed, matching the C++ `RandomUniformWeight`.
        lo: Range lower bound.
        hi: Range upper bound.
    """

    seed: int
    lo: float = -1.0
    hi: float = 1.0

    def __post_init__(self) -> None:
        """Reject a range width that host and device would not agree on.

        Raises:
            ValueError: If ``hi - lo`` is not a positive power of two.
        """
        width = self.hi - self.lo
        mantissa, _ = np.frexp(np.float32(width))
        if not (width > 0.0 and mantissa == 0.5):
            raise ValueError(
                f"UniformInit range ({self.lo}, {self.hi}) has width {width}, "
                "which is not a power of two. plastax-cpp's host and device range "
                "mapping then differ by up to 1 ULP (FMA contraction), so the "
                "vector's exact initial-weight comparison would pass on CPU and "
                "fail on GPU. Use a power-of-two width, e.g. (-1, 1) or (0, 1)."
            )

    def as_json(self) -> dict[str, Any]:
        """Return the JSON form the C++ runner parses."""
        return {
            "kind": "uniform",
            "engine": "philox",
            "seed": self.seed,
            "min": self.lo,
            "max": self.hi,
        }

    def weights(self, n_src: int, n_dst: int, base_conn_id: int) -> np.ndarray:
        """Return the (n_src, n_dst) weight matrix in plastax-cpp's edge order."""
        return fully_connected_weights(
            self.seed, n_src, n_dst, base_conn_id=base_conn_id, lo=self.lo, hi=self.hi
        )


@dataclasses.dataclass(frozen=True)
class Layer:
    """One fully connected layer."""

    units: int
    init: ConstantInit | UniformInit


@dataclasses.dataclass(frozen=True)
class Vector:
    """One conformance vector: a network, a driving sequence, and tolerances.

    Attributes:
        name: Golden filename stem; also the C++ test's parameter name.
        traits: Name of the matching fixture in parity_fixtures.hpp.
        description: One line on what this vector is for.
        net: The plastax Network subclass acting as the oracle.
        input_dim: Number of input units.
        layers: Fully connected layers, in order.
        steps: (inputs, targets) per step; targets is None for forward-only nets.
        learning_rate: Value plastax-cpp reads into GlobalState, or None.
        hyper: Extra optimizer hyperparameters this vector was generated with,
            recorded so the C++ runner can assert its compile-time constants
            still agree (they live in two places and would otherwise drift).
        rtol: Relative tolerance for computed floats.
        atol: Absolute tolerance, so exact-zero references are comparable.
    """

    name: str
    traits: str
    description: str
    net: type[px.Network[None]]
    input_dim: int
    layers: tuple[Layer, ...]
    steps: tuple[tuple[tuple[float, ...], tuple[float, ...] | None], ...]
    learning_rate: float | None = None
    hyper: dict[str, float] = dataclasses.field(default_factory=dict)
    rtol: float = 1e-4
    atol: float = 1e-6


def layer_plan(
    input_dim: int, layers: tuple[Layer, ...]
) -> list[tuple[int, int, Layer, int]]:
    """Resolve each layer's fan-in and its first global connection id.

    plastax-cpp's connection ids keep counting across layers, and
    `RandomUniformWeight` keys on that global id -- so a second layer with its
    own seed still starts at a non-zero counter. Getting the base wrong yields
    perfectly plausible weights on the wrong edges.

    Args:
        input_dim: Number of input units.
        layers: Fully connected layers, in order.

    Returns:
        One (n_src, n_dst, layer, base_conn_id) tuple per layer.
    """
    plan: list[tuple[int, int, Layer, int]] = []
    base = 0
    n_src = input_dim
    for layer in layers:
        plan.append((n_src, layer.units, layer, base))
        base += n_src * layer.units
        n_src = layer.units
    return plan


def build_topology(vector: Vector) -> Callable[[Any], Any]:
    """Build the topology callable for a vector, with plastax-cpp-identical weights.

    Args:
        vector: The vector to build.

    Returns:
        A topology function suitable for `NetworkBuilder.from_topology`.
    """
    blocks = [px.topology.input_units(vector.input_dim)]
    for n_src, n_dst, layer, base in layer_plan(vector.input_dim, vector.layers):
        weights = layer.init.weights(n_src, n_dst, base)

        def init(key: Any, shape: tuple[int, ...], _w: np.ndarray = weights) -> Any:
            # topology.dense calls init(key, (n_in, n_out)); the weights are
            # fully determined by the plastax-cpp seed, so the key is unused.
            del key
            assert shape == _w.shape, f"init shape {shape} != {_w.shape}"
            return jnp.asarray(_w)

        blocks.append(px.topology.dense(n_src, n_dst, init=init))
    return px.topology.sequential(*blocks)


# ---------------------------------------------------------------------------
# The vectors
# ---------------------------------------------------------------------------

_XOR = ((0.0, 0.0, 1.0), (0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0))
_XOR_TARGETS = (0.0, 1.0, 1.0, 0.0)


class _ManualFccNet(px.Network[None]):
    forward_pass = TanhForward()
    propagation = px.Propagation.TOPOLOGICAL


class _FccSigmoidNet(px.Network[None]):
    forward_pass = SigmoidForward()
    propagation = px.Propagation.TOPOLOGICAL


class _PipelineSigmoidNet(px.Network[None]):
    forward_pass = SigmoidForward()
    propagation = px.Propagation.PIPELINE


class _MlpNet(px.Network[None]):
    forward_pass = SigmoidForward()
    backward_pass = SigmoidBackward()
    loss = MSELoss()
    update_conn = px.optim.sgd(0.5, grad_field=GRAD_PRE_ACT).update_conn()
    extra_unit_fields = (GRAD_PRE_ACT, LOSS_GRAD)
    propagation = px.Propagation.TOPOLOGICAL


class _LinRegNet(px.Network[None]):
    forward_pass = LinearForward()
    loss = MSELoss()
    # No backward pass: with a single layer, dL/dActivation at the output *is*
    # dL/dz, so the update reads the loss's staged gradient directly. plastax-cpp
    # does the same by reading BackwardAcc, which nothing clears when the
    # backward pass is elided.
    update_conn = px.optim.sgd(0.01, grad_field=LOSS_GRAD).update_conn()
    extra_unit_fields = (LOSS_GRAD,)
    propagation = px.Propagation.TOPOLOGICAL


# ---------------------------------------------------------------------------
# Optimizer vectors (plastax-cpp notes/parity/02-optimizers.md)
# ---------------------------------------------------------------------------
#
# One net per plastax.optim bundle, all sharing _MlpNet's traits and differing
# only in `update_conn` -- the same showcase examples/mlp_xor.py makes, and the
# shape that lets the vectors attribute any divergence to the update rule
# rather than to anything around it.
#
# Because plastax's bundles are themselves validated against optax, pinning
# plastax-cpp to these vectors pins it transitively to optax. That is the point:
# there is no separate C++ Adam reference to disagree with.
#
# Hyperparameters other than the learning rate are the bundles' own defaults on
# both sides, restated in `hyper` only so the C++ runner can assert its
# compile-time constants still match (CheckOptimHyper in
# test_parity_plastax.cpp). The learning rate is plumbed through GlobalState so
# both implementations read the one value in the JSON.

_OPTIM_LR = 0.1
_ADAM_B1, _ADAM_B2, _ADAM_EPS = 0.9, 0.999, 1e-8
_ADAMW_DECAY = 1e-4
_MOMENTUM = 0.9
_RMSPROP_DECAY, _RMSPROP_EPS = 0.9, 1e-8


def _optim_net(update_conn: Any, extra_conn_fields: tuple[Any, ...]) -> Any:
    """Build an MLP Network class wired to one optimizer bundle.

    Args:
        update_conn: The bundle's UpdateConn policy.
        extra_conn_fields: The bundle's per-connection state columns.

    Returns:
        A Network subclass identical to _MlpNet apart from the update rule.
    """

    class _OptimNet(px.Network[None]):
        forward_pass = SigmoidForward()
        backward_pass = SigmoidBackward()
        loss = MSELoss()
        propagation = px.Propagation.TOPOLOGICAL
        extra_unit_fields = (GRAD_PRE_ACT, LOSS_GRAD)

    _OptimNet.update_conn = update_conn
    _OptimNet.extra_conn_fields = extra_conn_fields
    return _OptimNet


_SGD = px.optim.sgd(_OPTIM_LR, grad_field=GRAD_PRE_ACT)
_MOM = px.optim.momentum(_OPTIM_LR, _MOMENTUM, grad_field=GRAD_PRE_ACT)
_RMS = px.optim.rmsprop(
    _OPTIM_LR, grad_field=GRAD_PRE_ACT, decay=_RMSPROP_DECAY, eps=_RMSPROP_EPS
)
_ADAM = px.optim.adam(
    _OPTIM_LR, grad_field=GRAD_PRE_ACT, b1=_ADAM_B1, b2=_ADAM_B2, eps=_ADAM_EPS
)
_ADAMW = px.optim.adamw(
    _OPTIM_LR,
    grad_field=GRAD_PRE_ACT,
    weight_decay=_ADAMW_DECAY,
    b1=_ADAM_B1,
    b2=_ADAM_B2,
    eps=_ADAM_EPS,
)

_OPTIM_BUNDLES: tuple[tuple[str, str, Any, dict[str, float], str], ...] = (
    (
        "mlp_optim_sgd",
        "mlp_optim_sgd",
        _SGD,
        {},
        "Stateless SGD through the optim bundle rather than a hand-written rule.",
    ),
    (
        "mlp_optim_momentum",
        "mlp_optim_momentum",
        _MOM,
        {"momentum": _MOMENTUM},
        "Heavy-ball momentum: one state column (opt/v) carried per connection.",
    ),
    (
        "mlp_optim_rmsprop",
        "mlp_optim_rmsprop",
        _RMS,
        {"decay": _RMSPROP_DECAY, "eps": _RMSPROP_EPS},
        "RMSProp, eps INSIDE the sqrt -- the placement that distinguishes it "
        "from Adam's and changes the update wherever v is small.",
    ),
    (
        "mlp_optim_adam",
        "mlp_optim_adam",
        _ADAM,
        {"b1": _ADAM_B1, "b2": _ADAM_B2, "eps": _ADAM_EPS, "weight_decay": 0.0},
        "Adam: three state columns and a PER-EDGE step counter, so bias "
        "correction is exercised across all 24 steps.",
    ),
    (
        "mlp_optim_adamw",
        "mlp_optim_adamw",
        _ADAMW,
        {
            "b1": _ADAM_B1,
            "b2": _ADAM_B2,
            "eps": _ADAM_EPS,
            "weight_decay": _ADAMW_DECAY,
        },
        "AdamW: decoupled weight decay applied to the pre-update weight, not "
        "folded into the gradient.",
    ),
)


def _linreg_steps() -> tuple[tuple[tuple[float, ...], tuple[float, ...] | None], ...]:
    """Deterministic regression samples for y = 2*x1 - x2 + 0.5*x3."""
    rng = np.random.default_rng(20260830)
    steps = []
    for _ in range(24):
        x = rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
        y = float(2.0 * x[0] - 1.0 * x[1] + 0.5 * x[2])
        steps.append((tuple(float(v) for v in x), (y,)))
    return tuple(steps)


def _optim_vectors() -> tuple[Vector, ...]:
    """One vector per optim bundle, all on the same XOR problem.

    Returns:
        A vector for each entry in _OPTIM_BUNDLES.
    """
    steps = tuple((_XOR[i % 4], (_XOR_TARGETS[i % 4],)) for i in range(24))
    return tuple(
        Vector(
            name=name,
            traits=traits,
            description=(
                f"{description} Same net, inputs and seeds as mlp_xor_seeded, so "
                "the update rule is the only difference between these vectors."
            ),
            net=_optim_net(bundle.update_conn(), bundle.state_fields),
            input_dim=3,
            layers=(Layer(4, UniformInit(seed=1)), Layer(1, UniformInit(seed=2))),
            steps=steps,
            learning_rate=_OPTIM_LR,
            hyper=hyper,
        )
        for name, traits, bundle, hyper, description in _OPTIM_BUNDLES
    )


VECTORS: tuple[Vector, ...] = (
    Vector(
        name="manual_fcc",
        traits="manual_fcc_tanh",
        description=(
            "Topological forward with constant weights and no learning. The one "
            "vector with no RNG involvement at all, so it isolates the forward "
            "kernel from the initialiser."
        ),
        net=_ManualFccNet,
        input_dim=2,
        layers=(Layer(4, ConstantInit(0.5)), Layer(1, ConstantInit(-0.3))),
        steps=(
            ((0.1, 0.2), None),
            ((0.5, -0.5), None),
            ((1.0, 1.0), None),
            ((-0.75, 0.25), None),
        ),
    ),
    Vector(
        name="fcc_sigmoid_seeded",
        traits="fcc_sigmoid_forward",
        description=(
            "Topological forward from seeded weights. End-to-end check that the "
            "NumPy RNG port lands the same weight on the same edge as "
            "plastax::RandomUniformWeight, including the cross-layer connection-id "
            "offset."
        ),
        net=_FccSigmoidNet,
        input_dim=3,
        layers=(Layer(4, UniformInit(seed=1)), Layer(2, UniformInit(seed=2))),
        steps=(
            ((0.5, -0.25, 1.0), None),
            ((-1.0, 0.75, 0.5), None),
            ((0.0, 0.0, 0.0), None),
            ((2.0, -2.0, 1.5), None),
        ),
    ),
    Vector(
        name="mlp_xor_seeded",
        traits="mlp_sigmoid_mse_sgd",
        description=(
            "The full differentiable path -- forward, MSE loss, backward, SGD -- "
            "on XOR from seeded weights. The vector that actually exercises "
            "learning; weight drift here is cumulative, so it is the most "
            "sensitive of the set."
        ),
        net=_MlpNet,
        input_dim=3,
        layers=(Layer(4, UniformInit(seed=1)), Layer(1, UniformInit(seed=2))),
        steps=tuple((_XOR[i % 4], (_XOR_TARGETS[i % 4],)) for i in range(24)),
        learning_rate=0.5,
    ),
    Vector(
        name="linreg_mse",
        traits="linear_mse_sgd",
        description=(
            "Single layer, no backward pass: isolates loss-gradient staging and "
            "the update rule from the backward walk. Weights start at zero, so "
            "every value present is one the update produced."
        ),
        net=_LinRegNet,
        input_dim=3,
        layers=(Layer(1, ConstantInit(0.0)),),
        steps=_linreg_steps(),
        learning_rate=0.01,
    ),
    Vector(
        name="pipeline_sigmoid",
        traits="pipeline_sigmoid",
        description=(
            "Pipeline propagation: one flat sweep per step, so a signal advances "
            "exactly one layer per step and the network's output lags its input. "
            "A substantially different dispatch path from every topological "
            "vector, and the reason those are kept in separate files on both sides."
        ),
        net=_PipelineSigmoidNet,
        input_dim=3,
        layers=(Layer(4, UniformInit(seed=5)), Layer(2, UniformInit(seed=6))),
        steps=(
            ((1.0, 0.0, 0.5), None),
            ((0.0, 1.0, 0.5), None),
            ((0.0, 0.0, 0.0), None),
            ((-1.0, 1.0, 0.25), None),
            ((0.5, 0.5, 0.5), None),
            ((0.0, 0.0, 0.0), None),
        ),
    ),
    *_optim_vectors(),
)
