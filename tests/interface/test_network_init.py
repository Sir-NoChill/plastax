"""A `px.Network` with a topology: `build`, `init`, and the `Driver` it returns.

Plan `plans/20261121T113757Z-network-runner.md` (workspace root). The networks here
compute plain weighted sums, so a step's outputs can be compared with the same
arithmetic written in numpy. The learning test trains XOR with the policies of
examples/mlp_xor.py.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import plastax as px
import plastax.init as init
import plastax.layers as pxl
from plastax.layers import BuildError, Dense, Units

f32 = np.float32
RNG = np.random.default_rng(0)
KEY = jax.random.key(0)


class WeightedSum(px.policies.ForwardConnPolicy):
    """A unit's activation is the sum of weight times source activation."""

    def map(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: Any,
    ) -> jax.Array:
        return c[px.WEIGHT, cid] * u[px.ACTIVATION, src]


class Summed(px.policies.ForwardUnitPolicy):
    """A unit's activation is the sum of what its connections map."""

    combine = px.monoid.sum_

    def apply(
        self, u: px.UnitView, i: px.UnitIdx, g: Any, acc: jax.Array
    ) -> px.UnitWrite:
        return px.UnitWrite.of((px.ACTIVATION, acc))


W1 = RNG.normal(size=(4, 3)).astype(f32)
W2 = RNG.normal(size=(2, 4)).astype(f32)


class TwoLayers(pxl.Module):
    """3 inputs, 4 hidden units, 2 outputs, no bias, weights given."""

    def setup(self) -> None:
        self.hidden = Dense(4, bias=False, init=init.array(W1))
        self.out = Dense(2, bias=False, init=init.array(W2))

    def __call__(self, x: Units) -> Units:
        return self.out(self.hidden(x))


class Linear(px.Network[None]):
    topology = TwoLayers()
    input_shape = (3,)
    forward = px.policies.Forward(conn=WeightedSum(), unit=Summed())


def inputs_of(x: Any, targets: Any = None) -> px.StepInputs:
    return px.StepInputs(
        inputs=jnp.asarray(x, jnp.float32),
        targets=None if targets is None else jnp.asarray(targets, jnp.float32),
    )


# --------------------------------------------------------------------------- #
# init and step
# --------------------------------------------------------------------------- #
def test_step_returns_a_loss_and_no_outputs_unless_asked() -> None:
    result = Linear.init(KEY).step(inputs_of([1.0, 2.0, 3.0]))

    assert result.loss.shape == ()
    assert result.outputs is None
    assert result.records is None


def test_outputs_are_the_forward_pass_in_the_shape_the_topology_returned() -> None:
    x = RNG.normal(size=3).astype(f32)

    result = Linear.init(KEY).step(inputs_of(x), outputs=True)

    assert result.outputs is not None
    assert result.outputs.shape == (2,)
    np.testing.assert_allclose(result.outputs, W2 @ (W1 @ x), rtol=1e-5)


# --------------------------------------------------------------------------- #
# Declared shapes
# --------------------------------------------------------------------------- #
W_IMAGE = RNG.normal(size=(2, 5, 4 * 4 * 3)).astype(f32)


class Image(pxl.Module):
    """A (4, 4, 3) input read whole by 2 groups of 5 units."""

    def setup(self) -> None:
        self.fc = Dense(5, bias=False, init=init.array(W_IMAGE))

    def __call__(self, image: Units) -> Units:
        return self.fc(image.reshape(1, 48)[[0, 0]])  # the whole image, twice


class ImageNet(px.Network[None]):
    topology = Image()
    input_shape = (4, 4, 3)
    forward = px.policies.Forward(conn=WeightedSum(), unit=Summed())


def test_inputs_and_outputs_keep_the_shapes_the_topology_declared() -> None:
    image = RNG.normal(size=(4, 4, 3)).astype(f32)

    result = ImageNet.init(KEY).step(inputs_of(image), outputs=True)

    assert result.outputs is not None
    assert result.outputs.shape == (2, 5)
    np.testing.assert_allclose(result.outputs, W_IMAGE @ image.reshape(-1), rtol=1e-5)


def test_an_input_of_another_shape_is_refused_naming_the_declared_shape() -> None:
    driver = ImageNet.init(KEY)

    with pytest.raises(
        ValueError, match=r"inputs must have the declared shape \(4, 4, 3\)"
    ):
        driver.step(inputs_of(np.zeros(48, f32)))


def test_targets_of_another_shape_are_refused_naming_the_declared_shape() -> None:
    driver = ImageNet.init(KEY)

    with pytest.raises(
        ValueError, match=r"targets must have the declared shape \(2, 5\)"
    ):
        driver.step(inputs_of(np.zeros((4, 4, 3), f32), np.zeros(10, f32)))


# --------------------------------------------------------------------------- #
# Outputs are the forward pass's
# --------------------------------------------------------------------------- #
class Overwrite(px.policies.UpdateUnitPolicy):
    """Every unit's activation is replaced after the forward pass."""

    def update(self, u: px.UnitView, i: px.UnitIdx, g: Any) -> px.UnitWrite:
        return px.UnitWrite.of((px.ACTIVATION, jnp.float32(-1.0)))


class Overwritten(px.Network[None]):
    topology = TwoLayers()
    input_shape = (3,)
    forward = px.policies.Forward(conn=WeightedSum(), unit=Summed())
    update = px.policies.Update(unit=Overwrite())


def test_outputs_are_read_before_a_later_phase_rewrites_the_units() -> None:
    x = RNG.normal(size=3).astype(f32)
    driver = Overwritten.init(KEY)

    result = driver.step(inputs_of(x), outputs=True)

    np.testing.assert_allclose(result.outputs, W2 @ (W1 @ x), rtol=1e-5)
    held = np.asarray(driver.state.units["activation"])[list(driver.static.output_ids)]
    np.testing.assert_array_equal(held, [-1.0, -1.0])


def test_outputs_of_a_batched_step_are_not_available_yet() -> None:
    static, _ = Linear.build(KEY)

    with pytest.raises(NotImplementedError, match="batch_size"):
        px.make_step(Linear, static, batch_size=2, outputs=True)


# --------------------------------------------------------------------------- #
# Globals
# --------------------------------------------------------------------------- #
class Scaled(px.policies.ForwardConnPolicy):
    """Weight times source activation, multiplied by the global `scale`."""

    def map(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: dict[str, jax.Array],
    ) -> jax.Array:
        return g["scale"] * c[px.WEIGHT, cid] * u[px.ACTIVATION, src]


class OneWeight(pxl.Module):
    def setup(self) -> None:
        self.fc = Dense(1, bias=False, init=init.constant(1.0))

    def __call__(self, x: Units) -> Units:
        return self.fc(x)


class ScaledNet(px.Network[dict[str, Any]]):
    topology = OneWeight()
    input_shape = (1,)
    globals = {"scale": f32(2.0)}
    forward = px.policies.Forward(conn=Scaled(), unit=Summed())


def test_the_class_field_is_the_value_the_globals_start_from() -> None:
    result = ScaledNet.init(KEY).step(inputs_of([5.0]), outputs=True)

    np.testing.assert_allclose(result.outputs, [10.0])


def test_globals_given_to_init_replace_the_class_field_for_that_network() -> None:
    result = ScaledNet.init(KEY, globals={"scale": f32(3.0)}).step(
        inputs_of([5.0]), outputs=True
    )

    np.testing.assert_allclose(result.outputs, [15.0])


class Increment(px.policies.ResetGlobalPolicy):
    def reset(self, g: dict[str, jax.Array]) -> dict[str, jax.Array]:
        return {"scale": g["scale"] + 1.0}


class Rewritten(ScaledNet):
    reset_global = Increment()


def test_a_policy_that_writes_the_globals_leaves_the_class_field_alone() -> None:
    driver = Rewritten.init(KEY)

    driver.step(inputs_of([5.0]))
    second = driver.step(inputs_of([5.0]), outputs=True)

    np.testing.assert_allclose(second.outputs, [15.0])  # scale was 2, now 3
    np.testing.assert_allclose(driver.state.globals_["scale"], 4.0)
    assert Rewritten.globals == {"scale": f32(2.0)}
    np.testing.assert_allclose(  # a second network starts from the field again
        Rewritten.init(KEY).step(inputs_of([5.0]), outputs=True).outputs, [10.0]
    )


# --------------------------------------------------------------------------- #
# scan, and learning
# --------------------------------------------------------------------------- #
def _load_example(name: str) -> Any:
    path = Path(__file__).resolve().parents[2] / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


class XorTopology(pxl.Module):
    def setup(self) -> None:
        self.hidden = Dense(4)
        self.out = Dense(1)

    def __call__(self, x: Units) -> Units:
        return self.out(self.hidden(x))


def _xor_net() -> type[px.Network[None]]:
    """Sigmoid units, squared error, backprop and Adam, from examples/mlp_xor.py."""
    mlp_xor = _load_example("mlp_xor")
    base = mlp_xor.make_net(px.optim.adam(0.05, mlp_xor.GradPreAct), train=True)

    class Xor(base):  # type: ignore[misc, valid-type]
        topology = XorTopology()
        input_shape = (2,)

    return Xor


PATTERNS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], f32)
ANSWERS = np.array([[0], [1], [1], [0]], f32)


def test_xor_learns_step_by_step_and_scan_gives_the_same_losses() -> None:
    net = _xor_net()
    epochs = 300
    xs, ys = np.tile(PATTERNS, (epochs, 1)), np.tile(ANSWERS, (epochs, 1))

    by_step = net.init(KEY)
    losses = [
        float(by_step.step(inputs_of(x, y)).loss) for x, y in zip(xs, ys, strict=True)
    ]

    scanned = net.init(KEY).scan(inputs_of(xs, ys), outputs=True)

    assert sum(losses[:4]) > 0.4
    assert sum(losses[-4:]) < 0.05
    assert scanned.loss.shape == (4 * epochs,)
    np.testing.assert_allclose(scanned.loss, losses, rtol=1e-4, atol=1e-6)
    assert scanned.outputs is not None
    assert scanned.outputs.shape == (4 * epochs, 1)
    last = np.asarray(scanned.outputs)[-4:, 0]
    assert (last > 0.5).tolist() == [False, True, True, False]


def _step_count(result: px.StepResult[None]) -> jax.Array:
    return result.state.step


def test_scan_returns_what_record_kept_as_records() -> None:
    xs = RNG.normal(size=(5, 3)).astype(f32)

    plain = Linear.init(KEY).scan(inputs_of(xs))
    kept = Linear.init(KEY).scan(inputs_of(xs), record=_step_count)

    assert plain.records is None and plain.outputs is None
    np.testing.assert_array_equal(kept.records, [1, 2, 3, 4, 5])


def test_scan_needs_a_chunk_axis_in_front_of_the_declared_shape() -> None:
    with pytest.raises(ValueError, match="chunk axis"):
        ImageNet.init(KEY).scan(inputs_of(np.zeros((4, 4, 3), f32)))


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
def test_a_network_without_a_topology_cannot_be_built_this_way() -> None:
    class Bare(px.Network[None]):
        forward = px.policies.Forward(conn=WeightedSum(), unit=Summed())

    with pytest.raises(BuildError, match="has no topology"):
        Bare.build(KEY)


def test_a_topology_needs_an_input_shape() -> None:
    class Shapeless(px.Network[None]):
        topology = TwoLayers()
        forward = px.policies.Forward(conn=WeightedSum(), unit=Summed())

    with pytest.raises(BuildError, match="input_shape"):
        Shapeless.init(KEY)


def test_a_topology_with_two_inputs_is_not_supported_yet() -> None:
    class Two(pxl.Module):
        def __call__(self, a: Units, b: Units) -> Units:
            return a

    class TwoInputs(px.Network[None]):
        topology = Two()
        input_shape = (3,)
        forward = px.policies.Forward(conn=WeightedSum(), unit=Summed())

    with pytest.raises(BuildError, match="takes 2 arguments"):
        TwoInputs.build(KEY)


def test_a_topology_returning_two_handles_is_not_supported_yet() -> None:
    class Both(pxl.Module):
        def setup(self) -> None:
            self.a = Dense(1)
            self.b = Dense(1)

        def __call__(self, x: Units) -> tuple[Units, Units]:
            return self.a(x), self.b(x)

    class TwoOutputs(px.Network[None]):
        topology = Both()
        input_shape = (3,)
        forward = px.policies.Forward(conn=WeightedSum(), unit=Summed())

    with pytest.raises(NotImplementedError, match="returns 2 handles"):
        TwoOutputs.build(KEY)


def test_globals_that_are_not_numbers_and_arrays_are_refused() -> None:
    class Odd(px.Network[Any]):
        topology = TwoLayers()
        input_shape = (3,)
        globals = object()
        forward = px.policies.Forward(conn=WeightedSum(), unit=Summed())

    with pytest.raises(BuildError, match="not a pytree of numbers and arrays"):
        Odd.build(KEY)
