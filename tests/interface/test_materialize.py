"""`materialize`: a record becomes a state, checked by stepping it.

Plan `plans/20261119T201041Z-step6-materialize.md` (workspace root). Every network
here computes a plain weighted sum, so one step's outputs can be compared with the
same arithmetic written in numpy. The last test trains the XOR network with the
policies of examples/mlp_xor.py.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import plastax as px
import plastax.init as init
import plastax.layers as pxl
from plastax.layers import BuildError, CreateConnections, CreateUnits, Dense, Units

f32 = np.float32


class WeightedSum(px.policies.ForwardConnPolicy):
    """A unit's activation is the sum of weight times source activation."""

    def map(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: None,
    ) -> jax.Array:
        return c[px.WEIGHT, cid] * u[px.ACTIVATION, src]


class Summed(px.policies.ForwardUnitPolicy):
    """A unit's activation is the sum of what its connections map."""

    combine = px.monoid.sum_

    def apply(
        self, u: px.UnitView, i: px.UnitIdx, g: None, acc: jax.Array
    ) -> px.UnitWrite:
        return px.UnitWrite.of((px.ACTIVATION, acc))


class Linear(px.Network[None]):
    forward = px.policies.Forward(conn=WeightedSum(), unit=Summed())
    propagation = px.Propagation.TOPOLOGICAL


class Recurrent(px.Network[None]):
    forward = px.policies.Forward(conn=WeightedSum(), unit=Summed())
    propagation = px.Propagation.PIPELINE


def module(call: Callable[..., Any], **layers: Any) -> pxl.Module:
    """A topology with the given layers whose `__call__` is `call(module, **inputs)`."""

    class Net(pxl.Module):
        def setup(self) -> None:
            for name, layer in layers.items():
                setattr(self, name, layer)

    # `trace` reads the argument names of `__call__`, so it is the function given.
    Net.__call__ = call  # type: ignore[method-assign]
    return Net()


def built(
    topology: pxl.Module, inputs: dict[str, tuple[int, ...]], seed: int = 0
) -> tuple[pxl.Record, px.NetworkStatic, px.NetworkState[None]]:
    record = pxl.trace(topology, inputs)
    static, state = pxl.materialize(record, Linear, jax.random.key(seed), globals_=None)
    return record, static, state


def stepped(
    static: px.NetworkStatic, state: px.NetworkState[None], x: Any
) -> tuple[px.NetworkState[None], np.ndarray]:
    """Run one step; return the new state and the activations of the outputs."""
    step = px.make_step(Linear, static)
    # In the declared shape when the record has one input, one flat vector otherwise.
    shape = (-1,) if static.input_shape is None else static.input_shape
    inputs = px.StepInputs(
        inputs=jnp.asarray(x, jnp.float32).reshape(shape), targets=None
    )
    state = step(state, inputs).state
    return state, np.asarray(state.units["activation"])[list(static.output_ids)]


def rows(state: px.NetworkState[Any]) -> list[tuple[int, int, float]]:
    """Every live connection of a state as `(source, destination, weight)`."""
    out: list[tuple[int, int, float]] = []
    for bucket in state.conns:
        live = ~np.asarray(bucket["dead"])
        out += zip(
            np.asarray(bucket["from_id"])[live].tolist(),
            np.asarray(bucket["to_id"])[live].tolist(),
            np.asarray(bucket["weight"])[live].tolist(),
            strict=True,
        )
    return sorted(out)


def weights_of(state: px.NetworkState[Any]) -> np.ndarray:
    return np.array([w for _, _, w in rows(state)], f32)


RNG = np.random.default_rng(0)
W1 = RNG.normal(size=(4, 3)).astype(f32)
B1 = RNG.normal(size=(4,)).astype(f32)
W2 = RNG.normal(size=(2, 4)).astype(f32)
B2 = RNG.normal(size=(2,)).astype(f32)
X = np.array([0.5, -1.0, 2.0], f32)


def two_layers() -> pxl.Module:
    return module(
        lambda m, x: m.out(m.hidden(x)),
        hidden=Dense(4, init=init.array(W1), bias_init=init.array(B1)),
        out=Dense(2, init=init.array(W2), bias_init=init.array(B2)),
    )


# --------------------------------------------------------------------------- #
# The network that is built computes what the topology says
# --------------------------------------------------------------------------- #
def test_two_layers_with_biases_compute_what_numpy_computes() -> None:
    _, static, state = built(two_layers(), {"x": (3,)})

    _, y = stepped(static, state, X)

    np.testing.assert_allclose(y, W2 @ (W1 @ X + B1) + B2, rtol=1e-5)


def test_leading_axes_are_separate_positions_each_with_its_own_weights() -> None:
    w = RNG.normal(size=(5, 3, 4)).astype(f32)
    x = RNG.normal(size=(5, 4)).astype(f32)
    fc = Dense(3, bias=False, init=init.array(w))

    _, static, state = built(module(lambda m, x: m.fc(x), fc=fc), {"x": (5, 4)})
    _, y = stepped(static, state, x)

    np.testing.assert_allclose(
        y.reshape(5, 3), np.einsum("pji,pi->pj", w, x), rtol=1e-5
    )


def test_inputs_and_outputs_are_laid_out_in_the_order_they_were_given() -> None:
    """Two inputs fill a step's `inputs` one after the other; so do two outputs."""

    def call(m: Any, a: Units, b: Units) -> tuple[Units, Units]:
        return m.double(b), m.negate(a)

    topology = module(
        call,
        double=Dense(1, bias=False, init=init.constant(2.0)),
        negate=Dense(1, bias=False, init=init.constant(-1.0)),
    )
    _, static, state = built(topology, {"a": (2,), "b": (1,)})

    _, y = stepped(static, state, [1.0, 10.0, 100.0])  # a = (1, 10), b = (100,)

    np.testing.assert_allclose(y, [2 * 100.0, -(1.0 + 10.0)])


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
def test_a_constant_holds_the_value_it_was_given_and_is_read_every_step() -> None:
    def call(m: Any, x: Units) -> Units:
        y = m.units(x)
        m.pass_on(x, y)
        m.offset(m.half(shape=()), y)
        return y

    topology = module(
        call,
        units=CreateUnits(),
        half=CreateUnits(constant=0.5),
        pass_on=CreateConnections(init=init.constant(1.0)),
        offset=CreateConnections(init=init.constant(4.0)),
    )
    record, static, state = built(topology, {"x": (2,)})
    (half,) = record.constants

    assert np.asarray(state.units["activation"])[half] == 0.5
    for x in ([1.0, 2.0], [-3.0, 0.0]):
        state, y = stepped(static, state, x)
        np.testing.assert_allclose(y, np.array(x) + 4.0 * 0.5)
        assert np.asarray(state.units["activation"])[half] == 0.5


def test_a_connection_into_a_constant_is_refused() -> None:
    def call(m: Any, x: Units) -> Units:
        m.wrong(x, m.one(shape=(2,)))
        return x

    topology = module(call, one=CreateUnits(constant=1.0), wrong=CreateConnections())

    with pytest.raises(BuildError, match="block wrong connects .* is a constant"):
        built(topology, {"x": (2,)})


# --------------------------------------------------------------------------- #
# Explicit wiring, and two connections between the same pair
# --------------------------------------------------------------------------- #
def test_listed_pairs_are_the_only_connections_each_with_its_weight() -> None:
    src, dst = np.array([0, 1, 3, 3]), np.array([0, 0, 1, 0])
    w = np.array([1.0, 2.0, 3.0, 4.0], f32)

    def call(m: Any, x: Units) -> Units:
        y = m.units(shape=(2,))
        m.wire(x[src], y[dst])
        return y

    topology = module(
        call, units=CreateUnits(), wire=CreateConnections(init=init.array(w))
    )
    record, _, state = built(topology, {"x": (4,)})
    x_ids, y_ids = record.inputs["x"], record.outputs[0]

    assert rows(state) == sorted(
        zip(x_ids[src].tolist(), y_ids[dst].tolist(), w.tolist(), strict=True)
    )


def test_two_blocks_between_the_same_pair_are_two_connections_that_add() -> None:
    def call(m: Any, x: Units) -> Units:
        y = m.units(x)
        m.first(x, y)
        m.second(x, y)
        return y

    topology = module(
        call,
        units=CreateUnits(),
        first=CreateConnections(init=init.constant(0.25)),
        second=CreateConnections(init=init.constant(3.0)),
    )
    _, static, state = built(topology, {"x": (2,)})

    assert len(rows(state)) == 4
    _, y = stepped(static, state, [1.0, -2.0])
    np.testing.assert_allclose(y, 3.25 * np.array([1.0, -2.0]))


# --------------------------------------------------------------------------- #
# Keys
# --------------------------------------------------------------------------- #
def drawn_layers(*names: str) -> pxl.Module:
    """A chain of layers with the given names, each drawing its weights."""

    def call(m: Any, x: Units) -> Units:
        for name in names:
            x = getattr(m, name)(x)
        return x

    return module(
        call, **{name: Dense(3, bias=False, init=init.normal(1.0)) for name in names}
    )


def block_weights(topology: pxl.Module, seed: int) -> dict[str, np.ndarray]:
    """The weights of each block, by the name of its layer."""
    record, _, state = built(topology, {"x": (3,)}, seed=seed)
    by_pair = {(s, d): w for s, d, w in rows(state)}
    return {
        block.path[0]: np.array(
            [
                by_pair[s, d]
                for s, d in zip(block.src.tolist(), block.dst.tolist(), strict=True)
            ]
        )
        for block in record.blocks
    }


def test_a_key_replays_its_weights_and_another_key_gives_others() -> None:
    first = block_weights(drawn_layers("a", "b"), seed=1)

    np.testing.assert_array_equal(
        first["a"], block_weights(drawn_layers("a", "b"), seed=1)["a"]
    )
    assert not np.array_equal(
        first["a"], block_weights(drawn_layers("a", "b"), seed=2)["a"]
    )


def test_two_blocks_draw_different_weights_from_one_key() -> None:
    weights = block_weights(drawn_layers("a", "b"), seed=1)

    assert not np.array_equal(weights["a"], weights["b"])


def test_adding_a_layer_does_not_change_the_weights_of_another() -> None:
    """A block's key comes from its path, not from its place among the blocks."""
    alone = block_weights(drawn_layers("b"), seed=1)
    after_another = block_weights(drawn_layers("a", "b"), seed=1)

    np.testing.assert_array_equal(alone["b"], after_another["b"])


# --------------------------------------------------------------------------- #
# Cycles
# --------------------------------------------------------------------------- #
def looped() -> pxl.Module:
    def call(m: Any, x: Units) -> Units:
        y = m.fc(x)
        m.back(y, y[::-1])
        return y

    return module(call, fc=Dense(2, bias=False), back=CreateConnections())


def test_a_cycle_is_refused_naming_the_blocks_it_runs_through() -> None:
    with pytest.raises(BuildError, match="close a cycle, through back;") as excinfo:
        built(looped(), {"x": (3,)})

    assert "fc" not in str(excinfo.value)  # fc leads into the cycle; it is not on it


def test_without_the_check_a_cycle_is_left_to_the_builder() -> None:
    record = pxl.trace(looped(), {"x": (3,)})
    key = jax.random.key(0)

    with pytest.raises(ValueError, match="do not form a DAG") as excinfo:
        pxl.materialize(record, Linear, key, globals_=None, check_cycle=False)
    assert not isinstance(excinfo.value, BuildError)

    _, state = pxl.materialize(record, Recurrent, key, globals_=None, check_cycle=False)
    assert len(rows(state)) == 3 * 2 + 2


# --------------------------------------------------------------------------- #
# An initializer that cannot fill its block
# --------------------------------------------------------------------------- #
def test_an_initializer_written_without_its_call_is_refused() -> None:
    topology = module(lambda m, x: m.fc(x), fc=Dense(2, bias=False, init=init.zeros))

    with pytest.raises(BuildError, match=r"(?s)block fc/weight.*px\.init\.zeros\(\)"):
        built(topology, {"x": (3,)})


def test_an_array_of_another_shape_is_refused_naming_the_block() -> None:
    fc = Dense(2, bias=False, init=init.array(np.zeros((3, 2))))

    with pytest.raises(BuildError, match=r"block fc/weight: .*shape \(3, 2\)"):
        built(module(lambda m, x: m.fc(x), fc=fc), {"x": (3,)})


def test_an_initializer_that_returns_another_shape_is_refused() -> None:
    def flat(key: Any, fans: init.Fans) -> np.ndarray:
        return np.zeros(int(np.prod(fans.shape)), f32)

    topology = module(lambda m, x: m.fc(x), fc=Dense(2, bias=False, init=flat))

    with pytest.raises(BuildError, match=r"shape \(6,\), not weights of .* \(2, 3\)"):
        built(topology, {"x": (3,)})


# --------------------------------------------------------------------------- #
# The fans an initializer is handed
# --------------------------------------------------------------------------- #
def test_fans_are_counted_over_the_whole_record_and_a_bias_is_not_a_fan_in() -> None:
    """Dense(3) on x, and three more connections from z into the same units.

    Unit y0 is fed by 4 connections of the layer and 2 of `skip`, y1 by 4 and 1, y2
    by 4. Each also has a bias, which no fan-in counts. The constant feeds 3 units.
    """
    seen: dict[str, init.Fans] = {}

    def recording(name: str) -> init.Initializer:
        def initializer(key: Any, fans: init.Fans) -> np.ndarray:
            seen[name] = fans
            return np.zeros(fans.shape, f32)

        return initializer

    def call(m: Any, x: Units, z: Units) -> Units:
        y = m.fc(x)
        m.skip(z[np.array([0, 1, 1])], y[np.array([0, 0, 1])])
        return y

    topology = module(
        call,
        fc=Dense(3, init=recording("weight"), bias_init=recording("bias")),
        skip=CreateConnections(init=recording("skip")),
    )
    built(topology, {"x": (4,), "z": (2,)})

    np.testing.assert_array_equal(
        seen["weight"].fan_in, [[6, 6, 6, 6], [5, 5, 5, 5], [4, 4, 4, 4]]
    )
    np.testing.assert_array_equal(seen["weight"].fan_out, np.full((3, 4), 3))
    np.testing.assert_array_equal(seen["skip"].fan_in, [6, 6, 5])
    np.testing.assert_array_equal(seen["skip"].fan_out, [1, 2, 2])
    np.testing.assert_array_equal(seen["bias"].fan_in, [6, 5, 4])
    np.testing.assert_array_equal(seen["bias"].fan_out, [3, 3, 3])
    assert [seen[n].shape for n in ("weight", "skip", "bias")] == [(3, 4), (3,), (3,)]


def test_two_connections_between_one_pair_count_twice() -> None:
    seen: list[init.Fans] = []

    def recording(key: Any, fans: init.Fans) -> np.ndarray:
        seen.append(fans)
        return np.zeros(fans.shape, f32)

    def call(m: Any, x: Units) -> Units:
        y = m.units(x)
        m.first(x, y)
        m.second(x, y)
        return y

    topology = module(
        call,
        units=CreateUnits(),
        first=CreateConnections(init=recording),
        second=CreateConnections(init=recording),
    )
    built(topology, {"x": (2,)})

    for fans in seen:
        np.testing.assert_array_equal(fans.fan_in, [2, 2])
        np.testing.assert_array_equal(fans.fan_out, [2, 2])


# --------------------------------------------------------------------------- #
# With no initializer given
# --------------------------------------------------------------------------- #
def test_by_default_a_weight_is_drawn_as_lecun_normal_and_a_bias_is_zero() -> None:
    fan_in = 4
    record, _, state = built(
        module(lambda m, x: m.fc(x), fc=Dense(256)), {"x": (fan_in,)}
    )
    (one,) = record.constants
    weight = np.array([w for s, _, w in rows(state) if s != one])
    bias = np.array([w for s, _, w in rows(state) if s == one])

    stddev = 1 / np.sqrt(fan_in)
    np.testing.assert_allclose(weight.std(), stddev, rtol=0.1)
    assert 2.0 * stddev < np.abs(weight).max() <= 2 * stddev / 0.8796
    assert bias.shape == (256,)
    np.testing.assert_array_equal(bias, 0.0)


# --------------------------------------------------------------------------- #
# It learns
# --------------------------------------------------------------------------- #
def _load_example(name: str) -> Any:
    path = Path(__file__).resolve().parents[2] / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded
