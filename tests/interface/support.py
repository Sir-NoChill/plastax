"""Helpers shared by the interface tests: build a small network, run it, read its state.

Nothing here touches the interface at import time, so importing this module never fails.
Every use of `px` is inside a function.

The functions under "Reading state" are the only place the tests use `px.inspect`, which
the design document does not describe yet (see README.md). When that part of the
interface is decided, only this file changes.
"""

# NOTE: Could this be integrated nicely with the
# hydra config framework? Are there uses for that?
# I would like plastax to be aware of the hydra
# configuration tooling available, as well as
# weights and biases and openML tooling that could
# be useful to researchers.

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import numpy as np

import plastax as px

RTOL = 1e-5
ATOL = 1e-6


def close(
    actual: Any, expected: Any, *, rtol: float = RTOL, atol: float = ATOL
) -> None:
    """Assert two arrays are equal to float32 tolerance, shapes included."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #
def topology(call: Callable[..., Any], **layers: Any) -> Any:
    """A module with the given layers whose `__call__` is `call(module, x)`.

    For topologies with one input. A topology with named inputs is written out as a
    class in the test that needs it.
    """

    class Topology(px.layers.Module):
        def setup(self) -> None:
            for name, layer in layers.items():
                setattr(self, name, layer)

        def __call__(self, x: Any) -> Any:
            return call(self, x)

    return Topology()


def learned(lr: float = 0.1) -> Any:
    """Weighted-sum connections trained by plain SGD."""
    p = px.policies
    return p.Conn(
        forward=p.WeightedSum(),
        backward=p.WeightedSumBackward(),
        update=px.optim.sgd(lr=lr, grad=p.GRAD),
    )


def fixed() -> Any:
    """Weighted-sum connections that nothing updates, prunes or grows."""
    p = px.policies
    return p.Conn(
        forward=p.WeightedSum(),
        backward=p.WeightedSumBackward(),
        update=None,
        prune=None,
        add=None,
    )


def network(model: Any, **attrs: Any) -> Any:
    """A `px.Network` over `model`.

    Defaults: SGD-trained weighted sums, summing units, squared error, seed 0. Any of
    them can be replaced through `attrs`, which become attributes of the network class.
    """
    defaults: dict[str, Any] = {
        "conn_policy": learned(),
        "unit_policy": px.policies.summing,
        "loss": px.SquaredErrorLossPolicy(seed=px.policies.SEED),
        "config": px.Config(propagation=px.Propagation.TOPOLOGICAL, seed=0),
    }
    return type("TestNetwork", (px.Network,), {"topology": model, **defaults, **attrs})


# --------------------------------------------------------------------------- #
# Running and reading state
# --------------------------------------------------------------------------- #
class Runner:
    """One network instance: its state, its compiled step, and ways to read the
    state.
    """

    def __init__(self, net: Any, *, globals: Any = None, seed: int = 0) -> None:
        self.net = net
        extra = {} if globals is None else {"globals": globals}
        self.state = net.init(jax.random.PRNGKey(seed), **extra)
        self._step = px.make_step(net)

    def step(self, inputs: Any, targets: Any = None) -> Any:
        """Run one full step, keep the new state, return the step's result."""
        result = self._step(self.state, px.StepInputs(inputs=inputs, targets=targets))
        self.state = result.state
        return result

    def predict(self, inputs: Any, *, keep: bool = False) -> Any:
        """Run the forward policies only. Keeps what they wrote only when `keep`."""
        writes = px.Writes.KEEP if keep else px.Writes.DISCARD
        forward = px.make_step(self.net, phases=px.Phases.FORWARD, writes=writes)
        result = forward(self.state, px.StepInputs(inputs=inputs))
        if keep:
            self.state = result.state
        return result

    # ---- Reading state (the only uses of `px.inspect`) ----
    def field(self, field: Any, ref: Any) -> np.ndarray:
        """A unit field over the units `ref` names, in the shape of their handle.

        `ref` is a `px.UnitsRef`, a `px.UnitRef` or a `px.Input`. With `ref=None` the
        result is the field over every live unit, in unit order.
        """
        return np.asarray(px.inspect.unit_field(self.net, self.state, field, ref))

    def unit_ids(self, ref: Any) -> np.ndarray:
        """The indices of the units `ref` names, in the shape of their handle."""
        return np.asarray(px.inspect.unit_ids(self.net, ref))

    def conns(self, layer: Any = None) -> Any:
        """The live connections of `layer`, or of the whole network when it is None."""
        return px.inspect.connections(self.net, self.state, layer)

    def conn_field(self, layer: Any, field: Any) -> np.ndarray:
        """A connection field over the live connections of `layer`."""
        return np.asarray(self.conns(layer).field(field))

    def pairs(self, layer: Any = None) -> list[tuple[int, int]]:
        """The (source, destination) unit ids of the live connections of `layer`."""
        c = self.conns(layer)
        src, dst = np.asarray(c.src).tolist(), np.asarray(c.dst).tolist()
        return list(zip(src, dst, strict=True))

    def weights(self, layer: Any) -> np.ndarray:
        """The layer's weights as a (destinations, sources) matrix; 0 where unconnected.

        Sources and destinations are in the row-major order of their handles.
        """
        src_ids = np.asarray(px.inspect.sources(self.net, layer)).reshape(-1)
        dst_ids = np.asarray(px.inspect.destinations(self.net, layer)).reshape(-1)
        col = {int(u): k for k, u in enumerate(src_ids)}
        row = {int(u): k for k, u in enumerate(dst_ids)}
        matrix = np.zeros((len(dst_ids), len(src_ids)), np.float32)
        w = self.conn_field(layer, px.WEIGHT)
        for (s, d), value in zip(self.pairs(layer), w, strict=True):
            matrix[row[d], col[s]] += value
        return matrix

    def num_units(self) -> int:
        """How many units are alive."""
        return int(px.inspect.num_units(self.net, self.state))

    def num_connections(self) -> int:
        """How many connections are alive, over the whole network."""
        return int(px.inspect.num_connections(self.net, self.state))
