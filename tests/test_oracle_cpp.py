"""C++ oracle parity (M5).

Golden-file parity against the native C++ Plastix build. The two flagship
examples (mlp_xor, ipc_multilayer) both initialise weights from a PRNG, and
plastax's jax PRNG cannot reproduce the C++ std::mt19937 stream, so their
trajectories are only comparable in aggregate (see their acceptance tests).
The `manual-fcc` example is the one built-in example with NO randomness --
fixed weights (0.5 into the hidden layer, -0.3 into the output), a fixed
tanh forward pass, and three fixed inputs -- so it pins the topological
forward *kernel* against the oracle bit-for-bit (within reduction-order
tolerance).

Reproduce this network exactly with `from_topology` + constant
initialisers, drive it through the host `Driver` (mirroring the oracle's
sequential `for In : Inputs { Net.DoStep(In); }`), and compare the output
unit's activation to the golden values.

Golden values were generated once by the native binary and are pinned
below; regenerate with:

    plastix/build/examples/manual-fcc/manual_fcc

(built from plastix/examples/manual-fcc/manual_fcc.cpp). Tolerance is the
plan's topological rtol=1e-4 -- segment-reduction order differs from the
C++ level sweep -- with a small atol so the exact-zero row is not compared
by relative error alone.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import plastax as px

# 2 inputs -> 4 hidden (weight 0.5) -> 1 output (weight -0.3), tanh forward.
_HIDDEN_WEIGHT = 0.5
_OUTPUT_WEIGHT = -0.3
_NUM_INPUTS = 2
_NUM_HIDDEN = 4
_NUM_OUTPUT = 1

# (input pair -> output activation), from the native manual_fcc binary.
_GOLDEN: tuple[tuple[tuple[float, float], float], ...] = (
    ((0.1, 0.2), -0.176785),
    ((0.5, -0.5), 0.000000),
    ((1.0, 1.0), -0.723005),
)


class _TanhForward(px.ForwardPass):
    """manual_fcc.cpp's TanhForwardPass: acc = sum(weight * activation[src]),
    apply = tanh(acc)."""

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
        return c[px.WEIGHT, cid] * u[px.ACTIVATION, src]

    def apply(
        self, u: px.UnitView, i: px.UnitIdx, g: None, acc: jax.Array
    ) -> px.UnitWrite:
        del u, i, g
        return px.UnitWrite.of((px.ACTIVATION, jnp.tanh(acc)))


class _ManualFcc(px.Network[None]):
    forward_pass = _TanhForward()
    propagation = px.Propagation.TOPOLOGICAL


def _build() -> tuple[px.NetworkStatic, px.NetworkState[None]]:
    topology_fn = px.topology.sequential(
        px.topology.input_units(_NUM_INPUTS),
        px.topology.dense(
            _NUM_INPUTS, _NUM_HIDDEN, init=jax.nn.initializers.constant(_HIDDEN_WEIGHT)
        ),
        px.topology.dense(
            _NUM_HIDDEN, _NUM_OUTPUT, init=jax.nn.initializers.constant(_OUTPUT_WEIGHT)
        ),
    )
    # constant init is key-independent; the key only satisfies the signature.
    return px.NetworkBuilder.from_topology(
        _ManualFcc, topology_fn, jax.random.PRNGKey(0), globals_=None
    )


def test_manual_fcc_forward_matches_the_cpp_oracle() -> None:
    static, state = _build()
    assert static.propagation is px.Propagation.TOPOLOGICAL
    # inputs at level 0, hidden at level 1, output at level 2 -> 2 conn buckets.
    assert len(static.level_capacities) == 2

    driver = px.Driver(_ManualFcc, static, state)
    output_id = int(static.output_ids[0])

    for (x1, x2), golden in _GOLDEN:
        driver.step(
            px.StepInputs(inputs=jnp.asarray([x1, x2], dtype=jnp.float32), targets=None)
        )
        got = float(driver.state.units[px.ACTIVATION.name][output_id])
        np.testing.assert_allclose(
            got,
            golden,
            rtol=1e-4,
            atol=1e-5,
            err_msg=f"input ({x1}, {x2}): plastax {got} vs C++ oracle {golden}",
        )
