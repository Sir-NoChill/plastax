"""Donation contract.

make_step jits with donate_argnums=0 (step.py), so the returned step
function claims ownership of the whole state pytree: XLA is free to reuse
each input buffer for the corresponding output. That is sound only if the
step is shape-preserving on every leaf -- a leaf whose shape/dtype changes
cannot alias its input, and JAX signals the wasted donation with a "Some
donated buffers were not usable" warning.

These tests pin both halves of the contract: (1) donation actually happens
(the input state's buffers are deleted the moment the call returns -- the
same behaviour driver.py relies on and warns about in its docstring), and
(2) it is never wasted -- the pytree structure and every leaf's shape/dtype
survive the call, and feeding the donated output straight back into the
next step raises nothing under a filter that promotes that warning to an
error.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import plastax as px


class _TrivialForward(px.policies.ForwardConnPolicy, px.policies.ForwardUnitPolicy):
    """Minimal required trait: acc = sum(weight * activation[src]); apply
    writes it straight back to ACTIVATION (shape-preserving by construction).
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
        del dst, g
        return c[px.WEIGHT, cid] * u[px.ACTIVATION, src]

    def apply(
        self, u: px.UnitView, i: px.UnitIdx, g: None, acc: jax.Array
    ) -> px.UnitWrite:
        del u, i, g
        return px.UnitWrite.of((px.ACTIVATION, acc))


class _DonationNet(px.Network[None]):
    forward = _TrivialForward()
    propagation = px.Propagation.PIPELINE


def _build() -> tuple[px.NetworkStatic, px.NetworkState[None]]:
    builder = px.NetworkBuilder(_DonationNet, None)
    builder.add_unit()  # 0
    builder.add_unit()  # 1
    builder.add_unit()  # 2: destination
    builder.add_conn(0, 2, weight=0.5)
    builder.add_conn(1, 2, weight=0.5)
    return builder.finalize()


_DUMMY_INPUTS = px.StepInputs(inputs=jnp.zeros((0,), dtype=jnp.float32), targets=None)


def test_step_deletes_every_donated_input_buffer() -> None:
    static, state = _build()
    step = px.make_step(_DonationNet, static)
    input_leaves = jax.tree_util.tree_leaves(state)
    assert input_leaves  # a non-trivial pytree, so "all([])" can't pass vacuously

    result = step(state, _DUMMY_INPUTS)
    jax.block_until_ready(result.state)

    # donate_argnums=0: XLA consumes each input buffer, so every leaf of the
    # argument state is deleted once the call returns.
    assert all(leaf.is_deleted() for leaf in input_leaves)
