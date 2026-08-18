"""Donation contract (M5).

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

import warnings

import jax
import jax.numpy as jnp

import plastax as px


class _TrivialForward(px.ForwardPass):
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
    forward_pass = _TrivialForward()
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


def test_step_preserves_pytree_structure_and_every_leaf_shape_dtype() -> None:
    static, state = _build()
    step = px.make_step(_DonationNet, static)

    # Captured BEFORE the call: donation deletes the input leaves, so their
    # shape/dtype cannot be read afterward.
    in_structure = jax.tree_util.tree_structure(state)
    in_specs = [(leaf.shape, leaf.dtype) for leaf in jax.tree_util.tree_leaves(state)]

    result = step(state, _DUMMY_INPUTS)

    out_structure = jax.tree_util.tree_structure(result.state)
    out_specs = [
        (leaf.shape, leaf.dtype) for leaf in jax.tree_util.tree_leaves(result.state)
    ]
    assert out_structure == in_structure
    assert out_specs == in_specs


def test_donated_output_chains_into_the_next_step_without_warning() -> None:
    static, state = _build()
    step = px.make_step(_DonationNet, static)

    with warnings.catch_warnings():
        # The exact warning JAX emits when a donated buffer cannot be reused;
        # promoted to an error so a non-shape-preserving step fails loudly.
        warnings.filterwarnings("error", message=".*donated buffers were not usable.*")
        result = step(state, _DUMMY_INPUTS)
        # Feed the donated output straight back in -- the steady-state loop
        # the Driver runs, and the case that would surface a bad donation.
        result = step(result.state, _DUMMY_INPUTS)
        jax.block_until_ready(result.state)
