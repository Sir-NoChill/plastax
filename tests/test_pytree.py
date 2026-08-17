"""Pytree contract for NetworkState / NetworkStatic (M1).

NetworkState flatten/unflatten roundtrip; NetworkStatic meta fields hash/eq;
changing a meta field changes the PyTreeDef; changing a leaf does not.
Implemented when M1 lands (see tests/README.md, IMPLEMENTATION_PLAN.md).
"""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp

import plastax as px
from plastax.views import UnitWrite


class _SumForward(px.ForwardPass):
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
        return c[px.WEIGHT, cid] * u[px.ACTIVATION, src]

    def apply(
        self, u: px.UnitView, i: px.UnitIdx, g: None, acc: jax.Array
    ) -> UnitWrite:
        return UnitWrite.of((px.ACTIVATION, acc))


class _TinyNet(px.Network[None]):
    forward_pass = _SumForward()
    propagation = px.Propagation.TOPOLOGICAL


def _build_tiny(
    globals_: object = None,
) -> tuple[px.NetworkStatic, px.NetworkState[object]]:
    builder = px.NetworkBuilder(_TinyNet, globals_)
    a = builder.add_unit()
    b = builder.add_unit()
    builder.add_conn(a, b, weight=0.5)
    return builder.finalize()


def test_network_static_flattens_to_zero_leaves() -> None:
    static, _ = _build_tiny()
    leaves, treedef = jax.tree_util.tree_flatten(static)
    assert leaves == []
    # A CustomNode with no children: unflattening the empty leaf list from
    # its own treedef must round-trip to an equal (all-meta) object.
    assert jax.tree_util.tree_unflatten(treedef, []) == static


def test_network_static_hashable_and_eq() -> None:
    static, _ = _build_tiny()
    static_copy = dataclasses.replace(static)
    assert static == static_copy
    assert hash(static) == hash(static_copy)

    static_diff = dataclasses.replace(static, num_units=static.num_units + 1)
    assert static != static_diff
    assert hash(static) != hash(static_diff)


def test_network_state_flatten_leaves_are_arena_arrays_and_globals() -> None:
    static, state = _build_tiny(globals_={"tau": jnp.array(1.0)})
    leaves, _ = jax.tree_util.tree_flatten(state)

    num_unit_leaves = len(static.unit_fields)
    num_conn_leaves = len(static.conn_fields) * len(static.level_capacities)
    num_globals_leaves = 1  # {"tau": jnp.array(1.0)}
    num_needs_resort_leaves = 1
    expected = (
        num_unit_leaves + num_conn_leaves + num_globals_leaves + num_needs_resort_leaves
    )
    assert len(leaves) == expected
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)


def test_network_state_flatten_unflatten_roundtrip_is_identity() -> None:
    static, state = _build_tiny(globals_={"tau": jnp.array(2.5)})
    leaves, treedef = jax.tree_util.tree_flatten(state)
    roundtripped = jax.tree_util.tree_unflatten(treedef, leaves)

    assert roundtripped.units.keys() == state.units.keys()
    for name, column in state.units.items():
        assert jnp.array_equal(column, roundtripped.units[name])
        assert column.dtype == roundtripped.units[name].dtype

    assert len(roundtripped.conns) == len(state.conns)
    for original_bucket, roundtripped_bucket in zip(
        state.conns, roundtripped.conns, strict=True
    ):
        assert original_bucket.keys() == roundtripped_bucket.keys()
        for name, column in original_bucket.items():
            assert jnp.array_equal(column, roundtripped_bucket[name])

    assert roundtripped.globals_.keys() == state.globals_.keys()
    assert jnp.array_equal(roundtripped.globals_["tau"], state.globals_["tau"])
    assert bool(roundtripped.needs_resort) == bool(state.needs_resort)


def test_changing_a_meta_field_changes_the_pytreedef() -> None:
    static, _ = _build_tiny()
    _, treedef_a = jax.tree_util.tree_flatten(static)

    changed = dataclasses.replace(static, num_units=static.num_units + 1)
    _, treedef_b = jax.tree_util.tree_flatten(changed)
    assert treedef_a != treedef_b

    changed_ids = dataclasses.replace(static, input_ids=(*static.input_ids, 0))
    _, treedef_c = jax.tree_util.tree_flatten(changed_ids)
    assert treedef_a != treedef_c


def test_changing_a_leaf_does_not_change_the_pytreedef() -> None:
    _, state = _build_tiny()
    _, treedef_a = jax.tree_util.tree_flatten(state)

    changed = dataclasses.replace(
        state, units={**state.units, "activation": state.units["activation"] + 1.0}
    )
    _, treedef_b = jax.tree_util.tree_flatten(changed)
    assert treedef_a == treedef_b


def test_networkstatic_value_equality_drives_cache_reuse() -> None:
    """The load-bearing mechanism the design doc cites (rung0 section 1):
    `_make_step` is a `weakref_lru_cache` keyed on `NetworkStatic`, so two
    value-equal-but-distinct instances must collapse to one cache entry and
    a meta-field change must mint a new one. functools.lru_cache exercises
    the same hash/eq contract without needing M2's make_step.
    """
    static, _ = _build_tiny()
    static_value_equal = dataclasses.replace(static)  # equal, not the same object
    assert static_value_equal is not static
    static_different = dataclasses.replace(static, num_units=static.num_units + 1)

    builds = 0

    @functools.cache
    def build_step(s: px.NetworkStatic) -> int:
        nonlocal builds
        builds += 1
        return builds

    build_step(static)
    build_step(static_value_equal)  # equal by value -> cache hit, no new build
    assert builds == 1

    build_step(static_different)  # different meta field -> cache miss
    assert builds == 2
