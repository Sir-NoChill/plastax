"""Pytree contract for NetworkState / NetworkStatic.

NetworkState flatten/unflatten roundtrip; NetworkStatic meta fields hash/eq;
changing a meta field changes the PyTreeDef; changing a leaf does not.
"""

from __future__ import annotations

import dataclasses

import jax

import plastax as px
from plastax.views import UnitWrite


class _SumForward(px.policies.ForwardConnPolicy, px.policies.ForwardUnitPolicy):
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
    forward = _SumForward()
    propagation = px.Propagation.TOPOLOGICAL


def _build_tiny(
    globals_: object = None,
) -> tuple[px.NetworkStatic, px.NetworkState[object]]:
    builder = px.NetworkBuilder(_TinyNet, globals_)
    a = builder.add_unit()
    b = builder.add_unit()
    builder.add_conn(a, b, weight=0.5)
    return builder.finalize()


def test_network_static_hashable_and_eq() -> None:
    static, _ = _build_tiny()
    static_copy = dataclasses.replace(static)
    assert static == static_copy
    assert hash(static) == hash(static_copy)

    static_diff = dataclasses.replace(static, num_units=static.num_units + 1)
    assert static != static_diff
    assert hash(static) != hash(static_diff)
