"""Backward sweep (M3).

Direction reversal: accumulate into the source unit
(dispatch_cpu.hpp:232-258). Two tests: sweep.build_backward_sweep in
isolation (pipeline shape, single bucket) against a numpy reference, and
the topological reverse level-walk end to end through make_step against a
numpy reference of dispatch_cpu.hpp's `L = NumLevels..1` walk.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

import plastax as px
from plastax import sweep as sweep_mod
from plastax.views import UnitWrite

_GRAD = px.FieldSpec.float32("grad")


class _TrivialForward(px.ForwardPass):
    """Weighted-sum forward on ACTIVATION (test_forward_pipeline.py's
    _SumForward) -- required trait slot, not exercised by these tests
    (which only assert on the separate GRAD field)."""

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
    ) -> UnitWrite:
        return UnitWrite.of((px.ACTIVATION, acc))


class _ReplaceBackward(px.BackwardPass):
    """REPLACE-style apply: a unit's new GRAD is exactly its accumulator,
    discarding whatever was there before -- makes "was Apply really called
    on this unit" directly observable (a leaf with no outgoing edges has
    acc == identity == 0, so REPLACE zeroes any pre-seeded value). Used
    only for the direct build_backward_sweep test below, where pipeline's
    unconditional apply (dispatch_cpu.hpp:405-410, no NumInput skip) is
    exactly what that zeroing proves.

    `dst`/`src` follow sweep._accumulate_into's calling convention (first
    unit-id arg is always the accumulator TARGET): for backward that is the
    edge's FROM_ID, so `src` here is bound to the edge's TO_ID -- "the
    other side" -- not literally the edge's source.
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
        return c[px.WEIGHT, cid] * u[_GRAD, dst]

    def apply(
        self, u: px.UnitView, i: px.UnitIdx, g: None, acc: jnp.ndarray
    ) -> UnitWrite:
        return UnitWrite.of((_GRAD, acc))


class _SeedPreservingBackward(px.BackwardPass):
    """ADDITIVE apply: a unit's new GRAD is its accumulator added onto
    whatever it already held. Used for the topological level-walk test,
    where the top level is primed from a fresh identity accumulator
    (sweep.py's identity_accumulator) before any bucket is accumulated --
    additive apply is what lets a pre-seeded output-unit value (standing in
    for a Loss phase staging a gradient, "whatever Loss staged directly
    into BackwardAcc", dispatch_cpu.hpp:328-333) survive that first call
    instead of being zeroed by it.
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
        return c[px.WEIGHT, cid] * u[_GRAD, dst]

    def apply(
        self, u: px.UnitView, i: px.UnitIdx, g: None, acc: jnp.ndarray
    ) -> UnitWrite:
        return UnitWrite.of((_GRAD, u[_GRAD, i] + acc))


class _PipelineBackwardNet(px.Network[None]):
    forward_pass = _TrivialForward()
    backward_pass = _ReplaceBackward()
    propagation = px.Propagation.PIPELINE
    extra_unit_fields = (_GRAD,)


class _TopoBackwardNet(px.Network[None]):
    forward_pass = _TrivialForward()
    backward_pass = _SeedPreservingBackward()
    propagation = px.Propagation.TOPOLOGICAL
    extra_unit_fields = (_GRAD,)


# --- Direct build_backward_sweep test: 4 units, one bucket (pipeline). ----
# 0, 1 -> 2 -> 3; unit 3 is a leaf (no outgoing edges), pre-seeded.
_LEAF_SEED = 10.0
_PIPE_W02, _PIPE_W12, _PIPE_W23 = 0.5, -0.25, 2.0


def _build_pipeline_backward() -> tuple[px.NetworkStatic, px.NetworkState[None]]:
    builder = px.NetworkBuilder(_PipelineBackwardNet, None)
    builder.add_unit()  # 0
    builder.add_unit()  # 1
    builder.add_unit()  # 2
    builder.add_unit(grad=_LEAF_SEED)  # 3: leaf, no outgoing edges
    builder.add_conn(0, 2, weight=_PIPE_W02)
    builder.add_conn(1, 2, weight=_PIPE_W12)
    builder.add_conn(2, 3, weight=_PIPE_W23)
    return builder.finalize()


def test_build_backward_sweep_direction_reversal_and_unconditional_apply() -> None:
    static, state = _build_pipeline_backward()
    sweep = sweep_mod.build_backward_sweep(
        _ReplaceBackward(), num_units=static.num_units, indices_are_sorted=True
    )
    new_units = sweep(state.units, state.conns[0], state.globals_)
    got = np.asarray(new_units[_GRAD.name])

    # Direction reversal (dispatch_cpu.hpp:232-258): unit 0/1 accumulate via
    # their OWN outgoing edge (0->2 / 1->2), reading unit 2's (default,
    # 0.0) grad; unit 2 accumulates via ITS outgoing edge (2->3), reading
    # unit 3's SEEDED grad -- the direction only makes sense read backward
    # from a forward edge's perspective (from_id gets written, to_id is
    # read), the reverse of the forward sweep.
    expected = np.array(
        [
            _PIPE_W02 * 0.0,
            _PIPE_W12 * 0.0,
            _PIPE_W23 * _LEAF_SEED,
            0.0,
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)
    # The leaf (unit 3) is zeroed despite its seed -- proof Apply ran on it
    # unconditionally: no NumInput-style skip in the pipeline backward
    # sweep (dispatch_cpu.hpp:390-392's NumInput parameter is declared but
    # unused; its Apply loop is unconditional over [0, NumUnits)).
    assert float(got[3]) == 0.0


# --- Topological reverse level-walk test, end to end via make_step. ------
# 0, 1 (inputs, level 0) -> 2 (level 1) -> 3 (level 2) -> 4 (level 3,
# output, pre-seeded). 3 buckets/levels, so the walk must chain THROUGH two
# real accumulate steps (unlike the direct test above, which has only one
# bucket) -- this is what actually exercises the high-to-low apply-before-
# next-accumulate ordering dispatch_cpu.hpp:232-258 requires, the backward
# analogue of the forward level-walk's cross-bucket persistence.
_INPUT_SENTINEL = -7.0
_OUTPUT_SEED = 10.0
_TOPO_WA, _TOPO_WB, _TOPO_WC, _TOPO_WD = 0.5, -0.25, 1.5, -2.0


def _build_topo_backward() -> tuple[px.NetworkStatic, px.NetworkState[None]]:
    builder = px.NetworkBuilder(_TopoBackwardNet, None)
    builder.add_unit(grad=_INPUT_SENTINEL)  # 0: input
    builder.add_unit(grad=_INPUT_SENTINEL)  # 1: input
    builder.add_unit()  # 2: hidden
    builder.add_unit()  # 3: hidden2
    builder.add_unit(grad=_OUTPUT_SEED)  # 4: output, pre-seeded
    builder.mark_input(0)
    builder.mark_input(1)
    builder.mark_output(4)
    builder.add_conn(0, 2, weight=_TOPO_WA)
    builder.add_conn(1, 2, weight=_TOPO_WB)
    builder.add_conn(2, 3, weight=_TOPO_WC)
    builder.add_conn(3, 4, weight=_TOPO_WD)
    return builder.finalize()


def _numpy_backward_reference() -> np.ndarray:
    """Hand-computed reverse level walk (dispatch_cpu.hpp:232-258, `L =
    NumLevels..1`): prime the top level directly from its seed (no
    source-level bucket of its own, :328-333), then walk buckets 2, 1
    (high to low), each accumulate reading the JUST-finalized higher-level
    unit. Bucket 0 (units 0/1's own outgoing edges) is never reached --
    input units are excluded from Apply exactly like forward (:250 mirrors
    :59), so grad[0]/grad[1] stay exactly their pre-step (sentinel) value.
    """
    grad = np.array(
        [_INPUT_SENTINEL, _INPUT_SENTINEL, 0.0, 0.0, _OUTPUT_SEED], dtype=np.float32
    )
    grad[3] = grad[3] + _TOPO_WD * grad[4]  # bucket 2 -> finalize level 2
    grad[2] = grad[2] + _TOPO_WC * grad[3]  # bucket 1 -> finalize level 1
    return grad


def test_backward_topological_level_walk_matches_numpy_reverse_pass() -> None:
    static, state = _build_topo_backward()
    assert len(static.level_capacities) == 3
    step = px.make_step(_TopoBackwardNet, static)
    inputs = px.StepInputs(
        inputs=jnp.asarray([1.0, 1.0], dtype=jnp.float32), targets=None
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error", message=".*Some donated buffers were not usable.*"
        )
        result = step(state, inputs)

    got = np.asarray(result.state.units[_GRAD.name])
    expected = _numpy_backward_reference()
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)

    # Explicit input-skip check: units 0/1 (level 0) are untouched, unlike
    # pipeline backward's unconditional apply above.
    assert float(got[0]) == _INPUT_SENTINEL
    assert float(got[1]) == _INPUT_SENTINEL
    # Explicit chaining check: unit 2's grad depends on unit 3's, which
    # must already be finalized when bucket 1 (2->3's bucket) is
    # accumulated -- a wrong (low-to-high) walk order would read unit 3's
    # stale (0.0) grad instead, giving grad[2] == 0.0.
    assert float(got[2]) != 0.0


class _TargetIdProbe(px.BackwardPass):
    """Returns the id of its FIRST unit-id argument, whatever that turns out to be.

    Used to pin which endpoint that argument actually is, independently of what
    it is named.
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
        del u, dst, c, cid, g
        return src.astype(jnp.float32)

    def apply(
        self, u: px.UnitView, i: px.UnitIdx, g: None, acc: jnp.ndarray
    ) -> UnitWrite:
        del u, i, g
        return UnitWrite.of((_GRAD, acc))


class _ProbeNet(px.Network[None]):
    forward_pass = _TrivialForward()
    backward_pass = _TargetIdProbe()
    extra_unit_fields = (_GRAD,)
    propagation = px.Propagation.TOPOLOGICAL


def test_backward_map_first_unit_arg_is_the_accumulator_target() -> None:
    """`BackwardPass.map`'s first unit-id argument is the edge's SOURCE.

    sweep.py documents that map_fn's first unit-id argument is always the
    accumulator target, and backward accumulates into FROM_ID -- so the
    parameter must be `src`, not `dst`. The protocol named it `dst` in both
    directions, which contradicted the sweep and silently inverted the meaning
    of every backward map body written against it.

    The probe returns its first argument's id, so a unit accumulates
    out_degree(i) * i if and only if that argument is the source. Only HIDDEN
    units are checked: the backward phase excludes input units from apply, so
    their accumulator is never written.
    """
    # inputs 0,1 -> hidden 2,3 -> outputs 4,5
    from_ids = np.array([0, 0, 1, 2, 2, 3], dtype=np.int32)
    to_ids = np.array([2, 3, 2, 4, 5, 4], dtype=np.int32)
    static, state = px.NetworkBuilder.from_edges(
        _ProbeNet,
        6,
        from_ids,
        to_ids,
        weights=np.ones((len(from_ids),), dtype=np.float32),
        input_ids=(0, 1),
        output_ids=(4, 5),
        globals_=None,
    )
    result = px.make_step(_ProbeNet, static)(
        state,
        px.StepInputs(inputs=jnp.zeros((2,), jnp.float32), targets=None),
    )
    grad = np.asarray(result.state.units[_GRAD.name])
    for unit in (2, 3):
        out_degree = int(np.sum(from_ids == unit))
        assert grad[unit] == pytest.approx(out_degree * unit), (
            f"unit {unit}: expected {out_degree * unit}, got {grad[unit]} -- "
            "the first unit-id argument is not the accumulator target"
        )
