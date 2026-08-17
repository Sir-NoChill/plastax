"""UpdateConn two-pass ordering (M3b).

dispatch_cpu.hpp:450-469: a full `incoming` sweep over every live conn,
then a full `outgoing` sweep -- strictly sequenced, so `outgoing` observes
`incoming`'s writes. PruneConn / derived live counts stay in
test_update_prune.py, skipped pending M4 (that file also covers UpdateConn
in the plan's original one-file-per-milestone layout, but its M4 scope --
PruneConn, live-count derivation alongside UpdateConn -- is broader than
what M3b implements; this file exercises the UpdateConn phase alone, on
its own milestone).
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np

import plastax as px
from plastax import phases

_INCOMING_MARK = -123.5


class _TrivialForward(px.ForwardPass):
    """Required trait slot (Network.forward_pass is mandatory), not
    exercised by these tests, which only assert on WEIGHT."""

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
    ) -> px.UnitWrite:
        return px.UnitWrite.of((px.ACTIVATION, acc))


class _OrderProbeUpdateConn(px.UpdateConn):
    """incoming overwrites weight with a fixed marker; outgoing reads the
    CURRENT weight and adds 1000. If the two passes were not strictly
    sequenced (e.g. a single combined vmap computing both callbacks off the
    same pre-phase snapshot), outgoing would see each edge's ORIGINAL
    (distinct) weight and the results would differ per edge; correctly
    sequenced, every live edge converges on the identical
    _INCOMING_MARK + 1000 regardless of its starting weight."""

    def incoming(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: None,
    ) -> px.ConnWrite:
        del u, dst, src, c, cid, g
        return px.ConnWrite.of((px.WEIGHT, jnp.float32(_INCOMING_MARK)))

    def outgoing(
        self,
        u: px.UnitView,
        src: px.UnitIdx,
        dst: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: None,
    ) -> px.ConnWrite:
        del u, src, dst, g
        return px.ConnWrite.of((px.WEIGHT, c[px.WEIGHT, cid] + jnp.float32(1000.0)))


class _OrderProbeNet(px.Network[None]):
    forward_pass = _TrivialForward()
    update_conn = _OrderProbeUpdateConn()
    propagation = px.Propagation.PIPELINE


_DUMMY_INPUTS = px.StepInputs(inputs=jnp.zeros((0,), dtype=jnp.float32), targets=None)


def test_incoming_writes_land_before_outgoing_reads() -> None:
    builder = px.NetworkBuilder(_OrderProbeNet, None)
    builder.add_unit()  # 0
    builder.add_unit()  # 1
    builder.add_unit()  # 2
    builder.add_conn(0, 2, weight=1.0)
    builder.add_conn(1, 2, weight=-5.0)
    static, state = builder.finalize()

    phase = phases.build_update_conn_phase(_OrderProbeNet, static)
    new_state, loss = phase(state, _DUMMY_INPUTS)

    assert float(loss) == 0.0
    weights = np.asarray(new_state.conns[0][px.WEIGHT.name])
    dead = np.asarray(new_state.conns[0][px.DEAD.name])
    live_weights = weights[~dead]
    assert live_weights.size == 2
    expected = _INCOMING_MARK + 1000.0
    np.testing.assert_allclose(live_weights, expected, rtol=1e-6, atol=1e-6)


def test_dead_conns_are_never_updated_by_either_pass() -> None:
    """Dead slots (headroom beyond the 1 live edge, capacity_policy's
    min_bucket=64 default) must keep the FieldSpec default weight (0.0)
    through both passes -- the null-slot-discipline analogue (sweep.py's
    _build_conn_update docstring): a dead row's computed write is
    discarded, not merged."""
    builder = px.NetworkBuilder(_OrderProbeNet, None)
    builder.add_unit()
    builder.add_unit()
    builder.add_unit()
    builder.add_conn(0, 2, weight=1.0)
    static, state = builder.finalize()
    assert static.level_capacities[0] > 1  # real headroom beyond the 1 live edge

    phase = phases.build_update_conn_phase(_OrderProbeNet, static)
    new_state, _ = phase(state, _DUMMY_INPUTS)

    dead = np.asarray(new_state.conns[0][px.DEAD.name])
    weights = np.asarray(new_state.conns[0][px.WEIGHT.name])
    assert bool(dead[1:].all())  # exactly 1 conn was added; it sorts to slot 0
    np.testing.assert_array_equal(weights[1:], np.zeros_like(weights[1:]))


def test_two_pass_sweeps_every_bucket_in_topological_mode() -> None:
    """update_conn has no level structure of its own -- dispatch_cpu.hpp:
    450-469 takes no Ranges/NumLevels, unlike the forward/backward level
    walks -- so every bucket must be swept by both passes, not just bucket
    0. 3-level net (test_forward_topo.py's shape): 0,1 (L0) -> 2 (L1) -> 3
    (L2) -> 4 (L3); the marker must land in every one of the 3 buckets."""

    class _MarkNet(px.Network[None]):
        forward_pass = _TrivialForward()
        update_conn = _OrderProbeUpdateConn()
        propagation = px.Propagation.TOPOLOGICAL

    builder = px.NetworkBuilder(_MarkNet, None)
    for _ in range(5):
        builder.add_unit()
    builder.mark_input(0)
    builder.mark_input(1)
    builder.mark_output(4)
    builder.add_conn(0, 2, weight=1.0)
    builder.add_conn(1, 2, weight=2.0)
    builder.add_conn(2, 3, weight=3.0)
    builder.add_conn(3, 4, weight=4.0)
    static, state = builder.finalize()
    assert len(static.level_capacities) == 3

    phase = phases.build_update_conn_phase(_MarkNet, static)
    new_state, _ = phase(state, _DUMMY_INPUTS)

    for bucket in new_state.conns:
        dead = np.asarray(bucket[px.DEAD.name])
        weights = np.asarray(bucket[px.WEIGHT.name])
        live_weights = weights[~dead]
        assert live_weights.size > 0
        np.testing.assert_allclose(
            live_weights, _INCOMING_MARK + 1000.0, rtol=1e-6, atol=1e-6
        )


def test_update_conn_is_wired_into_build_phases_via_make_step() -> None:
    """End to end through make_step (not a directly-called isolated phase
    function): confirms build_phases' guard-raise removal actually wires
    update_conn into the compiled step rather than leaving
    build_update_conn_phase merely reachable-but-unused."""
    builder = px.NetworkBuilder(_OrderProbeNet, None)
    builder.add_unit()
    builder.add_unit()
    builder.add_unit()
    builder.add_conn(0, 2, weight=1.0)
    builder.add_conn(1, 2, weight=-5.0)
    static, state = builder.finalize()

    step = px.make_step(_OrderProbeNet, static)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error", message=".*Some donated buffers were not usable.*"
        )
        result = step(state, _DUMMY_INPUTS)

    weights = np.asarray(result.state.conns[0][px.WEIGHT.name])
    dead = np.asarray(result.state.conns[0][px.DEAD.name])
    np.testing.assert_allclose(
        weights[~dead], _INCOMING_MARK + 1000.0, rtol=1e-6, atol=1e-6
    )


# --- dst/src endpoint-binding convention, incoming and outgoing in isolation


class _IncomingEndpointProbe(px.UpdateConn):
    """weight <- activation[dst] - activation[src]; asymmetric, so a
    dst/src argument swap changes both the sign and magnitude, catching a
    wrong wire-up. outgoing is a genuine no-op so the final weight reflects
    ONLY incoming's own (dst, src) binding."""

    def incoming(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: None,
    ) -> px.ConnWrite:
        del c, cid, g
        diff = u[px.ACTIVATION, dst] - u[px.ACTIVATION, src]
        return px.ConnWrite.of((px.WEIGHT, diff))

    def outgoing(
        self,
        u: px.UnitView,
        src: px.UnitIdx,
        dst: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: None,
    ) -> px.ConnWrite:
        del u, src, dst, c, cid, g
        return px.ConnWrite.of()


class _OutgoingEndpointProbe(px.UpdateConn):
    """Mirror of _IncomingEndpointProbe with the roles reversed: incoming
    is the no-op, so the final weight reflects ONLY outgoing's own (src,
    dst) binding."""

    def incoming(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: None,
    ) -> px.ConnWrite:
        del u, dst, src, c, cid, g
        return px.ConnWrite.of()

    def outgoing(
        self,
        u: px.UnitView,
        src: px.UnitIdx,
        dst: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: None,
    ) -> px.ConnWrite:
        del c, cid, g
        diff = u[px.ACTIVATION, dst] - u[px.ACTIVATION, src]
        return px.ConnWrite.of((px.WEIGHT, diff))


class _IncomingEndpointNet(px.Network[None]):
    forward_pass = _TrivialForward()
    update_conn = _IncomingEndpointProbe()
    propagation = px.Propagation.PIPELINE


class _OutgoingEndpointNet(px.Network[None]):
    forward_pass = _TrivialForward()
    update_conn = _OutgoingEndpointProbe()
    propagation = px.Propagation.PIPELINE


def _build_endpoint_probe(
    net: type[px.Network[None]],
) -> tuple[px.NetworkStatic, px.NetworkState[None]]:
    builder = px.NetworkBuilder(net, None)
    builder.add_unit(activation=10.0)  # 0: src
    builder.add_unit(activation=20.0)  # 1: src
    builder.add_unit(activation=100.0)  # 2: dst
    builder.add_conn(0, 2, weight=0.0)
    builder.add_conn(1, 2, weight=0.0)
    return builder.finalize()


def test_incoming_binds_dst_then_src_matching_oracle_convention() -> None:
    """incoming(u, dst, src, ...) <-> UC::UpdateIncomingConnection(UnitAlloc,
    ToId, FromId, ...) (dispatch_cpu.hpp:458)."""
    static, state = _build_endpoint_probe(_IncomingEndpointNet)
    phase = phases.build_update_conn_phase(_IncomingEndpointNet, static)
    new_state, _ = phase(state, _DUMMY_INPUTS)

    weights = np.asarray(new_state.conns[0][px.WEIGHT.name])
    dead = np.asarray(new_state.conns[0][px.DEAD.name])
    live = np.sort(weights[~dead])
    # edge 0->2: dst(100) - src(10) = 90; edge 1->2: dst(100) - src(20) = 80.
    np.testing.assert_allclose(live, np.sort([90.0, 80.0]), rtol=1e-6, atol=1e-6)


def test_outgoing_binds_src_then_dst_matching_oracle_convention() -> None:
    """outgoing(u, src, dst, ...) <-> UC::UpdateOutgoingConnection(UnitAlloc,
    FromId, ToId, ...) (dispatch_cpu.hpp:466)."""
    static, state = _build_endpoint_probe(_OutgoingEndpointNet)
    phase = phases.build_update_conn_phase(_OutgoingEndpointNet, static)
    new_state, _ = phase(state, _DUMMY_INPUTS)

    weights = np.asarray(new_state.conns[0][px.WEIGHT.name])
    dead = np.asarray(new_state.conns[0][px.DEAD.name])
    live = np.sort(weights[~dead])
    np.testing.assert_allclose(live, np.sort([90.0, 80.0]), rtol=1e-6, atol=1e-6)
