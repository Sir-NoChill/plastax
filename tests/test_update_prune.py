"""UpdateConn / PruneConn composition.

PruneConn tombstones the predicted conns (dead set, `live_conn_count` drops
accordingly, derived from the mask rather than stored). UpdateConn's OWN
incoming/outgoing two-pass ordering is already fully exercised by
test_update_conn.py; this file additionally confirms that ordering holds
when prune_conn runs AFTERWARD in the SAME step
(phases.py module docstring's phase order: ... update_conn, prune_conn,
...) -- i.e. prune_conn's predicate observes update_conn's FRESH write,
not a stale pre-step snapshot.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np

import plastax as px

_DUMMY_INPUTS = px.StepInputs(inputs=jnp.zeros((0,), dtype=jnp.float32), targets=None)


class _TrivialForward(px.policies.ForwardConnPolicy, px.policies.ForwardUnitPolicy):
    """Required trait slot (Network.forward_pass is mandatory), not
    exercised by these tests, which only assert on WEIGHT/DEAD."""

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
    ) -> px.UnitWrite:
        return px.UnitWrite.of((px.ACTIVATION, acc))


_THRESHOLD = 2.0


class _ThresholdPrune(px.policies.PruneConnPolicy):
    """Prunes any conn whose CURRENT weight is at or below _THRESHOLD."""

    def predicate(
        self, u: px.UnitView, c: px.ConnView, cid: px.ConnIdx, g: None
    ) -> jax.Array:
        del u, g
        return c[px.WEIGHT, cid] <= jnp.float32(_THRESHOLD)


class _PruneNet(px.Network[None]):
    forward = _TrivialForward()
    prune = px.policies.Prune(conn=_ThresholdPrune())
    propagation = px.Propagation.PIPELINE


def _build_prune_net() -> tuple[px.NetworkStatic, px.NetworkState[None]]:
    builder = px.NetworkBuilder(_PruneNet, None)
    builder.add_unit()  # 0
    builder.add_unit()  # 1
    builder.add_unit()  # 2
    builder.add_unit()  # 3: dst for every conn below
    builder.add_conn(0, 3, weight=1.0)  # below threshold -> pruned
    builder.add_conn(1, 3, weight=5.0)  # above threshold -> survives
    builder.add_conn(2, 3, weight=2.0)  # at threshold ( <= prunes) -> pruned
    return builder.finalize()


def test_prune_conn_wired_into_build_phases_via_make_step() -> None:
    static, state = _build_prune_net()
    step = px.make_step(_PruneNet, static)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error", message=".*Some donated buffers were not usable.*"
        )
        result = step(state, _DUMMY_INPUTS)

    dead = np.asarray(result.state.conns[0][px.DEAD.name])
    from_id = np.asarray(result.state.conns[0][px.FROM_ID.name])
    to_id = np.asarray(result.state.conns[0][px.TO_ID.name])
    live_pairs = {
        (int(f), int(t)) for f, t, d in zip(from_id, to_id, dead, strict=True) if not d
    }
    assert live_pairs == {(1, 3)}
    assert bool(result.overflow) is False


_DECAY = 1.0
_COMPOSE_THRESHOLD = 2.0


class _DecayIncoming(px.policies.UpdateConnPolicy):
    """incoming shrinks weight by a fixed decay; outgoing is a no-op so the
    step applies exactly one decay, not two."""

    def incoming(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: None,
    ) -> px.ConnWrite:
        del u, dst, src, g
        return px.ConnWrite.of((px.WEIGHT, c[px.WEIGHT, cid] - jnp.float32(_DECAY)))

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


class _ComposeThresholdPrune(px.policies.PruneConnPolicy):
    def predicate(
        self, u: px.UnitView, c: px.ConnView, cid: px.ConnIdx, g: None
    ) -> jax.Array:
        del u, g
        return c[px.WEIGHT, cid] <= jnp.float32(_COMPOSE_THRESHOLD)


class _DecayThenPruneNet(px.Network[None]):
    forward = _TrivialForward()
    update = px.policies.Update(conn=_DecayIncoming())
    prune = px.policies.Prune(conn=_ComposeThresholdPrune())
    propagation = px.Propagation.PIPELINE


def test_update_conn_runs_before_prune_conn_so_pruning_sees_the_fresh_weight() -> None:
    """Phase order (phases.py module docstring): forward, loss, backward,
    update_conn, prune_conn, add_conn, reset_global. Conn A starts at 2.5:
    ABOVE _COMPOSE_THRESHOLD before decay, AT-OR-BELOW it after -- so it is
    pruned only if prune_conn's predicate reads update_conn's fresh
    (post-decay) weight within the SAME step, not a pre-step snapshot. Conn
    B (5.0) stays comfortably above the threshold either way: a control
    showing the step does not prune indiscriminately.
    """
    builder = px.NetworkBuilder(_DecayThenPruneNet, None)
    builder.add_unit()  # 0
    builder.add_unit()  # 1
    builder.add_unit()  # 2: dst for both conns below
    builder.add_conn(0, 2, weight=2.5)  # discriminates ordering
    builder.add_conn(1, 2, weight=5.0)  # control: survives regardless
    static, state = builder.finalize()

    step = px.make_step(_DecayThenPruneNet, static)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error", message=".*Some donated buffers were not usable.*"
        )
        result = step(state, _DUMMY_INPUTS)

    dead = np.asarray(result.state.conns[0][px.DEAD.name])
    weight = np.asarray(result.state.conns[0][px.WEIGHT.name])
    from_id = np.asarray(result.state.conns[0][px.FROM_ID.name])

    # Both conns share to_id=2; the builder's stable (dead, to_id) sort
    # keeps insertion order for the tie, so position 0 is (0, 2) and
    # position 1 is (1, 2) (matching test_update_conn.py's own reasoning
    # about stable-sort slot assignment).
    assert int(from_id[0]) == 0
    assert int(from_id[1]) == 1
    assert bool(dead[0]) is True
    np.testing.assert_allclose(float(weight[0]), 1.5, rtol=1e-6, atol=1e-6)
    assert bool(dead[1]) is False
    np.testing.assert_allclose(float(weight[1]), 4.0, rtol=1e-6, atol=1e-6)
