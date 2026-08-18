"""Host driver: owns the retrace protocol (rung0 design sections 4-5).

The step function itself is pure; the driver is the only component that
reads needs_resort/overflow on host, calls topo.resort or
state.grow_bucket, and swaps in the new (static, step). Retrace count is
observable for tests via jax.test_util.assert_num_jit_and_pmap_compilations.
"""

from __future__ import annotations

from plastax import topo
from plastax.phases import StepInputs
from plastax.state import NetworkState, NetworkStatic, grow_bucket, live_conn_count
from plastax.step import StepFn, make_step
from plastax.traits import Network


class Driver[GS]:
    def __init__(
        self,
        net: type[Network[GS]],
        static: NetworkStatic,
        state: NetworkState[GS],
    ) -> None:
        self._net = net
        self._static = static
        self._state = state
        self._step: StepFn[GS] = make_step(net, static)

    def step(self, inputs: StepInputs) -> None:
        """One network step. Order: run jitted step (donated state); if
        overflow -> grow_bucket + retry the same inputs; if needs_resort ->
        resort + continue (resort is between-steps, matching native
        NeedsResort semantics).

        Overflow retry loop -- NOT a replay from a pristine pre-attempt
        state (Deviation, IMPLEMENTATION_PLAN.md: `self._step` donates its
        `state` argument, jax/_src/interpreters/mlir.py:1177; XLA is free
        to invalidate that argument's buffers the moment the call returns,
        successful or not, so `self._state` is a dangling reference to
        deleted device memory as soon as `self._step(self._state, ...)`
        has been called once -- confirmed empirically, an
        "INVALID_ARGUMENT: Buffer has been deleted or donated" the first
        time this method tried to reuse it. Preserving a pristine copy
        would need an unconditional defensive copy on EVERY call, on the
        chance THIS ONE overflows, which permanently defeats the
        donation-based in-place update the whole framework is built on
        [D:6] for the common, non-overflowing case). Instead, "retry"
        grows the bucket(s) from -- and replays against -- the FAILED
        attempt's own valid output (`result.state`, not `self._state`):
        forward/backward/update_conn/prune_conn genuinely run an
        additional time on an overflow retry (not idempotent in general,
        e.g. a decaying UpdateConn), a real behavioral cost, but the
        alternative (an unconditional per-step copy) costs the donation
        invariant on every step to protect a rare, capacity-mistuned edge
        case; growing a bucket already means the network was configured
        below its live working set, an event `capacity_policy`'s headroom
        is meant to make rare in the first place. Fullness is checked
        against `result.state`'s OWN live counts (a candidate only
        overflows a bucket add_conn has already packed to exactly its
        capacity THIS call, build_add_conn_phase's `committed` claims
        every free slot before any candidate is ever dropped) against the
        PRE-grow `self._static.level_capacities`.
        """
        while True:
            result = self._step(self._state, inputs)
            if bool(result.overflow):
                state = result.state
                for level in range(len(self._static.level_capacities)):
                    live = int(live_conn_count(state, level))
                    if live == self._static.level_capacities[level]:
                        self._static, state = grow_bucket(self._static, state, level)
                self._state = state
                self._step = make_step(self._net, self._static)
                continue

            state = result.state
            if bool(state.needs_resort):
                self._static, self._state = topo.resort(self._static, state)
                self._step = make_step(self._net, self._static)
                return

            self._state = state
            return

    @property
    def state(self) -> NetworkState[GS]:
        return self._state

    @property
    def static(self) -> NetworkStatic:
        return self._static
