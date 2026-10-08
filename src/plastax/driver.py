"""Host driver + retrace protocol.

The step function returns flags for the network needing
either a resort or having overflowed memory. If either flag
is true, then we call into jax to redo the network trace and
recompile the code. The user can check the number of
retraces performed via
jax.test_util.assert_num_jit_and_pmap_compilations .

## Recommendations for Poor Performance

1. If your algorithm exhibits many overflow events then
   you should pre-allocate more VRAM. Note that in plastax-cpp
   this should not be an issue as VRAM is
   reallocated on demand
2. If your algorithm exhibits many retrace events, then you
   may want to consider implementing a pipelined version
   of your algorithm. Pipelined versions do not have to be
   sorted and execution order is arbitrary. If your algorithm
   is amenable to that paradigm, it will almost invariably
   perform better than a topologically sorted algorithm
3. Every step reads the two flags back to the host, which blocks dispatch.
   For large nets this is negligible; for small, launch-bound nets pass
   `check_every=N` to read them every N steps instead (see `Driver`).
"""

from __future__ import annotations

import jax.numpy as jnp

from plastax import topo
from plastax.phases import StepInputs
from plastax.state import NetworkState, NetworkStatic, grow_bucket, live_conn_count
from plastax.step import StepFn, make_step
from plastax.traits import Network


class Driver[GS]:
    """Runs a network's step loop, owning retrace on overflow and resort.

    With `check_every=1` (the default) the flags are read after every step:
    an overflowing step is retried after growing its full buckets, and a
    resort runs before the next step, so every step is exact.

    With `check_every=N > 1` the Driver reads the flags back only every N
    steps, keeping the host from blocking on the device in between. The
    overflow flag is OR-accumulated on device and `needs_resort` is already
    sticky in the state. Two semantics change, both bounded by N:

    - An overflowing step is not retried: the candidates it could not place
      are simply not grown. At the next check, every bucket with fewer free
      slots than the growth policy's `max_candidates` is grown.
    - A resort runs at the next check rather than before the next step. Until
      then, a committed edge that breaks the leveling invariant (a same-level
      or backward edge) may be skipped by the topological forward for up to
      N - 1 steps; level-preserving growth is unaffected.

    Type Args:
        GS: the global state type threaded through the network.
    """

    def __init__(
        self,
        net: type[Network[GS]],
        static: NetworkStatic,
        state: NetworkState[GS],
        *,
        check_every: int = 1,
    ) -> None:
        if check_every < 1:
            raise ValueError(f"Driver: check_every must be >= 1, got {check_every}")
        self._net = net
        self._static = static
        self._state = state
        self._step: StepFn[GS] = make_step(net, static)
        self._check_every = check_every
        self._since_check = 0
        self._overflowed = jnp.bool_(False)

    def step(self, inputs: StepInputs) -> None:
        """Run one step, handling overflow growth-and-retry and resort.

        On overflow, grow_bucket and retry the same inputs; on
        needs_resort, resort and continue -- resort happens between
        steps, matching native NeedsResort semantics.

        The retry replays against the failed attempt's own output
        (`result.state`), not a pristine `self._state`: the jitted step
        donates its state argument, so XLA is free to invalidate the
        pre-attempt buffers as soon as the call returns, successful or
        not. So forward/backward/update_conn/prune_conn genuinely re-run
        on a retry -- a real, non-idempotent cost (e.g. a decaying
        UpdateConn) -- but the alternative, an unconditional per-step
        defensive copy, would defeat the donation-based in-place update
        on every step to guard a rare, capacity-mistuned case; growing a
        bucket already means the network was configured below its live
        working set, which `capacity_policy`'s headroom is meant to make
        rare. Fullness is checked against `result.state`'s own live
        counts against the pre-grow `self._static.level_capacities`,
        since `build_add_conn_phase`'s `committed` claims every free
        slot before any candidate is dropped.

        Args:
            inputs: The external inputs for this step.
        """
        if self._check_every > 1:
            self._deferred_step(inputs)
            return
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

    def _deferred_step(self, inputs: StepInputs) -> None:
        """One `check_every > 1` step: no host read-back except every N steps."""
        result = self._step(self._state, inputs)
        self._state = result.state
        # Dispatched, not synced: the OR runs on device.
        self._overflowed = self._overflowed | result.overflow
        self._since_check += 1
        if self._since_check < self._check_every:
            return
        self._since_check = 0
        if bool(self._overflowed):
            self._overflowed = jnp.bool_(False)
            ac = self._net.add_conn
            want = ac.max_candidates if ac is not None else 1
            state = self._state
            for level in range(len(self._static.level_capacities)):
                capacity = self._static.level_capacities[level]
                if capacity - int(live_conn_count(state, level)) < want:
                    self._static, state = grow_bucket(self._static, state, level)
            self._state = state
            self._step = make_step(self._net, self._static)
        if bool(self._state.needs_resort):
            self._static, self._state = topo.resort(self._static, self._state)
            self._step = make_step(self._net, self._static)

    @property
    def state(self) -> NetworkState[GS]:
        """The current network state."""
        return self._state

    @property
    def static(self) -> NetworkStatic:
        """The current static configuration."""
        return self._static
