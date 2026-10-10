"""Host driver + retrace protocol.

The step function returns flags for the network needing
either a resort or having overflowed memory. If either flag
is true, then we call into jax to redo the network trace and
recompile the code. The user can check the number of
retraces performed via
jax.test_util.assert_num_jit_and_pmap_compilations .

The overflow flag is connection overflow: a selected growth candidate found
no slot it may claim -- its TOPOLOGICAL level's bucket is full, or the
PIPELINE bucket's never-used tail is -- which the driver fixes inside the same
step: it grows the full buckets and claims the dropped candidates, re-running
nothing but the growth phase's claim.
Unit overflow (`NetworkState.unit_overflow`, an AddUnit spawn that found no
free unit slot) is not a retrace event: `Network.unit_capacity` is fixed, so
the dropped spawn stays dropped and the flag is the caller's to read.

## Recommendations for Poor Performance

1. If your algorithm exhibits many overflow events then
   you should pre-allocate more VRAM. plastax-cpp regrows its
   buckets inside a fixed arena instead, so it retraces nothing
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

from typing import Literal

import jax.numpy as jnp

from plastax import topo
from plastax._types import Propagation
from plastax.phases import StepInputs
from plastax.state import NetworkState, NetworkStatic, grow_bucket, live_conn_count
from plastax.step import StepFn, make_growth_retry, make_step
from plastax.traits import Network


class Driver[GS]:
    """Runs a network's step loop, owning retrace on overflow and resort.

    With `check_every=1` (the default) the flags are read after every step:
    an overflowing step's growth is completed after growing its full buckets
    (only the growth claim re-runs; see `step`), and a resort runs before the
    next step, so every step is exact.

    The step is `make_step(net, static, batch_size=..., layout=...,
    fuse_prune=..., growth=...)`, rebuilt whenever a regrow or resort changes
    the static configuration; a batched or Scheme-A-sharded step overflows
    and recovers exactly like the streaming one.

    With `check_every=N > 1` the Driver reads the flags back only every N
    steps, keeping the host from blocking on the device in between. The
    overflow flag is OR-accumulated on device and `needs_resort` is already
    sticky in the state. Two semantics change, both bounded by N:

    - An overflowing step is not retried: the candidates it could not place
      are simply not grown. At the next check, every bucket with fewer free
      slots than the growth policy's `max_new_per_level` is grown.
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
        batch_size: int | None = None,
        layout: Literal["auto", "edge_list", "csr", "triton"] = "auto",
        fuse_prune: Literal["auto", "triton", "xla", "off"] = "auto",
        growth: Literal["auto", "xla", "triton"] = "auto",
    ) -> None:
        if check_every < 1:
            raise ValueError(f"Driver: check_every must be >= 1, got {check_every}")
        self._net = net
        self._static = static
        self._state = state
        self._batch_size = batch_size
        self._layout: Literal["auto", "edge_list", "csr", "triton"] = layout
        self._fuse_prune: Literal["auto", "triton", "xla", "off"] = fuse_prune
        self._growth: Literal["auto", "xla", "triton"] = growth
        self._step: StepFn[GS] = self._make_step()
        self._check_every = check_every
        self._since_check = 0
        self._overflowed = jnp.bool_(False)

    def step(self, inputs: StepInputs) -> None:
        """Run one step, completing an overflowing growth, then resort.

        The step runs once, as one fused jitted call. If its growth overflowed,
        only the growth is finished, inside the same step: every full bucket
        grows (`grow_bucket`), then the candidates the claim dropped
        (`StepResult.growth_remainder`) claim the new room
        (`make_growth_retry`), in the total order, until none is left. The
        selection is the step's own, so the result is exactly what the step
        would have committed had the buckets been that large from the start:
        forward, backward and the updates run once, the step counter advances
        once, and `state.overflow` reads False afterwards (`state.grown`
        counts every edge the step committed). Then, on needs_resort, the
        network resorts before the next step, matching native NeedsResort
        semantics.

        The completion works on the step's output state: the jitted step
        donates its input, so the pre-step buffers may be gone. That is why
        the step hands back its selection rather than being re-run on a
        regrown copy of its input. Every full bucket grows: a TOPOLOGICAL
        bucket with no dead slot left, or a PIPELINE bucket with no tail
        left, read off the output state against the pre-grow
        `self._static.level_capacities` -- an overflowing level's claim takes
        every slot it may use before dropping a candidate, so its bucket is
        always among them.

        Args:
            inputs: The external inputs for this step.
        """
        if self._check_every > 1:
            self._deferred_step(inputs)
            return
        result = self._step(self._state, inputs)
        state = result.state
        remainder = result.growth_remainder
        while bool(state.overflow):
            assert remainder is not None  # only growth overflows
            for level in range(len(self._static.level_capacities)):
                if self._free_slots(state, level) == 0:
                    self._static, state = grow_bucket(self._static, state, level)
            self._step = self._make_step()
            retry = make_growth_retry(self._net, self._static)
            state, remainder = retry(state, remainder)

        if bool(state.needs_resort):
            self._static, state = topo.resort(self._static, state)
            self._step = self._make_step()
        self._state = state

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
            want = getattr(ac, "max_new_per_level", None) or 1 if ac is not None else 1
            state = self._state
            for level in range(len(self._static.level_capacities)):
                if self._free_slots(state, level) < want:
                    self._static, state = grow_bucket(self._static, state, level)
            self._state = state
            self._step = self._make_step()
        if bool(self._state.needs_resort):
            self._static, self._state = topo.resort(self._static, self._state)
            self._step = self._make_step()

    def _make_step(self) -> StepFn[GS]:
        """The step for the current static configuration."""
        return make_step(
            self._net,
            self._static,
            batch_size=self._batch_size,
            layout=self._layout,
            fuse_prune=self._fuse_prune,
            growth=self._growth,
        )

    def _free_slots(self, state: NetworkState[GS], level: int) -> int:
        """The slots any level can still claim in bucket `level`.

        Every dead slot of a TOPOLOGICAL bucket; only the never-used tail of
        the PIPELINE bucket, whose dead slots serve the levels that left them.
        """
        capacity = self._static.level_capacities[level]
        if self._static.propagation is Propagation.PIPELINE:
            return capacity - int(state.tail_start)
        return capacity - int(live_conn_count(state, level))

    @property
    def state(self) -> NetworkState[GS]:
        """The current network state."""
        return self._state

    @property
    def static(self) -> NetworkStatic:
        """The current static configuration."""
        return self._static
