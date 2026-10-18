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

The resort flag is acted on under TOPOLOGICAL propagation only. A PIPELINE
net's levels are bookkeeping (the builder's and unit addition's), its single
bucket runs in any order, and plastax-cpp sorts by level only under
Topological, so the Driver never resorts a PIPELINE net: the flag stays
whatever growth left it and is the caller's to read.

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
3. Every `step` dispatches one jitted call and reads the two flags back to
   the host, which blocks dispatch. For large nets that is negligible; a
   small, launch-bound net is better run with `Driver.scan`, which keeps a
   whole chunk of steps on the device, exactly, and comes back to the host
   once per chunk. `check_every=N` is the older relaxation (flags read every
   N steps, with weaker semantics; see `Driver`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import jax
import jax.numpy as jnp
from jax import lax

from plastax import topo
from plastax._types import Propagation
from plastax.phases import GrowthRemainder, StepInputs
from plastax.state import NetworkState, NetworkStatic, grow_bucket, live_conn_count
from plastax.step import StepFn, StepResult, make_growth_retry, make_step
from plastax.traits import Network


def _loss_record(result: StepResult[Any]) -> jax.Array:
    """The default `Driver.scan` record: the step's loss."""
    return result.loss


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

    `scan` runs a chunk of steps as one device-side loop with the same
    semantics as `step`, reading back to the host once per chunk instead of
    once per step; it is the way to run a small, launch-bound net.

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

    `check_every` does not apply to `scan`, which is always exact.

    A PIPELINE net never resorts (see the module docstring): the Driver
    neither reads nor clears its `needs_resort`.

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
        self._resorts = net.propagation is Propagation.TOPOLOGICAL
        self._step: StepFn[GS] = self._make_step()
        self._check_every = check_every
        self._since_check = 0
        self._overflowed = jnp.bool_(False)
        # One compiled chunk loop per (static configuration, record callable).
        self._scan_fns: dict[tuple[NetworkStatic, Callable[..., Any]], Any] = {}

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
        counts every edge the step committed). Then, on needs_resort, a
        TOPOLOGICAL network resorts before the next step, matching native
        NeedsResort semantics; a PIPELINE network never resorts.

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
        state = self._complete_growth(result.state, result.growth_remainder)
        self._state = self._resort_if_due(state)

    def scan(
        self,
        inputs: StepInputs,
        *,
        record: Callable[[StepResult[GS]], Any] | None = None,
        steps: int | None = None,
    ) -> Any:
        """Run a chunk of steps as one device-side loop; exact, like `step`.

        `inputs` holds the chunk along a leading axis: `(K, num_inputs)`
        inputs and `(K, num_outputs)` targets (`(K, B, ...)` for a batched
        step). One jitted call runs the step K times in a `lax.while_loop`,
        with the state donated and resident on the device: no dispatch and
        no flag read-back per step. The loop stops itself right after a step
        that overflowed (or, in a TOPOLOGICAL network, that needs a resort);
        the host then completes that step exactly as `step` does (regrow and
        claim the dropped candidates, resort), rebuilds the loop for the new
        static configuration, and re-enters it at the next step of the same
        chunk. A chunk therefore costs one host round trip plus one per
        regrow or resort, and the result is step for step what K calls of
        `step` produce, whatever `check_every` is.

        `record` is evaluated on every step's `StepResult` on the device and
        the results come back stacked along the chunk axis, which is how a
        per-step signal (a prediction, the loss) leaves the device without a
        sync per step. The default records the loss. The compiled loop is
        cached per (static configuration, `record`), so pass the same
        callable object every time rather than a fresh lambda. A step that
        overflowed is re-recorded once its growth is complete. Each distinct
        chunk length compiles its own loop, so pad the last chunk to the
        common length and pass `steps` for the number to run.

        A `layout="csr"` batched step loses its cuSPARSE lowering inside the
        loop (see `make_step`), and jax_triton kernels are not validated in
        it under Scheme-A sharding.

        Args:
            inputs: The chunk's inputs, each leaf with the chunk axis first.
            record: What to keep from each step, as a function of its
                `StepResult`; None records the loss.
            steps: How many leading steps of the chunk to run; None runs
                them all.

        Returns:
            The per-step records, stacked along a leading axis of length K;
            entries at or beyond `steps` are zero.

        Raises:
            ValueError: If `inputs` has no chunk axis, its leaves disagree on
                the chunk length, or `steps` is outside `[0, K]`.
        """
        rec = _loss_record if record is None else record
        if inputs.inputs.ndim < 2:
            raise ValueError(
                "Driver.scan: inputs need a leading chunk axis, got shape "
                f"{inputs.inputs.shape}"
            )
        length = inputs.inputs.shape[0]
        if inputs.targets is not None and inputs.targets.shape[:1] != (length,):
            raise ValueError(
                f"Driver.scan: targets chunk axis {inputs.targets.shape[:1]} does "
                f"not match the inputs' ({length},)"
            )
        count = length if steps is None else steps
        if not 0 <= count <= length:
            raise ValueError(
                f"Driver.scan: steps must be in [0, {length}], got {steps}"
            )
        start = 0
        out: Any = None
        while True:
            loop = self._scan_fn(rec)
            state, status, remainder, loss, records = loop(
                self._state, inputs, jnp.int32(start), jnp.int32(count)
            )
            self._state = state
            out = records if out is None else _merge_from(out, records, start)
            # One host read: the steps done so far, and whether the loop
            # stopped itself after the last of them.
            done, halted = divmod(int(status), 2)
            if halted:
                # Finish step `done - 1` on the host, then re-record it.
                state = self._complete_growth(state, remainder)
                self._state = self._resort_if_due(state)
                completed = StepResult(
                    state=self._state, overflow=jnp.bool_(False), loss=loss
                )
                out = _set_at(out, done - 1, rec(completed))
            if done >= count:
                return out
            start = done

    def _scan_fn(self, record: Callable[[StepResult[GS]], Any]) -> Any:
        """The jitted chunk loop for the current static config and `record`."""
        key = (self._static, record)
        loop = self._scan_fns.get(key)
        if loop is not None:
            return loop
        step = self._step
        resorts = self._resorts

        def run(
            state: NetworkState[GS],
            inputs: StepInputs,
            start: jax.Array,
            count: jax.Array,
        ) -> tuple[NetworkState[GS], jax.Array, GrowthRemainder | None, jax.Array, Any]:
            length = inputs.inputs.shape[0]
            first = jax.tree_util.tree_map(lambda a: a[0], inputs)
            remainder0, records0 = _carry_templates(step, record, state, first, length)

            def keep_going(carry: tuple[Any, ...]) -> jax.Array:
                _, i, halted, _, _, _ = carry
                return jnp.logical_and(i < count, jnp.logical_not(halted))

            def body(carry: tuple[Any, ...]) -> tuple[Any, ...]:
                st, i, _, _, _, recs = carry
                x = jax.tree_util.tree_map(
                    lambda a: lax.dynamic_index_in_dim(a, i, keepdims=False), inputs
                )
                r = step(st, x)
                halt = r.overflow
                if resorts:
                    halt = halt | r.state.needs_resort
                recs = jax.tree_util.tree_map(
                    lambda buf, v: lax.dynamic_update_index_in_dim(buf, v, i, 0),
                    recs,
                    record(r),
                )
                return r.state, i + 1, halt, r.growth_remainder, r.loss, recs

            carry0 = (
                state,
                start,
                jnp.bool_(False),
                remainder0,
                jnp.float32(0.0),
                records0,
            )
            st, done, halted, remainder, loss, recs = lax.while_loop(
                keep_going, body, carry0
            )
            status = done * 2 + halted.astype(jnp.int32)
            return st, status, remainder, loss, recs

        loop = jax.jit(run, donate_argnums=0)
        self._scan_fns[key] = loop
        return loop

    def _complete_growth(
        self, state: NetworkState[GS], remainder: GrowthRemainder | None
    ) -> NetworkState[GS]:
        """Finish an overflowing step's growth: regrow, claim the remainder."""
        while bool(state.overflow):
            assert remainder is not None  # only growth overflows
            for level in range(len(self._static.level_capacities)):
                if self._free_slots(state, level) == 0:
                    self._static, state = grow_bucket(self._static, state, level)
            self._step = self._make_step()
            retry = make_growth_retry(self._net, self._static)
            state, remainder = retry(state, remainder)
        return state

    def _resort_if_due(self, state: NetworkState[GS]) -> NetworkState[GS]:
        """Resort a TOPOLOGICAL network whose flag is raised; PIPELINE never."""
        if self._resorts and bool(state.needs_resort):
            self._static, state = topo.resort(self._static, state)
            self._step = self._make_step()
        return state

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
        self._state = self._resort_if_due(self._state)

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


def _carry_templates(
    step: StepFn[Any],
    record: Callable[[StepResult[Any]], Any],
    state: NetworkState[Any],
    first: StepInputs,
    length: int,
) -> tuple[Any, Any]:
    """Zeros shaped like a step's growth remainder and like `length` records.

    Abstractly evaluated: the step runs nothing. The shapes come back flat
    and are rebuilt from the tree structures, so no state or result dataclass
    is constructed from shape structs (their constructors are type-checked
    under test).
    """
    trees: list[Any] = []

    def flat(s: NetworkState[Any], x: StepInputs) -> list[Any]:
        r = step(s, x)
        rem = jax.tree_util.tree_leaves(r.growth_remainder)
        rec = jax.tree_util.tree_leaves(record(r))
        trees.append(
            (
                jax.tree_util.tree_structure(r.growth_remainder),
                jax.tree_util.tree_structure(record(r)),
                len(rem),
            )
        )
        return [*rem, *rec]

    shapes = jax.eval_shape(flat, state, first)
    rem_tree, rec_tree, n_rem = trees[0]
    remainder0 = jax.tree_util.tree_unflatten(
        rem_tree, [jnp.zeros(a.shape, a.dtype) for a in shapes[:n_rem]]
    )
    records0 = jax.tree_util.tree_unflatten(
        rec_tree, [jnp.zeros((length, *a.shape), a.dtype) for a in shapes[n_rem:]]
    )
    return remainder0, records0


def _set_at(records: Any, index: int, record: Any) -> Any:
    """`records` with chunk entry `index` replaced by `record`."""
    return jax.tree_util.tree_map(lambda o, r: o.at[index].set(r), records, record)


def _merge_from(old: Any, new: Any, start: int) -> Any:
    """`new`'s records from chunk index `start` on, `old`'s before it."""

    def merge(o: jax.Array, n: jax.Array) -> jax.Array:
        index = jnp.arange(n.shape[0]).reshape((-1,) + (1,) * (n.ndim - 1))
        return jnp.where(index >= start, n, o)

    return jax.tree_util.tree_map(merge, old, new)
