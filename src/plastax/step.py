"""Step assembly: the monomorphization point.

One jit cache entry per (Network subclass, NetworkStatic); donation on the
whole state pytree (donate_argnums=0). Cached with the weakref_lru_cache
pattern.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
from collections.abc import Callable
from typing import Any, Literal, TypeVar, cast

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec
from jaxtyping import Array, Bool, Float

from plastax._types import ACTIVATION, Propagation
from plastax.distributed import scheme_a_mesh
from plastax.phases import (
    GrowthRemainder,
    Phase,
    PruneFusionPlan,
    PruneFusionRecord,
    StepInputs,
    build_batched_phases,
    build_growth_retry,
    build_phases,
    nvidia_triton_available,
    per_sample_written_fields,
    plan_prune_fusion,
    reduce_batch_units,
)
from plastax.state import NetworkState, NetworkStatic
from plastax.traits import Network, batch_written_reduction

# Module-scoped (not PEP 695) so it stays free inside the StepFn alias below;
# StepResult/make_step below shadow it with their own PEP 695 [GS] locally.
GS = TypeVar("GS")


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class StepResult[GS]:
    """One step's output: the new state plus framework-computed signals.

    Type Args:
        GS: the user's global-state pytree, opaque to the framework.

    Attributes:
        state: The network state after the step.
        overflow: growth overflow flag for the step (also on the state).
        loss: The scalar the loss policy returned for the step; 0.0 when the
            net has no loss phase.
        growth_remainder: The selected growth candidates the step's claim
            dropped for lack of room (`overflow`), which `make_growth_retry`
            claims after a regrow; None when the net does not grow
            connections.
    """

    state: NetworkState[GS]
    overflow: Bool[Array, ""]
    loss: Float[Array, ""]
    growth_remainder: GrowthRemainder | None = None


StepFn = Callable[[NetworkState[GS], StepInputs], StepResult[GS]]
GrowthRetryFn = Callable[
    [NetworkState[GS], GrowthRemainder], tuple[NetworkState[GS], GrowthRemainder]
]


def make_step[GS](
    net: type[Network[GS]],
    static: NetworkStatic,
    *,
    batch_size: int | None = None,
    layout: Literal["auto", "edge_list", "csr", "triton"] = "auto",
    fuse_prune: Literal["auto", "triton", "xla", "off"] = "auto",
    growth: Literal["auto", "xla", "triton"] = "auto",
) -> StepFn[GS]:
    """Assemble the present phases and jit them with donate_argnums=0.

    The returned callable must be shape-preserving on the state pytree so
    every leaf donates (CI promotes the donation warning to an error).

    plastax is built for streaming, one sample per step. `batch_size=B` is a
    convenience for mini-batch training and evaluation of feed-forward
    (TOPOLOGICAL) nets: `StepInputs` then carries `(B, num_inputs)` inputs and
    `(B, num_outputs)` targets; forward, loss, backward, and the unit update
    run per sample against the shared connections; the connection update is
    reduced over the batch (see `phases.build_batched_update_conn`: exact for
    the `optim/` bundles, mean-of-writes otherwise); prune, add, and reset run
    once on the reduced unit state, which is also what the returned state
    holds; and `StepResult.loss` is the batch mean.

    How the per-sample unit columns reduce to one state is the network's
    `batch_reduction` policy (`traits.BatchReduction`), which a batched step
    requires: there is no implicit average. When the step is first traced it
    finds the columns its per-sample phases write (the input scatter's
    `ACTIVATION`, the loss's seed field and every column a forward, backward or
    unit-update rule returns in its `UnitWrite`) and raises if the policy
    leaves one undeclared or declares it NOT_BATCHED. Set
    `batch_reduction = MeanFloatFirstRest()` for the batch mean of every
    floating column and sample 0 of every other one.

    `layout` picks how a batched step runs a *linear* forward or backward pass
    (one declaring `linear_input`, see `phases.linear_input_field`):
    "edge_list" runs it per sample over the edge arena; "csr" builds each
    bucket's CSR view on device every step (one radix sort) and runs the
    whole batch as one cuSPARSE sparse-dense product (fast on NVIDIA GPUs
    only; elsewhere jax falls back to generic kernels); "triton" runs each
    bucket as one edge-once Triton kernel through jax_triton (every edge read
    once for the whole batch, relaxed atomics into the targets) on an NVIDIA
    GPU with the `plastax[triton]` extra, and as the same edge-once product in
    plain XLA anywhere else. "auto" picks, on an NVIDIA GPU, "triton" for
    `2 <= batch_size <= 64` (when jax_triton is installed) and "csr" above
    it -- in batched SGD at 5.4M edges Triton is 1.7x the edge list at B = 2
    and 2.9x at 8, ahead of CSR through 64, and CSR is 4 % ahead at 128 (a
    tie at 50M) -- and on every other backend (AMD GPUs,
    TPU, CPU) the XLA edge-once product for `batch_size >= 2` (the speed of
    the per-sample edge list, with far smaller temporaries). Non-linear
    passes, and every streaming step, use the edge list. Under Scheme-A
    sharding, "triton" uses the XLA edge-once product (jax_triton is not
    validated inside shard_map).

    `fuse_prune` lets a streaming step evaluate the prune_conn predicate
    inside the forward's edge sweep, so each bucket's edge columns are read
    once rather than twice, when `phases.plan_prune_fusion` proves the
    predicate sees the same values there (decided once, when the step is
    first traced). The single-pass lowering is one Triton kernel per bucket
    (gather, atomic scatter-add, predicate, tombstones and add_conn's
    free-slot block counts), for an unsharded linear forward on an NVIDIA
    GPU with jax_triton; "auto" fuses only then, and otherwise keeps the
    two-pass step. "triton" also fuses where the kernel does not apply, as
    the same computation in plain XLA (which reads the edge columns twice,
    XLA being unable to fuse a scatter with a reduction), and "xla" always
    does; "off" never fuses. The tombstones and free-slot counts match the
    two-pass step exactly; forward sums may differ in summation order. The
    returned step carries the decision as `step.prune_fusion` (a
    `phases.PruneFusionRecord`). A batched step never fuses.

    `growth` picks the add_conn free-slot claim (see
    `phases.build_add_conn_phase`): "triton" claims and writes every growing
    bucket in one jax_triton kernel (plus one clearing the tombstones) on an
    NVIDIA GPU with the `plastax[triton]` extra, where "xla" is the portable
    claim (per-block free counts, a search, and one scatter per column, about
    20 kernels per bucket). Both pick the same slots. "auto" takes "triton"
    where it can run; a "triton" request that cannot (another backend, or
    Scheme-A sharding, where jax_triton is not validated) uses "xla".

    In a batched step a non-floating connection column written by an
    UpdateConn without the exact pair is stored from sample 0: such columns
    should agree across the batch. With layout "csr", call the step
    directly: the cuSPARSE lowering is scoped to its own calls, so wrapping
    it in an outer jit or scan lowers it outside that scope, onto the
    generic (much slower, still correct) kernel.

    Type Args:
        GS: the user's global-state pytree, opaque to the framework.

    Args:
        net: The network subclass to assemble phases for.
        static: The network's static configuration.
        batch_size: Samples per step, or None for the streaming step.
        layout: The batched linear-pass layout: "auto", "edge_list", "csr",
            or "triton".
        fuse_prune: Whether a streaming step fuses the prune predicate into
            the forward sweep: "auto", "triton", "xla", or "off".
        growth: The add_conn claim engine: "auto", "xla", or "triton".

    Returns:
        A jitted step function for the given network and static config.

    Raises:
        ValueError: If `batch_size` is below 1, or set for a PIPELINE net
            (whose carried unit state is per-sample recurrent state) or for a
            net without a `batch_reduction`, or if `layout`, `fuse_prune` or
            `growth` is not one of its values. The batched step raises it when
            first traced if a per-sample phase writes a unit column its
            `batch_reduction` leaves undeclared or declares NOT_BATCHED.
    """
    if batch_size is not None:
        if batch_size < 1:
            raise ValueError(f"make_step: batch_size must be >= 1, got {batch_size}")
        if net.propagation is Propagation.PIPELINE:
            raise ValueError(
                "make_step: batch_size is for TOPOLOGICAL (feed-forward) nets; a "
                "PIPELINE net carries per-sample recurrent state between steps"
            )
        if net.batch_reduction is None:
            raise ValueError(
                f"make_step: a batched step needs {net.__name__}.batch_reduction, "
                "the policy combining the per-sample unit columns (see "
                "plastax.BatchReduction); set batch_reduction = "
                "MeanFloatFirstRest() for the batch mean of every floating "
                "column and sample 0 of every other one"
            )
    if layout not in ("auto", "edge_list", "csr", "triton"):
        raise ValueError(f"make_step: unknown layout {layout!r}")
    if fuse_prune not in ("auto", "triton", "xla", "off"):
        raise ValueError(f"make_step: unknown fuse_prune {fuse_prune!r}")
    if growth not in ("auto", "xla", "triton"):
        raise ValueError(f"make_step: unknown growth engine {growth!r}")
    engine: str | None = None
    if batch_size is not None:
        triton_ok = nvidia_triton_available()
        if layout == "auto" and _nvidia_gpu():
            if batch_size > 64:
                engine = "csr"
            elif batch_size >= 2 and triton_ok:
                engine = "triton"
        elif layout == "auto" and batch_size >= 2:
            # Off NVIDIA (AMD GPU, TPU, CPU): the XLA edge-once product. Same
            # speed as the per-sample edge list on GPU, but it does not
            # materialise per-sample edge temporaries: compiled for TPU v5e,
            # a B = 32 Adam step at 5.4M edges needs 1.5 GB of temporaries
            # against 4.1 GB.
            engine = "xla"
        elif layout == "csr":
            engine = "csr"
        elif layout == "triton":
            engine = "triton" if triton_ok else "xla"
        # jax_triton is not validated inside shard_map; the portable XLA
        # edge-once product is (it all-reduces like every other sweep).
        if engine == "triton" and static.sharding is not None:
            engine = "xla"
    # mypy false positive: a parameterized generic base class fails the
    # structural Hashable check, though a class is always hashable by
    # identity; hence the cast.
    return cast(
        StepFn[GS],
        _cached_make_step(net, static, batch_size, engine, fuse_prune, growth),  # type: ignore[arg-type]
    )


def _spec(cls: type[Any], **fields: Any) -> Any:
    """Build a frozen-dataclass spec instance without running its __init__.

    The shard_map spec pytrees hold PartitionSpec leaves in the array-typed
    state/input/result container shapes, so the type-checked constructors
    (instrumented by jaxtyping under test) would reject them. This bypasses
    __init__ via object.__new__, yielding an instance with the same pytree
    structure whose leaves are the given specs.
    """
    obj = object.__new__(cls)
    for name, value in fields.items():
        object.__setattr__(obj, name, value)
    return obj


def _shard_map_step(
    step: Callable[[NetworkState[Any], StepInputs], StepResult[Any]],
    static: NetworkStatic,
) -> Callable[[NetworkState[Any], StepInputs], StepResult[Any]]:
    """Wrap `step` in a shard_map that shards connections across the mesh.

    Scheme A: the connection arenas are sharded on their capacity axis over
    the device mesh; units, globals, and the scalar signals are replicated.
    Each shard's sweep sees only its own edges, and the monoid collective in
    the sweep all-reduces the per-shard partial accumulators, so the sharded
    step is identical to the single-device step. The spec pytrees hold
    PartitionSpec leaves in the state/input/result container shapes; a bare
    replicated PartitionSpec at `globals_` is a prefix over the whole opaque
    globals subtree.
    """
    repl, state_spec = _scheme_a_specs(static)
    in_specs: Any = (state_spec, _spec(StepInputs, inputs=repl, targets=repl))
    # A bare replicated spec is a prefix over the whole remainder subtree
    # (replicated: the claim's masks and values are computed from replicated
    # units and collectives), and over None when the net does not grow.
    out_specs: Any = _spec(
        StepResult,
        state=state_spec,
        overflow=repl,
        loss=repl,
        growth_remainder=repl,
    )
    sharded: Any = jax.shard_map(
        step, mesh=scheme_a_mesh(static), in_specs=in_specs, out_specs=out_specs
    )
    return cast(Callable[[NetworkState[Any], StepInputs], StepResult[Any]], sharded)


def _scheme_a_specs(static: NetworkStatic) -> tuple[Any, Any]:
    """The replicated spec and the state's spec pytree under Scheme-A.

    PartitionSpec is untyped in jax's stubs; the spec pytrees deliberately
    hold PartitionSpec leaves in the array-typed state shapes, so they are
    built and threaded as Any.
    """
    sharding = static.sharding
    assert sharding is not None  # only called on the sharded branch
    repl: Any = PartitionSpec()  # type: ignore[no-untyped-call]
    conn: Any = PartitionSpec(sharding.axis_name)  # type: ignore[no-untyped-call]
    units_spec: Any = {spec.name: repl for spec in static.unit_fields}
    conns_spec: Any = tuple(
        {spec.name: conn for spec in static.conn_fields}
        for _ in static.level_capacities
    )
    state_spec: Any = _spec(
        NetworkState,
        units=units_spec,
        conns=conns_spec,
        globals_=repl,
        needs_resort=repl,
        step=repl,
        grown=repl,
        overflow=repl,
        units_added=repl,
        unit_overflow=repl,
        tail_start=repl,
    )
    return repl, state_spec


def make_growth_retry[GS](
    net: type[Network[GS]], static: NetworkStatic
) -> GrowthRetryFn[GS]:
    """The jitted growth-only retry for a regrown static (see `Driver.step`).

    `phases.build_growth_retry`, jitted with the state donated and, under
    Scheme-A sharding, wrapped in the same shard_map as the step. Cached on
    `(net, static)` like `make_step`.

    Type Args:
        GS: the user's global-state pytree, opaque to the framework.

    Args:
        net: The network subclass.
        static: The regrown static configuration.

    Returns:
        `(state, remainder) -> (state, still dropped)`.
    """
    return cast(GrowthRetryFn[GS], _cached_growth_retry(net, static))  # type: ignore[arg-type]


@functools.cache
def _cached_growth_retry(
    net: type[Network[Any]], static: NetworkStatic
) -> GrowthRetryFn[Any]:
    retry: Any = build_growth_retry(net, static)
    if static.sharding is not None:
        repl, state_spec = _scheme_a_specs(static)
        retry = jax.shard_map(
            retry,
            mesh=scheme_a_mesh(static),
            in_specs=(state_spec, repl),
            out_specs=(state_spec, repl),
        )
    jitted: Any = jax.jit(retry, donate_argnums=0)
    return cast(GrowthRetryFn[Any], jitted)


def _batched_step(
    net: type[Network[Any]],
    static: NetworkStatic,
    overflow_sink: list[Bool[Array, ""]],
    remainder_sink: list[Any],
    input_ids: jax.Array,
    batch_size: int,
    engine: str | None,
    growth: str = "auto",
) -> StepFn[Any]:
    """The jitted batched step (see `make_step`'s `batch_size` and `layout`)."""
    phases = build_batched_phases(
        net,
        static,
        overflow_sink=overflow_sink,
        engine=engine,
        growth=growth,
        remainder_sink=remainder_sink,
    )
    policy = net.batch_reduction
    assert policy is not None  # make_step checked
    specs = {spec.name: spec for spec in static.unit_fields}

    def per_sample(
        phase: Phase[Any],
        state: NetworkState[Any],
        units_b: Any,
        inputs: StepInputs,
    ) -> tuple[Any, jax.Array]:
        def one(units: Any, x: jax.Array, t: jax.Array | None) -> tuple[Any, jax.Array]:
            out, contribution = phase(
                dataclasses.replace(state, units=units), StepInputs(inputs=x, targets=t)
            )
            return out.units, contribution

        if inputs.targets is None:
            return jax.vmap(lambda u, x: one(u, x, None))(units_b, inputs.inputs)
        return jax.vmap(one)(units_b, inputs.inputs, inputs.targets)

    def step(state: NetworkState[Any], inputs: StepInputs) -> StepResult[Any]:
        # Shapes are static, so a mis-shaped batch fails at trace time instead
        # of broadcasting (an unbatched input) or vmapping the feature axis.
        want = (batch_size, len(static.input_ids))
        if inputs.inputs.shape != want:
            raise ValueError(
                f"batched step: inputs must be {want}, got {inputs.inputs.shape}"
            )
        if inputs.targets is not None and inputs.targets.shape != (
            batch_size,
            len(static.output_ids),
        ):
            raise ValueError(
                f"batched step: targets must be {(batch_size, len(static.output_ids))}"
                f", got {inputs.targets.shape}"
            )
        units_b = {
            name: jnp.broadcast_to(col, (batch_size, *col.shape))
            for name, col in state.units.items()
        }
        units_b[ACTIVATION.name] = (
            units_b[ACTIVATION.name].at[:, input_ids].set(inputs.inputs)
        )
        losses = jnp.zeros((batch_size,), jnp.float32)
        if phases.csr_forward is not None:
            units_b = phases.csr_forward(state, units_b)
        else:
            units_b, c = per_sample(phases.forward, state, units_b, inputs)
            losses = losses + c
        if phases.loss is not None:
            units_b, c = per_sample(phases.loss, state, units_b, inputs)
            losses = losses + c
        if phases.csr_backward is not None:
            units_b = phases.csr_backward(state, units_b)
        elif phases.backward is not None:
            units_b, c = per_sample(phases.backward, state, units_b, inputs)
            losses = losses + c
        if phases.update_unit is not None:
            units_b, _ = per_sample(phases.update_unit, state, units_b, inputs)
        if phases.update_conn is not None:
            state = phases.update_conn(state, units_b)
        units = reduce_batch_units(policy, static.unit_fields, units_b, state.units)
        state = dataclasses.replace(state, units=units)
        for phase in phases.structural:
            state, _ = phase(state, inputs)
        # One batched step = one framework step (phases saw the old value).
        state = dataclasses.replace(state, step=state.step + 1)
        return StepResult(
            state=state,
            overflow=overflow_sink[0],
            loss=losses.mean(),
            growth_remainder=remainder_sink[0],
        )

    traced = step if static.sharding is None else _shard_map_step(step, static)
    # The write check traces the per-sample phases on the whole (unsharded)
    # state, so it runs outside any shard_map.
    unsharded = dataclasses.replace(static, sharding=None)

    def checked(state: NetworkState[Any], inputs: StepInputs) -> StepResult[Any]:
        # Once per trace: every column a per-sample phase writes must have a
        # declared reduction.
        written = per_sample_written_fields(
            net, unsharded, state, inputs.targets is not None
        )
        for name, where in written.items():
            try:
                batch_written_reduction(policy, specs[name], where=where)
            except ValueError as err:
                raise ValueError(f"{net.__name__}.{err}") from None
        return traced(state, inputs)

    jitted = cast(StepFn[Any], jax.jit(checked, donate_argnums=0))
    if engine != "csr" or not _nvidia_gpu():
        # cuSPARSE exists only on NVIDIA GPUs; elsewhere BCSR uses XLA's
        # generic lowering and the flag would do nothing.
        return jitted
    return cast(StepFn[Any], _CusparseStep(jitted))


def _nvidia_gpu() -> bool:
    """Whether the default backend is an NVIDIA (CUDA) GPU."""
    if jax.default_backend() != "gpu":
        return False
    try:
        return "cuda" in jax.devices()[0].client.platform_version.lower()
    except Exception:  # noqa: BLE001 - unknown client: not known to be CUDA
        return False


class _CusparseStep:
    """A jitted step whose calls (and AOT trace/lower) lower BCSR to cuSPARSE.

    The switch is a global jax config flag read at lowering time (off by
    default, when BCSR products fall back to generic kernels 30-80x slower);
    it is scoped to this step's calls -- and to `.trace` / `.lower`, so the
    AOT API keeps working -- so nothing else in the process changes. If the
    private config handle moves in a future jax, the step still runs, on the
    default lowering.
    """

    def __init__(self, jitted: Any) -> None:
        self._jitted = jitted
        try:
            from jax._src.config import bcoo_cusparse_lowering
        except ImportError:  # pragma: no cover - depends on the jax version
            self._flag: Any = None
        else:
            self._flag = bcoo_cusparse_lowering

    def _scoped(self) -> Any:
        return self._flag(True) if self._flag is not None else contextlib.nullcontext()

    def __call__(self, state: NetworkState[Any], inputs: StepInputs) -> StepResult[Any]:
        with self._scoped():
            result: StepResult[Any] = self._jitted(state, inputs)
            return result

    def trace(self, *args: Any, **kwargs: Any) -> Any:
        with self._scoped():
            return _ScopedStage(self._jitted.trace(*args, **kwargs), self._scoped)

    def lower(self, *args: Any, **kwargs: Any) -> Any:
        with self._scoped():
            return self._jitted.lower(*args, **kwargs)


class _ScopedStage:
    """A traced stage whose `.lower()` runs under the cuSPARSE flag."""

    def __init__(self, traced: Any, scoped: Any) -> None:
        self._traced = traced
        self._scoped = scoped

    def lower(self, *args: Any, **kwargs: Any) -> Any:
        with self._scoped():
            return self._traced.lower(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._traced, name)


# jax.util.weakref_lru_cache is not cleanly importable off the pinned jax
# floor, so functools.cache is used instead: it gives the same hash/eq-keyed
# reuse, at the cost of strong (rather than weak) references to (net,
# static) and the cached StepFn -- benign for v1's low-cardinality,
# process-lifetime pairs.
@functools.cache
def _cached_make_step(
    net: type[Network[Any]],
    static: NetworkStatic,
    batch_size: int | None = None,
    engine: str | None = None,
    fuse_prune: str = "auto",
    growth: str = "auto",
) -> StepFn[Any]:
    # overflow_sink (see build_phases): a length-1 out-parameter
    # build_add_conn_phase (when net.add_conn is set) overwrites on every
    # call; stays [False] otherwise. Created once here (mirrors `phases`
    # itself), mutated once at trace time, read below into StepResult --
    # jax.jit traces step's body exactly once, so this is an ordinary data
    # dependency in the resulting jaxpr, not a stale Python-side read.
    overflow_sink: list[Bool[Array, ""]] = [jnp.bool_(False)]
    # Likewise the growth phase's dropped candidates (a GrowthRemainder);
    # stays [None] when the net does not grow connections.
    remainder_sink: list[Any] = [None]
    input_ids = jnp.asarray(static.input_ids, dtype=jnp.int32)
    if batch_size is not None:
        return _batched_step(
            net,
            static,
            overflow_sink,
            remainder_sink,
            input_ids,
            batch_size,
            engine,
            growth,
        )
    record = PruneFusionRecord()

    def step(state: NetworkState[Any], inputs: StepInputs) -> StepResult[Any]:
        # The fusion decision needs the globals' shapes (the predicate may
        # read them), so it is made here, once per trace, in Python: the
        # phase tuple is still fixed before any equation is emitted.
        if fuse_prune == "off":
            plan = PruneFusionPlan(False, "disabled (fuse_prune='off')")
        else:
            plan = plan_prune_fusion(net, static, state.globals_, engine=fuse_prune)
        record.plan = plan
        phases = build_phases(
            net,
            static,
            overflow_sink=overflow_sink,
            prune_fusion=plan,
            growth=growth,
            remainder_sink=remainder_sink,
        )

        # Step input scatter, before any phase: StepInputs.inputs onto
        # units[ACTIVATION] at the static input_ids.
        activation = state.units[ACTIVATION.name].at[input_ids].set(inputs.inputs)
        state = dataclasses.replace(
            state, units={**state.units, ACTIVATION.name: activation}
        )

        total_loss = jnp.float32(0.0)
        for phase in phases:
            state, contribution = phase(state, inputs)
            total_loss = total_loss + contribution

        # Completed-steps counter: phases above saw the pre-increment value.
        state = dataclasses.replace(state, step=state.step + 1)
        return StepResult(
            state=state,
            overflow=overflow_sink[0],
            loss=total_loss,
            growth_remainder=remainder_sink[0],
        )

    # Under Scheme-A sharding, wrap the step in a shard_map (connections
    # sharded, rest replicated) before jitting; single-device is unchanged.
    traced = step if static.sharding is None else _shard_map_step(step, static)

    # jax.jit's return type is opaque under follow_imports="skip" (pyproject,
    # jax.* -> Any); step's own signature is the true (and already checked)
    # contract, so cast rather than let strict mypy's no-any-return fire.
    jitted: Any = jax.jit(traced, donate_argnums=0)
    jitted.prune_fusion = record
    return cast(StepFn[Any], jitted)
