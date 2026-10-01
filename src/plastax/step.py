"""Step assembly: the monomorphization point.

One jit cache entry per (Network subclass, NetworkStatic); donation on the
whole state pytree (donate_argnums=0). Cached with the weakref_lru_cache
pattern.
"""

from __future__ import annotations

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
    Phase,
    StepInputs,
    batch_mean_units,
    build_batched_phases,
    build_phases,
)
from plastax.state import NetworkState, NetworkStatic
from plastax.traits import Network

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
        overflow: AddConn overflow flag for the step.
        loss: Reduced per-output loss for the step; 0.0 when the net has no
            loss phase.
    """

    state: NetworkState[GS]
    overflow: Bool[Array, ""]
    loss: Float[Array, ""]


StepFn = Callable[[NetworkState[GS], StepInputs], StepResult[GS]]


def make_step[GS](
    net: type[Network[GS]],
    static: NetworkStatic,
    *,
    batch_size: int | None = None,
    layout: Literal["auto", "edge_list", "csr", "pallas"] = "auto",
) -> StepFn[GS]:
    """Assemble the present phases and jit them with donate_argnums=0.

    The returned callable must be shape-preserving on the state pytree so
    every leaf donates (CI promotes the donation warning to an error).

    plastax is built for streaming, one sample per step. `batch_size=B` is a
    convenience for mini-batch training and evaluation of feed-forward
    (TOPOLOGICAL) nets: `StepInputs` then carries `(B, num_inputs)` inputs and
    `(B, num_outputs)` targets; forward, loss, and backward run per sample
    against the shared connections; the connection update is reduced over the
    batch (see `phases.build_batched_update_conn`: exact for the `optim/`
    bundles, mean-of-writes otherwise); prune, add, and reset run once on the
    batch-mean unit state, which is also what the returned state holds; and
    `StepResult.loss` is the batch mean.

    `layout` picks how a batched step runs a *linear* forward or backward pass
    (one declaring `linear_input`, see `phases.linear_input_field`):
    "edge_list" runs it per sample over the edge arena; "csr" builds each
    bucket's CSR view on device every step (one radix sort) and runs the
    whole batch as one cuSPARSE sparse-dense product; "pallas" runs each
    bucket as one edge-once Pallas kernel (every edge read once for the whole
    batch, atomics into the targets; GPU, or interpret mode elsewhere);
    "auto" picks, on a GPU backend, "pallas" for `2 <= batch_size <= 32` and
    "csr" above (measured on an RTX 5000 Ada, batched SGD training: at 5.4M
    edges Pallas is 2.4x the edge list at B = 8 and ties CSR at 32, while CSR
    is 6x Pallas at 128; at 50M edges Pallas still beats CSR at 32, whose
    per-step sort then dominates), else the edge list. Non-linear passes,
    and every streaming step, use the edge list. Under Scheme-A sharding,
    "pallas" falls back to "csr".

    In a batched step a non-floating unit column (a flag, a count) is stored
    from sample 0 rather than averaged, and so is a non-floating connection
    column written by an UpdateConn without the exact pair: such columns
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
            or "pallas".

    Returns:
        A jitted step function for the given network and static config.

    Raises:
        ValueError: If `batch_size` is below 1, or set for a PIPELINE net
            (whose carried unit state is per-sample recurrent state).
    """
    if batch_size is not None:
        if batch_size < 1:
            raise ValueError(f"make_step: batch_size must be >= 1, got {batch_size}")
        if net.propagation is Propagation.PIPELINE:
            raise ValueError(
                "make_step: batch_size is for TOPOLOGICAL (feed-forward) nets; a "
                "PIPELINE net carries per-sample recurrent state between steps"
            )
    if layout not in ("auto", "edge_list", "csr", "pallas"):
        raise ValueError(f"make_step: unknown layout {layout!r}")
    gpu = jax.default_backend() == "gpu"
    engine: str | None = None
    if batch_size is not None:
        if layout == "auto" and gpu:
            engine = "csr" if batch_size > 32 else "pallas" if batch_size >= 2 else None
        elif layout == "csr":
            engine = "csr"
        elif layout == "pallas":
            engine = "pallas" if gpu else "pallas_interpret"
        # The Pallas kernel is not yet validated inside shard_map (interpret
        # mode trips jax's varying-axes check there); sharded steps use CSR.
        if engine in ("pallas", "pallas_interpret") and static.sharding is not None:
            engine = "csr"
    # mypy false positive: a parameterized generic base class fails the
    # structural Hashable check, though a class is always hashable by
    # identity; hence the cast.
    return cast(
        StepFn[GS],
        _cached_make_step(net, static, batch_size, engine),  # type: ignore[arg-type]
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
    sharding = static.sharding
    assert sharding is not None  # only called on the sharded branch
    mesh = scheme_a_mesh(static)
    # PartitionSpec is untyped in jax's stubs; the spec pytrees deliberately
    # hold PartitionSpec leaves in the array-typed state/input/result shapes,
    # so they are built and threaded as Any.
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
    )
    in_specs: Any = (state_spec, _spec(StepInputs, inputs=repl, targets=repl))
    out_specs: Any = _spec(StepResult, state=state_spec, overflow=repl, loss=repl)
    sharded: Any = jax.shard_map(
        step, mesh=mesh, in_specs=in_specs, out_specs=out_specs
    )
    return cast(Callable[[NetworkState[Any], StepInputs], StepResult[Any]], sharded)


def _batched_step(
    net: type[Network[Any]],
    static: NetworkStatic,
    overflow_sink: list[Bool[Array, ""]],
    input_ids: jax.Array,
    batch_size: int,
    engine: str | None,
) -> StepFn[Any]:
    """The jitted batched step (see `make_step`'s `batch_size` and `layout`)."""
    phases = build_batched_phases(
        net, static, overflow_sink=overflow_sink, engine=engine
    )

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
        if phases.update_conn is not None:
            state = phases.update_conn(state, units_b)
        state = dataclasses.replace(state, units=batch_mean_units(units_b))
        for phase in phases.structural:
            state, _ = phase(state, inputs)
        return StepResult(state=state, overflow=overflow_sink[0], loss=losses.mean())

    traced = step if static.sharding is None else _shard_map_step(step, static)
    jitted = cast(StepFn[Any], jax.jit(traced, donate_argnums=0))
    if engine != "csr":
        return jitted
    return _with_cusparse(jitted)


def _with_cusparse(fn: StepFn[Any]) -> StepFn[Any]:
    """Run `fn` with jax.experimental.sparse lowering to cuSPARSE.

    The switch is a global jax config flag read at lowering time (off by
    default, when BCSR products fall back to generic kernels 30-80x slower);
    it is scoped to this step's calls so nothing else in the process changes.
    If the private config handle moves in a future jax, the step still runs,
    on the default lowering.
    """
    try:
        from jax._src.config import bcoo_cusparse_lowering
    except ImportError:  # pragma: no cover - depends on the jax version
        return fn

    @functools.wraps(fn)
    def call(state: NetworkState[Any], inputs: StepInputs) -> StepResult[Any]:
        with bcoo_cusparse_lowering(True):
            return fn(state, inputs)

    return call


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
) -> StepFn[Any]:
    # overflow_sink (see build_phases): a length-1 out-parameter
    # build_add_conn_phase (when net.add_conn is set) overwrites on every
    # call; stays [False] otherwise. Created once here (mirrors `phases`
    # itself), mutated once at trace time, read below into StepResult --
    # jax.jit traces step's body exactly once, so this is an ordinary data
    # dependency in the resulting jaxpr, not a stale Python-side read.
    overflow_sink: list[Bool[Array, ""]] = [jnp.bool_(False)]
    input_ids = jnp.asarray(static.input_ids, dtype=jnp.int32)
    if batch_size is not None:
        return _batched_step(net, static, overflow_sink, input_ids, batch_size, engine)
    phases = build_phases(net, static, overflow_sink=overflow_sink)

    def step(state: NetworkState[Any], inputs: StepInputs) -> StepResult[Any]:
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

        return StepResult(state=state, overflow=overflow_sink[0], loss=total_loss)

    # Under Scheme-A sharding, wrap the step in a shard_map (connections
    # sharded, rest replicated) before jitting; single-device is unchanged.
    traced = step if static.sharding is None else _shard_map_step(step, static)

    # jax.jit's return type is opaque under follow_imports="skip" (pyproject,
    # jax.* -> Any); step's own signature is the true (and already checked)
    # contract, so cast rather than let strict mypy's no-any-return fire.
    return cast(StepFn[Any], jax.jit(traced, donate_argnums=0))
