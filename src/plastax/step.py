"""Step assembly: the monomorphization point.

One jit cache entry per (Network subclass, NetworkStatic); donation on the
whole state pytree (donate_argnums=0). Cached with the weakref_lru_cache
pattern.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable
from typing import Any, TypeVar, cast

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec
from jaxtyping import Array, Bool, Float

from plastax._types import ACTIVATION, Propagation
from plastax.distributed import scheme_a_mesh
from plastax.phases import (
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
    net: type[Network[GS]], static: NetworkStatic, *, batch_size: int | None = None
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

    Type Args:
        GS: the user's global-state pytree, opaque to the framework.

    Args:
        net: The network subclass to assemble phases for.
        static: The network's static configuration.
        batch_size: Samples per step, or None for the streaming step.

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
    # mypy false positive: a parameterized generic base class fails the
    # structural Hashable check, though a class is always hashable by
    # identity; hence the cast.
    return cast(
        StepFn[GS],
        _cached_make_step(net, static, batch_size),  # type: ignore[arg-type]
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
) -> StepFn[Any]:
    """The jitted batched step (see `make_step`'s `batch_size`)."""
    phases = build_batched_phases(net, static, overflow_sink=overflow_sink)

    def one_sample(
        state: NetworkState[Any], x: jax.Array, target: jax.Array | None
    ) -> tuple[Any, jax.Array]:
        activation = state.units[ACTIVATION.name].at[input_ids].set(x)
        state = dataclasses.replace(
            state, units={**state.units, ACTIVATION.name: activation}
        )
        inputs = StepInputs(inputs=x, targets=target)
        total_loss = jnp.float32(0.0)
        for phase in phases.per_sample:
            state, contribution = phase(state, inputs)
            total_loss = total_loss + contribution
        return state.units, total_loss

    def step(state: NetworkState[Any], inputs: StepInputs) -> StepResult[Any]:
        if inputs.targets is None:
            units_b, losses = jax.vmap(lambda x: one_sample(state, x, None), in_axes=0)(
                inputs.inputs
            )
        else:
            units_b, losses = jax.vmap(
                lambda x, t: one_sample(state, x, t), in_axes=(0, 0)
            )(inputs.inputs, inputs.targets)
        if phases.update_conn is not None:
            state = phases.update_conn(state, units_b)
        state = dataclasses.replace(state, units=batch_mean_units(units_b))
        for phase in phases.structural:
            state, _ = phase(state, inputs)
        return StepResult(state=state, overflow=overflow_sink[0], loss=losses.mean())

    traced = step if static.sharding is None else _shard_map_step(step, static)
    return cast(StepFn[Any], jax.jit(traced, donate_argnums=0))


# jax.util.weakref_lru_cache is not cleanly importable off the pinned jax
# floor, so functools.cache is used instead: it gives the same hash/eq-keyed
# reuse, at the cost of strong (rather than weak) references to (net,
# static) and the cached StepFn -- benign for v1's low-cardinality,
# process-lifetime pairs.
@functools.cache
def _cached_make_step(
    net: type[Network[Any]], static: NetworkStatic, batch_size: int | None = None
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
        return _batched_step(net, static, overflow_sink, input_ids)
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
