"""Host-side network construction (pre-jit, eager, plain numpy).

Mirrors the manual construction path of the C++ examples: add units and
connections imperatively, then finalize into (NetworkStatic, NetworkState).
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
from jaxtyping import PRNGKeyArray

from plastax import topo
from plastax._types import (
    ACTIVATION,
    DEAD,
    FROM_ID,
    LEVEL,
    TO_ID,
    WEIGHT,
    FieldSpec,
    Propagation,
)
from plastax.state import Columns, NetworkState, NetworkStatic
from plastax.topology import Topology
from plastax.traits import Network

# Values a caller may pass into add_unit/add_conn **kwargs, widened with the
# numpy-scalar FieldSpec.default that fills in unset fields (kept private to
# the module).
FieldValue = float | int | bool | np.generic


class NetworkBuilder[GS]:
    """Construct a network host-side, unit by unit and connection by connection.

    Callers add units and connections imperatively, then call finalize() to
    freeze the result into a (NetworkStatic, NetworkState) pair.

    Type Args:
        GS: The network's globals type.
    """

    def __init__(self, net: type[Network[GS]], globals_: GS) -> None:
        self.net = net
        self.globals_ = globals_
        self._unit_fields: tuple[FieldSpec[np.generic], ...] = (
            ACTIVATION,
            LEVEL,
            *net.extra_unit_fields,
        )
        self._conn_fields: tuple[FieldSpec[np.generic], ...] = (
            FROM_ID,
            TO_ID,
            DEAD,
            WEIGHT,
            *net.extra_conn_fields,
        )
        # level is derived at finalize() time (topo.initial_levels), never
        # user-settable; from_id/to_id/dead are add_conn's positional args
        # and derived tombstone state, not settable via **field_values.
        self._settable_unit_fields: dict[str, FieldSpec[np.generic]] = {
            spec.name: spec for spec in self._unit_fields if spec.name != LEVEL.name
        }
        self._settable_conn_fields: dict[str, FieldSpec[np.generic]] = {
            spec.name: spec
            for spec in self._conn_fields
            if spec.name not in (FROM_ID.name, TO_ID.name, DEAD.name)
        }
        self._units: list[dict[str, FieldValue]] = []
        self._conn_src: list[int] = []
        self._conn_dst: list[int] = []
        self._conn_values: list[dict[str, FieldValue]] = []
        self._input_ids: list[int] = []
        self._output_ids: list[int] = []

    @classmethod
    def from_topology(
        cls,
        net: type[Network[GS]],
        topology_fn: Callable[[PRNGKeyArray], Topology],
        key: PRNGKeyArray,
        *,
        globals_: GS,
    ) -> tuple[NetworkStatic, NetworkState[GS]]:
        """Expand a plastax.topology spec into arenas.

        Adds units, marks the first/last blocks as inputs/outputs, bulk
        add_conn's the edge set, then finalizes.

        Args:
            net: The network type to build.
            topology_fn: Draws a topology spec from a PRNG key.
            key: PRNG key passed to topology_fn.
            globals_: The network's globals value.

        Returns:
            The finalized (NetworkStatic, NetworkState) pair.
        """
        spec = topology_fn(key)
        builder = cls(net, globals_)
        for _ in range(spec.num_units):
            builder.add_unit()
        for unit_id in spec.input_ids:
            builder.mark_input(unit_id)
        for unit_id in spec.output_ids:
            builder.mark_output(unit_id)
        edges = spec.edges
        for from_id, to_id, weight in zip(
            edges.from_ids.tolist(),
            edges.to_ids.tolist(),
            edges.weights.tolist(),
            strict=True,
        ):
            builder.add_conn(from_id, to_id, weight=weight)
        return builder.finalize()

    def add_unit(self, **field_values: float | int | bool) -> int:
        """Add a unit, filling unset fields with their defaults.

        Args:
            **field_values: Settable unit field values, keyed by field name.

        Returns:
            The new unit's global id (dense, 0-based).

        Raises:
            ValueError: A field name is unknown or not settable.
        """
        for name in field_values:
            if name not in self._settable_unit_fields:
                raise ValueError(
                    f"add_unit: unknown or non-settable unit field {name!r}"
                )
        row: dict[str, FieldValue] = {
            name: field_values.get(name, spec.default)
            for name, spec in self._settable_unit_fields.items()
        }
        self._units.append(row)
        return len(self._units) - 1

    def add_conn(self, src: int, dst: int, **field_values: float | int | bool) -> None:
        """Add a connection, filling unset fields with their defaults.

        Args:
            src: Source unit id.
            dst: Destination unit id.
            **field_values: Settable conn field values, keyed by field name.

        Raises:
            ValueError: A field name is unknown or not settable.
        """
        for name in field_values:
            if name not in self._settable_conn_fields:
                raise ValueError(
                    f"add_conn: unknown or non-settable conn field {name!r}"
                )
        row: dict[str, FieldValue] = {
            name: field_values.get(name, spec.default)
            for name, spec in self._settable_conn_fields.items()
        }
        self._conn_src.append(src)
        self._conn_dst.append(dst)
        self._conn_values.append(row)

    def mark_input(self, unit_id: int) -> None:
        """Mark a unit id as a network input."""
        self._input_ids.append(unit_id)

    def mark_output(self, unit_id: int) -> None:
        """Mark a unit id as a network output."""
        self._output_ids.append(unit_id)

    def finalize(self) -> tuple[NetworkStatic, NetworkState[GS]]:
        """Freeze the accumulated units and connections into arenas.

        Computes initial levels (topo.initial_levels), buckets conns by
        source level with capacity_policy headroom, allocates arenas, and
        freezes the static config.

        Returns:
            The (NetworkStatic, NetworkState) pair for the built network.

        Raises:
            ValueError: A referenced unit id is out of range.
        """
        num_units = len(self._units)
        for unit_id in (
            *self._input_ids,
            *self._output_ids,
            *self._conn_src,
            *self._conn_dst,
        ):
            if not 0 <= unit_id < num_units:
                raise ValueError(
                    f"finalize: unit id {unit_id} out of range [0, {num_units})"
                )

        if self._conn_src:
            src_arr = np.asarray(self._conn_src, dtype=np.int32)
            dst_arr = np.asarray(self._conn_dst, dtype=np.int32)
        else:
            src_arr = np.zeros((0,), dtype=np.int32)
            dst_arr = np.zeros((0,), dtype=np.int32)
        edges = (
            np.stack([src_arr, dst_arr], axis=1)
            if src_arr.size
            else np.zeros((0, 2), np.int32)
        )
        levels = topo.initial_levels(num_units, edges)

        unit_cols: Columns = {}
        for spec in self._unit_fields:
            if spec.name == LEVEL.name:
                unit_cols[spec.name] = jnp.asarray(levels, dtype=spec.dtype)
            else:
                values = [row[spec.name] for row in self._units]
                unit_cols[spec.name] = jnp.asarray(values, dtype=spec.dtype)

        if self.net.propagation is Propagation.PIPELINE:
            # Single flat arena: every conn lands in bucket 0 regardless of
            # source level.
            bucket_of_conn = np.zeros_like(src_arr)
            num_buckets = 1
        else:
            # Every edge goes strictly from a lower to a higher level
            # (level(v) = max over incoming edges of level(u) + 1), so the
            # highest level ever used as a SOURCE is max(levels) - 1, i.e.
            # exactly max(levels) buckets (0-indexed); max(., 1) covers the
            # no-edges degenerate case.
            bucket_of_conn = levels[src_arr]
            num_buckets = max(int(levels.max()) if levels.size else 0, 1)

        conns: list[Columns] = []
        level_capacities: list[int] = []
        for level_idx in range(num_buckets):
            idx = np.flatnonzero(bucket_of_conn == level_idx)
            order = idx[np.argsort(dst_arr[idx], kind="stable")]
            live = int(order.size)
            capacity = topo.capacity_policy(live)
            pad = capacity - live

            cols: Columns = {}
            for spec in self._conn_fields:
                live_vals: np.ndarray
                if spec.name == FROM_ID.name:
                    live_vals = src_arr[order]
                elif spec.name == TO_ID.name:
                    live_vals = dst_arr[order]
                elif spec.name == DEAD.name:
                    live_vals = np.zeros((live,), dtype=np.bool_)
                else:
                    live_vals = np.asarray(
                        [self._conn_values[j][spec.name] for j in order.tolist()]
                    )
                pad_vals = np.full((pad,), spec.default, dtype=spec.dtype)
                full = np.concatenate([live_vals.astype(spec.dtype), pad_vals])
                cols[spec.name] = jnp.asarray(full)
            conns.append(cols)
            level_capacities.append(capacity)

        static = NetworkStatic(
            num_units=num_units,
            propagation=self.net.propagation,
            unit_fields=self._unit_fields,
            conn_fields=self._conn_fields,
            level_capacities=tuple(level_capacities),
            kahn_max_depth=self.net.kahn_max_depth,
            input_ids=tuple(self._input_ids),
            output_ids=tuple(self._output_ids),
        )
        state = NetworkState(
            units=unit_cols,
            conns=tuple(conns),
            globals_=self.globals_,
            needs_resort=jnp.bool_(False),
        )
        return static, state
