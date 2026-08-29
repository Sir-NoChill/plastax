"""Host-side network construction (pre-jit, eager, plain numpy).

Mirrors the manual construction path of the C++ examples: add units and
connections imperatively, then finalize into (NetworkStatic, NetworkState).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

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

        Draws the topology, then hands its EdgeSet straight to `from_edges`
        for vectorized column assembly -- no per-edge Python pass.

        Args:
            net: The network type to build.
            topology_fn: Draws a topology spec from a PRNG key.
            key: PRNG key passed to topology_fn.
            globals_: The network's globals value.

        Returns:
            The finalized (NetworkStatic, NetworkState) pair.
        """
        spec = topology_fn(key)
        return cls.from_edges(
            net,
            spec.num_units,
            spec.edges.from_ids,
            spec.edges.to_ids,
            weights=spec.edges.weights,
            input_ids=spec.input_ids,
            output_ids=spec.output_ids,
            globals_=globals_,
        )

    @classmethod
    def from_edges(
        cls,
        net: type[Network[GS]],
        num_units: int,
        from_ids: np.ndarray,
        to_ids: np.ndarray,
        *,
        weights: np.ndarray | None = None,
        input_ids: Sequence[int],
        output_ids: Sequence[int],
        globals_: GS,
        extra_conn_columns: Mapping[str, np.ndarray] | None = None,
    ) -> tuple[NetworkStatic, NetworkState[GS]]:
        """Build arenas from whole edge-column arrays, no per-edge Python.

        The vectorized construction path: the caller supplies parallel
        ``(E,)`` source/destination arrays (plus optional per-edge weight and
        extra-field columns), and every column is bucketed, sorted, and padded
        with numpy fancy-indexing instead of the ``add_unit``/``add_conn`` loop
        `finalize` walks. Output is byte-identical to the imperative path (both
        route through the same assembler), so a large sparse or dense net
        builds in one vectorized pass rather than E Python calls.

        Units carry field defaults (topology construction sets no per-unit
        values); a conn field is filled from ``weights``/``extra_conn_columns``
        if given, else its spec default. All edges live (``DEAD`` false); the
        assembler tombstones only the per-bucket capacity padding.

        Args:
            net: The network type to build.
            num_units: Total unit count; ids must fall in ``[0, num_units)``.
            from_ids: ``(E,)`` source unit ids.
            to_ids: ``(E,)`` destination unit ids (parallel to ``from_ids``).
            weights: ``(E,)`` initial edge weights, or None for the ``WEIGHT``
                default.
            input_ids: Unit ids StepInputs scatters onto.
            output_ids: Unit ids the loss clamps targets to.
            globals_: The network's globals value.
            extra_conn_columns: Optional per-edge values for settable conn
                fields (e.g. an optimizer's ``opt/…`` columns), each ``(E,)``;
                unset fields take their spec default.

        Returns:
            The finalized (NetworkStatic, NetworkState) pair.

        Raises:
            ValueError: On mismatched array lengths, an out-of-range unit id,
                or an unknown/non-settable ``extra_conn_columns`` key.
        """
        builder = cls(net, globals_)
        src_arr = np.asarray(from_ids, dtype=np.int32)
        dst_arr = np.asarray(to_ids, dtype=np.int32)
        if src_arr.ndim != 1 or dst_arr.shape != src_arr.shape:
            raise ValueError(
                "from_edges: from_ids and to_ids must be 1-D arrays of equal "
                f"length, got {src_arr.shape} and {dst_arr.shape}"
            )
        num_edges = int(src_arr.shape[0])

        extra = dict(extra_conn_columns or {})
        for name in extra:
            if name not in builder._settable_conn_fields or name == WEIGHT.name:
                raise ValueError(
                    f"from_edges: unknown or non-settable conn field {name!r} "
                    "in extra_conn_columns"
                )

        conn_columns: dict[str, np.ndarray] = {}
        for name, spec in builder._settable_conn_fields.items():
            if name == WEIGHT.name and weights is not None:
                supplied: np.ndarray | None = np.asarray(weights)
            else:
                supplied = extra.get(name)
            if supplied is None:
                conn_columns[name] = np.full(
                    (num_edges,), spec.default, dtype=spec.dtype
                )
            elif supplied.shape != (num_edges,):
                raise ValueError(
                    f"from_edges: conn column {name!r} has shape {supplied.shape}, "
                    f"expected ({num_edges},)"
                )
            else:
                conn_columns[name] = supplied
        unit_columns: dict[str, np.ndarray] = {
            name: np.full((num_units,), spec.default, dtype=spec.dtype)
            for name, spec in builder._settable_unit_fields.items()
        }
        return builder._assemble(
            num_units,
            src_arr,
            dst_arr,
            conn_columns,
            unit_columns,
            tuple(input_ids),
            tuple(output_ids),
        )

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

        Re-views the imperative per-edge / per-unit rows as one array per
        settable field (in insertion order) and hands them to the shared
        `_assemble` core, which validates ids, buckets conns by source level
        with capacity_policy headroom, allocates arenas, and freezes the static
        config (raising ValueError via `_assemble` on an out-of-range id).

        Returns:
            The (NetworkStatic, NetworkState) pair for the built network.
        """
        num_units = len(self._units)
        if self._conn_src:
            src_arr = np.asarray(self._conn_src, dtype=np.int32)
            dst_arr = np.asarray(self._conn_dst, dtype=np.int32)
        else:
            src_arr = np.zeros((0,), dtype=np.int32)
            dst_arr = np.zeros((0,), dtype=np.int32)

        # Column-major re-view of the row dicts: one (E,) / (num_units,) array
        # per settable field, in insertion order -- the assembler's contract.
        conn_columns = {
            name: np.asarray([row[name] for row in self._conn_values])
            for name in self._settable_conn_fields
        }
        unit_columns = {
            name: np.asarray([row[name] for row in self._units])
            for name in self._settable_unit_fields
        }
        return self._assemble(
            num_units,
            src_arr,
            dst_arr,
            conn_columns,
            unit_columns,
            tuple(self._input_ids),
            tuple(self._output_ids),
        )

    def _assemble(
        self,
        num_units: int,
        src_arr: np.ndarray,
        dst_arr: np.ndarray,
        conn_columns: dict[str, np.ndarray],
        unit_columns: dict[str, np.ndarray],
        input_ids: tuple[int, ...],
        output_ids: tuple[int, ...],
    ) -> tuple[NetworkStatic, NetworkState[GS]]:
        """Bucket, sort, and pad whole edge columns into frozen arenas.

        The single construction core both `finalize` (imperative rows) and
        `from_edges` (vectorized arrays) feed: computes initial levels, buckets
        edges by source level, stable-sorts each bucket by destination id, pads
        to a capacity_policy capacity with tombstoned slots, and freezes the
        static config. The built-in `from_id`/`to_id`/`dead` columns and the
        derived `level` are filled here; callers supply only settable fields.

        Args:
            num_units: Total unit count.
            src_arr: (E,) int32 source ids in insertion order.
            dst_arr: (E,) int32 destination ids (parallel to src_arr).
            conn_columns: Settable conn field name -> (E,) values.
            unit_columns: Settable unit field name -> (num_units,) values.
            input_ids: Input unit ids.
            output_ids: Output unit ids.

        Returns:
            The (NetworkStatic, NetworkState) pair for the built network.

        Raises:
            ValueError: A referenced unit id or edge endpoint is out of range.
        """
        for unit_id in (*input_ids, *output_ids):
            if not 0 <= unit_id < num_units:
                raise ValueError(
                    f"finalize: unit id {unit_id} out of range [0, {num_units})"
                )
        if src_arr.size:
            lo = min(int(src_arr.min()), int(dst_arr.min()))
            hi = max(int(src_arr.max()), int(dst_arr.max()))
            if lo < 0 or hi >= num_units:
                raise ValueError(
                    f"finalize: an edge endpoint is out of range [0, {num_units})"
                )

        edges = (
            np.stack([src_arr, dst_arr], axis=1)
            if src_arr.size
            else np.zeros((0, 2), np.int32)
        )
        # Pipeline propagation admits recurrent (cyclic) structure -- an
        # echo state network's reservoir is the canonical case -- since
        # every conn lands in the single flat bucket and each step is one
        # synchronous sweep reading the previous step's activations.
        # Topological propagation still requires a DAG.
        levels = topo.initial_levels(
            num_units,
            edges,
            allow_cycles=self.net.propagation is Propagation.PIPELINE,
        )

        unit_cols: Columns = {}
        for spec in self._unit_fields:
            if spec.name == LEVEL.name:
                unit_cols[spec.name] = jnp.asarray(levels, dtype=spec.dtype)
            else:
                unit_cols[spec.name] = jnp.asarray(
                    unit_columns[spec.name], dtype=spec.dtype
                )

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
                    live_vals = conn_columns[spec.name][order]
                pad_vals = np.full((pad,), spec.default, dtype=spec.dtype)
                full = np.concatenate(
                    [np.asarray(live_vals).astype(spec.dtype), pad_vals]
                )
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
            input_ids=input_ids,
            output_ids=output_ids,
            sharding=self.net.sharding,
        )
        state = NetworkState(
            units=unit_cols,
            conns=tuple(conns),
            globals_=self.globals_,
            needs_resort=jnp.bool_(False),
        )
        return static, state
