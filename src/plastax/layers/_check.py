"""The checks that run when a network is built, before anything is materialized.

Every policy in a network's resolved table is traced once with abstract values:
nothing is computed and no device is used. What each returns is compared with
what will receive it.

- **Accumulators.** What a connection's `map` returns against the `combine` of
  the units that reduce it: the units it ends in going forward, the units it
  starts from going backward.
- **Writes.** Every field a policy writes: that the element has the field, that
  the value has the field's dtype, and that the value is a scalar.
- **Field names.** A field written under a name that belongs to a different
  field.

A fault does not stop the check. Every fault is found and returned, so that one
run shows everything there is to fix.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import jax
import numpy as np

from plastax._types import (
    ACTIVATION,
    DEAD,
    FROM_ID,
    LEVEL,
    PRUNED,
    TO_ID,
    WEIGHT,
    ConnIdx,
    FieldSpec,
    UnitIdx,
)
from plastax.layers._module import _show
from plastax.layers._policies import ConnPolicies, UnitPolicies
from plastax.layers._record import Record
from plastax.layers._resolve import PolicyTable
from plastax.monoid import Monoid
from plastax.views import ConnView, ConnWrite, UnitView, UnitWrite

if TYPE_CHECKING:
    from plastax.network import Network


@dataclass(frozen=True)
class Fault:
    """One thing wrong with a network's policies, found before it is built.

    Attributes:
        kind: What is wrong. `accumulator`: a `map` and a `combine` disagree.
            `write`: a policy writes a value its field cannot hold, or a field
            the element does not have. `field`: a field is written under
            another field's name. `trace`: a policy could not be traced at all.
        where: The layer and the phase, as `middle.weight, backward`; `the
            network` for a policy no layer overrides.
        policy: The class and method at fault, as `DeepRUpdate.incoming`.
        message: What was given against what was expected, and what to do.
    """

    kind: Literal["accumulator", "write", "field", "trace"]
    where: str
    policy: str
    message: str

    def __str__(self) -> str:
        """The fault as it is printed in a build error."""
        body = "\n".join(f"      {line}" for line in self.message.splitlines())
        return f"  {self.where}: {self.policy}\n{body}"


@dataclass(frozen=True)
class _Leaf:
    """One value a policy returned, as the trace saw it.

    Attributes:
        dtype: The value's dtype.
        shape: The value's shape; `()` for a scalar.
        weak: Whether the value is a Python number, or was made from one alone.
            Such a value takes the dtype of what it is written to.
    """

    dtype: np.dtype[Any]
    shape: tuple[int, ...]
    weak: bool


def _leaf(value: Any) -> _Leaf:
    """Describe one returned value. Called inside the trace.

    Args:
        value: A Python number, a numpy value, or a JAX value or tracer.

    Returns:
        Its dtype and shape, and whether it is weakly typed.
    """
    if isinstance(value, bool | int | float):
        kind = bool if isinstance(value, bool) else type(value)
        return _Leaf(np.dtype(kind), (), True)
    weak = bool(getattr(getattr(value, "aval", value), "weak_type", False))
    return _Leaf(np.dtype(value.dtype), tuple(value.shape), weak)


def _tree(value: object) -> Any:
    """Describe what a `map` returned: tuples and dicts of leaves.

    Args:
        value: The returned value. Called inside the trace.

    Returns:
        A tuple, a dict, or a `_Leaf`, nested as the value is.
    """
    if isinstance(value, tuple | list):
        return tuple(_tree(v) for v in value)
    if isinstance(value, dict):
        return {k: _tree(v) for k, v in value.items()}
    return _leaf(value)


def _matches(returned: Any, combine: Any) -> bool:
    """Whether what a `map` returned has the structure a `combine` reduces.

    Args:
        returned: The description `_tree` gave of the returned value.
        combine: The unit half's tree of monoids.

    Returns:
        True when the two have the same tuples and dicts, with a scalar at
        every monoid.
    """
    if isinstance(combine, Monoid):
        return isinstance(returned, _Leaf) and returned.shape == ()
    if isinstance(combine, tuple):
        return (
            isinstance(returned, tuple)
            and len(returned) == len(combine)
            and all(_matches(r, c) for r, c in zip(returned, combine, strict=True))
        )
    if isinstance(combine, dict):
        return (
            isinstance(returned, dict)
            and returned.keys() == combine.keys()
            and all(_matches(returned[k], combine[k]) for k in combine)
        )
    return False


def _say(tree: Any, leaf: str) -> str:
    """Put a tree of returned values, or of monoids, into words.

    Args:
        tree: A description from `_tree`, or a tree of monoids.
        leaf: What one leaf is called: "scalar" or "monoid".

    Returns:
        For example "one scalar", "a tuple of 6 scalars", "a dict with keys a, b".
    """
    if isinstance(tree, tuple):
        if all(not isinstance(t, tuple | dict) for t in tree):
            shapes = {t.shape for t in tree if isinstance(t, _Leaf)}
            odd = "" if shapes <= {()} else f", of shapes {sorted(shapes)}"
            return f"a tuple of {len(tree)} {leaf}s{odd}"
        return "a tuple of (" + ", ".join(_say(t, leaf) for t in tree) + ")"
    if isinstance(tree, dict):
        return "a dict with keys " + ", ".join(sorted(map(str, tree)))
    if isinstance(tree, _Leaf) and tree.shape != ():
        return f"an array of shape {tree.shape}"
    return f"one {leaf}"


def _fits(leaf: _Leaf, field: np.dtype[Any]) -> bool:
    """Whether a written value may go into a field of this dtype.

    Args:
        leaf: The written value.
        field: The field's dtype.

    Returns:
        True when the value has the field's dtype exactly; or when it is weakly
        typed and converts without changing kind: a bool into a bool field, an
        int into an int or a float field, a float into a float field.
    """
    if not leaf.weak:
        return leaf.dtype == field
    into = {"b": "b", "i": "iuf", "u": "iuf", "f": "f"}.get(leaf.dtype.kind, "")
    return field.kind in into


class _Tracer:
    """Calls policy methods on abstract values, once each, and keeps the results.

    `unit_specs` and `conn_specs` hold the network's unit and connection columns,
    by name.

    Args:
        net: The network class whose policies are traced.
        record: Its traced topology, for the number of units and connections.
        g: The network's globals, as arrays.
    """

    def __init__(self, net: type[Network[Any]], record: Record, g: Any) -> None:
        resolved = net.resolved()
        units = (
            ACTIVATION,
            LEVEL,
            *((PRUNED,) if net.unit_capacity is not None else ()),
            *resolved.unit_fields,
        )
        conns = (FROM_ID, TO_ID, DEAD, WEIGHT, *resolved.conn_fields)
        self.unit_specs: dict[str, FieldSpec[Any]] = {s.name: s for s in units}
        self.conn_specs: dict[str, FieldSpec[Any]] = {s.name: s for s in conns}
        n = max(record.num_units, 1)
        e = max(sum(len(b.src) for b in record.blocks), 1)
        self._units = {s.name: np.zeros(n, s.dtype) for s in units}
        self._conns = {s.name: np.zeros(e, s.dtype) for s in conns}
        self._g = g
        self._done: dict[tuple[int, str], tuple[Any, str | None]] = {}

    def run(
        self,
        policy: object,
        method: str,
        call: Callable[[UnitView, ConnView, Any, Any, Any], object],
        describe: Callable[[Any], Any],
        extra: Any = None,
    ) -> tuple[Any, str | None, bool]:
        """Trace one method of one policy, unless it has been traced already.

        Args:
            policy: The policy object.
            method: The method's name, which with the policy keys the result.
            call: Calls the method, given the unit view, the connection view,
                the globals, an index to use for every element argument, and
                `extra`.
            describe: Turns what the method returned into a description, while
                the trace is still open.
            extra: A tree of arrays the method is given as traced values: the
                accumulator of an `apply`. None for a method that takes none.

        Returns:
            The description, or None if the method raised; the error's message,
            or None; and whether this call is the first for this method.
        """
        key = (id(policy), method)
        if key in self._done:
            return (*self._done[key], False)
        seen: list[Any] = []

        def probe(u: Any, c: Any, g: Any, i: Any, x: Any) -> int:
            seen.append(describe(call(UnitView(u), ConnView(c), g, i, x)))
            return 0

        try:
            index = np.zeros((), np.int32)
            jax.eval_shape(probe, self._units, self._conns, self._g, index, extra)
            self._done[key] = (seen[0], None)
        except Exception as error:  # noqa: BLE001 - whatever the policy raised
            self._done[key] = (None, f"{type(error).__name__}: {error}")
        return (*self._done[key], True)


def _cell(row: object, phase: str) -> Any:
    """One cell of a resolved row: a policy, or None. Never `INHERIT`."""
    return getattr(row, phase)


def _named(policy: object, method: str) -> str:
    """How a policy's method is named in a fault."""
    return f"{type(policy).__name__}.{method}"


def _untraced(where: str, policy: object, method: str, error: str) -> Fault:
    """The fault for a method that raised while it was traced.

    Args:
        where: The layer and phase.
        policy: The policy object.
        method: The method that raised.
        error: The error's type and message.

    Returns:
        A fault of kind `trace`.
    """
    first = error.strip().splitlines()[0] if error.strip() else error
    return Fault(
        "trace",
        where,
        _named(policy, method),
        f"could not be traced: {first}\n"
        f"The method is called for one element with the network's fields; this "
        f"is the error it would raise inside the first step.",
    )


def _unit_rows(
    record: Record, table: PolicyTable
) -> list[tuple[str, UnitPolicies, np.ndarray | None]]:
    """The network's unit row, then each block of units with its ids.

    Args:
        record: The traced topology.
        table: Its resolved policies.

    Returns:
        A name, the policies and the unit ids of each row. The network's row has
        no ids: it stands for the units in no block, the inputs.
    """
    rows: list[tuple[str, UnitPolicies, np.ndarray | None]] = [
        ("the network", table.network_units, None)
    ]
    for block, policies in zip(record.unit_blocks, table.units, strict=True):
        size = int(np.prod(block.shape, dtype=np.int64))
        ids = np.arange(block.start, block.start + size)
        rows.append((_show(block.path), policies, ids))
    return rows


def _reduced_at(
    ends: np.ndarray, rows: list[tuple[str, UnitPolicies, np.ndarray | None]]
) -> list[tuple[str, UnitPolicies]]:
    """The rows of units that a set of connection ends falls in.

    Args:
        ends: The unit ids at one end of a block of connections.
        rows: The unit rows, the network's first.

    Returns:
        Each row that holds at least one of the ids. An id in no block belongs
        to the network's row, named `the inputs`.
    """
    found: list[tuple[str, UnitPolicies]] = []
    claimed = np.zeros(len(ends), bool)
    for name, policies, ids in rows[1:]:
        assert ids is not None
        inside = np.isin(ends, ids)
        if inside.any():
            found.append((f"the units of `{name}`", policies))
        claimed |= inside
    if not claimed.all():
        found.insert(0, ("the inputs", rows[0][1]))
    return found


def _accumulators(tracer: _Tracer, record: Record, table: PolicyTable) -> list[Fault]:
    """Compare every block's `map` with the `combine` that reduces it.

    Args:
        tracer: The tracer, which holds the abstract columns.
        record: The traced topology.
        table: Its resolved policies.

    Returns:
        A fault for each mismatch, and for each `map` that could not be traced.
    """
    faults: list[Fault] = []
    rows = _unit_rows(record, table)
    for block, policies in zip(record.blocks, table.conns, strict=True):
        # Forward, a map is reduced where the connection ends; backward, where
        # it starts.
        for phase, ends, far in (
            ("forward", block.dst, "ends in"),
            ("backward", block.src, "starts from"),
        ):
            conn = getattr(policies, phase)
            if conn is None or len(ends) == 0:
                continue
            where = f"{_show(block.path)}, {phase}"

            def call(
                u: UnitView, c: ConnView, g: Any, i: Any, x: Any, conn: Any = conn
            ) -> Any:
                return conn.map(u, UnitIdx(i), UnitIdx(i), c, ConnIdx(i), g)

            returned, error, first = tracer.run(conn, f"{phase} map", call, _tree)
            if error is not None:
                if first:
                    faults.append(_untraced(where, conn, "map", error))
                continue
            for at, unit_policies in _reduced_at(ends, rows):
                unit = getattr(unit_policies, phase)
                # Units with no half in this phase reduce nothing, so what a
                # `map` sends them is dropped: sound, and not a fault.
                if unit is None or _matches(returned, unit.combine):
                    continue
                faults.append(
                    Fault(
                        "accumulator",
                        where,
                        _named(conn, "map"),
                        f"returns  {_say(returned, 'scalar')}\n"
                        f"{_named(unit, 'combine')} reduces  "
                        f"{_say(unit.combine, 'monoid')}\n"
                        f"at  {at}\n"
                        f"A connection's {phase} `map` must return what the units "
                        f"it {far} combine.",
                    )
                )
    return faults


def _written(
    where: str,
    policy: object,
    method: str,
    write: Any,
    kind: type,
    specs: dict[str, FieldSpec[Any]],
    element: str,
) -> list[Fault]:
    """Check one traced write against the element's columns.

    Args:
        where: The layer and phase.
        policy: The policy object.
        method: The method that returned the write.
        write: The description the trace kept: the written leaves by name and
            the specs by name, or the class name of what was returned instead.
        kind: `UnitWrite` or `ConnWrite`.
        specs: The element's columns, by name.
        element: "unit" or "connection", for the message.

    Returns:
        A fault for each field that is missing, of another dtype, not a scalar,
        or written under another field's name.
    """
    name = _named(policy, method)
    if isinstance(write, str):
        return [
            Fault(
                "write",
                where,
                name,
                f"returns  {write}\nexpected  a {kind.__name__}, as "
                f"`px.{kind.__name__}.of((field, value), ...)`",
            )
        ]
    leaves, given = write
    faults: list[Fault] = []
    for field, leaf in leaves.items():
        column = specs.get(field)
        if column is None:
            faults.append(
                Fault(
                    "write",
                    where,
                    name,
                    f"writes the field  `{field}`\n"
                    f"a {element} has  {', '.join(sorted(specs))}\n"
                    f"Declare the field on the network, or write one it has.",
                )
            )
            continue
        spec = given.get(field)
        if spec is not None and spec != column:
            faults.append(
                Fault(
                    "field",
                    where,
                    name,
                    f"writes  {spec!r}\n"
                    f"the {element}'s `{field}` is  {column!r}\n"
                    f"Two different fields have one name. Rename one, or pass the "
                    f"same field object to both.",
                )
            )
            continue
        if leaf.shape != ():
            faults.append(
                Fault(
                    "write",
                    where,
                    name,
                    f"writes to `{field}`  an array of shape {leaf.shape}\n"
                    f"expected  one value\n"
                    f"A policy is written for one {element}; the framework maps "
                    f"it over all of them.",
                )
            )
        elif not _fits(leaf, np.dtype(column.dtype)):
            faults.append(
                Fault(
                    "write",
                    where,
                    name,
                    f"writes  {leaf.dtype}\n"
                    f"to the field `{field}`  {np.dtype(column.dtype)}\n"
                    f"Convert the value, or declare the field with the dtype it "
                    f"holds.",
                )
            )
    return faults


def _describe_write(kind: type[Any]) -> Callable[[Any], Any]:
    """How a returned write is kept while the trace is open.

    Args:
        kind: `UnitWrite` or `ConnWrite`.

    Returns:
        A function giving the written leaves and specs by name, or the class
        name of a value that is not a write of this kind.
    """

    def describe(write: Any) -> Any:
        if not isinstance(write, kind):
            return type(write).__name__
        leaves = {name: _leaf(value) for name, value in write.fields.items()}
        return leaves, dict(write.specs)

    return describe


def _acc(combine: Any) -> Any:
    """An accumulator with the structure of a `combine`, for tracing `apply`.

    Args:
        combine: The unit half's tree of monoids.

    Returns:
        The same tuples and dicts, with a float32 zero at every monoid.
    """
    if isinstance(combine, tuple):
        return tuple(_acc(c) for c in combine)
    if isinstance(combine, dict):
        return {k: _acc(c) for k, c in combine.items()}
    return np.zeros((), np.float32)


def _writes(tracer: _Tracer, record: Record, table: PolicyTable) -> list[Fault]:
    """Check every write of every policy in the table.

    Args:
        tracer: The tracer, which holds the abstract columns.
        record: The traced topology.
        table: Its resolved policies.

    Returns:
        A fault for each write that does not fit, and for each method that could
        not be traced. A policy used by several blocks is checked once.
    """
    faults: list[Fault] = []

    def one(
        where: str,
        policy: Any,
        method: str,
        # Called as `call(units, conns, g, index, extra)`. The callers bind the
        # policy as a default argument, which mypy cannot match to five arguments.
        call: Callable[..., object],
        kind: type,
        extra: Any = None,
    ) -> None:
        unit = kind is UnitWrite
        specs = tracer.unit_specs if unit else tracer.conn_specs
        write, error, first = tracer.run(
            policy, method, call, _describe_write(kind), extra
        )
        if not first:
            return
        if error is not None:
            faults.append(_untraced(where, policy, method, error))
            return
        element = "unit" if unit else "connection"
        faults.extend(_written(where, policy, method, write, kind, specs, element))

    p: Any
    for name, units, _ in _unit_rows(record, table):
        for phase in ("forward", "backward"):
            p = getattr(units, phase)
            if p is not None:
                one(
                    f"{name}, {phase}",
                    p,
                    "apply",
                    lambda u, c, g, i, x, p=p: p.apply(u, UnitIdx(i), g, x),
                    UnitWrite,
                    _acc(p.combine),
                )
        p = _cell(units, "update")
        if p is not None:
            one(
                f"{name}, update",
                p,
                "update",
                lambda u, c, g, i, x, p=p: p.update(u, UnitIdx(i), g),
                UnitWrite,
            )
        p = _cell(units, "grow")
        if p is not None:
            one(
                f"{name}, grow",
                p,
                "init",
                lambda u, c, g, i, x, p=p: p.init(u, UnitIdx(i), UnitIdx(i), g),
                UnitWrite,
            )

    conn_rows: list[tuple[str, ConnPolicies]] = [("the network", table.network_conns)]
    conn_rows += [
        (_show(block.path), policies)
        for block, policies in zip(record.blocks, table.conns, strict=True)
    ]
    for name, conns in conn_rows:
        p = _cell(conns, "update")
        if p is not None:
            for method in ("incoming", "outgoing"):
                one(
                    f"{name}, update",
                    p,
                    method,
                    lambda u, c, g, i, x, p=p, method=method: getattr(p, method)(
                        u, UnitIdx(i), UnitIdx(i), c, ConnIdx(i), g
                    ),
                    ConnWrite,
                )
        p = _cell(conns, "grow")
        if p is not None:
            one(
                f"{name}, grow",
                p,
                "init",
                lambda u, c, g, i, x, p=p: p.init(u, UnitIdx(i), UnitIdx(i), g),
                ConnWrite,
            )
    return faults


def check(
    net: type[Network[Any]], record: Record, table: PolicyTable, globals_: Any = None
) -> list[Fault]:
    """Find every fault in a network's policies, before it is materialized.

    Each policy method is traced once with abstract values. A `map` is compared
    with the `combine` of the units that reduce it; every write is compared with
    the field it is written to.

    Args:
        net: The network class.
        record: Its traced topology.
        table: Its resolved policies, from `resolve`.
        globals_: The network's global state, as arrays. None for a network with
            none.

    Returns:
        The faults: the accumulator mismatches in the record's order, then the
        writes. Empty when the policies are sound.
    """
    tracer = _Tracer(net, record, globals_)
    return [
        *_accumulators(tracer, record, table),
        *_writes(tracer, record, table),
    ]
