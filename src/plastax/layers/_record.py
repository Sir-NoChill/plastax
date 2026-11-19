"""The build record: what tracing a topology writes down.

A record is plain numpy and Python. It says which units exist and which connections
run between them, and nothing about how they are stored. Materialization, a separate
step, turns a record into a state. Nothing here imports JAX or any other part of
plastax.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field

import numpy as np

from plastax.layers._init import ConstInit
from plastax.layers._module import BuildError, Module, setup_tree
from plastax.layers._units import ShapeError, Units


def _no_ids() -> np.ndarray:
    return np.zeros(0, np.int32)


@dataclass(eq=False)
class UnitBlock:
    """The units one `CreateUnits` call wrote down: a run of consecutive ids.

    Attributes:
        path: The path of the layer that made the block.
        start: The id of the block's first unit. The block is the ids `start` to
            `start + prod(shape) - 1`.
        shape: The arrangement of the units in the handle the call returned.
        constant: The value the units always hold, or None for ordinary units.
    """

    path: tuple[str, ...]
    start: int
    shape: tuple[int, ...]
    constant: float | None = None


@dataclass(eq=False)
class ConnBlock:
    """The connections one `CreateConnections` call wrote down.

    A block is identified by its path.

    Attributes:
        path: The path of the layer that made the block.
        src: The source unit of each connection, as `(E,)` record ids.
        dst: The destination unit of each connection, as `(E,)` record ids.
        init: How the block's weights get their first values, or None for the
            layer's default. Stored here and not called while tracing.
    """

    path: tuple[str, ...]
    src: np.ndarray
    dst: np.ndarray
    init: ConstInit | None = None


@dataclass(eq=False)
class Record:
    """What a traced topology consists of.

    Unit ids in a record count up from 0 in the order the units were created. They
    are the record's own names for units: materialization may renumber them.

    Attributes:
        num_units: How many units exist.
        inputs: For each input name, its unit ids in the input's shape.
        unit_blocks: The units layers created, one block per call, in the order
            written. Input units are not in a block.
        blocks: The connections, one block per layer call, in the order written.
        outputs: The unit ids of each handle the topology returned.
    """

    num_units: int = 0
    inputs: dict[str, np.ndarray] = field(default_factory=dict)
    unit_blocks: list[UnitBlock] = field(default_factory=list)
    blocks: list[ConnBlock] = field(default_factory=list)
    outputs: tuple[np.ndarray, ...] = ()
    _sealed: bool = field(default=False, repr=False)
    # The modules this trace set up, by `id`, with their paths; and those of them
    # that have been called. Both belong to the trace, not to the modules, so a
    # module can be traced again.
    _paths: dict[int, tuple[str, ...]] = field(default_factory=dict, repr=False)
    _called: set[int] = field(default_factory=set, repr=False)

    @property
    def constants(self) -> np.ndarray:
        """The ids of the units that always hold one value, as an int32 array."""
        runs = [
            np.arange(b.start, b.start + int(np.prod(b.shape, dtype=np.int64)))
            for b in self.unit_blocks
            if b.constant is not None
        ]
        return np.concatenate([_no_ids(), *runs]).astype(np.int32)

    def _check_open(self) -> None:
        """Refuse to change a record whose trace has finished.

        Raises:
            BuildError: If the trace that made this record has finished.
        """
        if self._sealed:
            raise BuildError(
                "this record is finished: a handle was used after the trace that "
                "made it returned"
            )

    def _claim(self, layer: Module) -> tuple[str, ...]:
        """Note that a layer is being called, and give its path in this trace.

        A layer calls this before it creates anything.

        Args:
            layer: The layer being called.

        Returns:
            The layer's path.

        Raises:
            BuildError: If the trace has finished, or the layer is not part of the
                topology being traced.
            NotImplementedError: If the layer was already called in this trace.
        """
        self._check_open()
        name = type(layer).__name__
        path = self._paths.get(id(layer))
        if path is None:
            raise BuildError(
                f"this {name} is not part of the topology being traced; a layer is "
                f"assigned to an attribute in `setup`"
            )
        if id(layer) in self._called:
            raise NotImplementedError(
                f"{name} at {'.'.join(path) or 'the root'} was called a second time; "
                f"calling a layer more than once is not supported yet"
            )
        self._called.add(id(layer))
        return path

    def _add_unit_block(
        self, path: tuple[str, ...], shape: tuple[int, ...], constant: float | None
    ) -> Units:
        """Create one block of units and write it down.

        Args:
            path: The path of the layer creating the units.
            shape: The shape of the new handle.
            constant: The value the units always hold, or None.

        Returns:
            A handle to the new units.
        """
        start = self.num_units
        units = self._new_units(shape)
        self.unit_blocks.append(UnitBlock(path, start, units.shape, constant))
        return units

    def _add_block(self, block: ConnBlock) -> None:
        """Write down one block of connections.

        Args:
            block: The block.

        Raises:
            BuildError: If the trace that made this record has finished.
        """
        self._check_open()
        self.blocks.append(block)

    def _new_units(self, shape: tuple[int, ...]) -> Units:
        """Create units and return a handle to them, numbered in row-major order.

        Args:
            shape: The shape of the new handle.

        Returns:
            A handle to `prod(shape)` new units.

        Raises:
            BuildError: If the trace that made this record has finished.
            ShapeError: If an axis length is not a non-negative integer.
        """
        self._check_open()
        # The annotation is not enforced when a shape comes from user code, so the
        # check stays although mypy can prove it from the annotation.
        if not all(
            isinstance(n, int | np.integer) and n >= 0  # type: ignore[redundant-expr]
            for n in shape
        ):
            raise ShapeError(f"a shape is non-negative integers, got {shape!r}")
        count = int(np.prod(shape, dtype=np.int64))
        ids = np.arange(self.num_units, self.num_units + count, dtype=np.int32)
        self.num_units += count
        return Units(ids.reshape(shape), self)


def trace(module: Module, inputs: dict[str, tuple[int, ...]]) -> Record:
    """Run a topology over handles and write down the structure it describes.

    Sets up the module tree, creates the input units (one input after another, in
    the order of `inputs`), calls the module with one handle per input, and records
    what it returned. Nothing is computed and no weights are drawn, so tracing the
    same module twice gives the same record.

    Args:
        module: The topology.
        inputs: The shape of each input, by name. Each name is the name of one
            argument of the module's `__call__`.

    Returns:
        The finished record.

    Raises:
        BuildError: If the module has no `__call__`, if its arguments and `inputs`
            do not have the same names, if it returns anything other than handles
            of this trace, or if one of its modules is already being traced.
    """
    name = type(module).__name__
    tree = setup_tree(module)
    if not callable(module):
        raise BuildError(f"{name} defines no __call__")

    parameters = inspect.signature(module.__call__).parameters
    variadic = [p.name for p in parameters.values() if p.kind in _VARIADIC]
    if variadic:
        raise BuildError(
            f"{name}.__call__ must name each input; it takes *{variadic[0]}"
        )
    unfed = [p for p in parameters if p not in inputs]
    unread = [i for i in inputs if i not in parameters]
    if unfed or unread:
        problems = []
        if unfed:
            problems.append(f"arguments {unfed} have no input")
        if unread:
            problems.append(f"inputs {unread} have no argument")
        raise BuildError(f"{name}.__call__: " + " and ".join(problems))

    record = Record(_paths={id(m): path for m, path in tree})
    handles = {key: record._new_units(tuple(shape)) for key, shape in inputs.items()}
    record.inputs = {key: handle.ids for key, handle in handles.items()}

    # For the length of the trace every module in the tree knows the record, so
    # that a layer called with no handle (`CreateUnits(shape=...)`) can reach it.
    busy = [m for m, _ in tree if m._record is not None]
    if busy:
        raise BuildError(
            f"{type(busy[0]).__name__} is already being traced; a module can be "
            f"part of one trace at a time"
        )
    for m, _ in tree:
        m._record = record
    try:
        returned = module(*(handles[p] for p in parameters))
    finally:
        for m, _ in tree:
            m._record = None
    outputs = returned if isinstance(returned, tuple) else (returned,)
    ids: list[np.ndarray] = []
    for output in outputs:
        if not isinstance(output, Units):
            raise BuildError(
                f"{name}.__call__ must return a handle or a tuple of handles, "
                f"got {type(output).__name__}"
            )
        if output._record is not record:
            raise BuildError(f"{name}.__call__ returned a handle from another trace")
        ids.append(output.ids)
    record.outputs = tuple(ids)
    record._sealed = True
    return record


_VARIADIC = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
