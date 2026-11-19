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
class ConnBlock:
    """The connections one layer call wrote down.

    Attributes:
        path: The path of the layer that made the block.
        name: Which of the layer's blocks this is: `"weight"` or `"bias"`. A path and
            a name together identify one block of a record.
        src: The source unit of each connection, as `(E,)` record ids.
        dst: The destination unit of each connection, as `(E,)` record ids.
        init: How the block's weights get their first values, or None for the
            layer's default. Stored here and not called while tracing.
    """

    path: tuple[str, ...]
    name: str
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
        constants: The ids of the units that are always 1.
        blocks: The connections, one block per layer call, in the order written.
        outputs: The unit ids of each handle the topology returned.
    """

    num_units: int = 0
    inputs: dict[str, np.ndarray] = field(default_factory=dict)
    constants: np.ndarray = field(default_factory=_no_ids)
    blocks: list[ConnBlock] = field(default_factory=list)
    outputs: tuple[np.ndarray, ...] = ()
    _sealed: bool = field(default=False, repr=False)
    # The modules this trace set up, by `id`, with their paths; and those of them
    # that have been called. Both belong to the trace, not to the modules, so a
    # module can be traced again.
    _paths: dict[int, tuple[str, ...]] = field(default_factory=dict, repr=False)
    _called: set[int] = field(default_factory=set, repr=False)

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

    def _new_constant(self) -> int:
        """Create one unit that is always 1.

        Returns:
            The new unit's id.

        Raises:
            BuildError: If the trace that made this record has finished.
        """
        self._check_open()
        unit = self.num_units
        self.num_units += 1
        self.constants = np.append(self.constants, np.int32(unit))
        return unit

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
            isinstance(n, (int, np.integer)) and n >= 0  # type: ignore[redundant-expr]
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
            do not have the same names, or if it returns anything other than handles
            of this trace.
    """
    name = type(module).__name__
    paths = setup_tree(module)
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

    record = Record(_paths=paths)
    handles = {key: record._new_units(tuple(shape)) for key, shape in inputs.items()}
    record.inputs = {key: handle.ids for key, handle in handles.items()}

    returned = module(*(handles[p] for p in parameters))
    outputs = returned if isinstance(returned, tuple) else (returned,)
    for output in outputs:
        if not isinstance(output, Units):
            raise BuildError(
                f"{name}.__call__ must return a handle or a tuple of handles, "
                f"got {type(output).__name__}"
            )
        if output._record is not record:
            raise BuildError(f"{name}.__call__ returned a handle from another trace")
    record.outputs = tuple(output.ids for output in outputs)
    record._sealed = True
    return record


_VARIADIC = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
