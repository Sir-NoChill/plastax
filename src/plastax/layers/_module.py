"""The base class a topology is written as.

A module is a description: it holds no arrays and no unit ids, and no record outside
a trace. Tracing it (`plastax.layers.trace`) is what produces a record; for the length
of that trace each module in the tree is told which record it is. Nothing here imports
JAX or any other part of plastax.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from plastax.layers._units import Units

if TYPE_CHECKING:
    from plastax.layers._record import Record


class BuildError(ValueError):
    """A topology was written in a way that cannot be built.

    Raised while a module is traced, naming what the user wrote.
    """


# One registered child: the path parts below its parent, and the child. An attribute
# holds `("conv",)`; element 2 of a list attribute holds `("blocks", "2")`.
type _Child = tuple[tuple[str, ...], "Module"]

# Instance attributes the base class sets, which are never children.
_INTERNAL = frozenset({"_path", "_children", "_record"})


class Module:
    """A piece of initial structure: layers declared in `setup`, wired in `__call__`.

    A subclass writes two methods:

    - `setup(self)` assigns layers and other modules to attributes.
    - `__call__(self, ...)` takes one handle per input, passes them through the
      layers, and returns the handle or handles of the outputs.

    A module that takes arguments of its own stores them in a plain `__init__`
    (`self.width = width`) and reads them in `setup`. The base class has no
    `__init__` to call.

    `setup` runs once, the first time the module is traced. Every attribute that is
    a module is then registered under its name, and a list or tuple of modules under
    its name and each index, which gives every module a path from the root:
    `("blocks", "0", "attention")`. The path names the module in error messages.
    """

    # Declared, not defined: a subclass's `__call__` may take any named handles, and
    # a method defined here would fix one signature for all of them.
    # A layer that only connects existing units (`CreateConnections`) returns None.
    __call__: Callable[..., Units | tuple[Units, ...] | None]

    # Class-level defaults, so that a subclass's `__init__` need not call the base.
    _path: tuple[str, ...] | None = None
    _children: tuple[_Child, ...] | None = None  # None until `setup` has run
    _record: Record | None = None  # the record being traced; None outside a trace

    def setup(self) -> None:
        """Declare this module's layers and sub-modules as attributes.

        The default declares nothing.
        """

    @property
    def path(self) -> tuple[str, ...]:
        """The module's path from the root of the last trace it was part of.

        The root's path is `()`.

        Raises:
            BuildError: If the module has not been part of a trace.
        """
        if self._path is None:
            raise BuildError(
                f"{type(self).__name__} has no path: it has not been part of a trace"
            )
        return self._path


def _collect_children(module: Module) -> tuple[_Child, ...]:
    """Find the modules among a module's attributes, in the order they were assigned.

    Args:
        module: The module, after its `setup` has run.

    Returns:
        Each child with its path parts below `module`.
    """
    children: list[_Child] = []
    for name, value in vars(module).items():
        if name in _INTERNAL:
            continue
        if isinstance(value, Module):
            children.append(((name,), value))
        elif isinstance(value, list | tuple):
            children.extend(
                ((name, str(i)), item)
                for i, item in enumerate(value)
                if isinstance(item, Module)
            )
    return tuple(children)


def setup_tree(root: Module) -> list[tuple[Module, tuple[str, ...]]]:
    """Run `setup` where it has not run, and give every module its path from `root`.

    Args:
        root: The module being traced.

    Returns:
        Every module in the tree with its path.

    Raises:
        BuildError: If one module object appears at two places in the tree.
    """
    seen: dict[int, tuple[str, ...]] = {}
    tree: list[tuple[Module, tuple[str, ...]]] = []
    pending: list[tuple[tuple[str, ...], Module]] = [((), root)]
    while pending:
        path, module = pending.pop()
        if id(module) in seen:
            raise BuildError(
                f"the same {type(module).__name__} is declared at "
                f"{_show(seen[id(module)])} and at {_show(path)}; "
                f"a module has one place in a topology"
            )
        seen[id(module)] = path
        tree.append((module, path))
        if module._children is None:
            module.setup()
            module._children = _collect_children(module)
        module._path = path
        pending.extend((path + parts, child) for parts, child in module._children)
    return tree


def _show(path: tuple[str, ...]) -> str:
    return ".".join(path) if path else "the root"
