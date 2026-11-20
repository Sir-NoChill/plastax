"""Declaring a network's initial structure: modules, layers and handles.

Under construction: see `plans/20261116T132654Z-layers-module.md` at the workspace
root. So far: handles, modules, tracing a module to a record, the two primitives
`CreateUnits` and `CreateConnections`, `Dense`, a helper written with them, and
`materialize`, which turns a record into a state that today's engine can step.
Weights are drawn by the initializers of `plastax.init`.
"""

from plastax.layers._dense import Dense
from plastax.layers._draw import draw
from plastax.layers._materialize import materialize
from plastax.layers._module import BuildError, Module
from plastax.layers._primitives import CreateConnections, CreateUnits
from plastax.layers._record import ConnBlock, Record, UnitBlock, trace
from plastax.layers._units import ShapeError, Units

__all__ = [
    "BuildError",
    "ConnBlock",
    "CreateConnections",
    "CreateUnits",
    "Dense",
    "Module",
    "Record",
    "ShapeError",
    "UnitBlock",
    "Units",
    "draw",
    "materialize",
    "trace",
]
