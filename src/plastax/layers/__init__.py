"""Declaring a network's initial structure: modules, layers and handles.

Under construction: see `plans/20261116T132654Z-layers-module.md` at the workspace
root. So far: handles, modules, tracing a module to a record, the two primitives
`CreateUnits` and `CreateConnections`, and `Dense`, a helper written with them.
"""

from plastax.layers._dense import Dense
from plastax.layers._draw import draw
from plastax.layers._init import ConstInit
from plastax.layers._module import BuildError, Module
from plastax.layers._primitives import CreateConnections, CreateUnits
from plastax.layers._record import ConnBlock, Record, UnitBlock, trace
from plastax.layers._units import ShapeError, Units

__all__ = [
    "BuildError",
    "ConnBlock",
    "ConstInit",
    "CreateConnections",
    "CreateUnits",
    "Dense",
    "Module",
    "Record",
    "ShapeError",
    "UnitBlock",
    "Units",
    "draw",
    "trace",
]
