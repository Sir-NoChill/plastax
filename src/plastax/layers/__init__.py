"""Declaring a network's initial structure: modules, layers and handles.

Under construction: see `plans/20261116T132654Z-layers-module.md` at the workspace
root. So far: handles, modules and tracing a module to a record. There are no layers
yet.
"""

from plastax.layers._init import ConstInit
from plastax.layers._module import BuildError, Module
from plastax.layers._record import ConnBlock, Record, trace
from plastax.layers._units import ShapeError, Units

__all__ = [
    "BuildError",
    "ConnBlock",
    "ConstInit",
    "Module",
    "Record",
    "ShapeError",
    "Units",
    "trace",
]
