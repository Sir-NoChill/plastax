"""Declaring a network's initial structure: modules, layers and handles.

Under construction: see `plans/20261116T132654Z-layers-module.md` at the workspace
root. So far: handles, modules, tracing a module to a record, and one layer, `Dense`
with private weights.
"""

from plastax.layers._dense import Dense
from plastax.layers._draw import draw
from plastax.layers._init import ConstInit
from plastax.layers._module import BuildError, Module
from plastax.layers._record import ConnBlock, Record, trace
from plastax.layers._units import ShapeError, Units

__all__ = [
    "BuildError",
    "ConnBlock",
    "ConstInit",
    "Dense",
    "Module",
    "Record",
    "ShapeError",
    "Units",
    "draw",
    "trace",
]
