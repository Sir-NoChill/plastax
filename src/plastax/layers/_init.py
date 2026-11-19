"""How a block's weights get their first values.

A placeholder. Initializers are to be designed later, reusing `jax.nn.initializers`
behind an adapter; until then a block can only be given one constant. Nothing here
imports JAX or any other part of plastax.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConstInit:
    """Every weight of a block starts at the same value.

    Attributes:
        value: The value.
    """

    value: float
