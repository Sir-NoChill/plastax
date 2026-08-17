"""plastax: declarative plastic-network traits for JAX."""
from plastax._types import (
    ACTIVATION,
    DEAD,
    FROM_ID,
    LEVEL,
    TO_ID,
    WEIGHT,
    ConnIdx,
    FieldSpec,
    Propagation,
    UnitIdx,
)
from plastax import monoid, topology
from plastax.builder import NetworkBuilder
from plastax.driver import Driver
from plastax.state import NetworkState, NetworkStatic, make_empty_state
from plastax.step import StepResult, make_step
from plastax.traits import (
    AddConn,
    BackwardPass,
    ForwardPass,
    Loss,
    Network,
    PruneConn,
    ResetGlobal,
    UpdateConn,
)
from plastax.views import ConnView, ConnWrite, UnitView, UnitWrite

__all__ = [
    "ACTIVATION", "DEAD", "FROM_ID", "LEVEL", "TO_ID", "WEIGHT",
    "AddConn", "BackwardPass", "ConnIdx", "ConnView", "ConnWrite",
    "Driver", "FieldSpec", "ForwardPass", "Loss", "Network",
    "NetworkBuilder", "NetworkState", "NetworkStatic", "Propagation",
    "PruneConn", "ResetGlobal", "StepResult", "UnitIdx", "UnitView",
    "UnitWrite", "UpdateConn", "make_empty_state", "make_step", "monoid",
    "topology",
]
