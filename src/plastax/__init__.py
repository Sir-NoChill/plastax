"""plastax: declarative plastic-network traits for JAX."""

from plastax import monoid, optim, topology
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
    ShardSpec,
    UnitIdx,
)
from plastax.builder import NetworkBuilder
from plastax.driver import Driver
from plastax.phases import StepInputs
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
    "ACTIVATION",
    "DEAD",
    "FROM_ID",
    "LEVEL",
    "TO_ID",
    "WEIGHT",
    "AddConn",
    "BackwardPass",
    "ConnIdx",
    "ConnView",
    "ConnWrite",
    "Driver",
    "FieldSpec",
    "ForwardPass",
    "Loss",
    "Network",
    "NetworkBuilder",
    "NetworkState",
    "NetworkStatic",
    "Propagation",
    "PruneConn",
    "ResetGlobal",
    "ShardSpec",
    "StepInputs",
    "StepResult",
    "UnitIdx",
    "UnitView",
    "UnitWrite",
    "UpdateConn",
    "make_empty_state",
    "make_step",
    "monoid",
    "optim",
    "topology",
]
