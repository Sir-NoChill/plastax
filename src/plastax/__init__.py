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
from plastax.distributed import distribute_state, scheme_a_mesh
from plastax.driver import Driver
from plastax.phases import (
    ShortlistCoverage,
    StepInputs,
    recommended_shortlist,
    shortlist_coverage,
)
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
    "AddConn",
    "BackwardPass",
    "ConnIdx",
    "ConnView",
    "ConnWrite",
    "DEAD",
    "Driver",
    "FROM_ID",
    "FieldSpec",
    "ForwardPass",
    "LEVEL",
    "Loss",
    "Network",
    "NetworkBuilder",
    "NetworkState",
    "NetworkStatic",
    "Propagation",
    "PruneConn",
    "ResetGlobal",
    "ShardSpec",
    "ShortlistCoverage",
    "StepInputs",
    "StepResult",
    "TO_ID",
    "UnitIdx",
    "UnitView",
    "UnitWrite",
    "UpdateConn",
    "WEIGHT",
    "distribute_state",
    "make_empty_state",
    "make_step",
    "monoid",
    "optim",
    "recommended_shortlist",
    "scheme_a_mesh",
    "shortlist_coverage",
    "topology",
]
