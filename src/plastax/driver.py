"""Host driver: owns the retrace protocol (rung0 design sections 4-5).

The step function itself is pure; the driver is the only component that
reads needs_resort/overflow on host, calls topo.resort or
state.grow_bucket, and swaps in the new (static, step). Retrace count is
observable for tests via jax.test_util.assert_num_jit_and_pmap_compilations.
"""
from __future__ import annotations

from typing import Generic, TypeVar

from plastax.phases import StepInputs
from plastax.state import NetworkState, NetworkStatic
from plastax.traits import Network

GS = TypeVar("GS")


class Driver(Generic[GS]):
    def __init__(
        self,
        net: type[Network[GS]],
        static: NetworkStatic,
        state: NetworkState[GS],
    ) -> None:
        raise NotImplementedError

    def step(self, inputs: StepInputs) -> None:
        """One network step. Order: run jitted step (donated state); if
        overflow -> grow_bucket + retry the same inputs; if needs_resort ->
        resort + continue (resort is between-steps, matching native
        NeedsResort semantics)."""
        raise NotImplementedError

    @property
    def state(self) -> NetworkState[GS]:
        raise NotImplementedError

    @property
    def static(self) -> NetworkStatic:
        raise NotImplementedError
