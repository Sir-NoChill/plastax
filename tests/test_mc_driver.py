"""Multi-controller Driver structural events, in a fan-out of subprocesses.

The check lives in `mc_driver_equiv.py`, which launches N separate processes via
`jax.distributed` (gloo on CPU) and drives the Driver's overflow -> grow_bucket
-> retrace loop and topo.resort on a state distributed across those processes.
Subprocessed for the same reasons as the other sharding tests (shard_map is
incompatible with jaxtyping instrumentation; each worker needs its own device).

Marked `slow`: spawns a process per shard and pays four backend inits.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parent / "mc_driver_equiv.py"


@pytest.mark.slow
def test_multi_controller_driver_structural_events() -> None:
    result = subprocess.run(
        [sys.executable, str(_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "MC DRIVER EQUIVALENCE PASS" in result.stdout, (
        result.stdout + "\n" + result.stderr
    )
