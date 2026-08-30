"""Multi-controller per-shard construction, in a fan-out of subprocesses.

The check lives in `mc_construct_equiv.py`, which launches N separate processes
via `jax.distributed` (gloo on CPU) and has each build only its own shard with
`from_edges(..., sharding=)`. Subprocessed for the same reasons as the other
sharding tests (shard_map is incompatible with jaxtyping instrumentation; each
worker needs its own device).

Marked `slow`: spawns a process per shard and pays four backend inits.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parent / "mc_construct_equiv.py"


@pytest.mark.slow
def test_multi_controller_per_shard_construction() -> None:
    result = subprocess.run(
        [sys.executable, str(_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "MC CONSTRUCT EQUIVALENCE PASS" in result.stdout, (
        result.stdout + "\n" + result.stderr
    )
