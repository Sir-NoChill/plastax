"""Multi-controller Scheme-A equivalence, in a fan-out of clean subprocesses.

The real check lives in `mc_sharding_equiv.py`, which launches N separate
processes via `jax.distributed` (gloo on CPU), one device each -- the local
stand-in for a one-process-per-node Narval run. It runs here as a subprocess for
two reasons, both matching the other sharding tests: shard_map is incompatible
with pytest's jaxtyping instrumentation, and the launcher needs each worker to
own exactly one device (its own JAX backend), which a re-exec gives cleanly.

Marked `slow`: it spawns a process per shard and pays four backend inits.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parent / "mc_sharding_equiv.py"


@pytest.mark.slow
def test_multi_controller_sharding_equivalence() -> None:
    result = subprocess.run(
        [sys.executable, str(_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "MC SHARDING EQUIVALENCE PASS" in result.stdout, (
        result.stdout + "\n" + result.stderr
    )
