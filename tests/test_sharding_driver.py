"""Driver structural-event protocol under Scheme-A, in a clean subprocess.

Like test_sharding.py, the real check lives in `sharding_driver_equiv.py` and
runs in a separate interpreter without pytest's jaxtyping instrumentation
(shard_map is incompatible with it). It verifies the Driver's overflow ->
grow_bucket -> retrace loop and topo.resort both match single-device under
single-controller shard_map.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import jax
import pytest

_SCRIPT = Path(__file__).parent / "sharding_driver_equiv.py"
_N_SHARDS = 4


@pytest.mark.skipif(
    len(jax.devices()) < _N_SHARDS,
    reason=f"needs >= {_N_SHARDS} devices",
)
def test_driver_structural_events_under_scheme_a() -> None:
    result = subprocess.run(
        [sys.executable, str(_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "DRIVER SHARDING CHECK PASS" in result.stdout
