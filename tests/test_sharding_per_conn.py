"""Per-connection proposals under Scheme-A, checked in a clean subprocess.

Like test_sharding_churn.py, the real check lives in
`sharding_per_conn_equiv.py` and runs in a separate interpreter without
pytest's jaxtyping instrumentation (shard_map reconstructs the registered
NetworkState pytree with placeholder leaves, which beartype rejects -- a
test-only artifact). It verifies that per-connection growth over 2 and 4
shards matches single-device bit for bit over several churn steps.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import jax
import pytest

_SCRIPT = Path(__file__).parent / "sharding_per_conn_equiv.py"
_N_SHARDS = 4


@pytest.mark.skipif(
    len(jax.devices()) < _N_SHARDS,
    reason=f"needs >= {_N_SHARDS} devices",
)
def test_per_connection_growth_under_scheme_a() -> None:
    result = subprocess.run(
        [sys.executable, str(_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "PER-CONNECTION SHARDING CHECK PASS" in result.stdout
