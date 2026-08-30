"""Scheme-A across DST phases, checked in a clean subprocess.

Like test_sharding.py, the real check lives in `sharding_churn_equiv.py` and
runs in a separate interpreter without pytest's jaxtyping instrumentation
(shard_map reconstructs the registered NetworkState pytree with placeholder
leaves, which beartype rejects -- a test-only artifact). It verifies that a
full train step and a prune step match single-device under Scheme-A, and pins
the current limitation that add_conn growth does not shard.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import jax
import pytest

_SCRIPT = Path(__file__).parent / "sharding_churn_equiv.py"
_N_SHARDS = 4


@pytest.mark.skipif(
    len(jax.devices()) < _N_SHARDS,
    reason=f"needs >= {_N_SHARDS} devices",
)
def test_dst_phases_under_scheme_a() -> None:
    result = subprocess.run(
        [sys.executable, str(_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "CHURN SHARDING CHECK PASS" in result.stdout
