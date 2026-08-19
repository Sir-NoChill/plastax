"""Scheme-A sharding equivalence, checked in a clean subprocess.

The check itself lives in `sharding_equiv.py` and runs in a separate
interpreter without pytest's jaxtyping instrumentation: shard_map reconstructs
the registered NetworkState pytree with placeholder leaves internally, which
the instrumentation's beartype layer rejects -- a test-only artifact (there is
no beartype in production). The subprocess validates the real behavior.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import jax
import pytest

_SCRIPT = Path(__file__).parent / "sharding_equiv.py"
_N_SHARDS = 4


@pytest.mark.skipif(
    len(jax.devices()) < _N_SHARDS,
    reason=f"needs >= {_N_SHARDS} devices",
)
def test_scheme_a_matches_single_device() -> None:
    result = subprocess.run(
        [sys.executable, str(_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "SHARDING EQUIVALENCE PASS" in result.stdout
