"""Shared pytest configuration.

CPU-only for determinism (TOOLING.md): float reductions are reproducible and
CI needs no accelerator, and the oracle tolerances in tests/README.md assume
it. Set before JAX is imported by any test module.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
