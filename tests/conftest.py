"""Shared pytest configuration.

CPU-only for determinism (TOOLING.md): float reductions are reproducible and
CI needs no accelerator, and the oracle tolerances in tests/README.md assume
it. Set before JAX is imported by any test module.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
# Fake multi-device CPU so the Scheme-A sharding tests can run without a GPU;
# harmless to single-device tests, which still default to device 0.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
