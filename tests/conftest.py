"""Shared pytest configuration.

CPU-only for determinism (docs/development/tooling.md): float reductions are
reproducible and CI needs no accelerator, and the oracle tolerances assume it.
Set before JAX is imported by any test module.

JAX's persistent compilation cache is on (under ``.cache/jax`` in the repo
root, git-ignored): most test time is XLA compilation, and a warm cache skips
it. Entries are keyed by the HLO module, compile options, XLA flags, backend
and jax/jaxlib versions, so a code change is a cache miss, never a stale hit.
Tracing, lowering (and its donation warning) and execution still run every
time. The variables are also inherited by the subprocess-based tests. Set
``JAX_ENABLE_COMPILATION_CACHE=false`` to turn it off.
"""

import os
import pathlib
import shutil

import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
# Fake multi-device CPU so the Scheme-A sharding tests can run without a GPU;
# harmless to single-device tests, which still default to device 0.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

_JAX_CACHE = pathlib.Path(__file__).resolve().parent.parent / ".cache" / "jax"
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(_JAX_CACHE))
# Cache every executable: most test programs compile in well under jax's
# default 1 s threshold, and they are the bulk of the suite's compile time.
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")

# Cap at 1 GiB, pruned at session start: dropping the whole cache costs one
# cold run. (jax's own size bound is not used: it rescans the directory under
# a global file lock on every write, which serialises pytest-xdist workers.)
_JAX_CACHE_MAX_BYTES = 1 << 30


def pytest_sessionstart(session: pytest.Session) -> None:
    """Drop the compilation cache once it outgrows its cap (controller only)."""
    if hasattr(session.config, "workerinput") or not _JAX_CACHE.is_dir():
        return
    size = sum(p.stat().st_size for p in _JAX_CACHE.iterdir() if p.is_file())
    if size > _JAX_CACHE_MAX_BYTES:
        shutil.rmtree(_JAX_CACHE, ignore_errors=True)
