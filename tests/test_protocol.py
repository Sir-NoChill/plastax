"""The pre-registered evaluation protocol (examples/protocol.py).

These tests exist to stop the protocol drifting after results are seen, which is
the exact failure it was written to prevent: a single-seed recovery-time slope
once showed CBP cutting plasticity loss by 40%, and the same arm re-run gave the
opposite sign.

The seed set is pinned by VALUE, not by length -- a test that only checked
`len(SEEDS) == 30` would happily pass after someone re-rolled the seeds until an
arm behaved.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

_EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def _load_example(name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, _EXAMPLES_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


protocol = _load_example("protocol")


def test_seed_set_is_pinned_by_value() -> None:
    """The seeds are fixed, distinct, and not silently re-rollable."""
    assert protocol.SEEDS == tuple(range(30))
    assert len(set(protocol.SEEDS)) == len(protocol.SEEDS)


def test_summary_reports_percentiles_not_standard_error() -> None:
    """The summary is order statistics, so it tracks spread rather than shrinking.

    Standard error would fall by sqrt(N) between these two samples even though
    they have the same spread; IPR-90 must not.
    """
    small = protocol.summarize(list(np.linspace(0.0, 100.0, 11)))
    large = protocol.summarize(list(np.linspace(0.0, 100.0, 1001)))
    assert small.median == pytest.approx(50.0)
    assert large.median == pytest.approx(50.0)
    assert small.ipr90 == pytest.approx(large.ipr90, rel=0.05)
    assert small.ipr90 == pytest.approx(90.0, rel=0.05)


def test_relative_ipr_and_zero_exclusion() -> None:
    """The two derived quantities the decision rule reads."""
    positive = protocol.summarize([2.0, 3.0, 4.0, 5.0])
    assert positive.excludes_zero
    assert positive.relative_ipr > 0.0
    straddling = protocol.summarize([-3.0, -1.0, 1.0, 3.0])
    assert not straddling.excludes_zero


def test_pairing_cancels_shared_variance() -> None:
    """Pairing is the whole reason arms share seeds.

    Both arms here carry a large per-seed offset and differ by a constant 1.0.
    Unpaired, that offset swamps the effect; paired, it cancels exactly.
    """
    rng = np.random.default_rng(0)
    offsets = rng.normal(0.0, 50.0, size=30)
    control = list(offsets)
    treatment = list(offsets - 1.0)
    unpaired = protocol.summarize(treatment)
    paired = protocol.paired_difference(treatment, control)
    assert paired.median == pytest.approx(-1.0)
    assert paired.ipr90 == pytest.approx(0.0, abs=1e-9)
    assert unpaired.ipr90 > 50.0, "unpaired spread should be dominated by seeds"
    assert paired.excludes_zero


def test_paired_difference_rejects_mismatched_lengths() -> None:
    """A dropped seed must be an error, never a silent realignment."""
    with pytest.raises(ValueError):
        protocol.paired_difference([1.0, 2.0], [1.0])


def test_verdict_follows_the_prereigstered_rule() -> None:
    """Overlap with zero is 'no effect', never reported as a trend."""
    better = protocol.summarize([-3.0, -2.0, -1.0])
    worse = protocol.summarize([1.0, 2.0, 3.0])
    overlapping = protocol.summarize([-2.0, 0.0, 2.0])
    assert protocol.verdict(better) == "better"
    assert protocol.verdict(worse) == "worse"
    assert protocol.verdict(overlapping) == "no effect at this power"
    # lower_is_better=False flips which sign counts as an improvement
    assert protocol.verdict(worse, lower_is_better=False) == "better"


def test_summarize_rejects_empty() -> None:
    """An arm that produced no runs must not summarise to anything."""
    with pytest.raises(ValueError):
        protocol.summarize([])
