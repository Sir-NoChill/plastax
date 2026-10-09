"""Pin the NumPy port of plastax-cpp's initialiser to the C++ golden.

`tests/_plastax_cpp_rng.py` reproduces `plastax::UniformReal` so the conformance
vectors can start from weights identical to the C++ ones. This test asserts the
port is *bit* exact, not merely close: initial weights are an input to every
other vector, so a one-ULP difference here would quietly eat into the tolerance
budget of everything downstream and make a real divergence look like rounding.

The golden lives in the plastax-cpp tree (`tests/golden/rng_uniform.json`,
emitted by `tests/tools/emit_rng_golden.cpp`) and plastax-cpp's own
`test_parity_rng.cpp` asserts against the same file, so neither implementation
can drift alone.

Skipped when the plastax-cpp checkout is absent (see `_plastax_cpp`) -- the
port is still importable and usable, it just cannot be verified from here.
"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from _plastax_cpp import plastax_cpp_dir
from _plastax_cpp_rng import fully_connected_weights, mix_seed, uniform_real

_GOLDEN = plastax_cpp_dir() / "tests" / "golden" / "rng_uniform.json"

pytestmark = pytest.mark.skipif(
    not _GOLDEN.is_file(),
    reason=(
        f"no plastax-cpp RNG golden at {_GOLDEN}; point PLASTAX_CPP_DIR at a "
        "plastax-cpp checkout"
    ),
)


def _bits(value: np.float32) -> str:
    """Return the raw float32 bit pattern as 8 lowercase hex digits."""
    return format(struct.unpack("<I", struct.pack("<f", np.float32(value)))[0], "08x")


def _golden() -> dict:
    return json.loads(_GOLDEN.read_text())


def test_uniform_real_is_bit_exact() -> None:
    """Every (seed, counter, range) in the golden reproduces exactly."""
    golden = _golden()
    ranges = golden["ranges"]
    samples = golden["samples"]
    assert len(samples) > 100, "golden is suspiciously small; did the emitter change?"

    mismatches: list[str] = []
    for sample in samples:
        seed, counter = sample["seed"], sample["counter"]
        for name, (lo, hi) in ranges.items():
            got = _bits(uniform_real(np.uint64(seed), np.uint64(counter), lo, hi))
            if got != sample[name]:
                mismatches.append(
                    f"seed={seed} counter={counter} range={name}({lo},{hi}): "
                    f"C++ {sample[name]} vs port {got}"
                )
    assert not mismatches, "\n".join(
        ["NumPy port diverged from plastax::UniformReal:", *mismatches[:20]]
    )


def test_both_lerp_branches_are_covered() -> None:
    """The golden must exercise a zero-straddling and a same-sign range.

    The two ranges take different branches of std::lerp, and a port that only
    handles the first is wrong for every positive-only initialiser. Assert the
    coverage rather than assuming the emitter kept it.
    """
    ranges = _golden()["ranges"].values()
    assert any(lo <= 0 <= hi for lo, hi in ranges), "no zero-straddling range"
    assert any(lo > 0 or hi < 0 for lo, hi in ranges), "no same-sign range"


def test_vectorises_over_counters() -> None:
    """Array and scalar counters agree, so a whole layer can be drawn at once."""
    counters = np.arange(64, dtype=np.uint64)
    batched = uniform_real(np.uint64(7), counters, -1.0, 1.0)
    scalar = np.array(
        [uniform_real(np.uint64(7), np.uint64(c), -1.0, 1.0) for c in counters],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(batched, scalar)


def test_mix_seed_is_uint32() -> None:
    """MixSeed truncates to 32 bits, as the C++ return type does."""
    out = mix_seed(np.uint64(0xFFFFFFFFFFFFFFFF), np.uint64(12345))
    assert out.dtype == np.uint32


def test_fully_connected_weights_uses_destination_major_ids() -> None:
    """w[src, dst] must come from conn id base + dst * n_src + src.

    This is the mapping between plastax-cpp's allocation order and plastax's edge
    order. Getting it wrong still produces plausible random weights -- just
    permuted onto the wrong edges -- so pin it explicitly rather than trusting
    a conformance vector to notice.
    """
    n_src, n_dst, base, seed = 3, 4, 12, 2
    w = fully_connected_weights(seed, n_src, n_dst, base_conn_id=base, engine="minstd")
    assert w.shape == (n_src, n_dst)
    for dst in range(n_dst):
        for src in range(n_src):
            expected = uniform_real(
                np.uint64(seed), np.uint64(base + dst * n_src + src), -1.0, 1.0
            )
            assert _bits(w[src, dst]) == _bits(expected), f"({src}, {dst})"


def test_fully_connected_weights_philox_engine_matches_the_scalar_form() -> None:
    """The default (philox) engine: min + (max-min) * unit_float(word).

    Same destination-major id mapping as the minstd branch; the scalar form is
    pinned to plastax-cpp's rng_philox32.json through the parity reference.
    """
    from reference import philox32, unit_float

    n_src, n_dst, base, seed = 3, 4, 12, 2
    w = fully_connected_weights(seed, n_src, n_dst, base_conn_id=base)
    assert w.shape == (n_src, n_dst)
    for dst in range(n_dst):
        for src in range(n_src):
            word = philox32(seed, base + dst * n_src + src)
            expected = np.float32(-1.0) + np.float32(2.0) * unit_float(word)
            assert _bits(w[src, dst]) == _bits(expected), f"({src}, {dst})"


def test_samples_stay_in_range() -> None:
    """Sanity bound on both branches, independent of the golden."""
    counters = np.arange(2048, dtype=np.uint64)
    signed = uniform_real(np.uint64(3), counters, -1.0, 1.0)
    assert signed.min() >= -1.0 and signed.max() < 1.0
    positive = uniform_real(np.uint64(3), counters, 2.0, 5.0)
    assert positive.min() >= 2.0 and positive.max() < 5.0
