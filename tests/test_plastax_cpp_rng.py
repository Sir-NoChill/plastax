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

The golden-reading tests skip when the plastax-cpp checkout is absent (see
`_plastax_cpp`) -- the port is still importable and usable, it just cannot be
verified against the C++ from here; the golden-independent tests always run.
"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from _plastax_cpp import plastax_cpp_dir
from _plastax_cpp_rng import fully_connected_weights, uniform_real

_GOLDEN = plastax_cpp_dir() / "tests" / "golden" / "rng_uniform.json"

# Only the golden-reading tests need the checkout; the rest pin the port on its
# own and must run where it is absent too (CI has no plastax-cpp checkout).
needs_golden = pytest.mark.skipif(
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


@needs_golden
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
