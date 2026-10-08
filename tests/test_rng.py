"""Pin `plastax.rng` to the plastax-cpp Philox goldens and its own contract.

The raw core (`philox32`) and the uniform derivation are asserted *bit*-exact
against goldens emitted by plastax-cpp's `emit_rng_golden` (`rng_philox32.json`
raw words; `rng_uniform_philox.json` range-mapped floats), so neither port can
drift alone. The site-keying, sub-counter and derived-draw behaviour is pinned
by contract tests that need no checkout.
"""

from __future__ import annotations

import json
import struct

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _plastax_cpp import plastax_cpp_dir
from plastax.rng import Rng, conn_key, philox32, unit_float

_GOLDEN_DIR = plastax_cpp_dir() / "tests" / "golden"

needs_philox_golden = pytest.mark.skipif(
    not (_GOLDEN_DIR / "rng_philox32.json").is_file(),
    reason=(
        f"no plastax-cpp Philox golden at {_GOLDEN_DIR}; point PLASTAX_CPP_DIR "
        "at a plastax-cpp checkout"
    ),
)
needs_uniform_golden = pytest.mark.skipif(
    not (_GOLDEN_DIR / "rng_uniform_philox.json").is_file(),
    reason=(
        f"no plastax-cpp uniform golden at {_GOLDEN_DIR}; point PLASTAX_CPP_DIR "
        "at a plastax-cpp checkout"
    ),
)


def _limbs(values: list[int]) -> tuple[jax.Array, jax.Array]:
    arr = np.asarray(values, dtype=np.uint64)
    return (
        jnp.asarray((arr >> np.uint64(32)).astype(np.uint32)),
        jnp.asarray((arr & np.uint64(0xFFFFFFFF)).astype(np.uint32)),
    )


@needs_philox_golden
def test_philox32_is_bit_exact_against_the_cpp_golden() -> None:
    """Every (seed, counter) in the golden reproduces the exact word."""
    golden = json.loads((_GOLDEN_DIR / "rng_philox32.json").read_text())
    samples = golden["samples"]
    assert len(samples) > 100, "golden is suspiciously small"
    seed_hi, seed_lo = _limbs([s["seed"] for s in samples])
    ctr_hi, ctr_lo = _limbs([s["counter"] for s in samples])
    want = np.asarray([int(s["word"], 16) for s in samples], dtype=np.uint32)
    got = np.asarray(jax.jit(philox32)(seed_hi, seed_lo, ctr_hi, ctr_lo))
    assert got.dtype == np.uint32
    np.testing.assert_array_equal(got, want)


@needs_uniform_golden
def test_uniform_derivation_matches_the_cpp_golden_bitwise() -> None:
    """min + (max-min) * unit_float(word) reproduces UniformReal's bits."""
    golden = json.loads((_GOLDEN_DIR / "rng_uniform_philox.json").read_text())
    assert golden["engine"] == "philox"
    samples = golden["samples"]
    assert len(samples) > 100, "golden is suspiciously small"
    seed_hi, seed_lo = _limbs([s["seed"] for s in samples])
    ctr_hi, ctr_lo = _limbs([s["counter"] for s in samples])
    words = jax.jit(philox32)(seed_hi, seed_lo, ctr_hi, ctr_lo)
    for name, (lo, hi) in golden["ranges"].items():
        got = np.asarray(jnp.float32(lo) + jnp.float32(hi - lo) * unit_float(words))
        want_bits = np.asarray([int(s[name], 16) for s in samples], dtype=np.uint32)
        got_bits = np.frombuffer(got.astype("<f4").tobytes(), dtype=np.uint32)
        np.testing.assert_array_equal(got_bits, want_bits, err_msg=f"range {name}")


def _site(**overrides: object) -> Rng:
    kwargs: dict = dict(
        network_seed=0,
        step=jnp.int32(3),
        stream=1,
        proposer_key=jnp.uint32(7),
        j=jnp.uint32(2),
    )
    kwargs.update(overrides)
    return Rng.for_site(**kwargs)  # type: ignore[arg-type]


def test_same_site_replays_the_same_stream() -> None:
    a = [_site().uniform() for _ in range(1)]
    rng1, rng2 = _site(), _site()
    seq1 = [float(rng1.uniform()) for _ in range(4)]
    seq2 = [float(rng2.uniform()) for _ in range(4)]
    assert seq1 == seq2
    assert float(a[0]) == seq1[0]


def test_sub_counter_advances_per_draw() -> None:
    rng = _site()
    draws = [float(rng.uniform()) for _ in range(8)]
    assert len(set(draws)) == len(draws), "successive draws collided"
    assert all(0.0 <= d < 1.0 for d in draws)


def test_distinct_site_coordinates_give_distinct_streams() -> None:
    base = float(_site().uniform())
    for name, value in [
        ("step", jnp.int32(4)),
        ("stream", 2),
        ("proposer_key", jnp.uint32(8)),
        ("j", jnp.uint32(3)),
        ("network_seed", 1),
    ]:
        other = float(_site(**{name: value}).uniform())
        assert other != base, f"varying {name} did not change the stream"


def test_normal_consumes_two_sub_counters_and_has_sane_moments() -> None:
    rng = _site()
    _ = rng.normal()
    after_normal = float(rng.uniform())
    rng2 = _site()
    _, _ = rng2.uniform(), rng2.uniform()
    assert after_normal == float(rng2.uniform()), "normal() must consume 2 subs"

    # Moments over vmapped sites: loose bounds, this is a wiring check.
    ids = jnp.arange(4096, dtype=jnp.uint32)
    normals = jax.vmap(
        lambda k: Rng.for_site(
            network_seed=0,
            step=jnp.int32(0),
            stream=1,
            proposer_key=k,
            j=jnp.uint32(0),
        ).normal()
    )(ids)
    assert np.isfinite(np.asarray(normals)).all()
    assert abs(float(normals.mean())) < 0.05
    assert abs(float(normals.var()) - 1.0) < 0.05


def test_uniform_int_bounds_and_determinism() -> None:
    ids = jnp.arange(2048, dtype=jnp.uint32)

    def draw(k: jax.Array) -> jax.Array:
        return Rng.for_site(
            network_seed=5,
            step=jnp.int32(1),
            stream=1,
            proposer_key=k,
            j=jnp.uint32(0),
        ).uniform_int(13)

    a = np.asarray(jax.vmap(draw)(ids))
    b = np.asarray(jax.vmap(draw)(ids))
    np.testing.assert_array_equal(a, b)
    assert a.min() >= 0 and a.max() <= 12
    assert len(np.unique(a)) == 13, "13 buckets should all be hit over 2048 sites"


def test_bernoulli_edges() -> None:
    rng = _site()
    assert not bool(rng.bernoulli(0.0))
    rng2 = _site()
    assert bool(rng2.bernoulli(1.0)), "uniform() < 1.0 must hold (draws are in [0,1))"


def test_vmapped_sites_match_scalar_sites() -> None:
    keys = jnp.arange(16, dtype=jnp.uint32)

    def draw(k: jax.Array) -> jax.Array:
        return Rng.for_site(
            network_seed=9,
            step=jnp.int32(2),
            stream=1,
            proposer_key=k,
            j=jnp.uint32(1),
        ).uniform()

    batched = np.asarray(jax.vmap(draw)(keys))
    scalar = np.asarray([float(draw(k)) for k in keys], dtype=np.float32)
    np.testing.assert_array_equal(batched, scalar)


def test_conn_key_distinguishes_edges_and_occurrences() -> None:
    k = conn_key(jnp.uint32(3), jnp.uint32(5), jnp.uint32(0))
    assert k.dtype == jnp.uint32
    others = [
        conn_key(jnp.uint32(5), jnp.uint32(3), jnp.uint32(0)),  # reversed edge
        conn_key(jnp.uint32(3), jnp.uint32(5), jnp.uint32(1)),  # parallel edge
        conn_key(jnp.uint32(3), jnp.uint32(6), jnp.uint32(0)),  # other dst
    ]
    assert all(int(o) != int(k) for o in others)


def test_unit_float_range_and_precision() -> None:
    assert float(unit_float(jnp.uint32(0))) == 0.0
    top = float(unit_float(jnp.uint32(0xFFFFFFFF)))
    assert top < 1.0
    assert top == float(np.float32(16777215) * np.float32(1.0 / 16777216.0))
    assert struct.calcsize("<f") == 4  # float32 bit comparisons above rely on it
