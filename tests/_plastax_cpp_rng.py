"""NumPy port of plastax-cpp's counter-based weight initialiser.

plastax-cpp's `RandomUniformWeight` does not draw from a PRNG *stream*. It calls
`plastax::UniformReal(seed, counter)` (``include/plastax/random.hpp`` in
plastax-cpp),
a pure function of ``(seed, counter)``: a SplitMix64 mix, one minstd step, and
a ``std::lerp`` into the requested range. A connection at id ``c`` therefore
always gets the same weight regardless of scheduling, host or device -- and,
usefully here, regardless of language.

That is what makes this port possible, and it matters more than it sounds. The
conformance vectors compare a *trajectory*: if the two implementations started
from even slightly different weights, every later divergence would be
uninterpretable. Reproducing the initialiser exactly means the vectors start
from an identical state, so any drift they report was genuinely accumulated by
the algorithm rather than inherited from the initial conditions.

This is the one place in the harness where the comparison is bit-exact.
Everything downstream is tolerance-based, because plastax-cpp walks levels while
plastax segment-reduces and the reduction orders differ by construction.

Pinned by ``tests/test_plastax_cpp_rng.py`` against
``tests/golden/rng_uniform.json`` in plastax-cpp, whose own
``test_parity_rng.cpp`` asserts the same file. Neither side can drift alone.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# Scalars or arrays: every function here is elementwise, so a whole layer's
# connection ids can be drawn in one call.
Counters = npt.ArrayLike

# minstd (a == 48271, m == 2**31 - 1), which is what thrust::default_random_engine
# is on the CUDA path; random.hpp's host branch replicates it exactly so host and
# device builds initialise identically.
_MINSTD_M = np.uint32(2147483647)  # 2**31 - 1
_MINSTD_A = np.uint64(48271)
# 1 + (max - min) for thrust::uniform_real_distribution's [1, m-1] engine range.
_DRAW_DENOM = np.float32(2147483646.0)

_GOLDEN_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_MIX_C1 = np.uint64(0xFF51AFD7ED558CCD)
_MIX_C2 = np.uint64(0xC4CEB9FE1A85EC53)
_SHIFT_33 = np.uint64(33)


def mix_seed(seed: Counters, counter: Counters) -> npt.NDArray[np.uint32]:
    """SplitMix64 finalizer over a Weyl-mixed ``(seed, counter)`` pair.

    Mirrors ``plastax::detail::MixSeed``. Arithmetic is unsigned 64-bit with
    wraparound, which NumPy gives us natively -- the ``errstate`` block only
    silences the overflow warnings that wraparound legitimately raises.

    Args:
        seed: The initialiser's seed, scalar or array.
        counter: The connection id, scalar or array.

    Returns:
        The low 32 bits of the mixed state, as ``uint32``.
    """
    seed_u = np.asarray(seed, dtype=np.uint64)
    counter_u = np.asarray(counter, dtype=np.uint64)
    with np.errstate(over="ignore"):
        x = seed_u + _GOLDEN_GAMMA * (counter_u + np.uint64(1))
        x = x ^ (x >> _SHIFT_33)
        x = x * _MIX_C1
        x = x ^ (x >> _SHIFT_33)
        x = x * _MIX_C2
        x = x ^ (x >> _SHIFT_33)
    return np.asarray(x & np.uint64(0xFFFFFFFF)).astype(np.uint32)


def _lerp_f32(a: float, b: float, t: npt.ArrayLike) -> npt.NDArray[np.float32]:
    """float32 ``std::lerp``, following libstdc++'s branch structure.

    Reproducing the branches matters: the first (``a <= 0 <= b``) evaluates
    ``t*b + (1-t)*a``, the second ``a + t*(b-a)`` with a monotonicity clamp.
    A range straddling zero (the usual ``[-1, 1]`` weight init) takes the
    first; a positive-only range takes the second, and the two do not agree
    to the last bit. A port that only checks a signed range will pass while
    being wrong for every positive-only initialiser.

    Args:
        a: Range lower bound.
        b: Range upper bound.
        t: Interpolation parameter in ``[0, 1)``, float32.

    Returns:
        The interpolated values as float32.
    """
    a32 = np.float32(a)
    b32 = np.float32(b)
    t32 = np.asarray(t, dtype=np.float32)
    one = np.float32(1.0)

    if (a32 <= 0 and b32 >= 0) or (a32 >= 0 and b32 <= 0):
        return np.asarray(t32 * b32 + (one - t32) * a32).astype(np.float32)

    x = np.asarray(a32 + t32 * (b32 - a32)).astype(np.float32)
    # t is always in [0, 1) here, so `(t > 1) == (b > a)` reduces to
    # `False == (b > a)`, a compile-time-constant branch for a fixed range.
    if not (b32 > a32):
        clamped = np.where(b32 < x, x, b32)
    else:
        clamped = np.where(b32 > x, x, b32)
    # `if (t == 1) return b` -- unreachable for our t, kept for fidelity.
    return np.where(t32 == one, b32, clamped).astype(np.float32)


def uniform_real(
    seed: Counters,
    counter: Counters,
    lo: float = -1.0,
    hi: float = 1.0,
) -> npt.NDArray[np.float32]:
    """Reproduce ``plastax::UniformReal(seed, counter, lo, hi)``.

    Args:
        seed: The initialiser's seed.
        counter: The connection id (or an array of them).
        lo: Range lower bound.
        hi: Range upper bound.

    Returns:
        float32 sample(s) in ``[lo, hi)``, bit-identical to the C++ result.
    """
    x = mix_seed(seed, counter) % _MINSTD_M
    # The C++ guards a zero state, which minstd cannot leave.
    x = np.where(x == np.uint32(0), np.uint32(1), x).astype(np.uint32)
    x = ((_MINSTD_A * x.astype(np.uint64)) % _MINSTD_M.astype(np.uint64)).astype(
        np.uint32
    )
    t = ((x - np.uint32(1)).astype(np.float32) / _DRAW_DENOM).astype(np.float32)
    return _lerp_f32(lo, hi, t)


def fully_connected_weights(
    seed: int,
    n_src: int,
    n_dst: int,
    *,
    base_conn_id: int = 0,
    lo: float = -1.0,
    hi: float = 1.0,
) -> npt.NDArray[np.float32]:
    """Weights for one `plastax::FullyConnected` layer, as a ``(n_src, n_dst)`` matrix.

    Two details decide whether this lands the right weight on the right edge,
    and both are easy to get silently wrong:

    * **Allocation order is destination-major.** `FullyConnected::operator()`
      (``include/plastax/layers.hpp``) runs
      ``for dst in new_units: for src in prev_layer:``, so within a layer the
      connection id is ``base + dst_local * n_src + src_local``. plastax's
      `topology.dense` enumerates source-major, which is why this returns a
      matrix for the caller to index rather than a flat array.
    * **The counter is the *global* connection id.** `RandomUniformWeight`
      passes the allocator id straight through, and that id keeps counting
      across layers. So layer 1 of a 3->4->1 net starts at ``base_conn_id=12``,
      not 0, even though it has its own seed.

    Args:
        seed: The layer's initialiser seed.
        n_src: Number of source (previous-layer) units.
        n_dst: Number of destination (new-layer) units.
        base_conn_id: Global connection id of this layer's first edge.
        lo: Range lower bound.
        hi: Range upper bound.

    Returns:
        A ``(n_src, n_dst)`` float32 matrix, indexable as ``w[src, dst]``.
    """
    dst_idx, src_idx = np.meshgrid(
        np.arange(n_dst, dtype=np.uint64),
        np.arange(n_src, dtype=np.uint64),
        indexing="ij",
    )
    conn_ids = np.uint64(base_conn_id) + dst_idx * np.uint64(n_src) + src_idx
    # (n_dst, n_src) in allocation order -> (n_src, n_dst) for w[src, dst].
    return uniform_real(seed, conn_ids, lo, hi).T.copy()
