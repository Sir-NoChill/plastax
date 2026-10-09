"""Counter-based framework RNG: Philox-4x32-10, shared with plastax-cpp.

Every draw is a pure function of ``(seed, counter)``: the same pair gives the
same bits on CPU, GPU and TPU, under ``jit``, ``vmap`` and ``shard_map``, and
in plastax-cpp. The core is a bit-exact port of plastax-cpp's ``Philox32``
(``include/plastax/random.hpp``): Philox-4x32-10 with the key loaded from the
64-bit seed, the counter block loaded from ``(counter, seed)``, and the first
output word returned.

Cross-library contract
----------------------

This docstring is the normative spec; plastax-cpp mirrors it. All quantities
are unsigned; 64-bit values are handled as ``(hi, lo)`` uint32 limb pairs so
the implementation never needs 64-bit arrays.

1. ``philox32(seed, counter) -> uint32``: the core generator, bit-exact with
   plastax-cpp ``Philox32(Seed, Counter)``.
2. ``unit_float(word) = float32(word >> 8) * 2**-24``: uint32 to [0, 1) with
   24 bits of mantissa, bit-exact with plastax-cpp ``UnitFloat``.
3. A proposal site is addressed by ``(network_seed, step, stream,
   proposer_key, j)``:

   - ``network_seed``: ``Network.seed``, 64-bit.
   - ``step``: the framework step counter (``NetworkState.step``), 32-bit.
   - ``stream``: the phase stream id, 8-bit; growth draws on stream 1.
   - ``proposer_key``: one uint32 word for the proposing entity: the unit id
     for per-unit proposers; ``philox32((src << 32) | dst, occurrence)`` for
     per-connection proposers (``conn_key``); 0 for a global proposer.
   - ``j``: the proposal index within the proposer, < 2**32.

   The site's draw stream is::

       site_seed = (philox32(network_seed, (step << 8) | stream) << 32)
                   | proposer_key                       # 64-bit
       draw(sub) = philox32(site_seed, (j << 16) | sub) # sub < 2**16

   where ``sub`` is the per-site sub-counter: every ``uniform`` /
   ``uniform_int`` / ``bernoulli`` call consumes one sub-counter and
   ``normal`` consumes two, in call order, starting at 0.
4. Derived draws (all consume ``uniform`` words as in plastax-cpp):

   - ``uniform() = unit_float(draw(sub))``, in [0, 1).
   - ``uniform_int(n) = min(uint32(uniform() * n), n - 1)`` — the float path
     plastax-cpp uses in ``SampleKofN``; the clamp guards the ``u -> 1.0``
     rounding edge. Uniform enough for n << 2**24.
   - ``normal() = sqrt(-2 ln(1 - u1)) * cos(2 pi u2)`` with ``u1`` at
     ``sub`` and ``u2`` at ``sub + 1`` (deterministic Box-Muller;
     ``1 - u1 > 0`` always, so the log is finite). The formula is shared
     bit-for-bit at the uniform-word level; the transcendental result may
     differ from plastax-cpp by float32 libm ULPs across platforms.
   - ``bernoulli(p) = uniform() < p``.

The sub-counter advances at trace time (a Python int), so a rule body makes a
fixed number of draws per call site — which is exactly the counter-based
model: the jaxpr bakes in one ``(site, sub)`` address per textual draw.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float32, Int32, UInt32

__all__ = ["Rng", "conn_key", "philox32", "unit_float"]

_MASK16 = jnp.uint32(0xFFFF)
_M0 = jnp.uint32(0xD2511F53)
_M1 = jnp.uint32(0xCD9E8D57)
_W0 = jnp.uint32(0x9E3779B9)
_W1 = jnp.uint32(0xBB67AE85)


def _mulhi32(a: UInt32[Array, ...], b: UInt32[Array, ...]) -> UInt32[Array, ...]:
    """High 32 bits of the 32x32 product, in pure uint32 (no uint64).

    Args:
        a: left factor.
        b: right factor.

    Returns:
        ``(a * b) >> 32`` as uint32.
    """
    a_hi, a_lo = a >> 16, a & _MASK16
    b_hi, b_lo = b >> 16, b & _MASK16
    lo = a_lo * b_lo
    mid1 = a_hi * b_lo
    mid2 = a_lo * b_hi
    # Carries out of the low 32 bits: each term is < 2**16 after the shifts
    # and masks, so the sum fits in uint32.
    carry = ((lo >> 16) + (mid1 & _MASK16) + (mid2 & _MASK16)) >> 16
    return jnp.asarray(a_hi * b_hi + (mid1 >> 16) + (mid2 >> 16) + carry)


def _split64(value: int) -> tuple[UInt32[Array, ""], UInt32[Array, ""]]:
    """Split a Python int (taken mod 2**64) into (hi, lo) uint32 limbs.

    Args:
        value: the 64-bit quantity (wider ints are truncated mod 2**64).

    Returns:
        The (hi, lo) uint32 limb pair.
    """
    value &= (1 << 64) - 1
    return jnp.uint32(value >> 32), jnp.uint32(value & 0xFFFFFFFF)


def philox32(
    seed_hi: UInt32[Array, ...],
    seed_lo: UInt32[Array, ...],
    counter_hi: UInt32[Array, ...],
    counter_lo: UInt32[Array, ...],
) -> UInt32[Array, ...]:
    """Philox-4x32-10 first output word, bit-exact with plastax-cpp.

    The 64-bit seed and counter arrive as ``(hi, lo)`` uint32 limbs. The
    counter block is loaded as ``(counter_lo, counter_hi, seed_lo, seed_hi)``
    and the key as ``(seed_lo, seed_hi)``; ten rounds with the standard Weyl
    key schedule.

    Args:
        seed_hi: seed bits 32-63.
        seed_lo: seed bits 0-31.
        counter_hi: counter bits 32-63.
        counter_lo: counter bits 0-31.

    Returns:
        The first Philox output word as uint32.
    """
    x0, x1, x2, x3 = counter_lo, counter_hi, seed_lo, seed_hi
    k0, k1 = seed_lo, seed_hi
    for _ in range(10):
        lo0 = _M0 * x0
        hi0 = _mulhi32(_M0, x0)
        lo1 = _M1 * x2
        hi1 = _mulhi32(_M1, x2)
        x0, x1, x2, x3 = hi1 ^ x1 ^ k0, lo1, hi0 ^ x3 ^ k1, lo0
        k0 = k0 + _W0
        k1 = k1 + _W1
    return x0


def unit_float(word: UInt32[Array, ...]) -> Float32[Array, ...]:
    """Map a uint32 word to [0, 1) with 24-bit precision (cpp ``UnitFloat``).

    Args:
        word: the Philox output word.

    Returns:
        ``float32(word >> 8) * 2**-24``.
    """
    return jnp.asarray((word >> 8).astype(jnp.float32) * jnp.float32(1.0 / 16777216.0))


def conn_key(
    src: UInt32[Array, ...] | Int32[Array, ...],
    dst: UInt32[Array, ...] | Int32[Array, ...],
    occurrence: UInt32[Array, ...] | Int32[Array, ...],
) -> UInt32[Array, ...]:
    """The proposer-key word of a connection site.

    ``philox32((src << 32) | dst, occurrence)``: src is the seed's high limb,
    dst the low limb, and the parallel-edge occurrence index the counter.

    Args:
        src: source unit id.
        dst: destination unit id.
        occurrence: parallel-edge occurrence index (0 for the first edge).

    Returns:
        A uint32 proposer key.
    """
    occ = occurrence.astype(jnp.uint32)
    return philox32(
        src.astype(jnp.uint32),
        dst.astype(jnp.uint32),
        jnp.zeros_like(occ),
        occ,
    )


@dataclasses.dataclass
class Rng:
    """One proposal site's draw stream (see the module docstring, item 3).

    Constructed by the framework via :meth:`for_site` and handed to proposal
    rules. Each ``uniform`` / ``uniform_int`` / ``bernoulli`` call consumes
    one sub-counter and ``normal`` two, advancing at trace time, so a rule
    makes a fixed number of draws per call site.

    Attributes:
        seed_hi: site-seed bits 32-63 (the keyed step/stream word).
        seed_lo: site-seed bits 0-31 (the proposer key).
        base_hi: draw-counter bits 32-63 (``j >> 16``).
        base_lo: draw-counter bits 0-31 (``(j & 0xFFFF) << 16``); the low 16
            bits hold the sub-counter.
    """

    seed_hi: UInt32[Array, ...]
    seed_lo: UInt32[Array, ...]
    base_hi: UInt32[Array, ...]
    base_lo: UInt32[Array, ...]
    _sub: int = 0

    @classmethod
    def for_site(
        cls,
        network_seed: int,
        step: Int32[Array, ...] | UInt32[Array, ...],
        stream: int,
        proposer_key: UInt32[Array, ...],
        j: UInt32[Array, ...] | Int32[Array, ...],
    ) -> Rng:
        """Build the Rng of one proposal site.

        Args:
            network_seed: ``Network.seed`` (64-bit, a static Python int).
            step: the framework step counter (``NetworkState.step``).
            stream: the phase stream id (static; growth is stream 1).
            proposer_key: the proposing entity's key word (unit id,
                :func:`conn_key`, or 0 for a global proposer).
            j: the proposal index within the proposer.

        Returns:
            The site's draw stream, with the sub-counter at 0.
        """
        ns_hi, ns_lo = _split64(network_seed)
        step_u = step.astype(jnp.uint32)
        # (step << 8) | stream, as 64-bit limbs: step contributes 32 bits, so
        # its top 8 bits carry into the high limb.
        ctr_hi = step_u >> 24
        ctr_lo = (step_u << 8) | jnp.uint32(stream & 0xFF)
        keyed = philox32(
            jnp.broadcast_to(ns_hi, ctr_lo.shape),
            jnp.broadcast_to(ns_lo, ctr_lo.shape),
            ctr_hi,
            ctr_lo,
        )
        j_u = j.astype(jnp.uint32)
        return cls(
            seed_hi=keyed,
            seed_lo=jnp.broadcast_to(proposer_key.astype(jnp.uint32), keyed.shape),
            base_hi=j_u >> 16,
            base_lo=j_u << 16,
        )

    def _draw(self) -> UInt32[Array, ...]:
        """One raw word at the current sub-counter, which then advances."""
        word = philox32(
            self.seed_hi,
            self.seed_lo,
            self.base_hi,
            self.base_lo | jnp.uint32(self._sub),
        )
        self._sub += 1
        if self._sub >= 1 << 16:
            raise ValueError("Rng sub-counter exhausted (>= 2**16 draws)")
        return word

    def uniform(self) -> Float32[Array, ...]:
        """One float32 draw in [0, 1); consumes one sub-counter.

        Returns:
            ``unit_float`` of the next word.
        """
        return unit_float(self._draw())

    def uniform_int(self, n: int) -> UInt32[Array, ...]:
        """One integer draw in [0, n); consumes one sub-counter.

        The float path of plastax-cpp's ``SampleKofN``: ``uint32(u * n)``
        clamped to ``n - 1`` against the ``u -> 1.0`` rounding edge.

        Args:
            n: exclusive upper bound, ``1 <= n``; uniform enough for
                ``n << 2**24``.

        Returns:
            A uint32 draw in [0, n).
        """
        scaled = (self.uniform() * jnp.float32(n)).astype(jnp.uint32)
        return jnp.minimum(scaled, jnp.uint32(n - 1))

    def normal(self) -> Float32[Array, ...]:
        """One standard-normal draw; consumes two sub-counters.

        Deterministic Box-Muller on the next two uniforms (the module
        docstring, item 4): bit-exact at the uniform-word level, float32
        libm ULP differences possible across platforms.

        Returns:
            A float32 standard-normal draw.
        """
        u1 = self.uniform()
        u2 = self.uniform()
        r = jnp.sqrt(jnp.float32(-2.0) * jnp.log(jnp.float32(1.0) - u1))
        return r * jnp.cos(jnp.float32(6.28318530717958648) * u2)

    def bernoulli(self, p: Float32[Array, ...] | float) -> Bool[Array, ...]:
        """One Bernoulli(p) draw; consumes one sub-counter.

        Args:
            p: success probability.

        Returns:
            ``uniform() < p``.
        """
        return jnp.asarray(self.uniform() < jnp.float32(p))
