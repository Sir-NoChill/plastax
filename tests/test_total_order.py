"""The two formulations of the total candidate order agree exactly.

`total_order` sorts with one four-key comparison sort, or, on a GPU backend
above `RADIX_TOTAL_ORDER_MIN` candidates, with three stable one-key passes
that XLA lowers to radix sorts. The choice must never change the order, so
both formulations are called directly here (on whatever backend runs the
tests) and compared, permutation for permutation, with each other and with a
numpy lexsort oracle. The scores mix heavy ties with NaN (either sign),
+-inf, +-0.0 and subnormals; the ids repeat heavily.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from plastax import phases
from plastax.phases import comparison_total_order, radix_total_order, total_order

_SPECIAL = np.array(
    [
        np.nan,
        -np.nan,
        np.inf,
        -np.inf,
        0.0,
        -0.0,
        1e-45,  # the smallest subnormal
        -1e-45,
        np.finfo(np.float32).max,
        -np.finfo(np.float32).max,
        1.0,
        -1.0,
    ],
    dtype=np.float32,
)


def _oracle(scores: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """The total order from its definition: (-score, src, dst, index).

    A backend that flushes subnormals to zero (XLA:CPU does) ties them with
    the zeros, in both formulations alike; the oracle follows the backend.
    """
    flushes = bool(jax.jit(lambda x: x == 0)(jnp.float32(1e-45)))
    if flushes:
        scores = np.where(np.abs(scores) < np.finfo(np.float32).tiny, 0, scores)
    neg = np.where(np.isnan(scores), np.float32(np.inf), -scores).astype(np.float32)
    neg = neg + np.float32(0.0)  # -0.0 + 0.0 is +0.0: the two zeros tie
    idx = np.arange(scores.shape[0])
    return np.lexsort((idx, dst, src, neg))


def _candidates(
    seed: int, n: int, distinct_scores: int, id_range: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pool = np.concatenate(
        [_SPECIAL, rng.standard_normal(distinct_scores).astype(np.float32)]
    )
    scores = pool[rng.integers(0, pool.shape[0], n)]
    src = rng.integers(0, id_range, n).astype(np.int32)
    dst = rng.integers(0, id_range, n).astype(np.int32)
    return scores, src, dst


@pytest.mark.parametrize(
    ("seed", "n", "distinct_scores", "id_range"),
    [
        (0, 1, 1, 1),
        (1, 2, 0, 1),
        (2, 17, 0, 2),
        (3, 1000, 0, 3),  # nothing but special values and near-total ties
        (4, 1000, 4, 1000),
        (5, 4096, 2, 4),
        (6, 4099, 50, 64),
        (7, 20000, 0, 7),
        (8, 65537, 1000, 1 << 16),
        (9, 131072, 3, 300),
        (10, 140001, 100000, 1 << 30),
    ],
)
def test_both_formulations_give_the_same_permutation(
    seed: int, n: int, distinct_scores: int, id_range: int
) -> None:
    scores, src, dst = _candidates(seed, n, distinct_scores, id_range)
    want = _oracle(scores, src, dst)
    args = (jnp.asarray(scores), jnp.asarray(src), jnp.asarray(dst))
    comparison = np.asarray(jax.jit(comparison_total_order)(*args))
    radix = np.asarray(jax.jit(radix_total_order)(*args))
    np.testing.assert_array_equal(comparison, want)
    np.testing.assert_array_equal(radix, want)
    np.testing.assert_array_equal(np.asarray(total_order(*args)), want)


def test_the_zeros_and_nans_tie_and_fall_back_to_src_dst_index() -> None:
    scores = np.array(
        [-0.0, 0.0, np.nan, -np.nan, -np.inf, 0.0, -0.0, np.nan], dtype=np.float32
    )
    src = np.array([1, 1, 0, 0, 0, 0, 1, 0], dtype=np.int32)
    dst = np.array([2, 2, 5, 3, 0, 9, 1, 3], dtype=np.int32)
    args = (jnp.asarray(scores), jnp.asarray(src), jnp.asarray(dst))
    # zeros first (src 0 before src 1, then dst, then index), then -inf (as
    # +inf after negation) tied with the NaNs, ordered by src, dst, index.
    want = [5, 6, 0, 1, 4, 3, 7, 2]
    assert np.asarray(comparison_total_order(*args)).tolist() == want
    assert np.asarray(radix_total_order(*args)).tolist() == want


def _sorts(fn: Any, n: int) -> list[int]:
    """The `num_keys` of every sort in `fn`'s jaxpr over n candidates."""
    scores = jax.ShapeDtypeStruct((n,), jnp.float32)
    ids = jax.ShapeDtypeStruct((n,), jnp.int32)
    # A fresh wrapper per call: make_jaxpr caches traces by function.
    jaxpr = jax.make_jaxpr(lambda *a: fn(*a))(scores, ids, ids)
    return [e.params["num_keys"] for e in jaxpr.eqns if e.primitive.name == "sort"]


def test_total_order_takes_the_radix_passes_only_on_a_gpu_above_the_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    big = phases.RADIX_TOTAL_ORDER_MIN
    small = big - 1
    if jax.default_backend() != "gpu":
        assert _sorts(total_order, big) == [4]
        monkeypatch.setattr(phases.jax, "default_backend", lambda: "gpu")
    assert _sorts(total_order, small) == [4]
    assert _sorts(total_order, big) == [1, 1, 1]
