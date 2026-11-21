"""Balanced level-cut DP (Scheme-B host-side sharding)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from plastax.shard import balanced_level_cut, band_weights


def _brute_force_bottleneck(weights: np.ndarray, num_shards: int) -> float:
    """Exhaustive min-max over all contiguous partitions -- test oracle."""
    length = len(weights)
    best = np.inf
    # Choose num_shards - 1 interior cut positions from {1, ..., length-1}.
    for cuts in itertools.combinations(range(1, length), num_shards - 1):
        bounds = (0, *cuts, length)
        weight = max(
            weights[bounds[d] : bounds[d + 1]].sum() for d in range(num_shards)
        )
        best = min(best, weight)
    return float(best)


# --- core DP -----------------------------------------------------------


@pytest.mark.parametrize("seed", range(20))
def test_bottleneck_matches_brute_force_oracle(seed: int) -> None:
    rng = np.random.default_rng(seed)
    length = int(rng.integers(2, 7))
    num_shards = int(rng.integers(1, length + 1))
    weights = rng.uniform(0.0, 10.0, size=length)

    boundaries = balanced_level_cut(weights, num_shards)
    assert boundaries[0] == 0
    assert boundaries[-1] == length
    assert len(boundaries) == num_shards + 1
    assert all(a < b for a, b in itertools.pairwise(boundaries))

    achieved = max(band_weights(weights, boundaries))
    expected = _brute_force_bottleneck(weights, num_shards)
    assert achieved == pytest.approx(expected)


def test_bottleneck_matches_brute_force_with_zero_weights() -> None:
    weights = np.array([0.0, 5.0, 0.0, 0.0, 3.0, 0.0])
    num_shards = 3
    boundaries = balanced_level_cut(weights, num_shards)
    achieved = max(band_weights(weights, boundaries))
    expected = _brute_force_bottleneck(weights, num_shards)
    assert achieved == pytest.approx(expected)


# --- tie-break with cross_band_edges ------------------------------------


def test_tie_break_three_shards_selects_min_comm_boundaries() -> None:
    weights = [1.0] * 9
    # All splits into 3 equal bands of weight 3 tie the bottleneck; only
    # boundaries at (3, 6) are cheap in the crossing-edge cost.
    edges = np.zeros(10)
    edges[3] = 1.0
    edges[6] = 1.0
    # Every other interior position is expensive.
    for p in range(1, 9):
        if p not in (3, 6):
            edges[p] = 50.0
    boundaries = balanced_level_cut(weights, 3, cross_band_edges=edges)
    assert boundaries == (0, 3, 6, 9)


def test_tie_break_does_not_override_bottleneck() -> None:
    # The lowest-communication split is NOT bottleneck-optimal; the
    # bottleneck must still win.
    weights = [10.0, 1.0, 1.0, 1.0]
    edges = np.array([0.0, 0.0, 100.0, 100.0, 0.0])  # cheapest cut at p=1
    boundaries = balanced_level_cut(weights, 2, cross_band_edges=edges)
    # Optimal bottleneck is achieved only by cutting after the heavy level.
    assert boundaries == (0, 1, 4)
    assert max(band_weights(weights, boundaries)) == 10.0


def test_cross_band_edges_wrong_shape_raises() -> None:
    with pytest.raises(ValueError):
        balanced_level_cut([1.0] * 4, 2, cross_band_edges=np.zeros(3))


# --- errors ---------------------------------------------------------------


def test_empty_weights_raises() -> None:
    with pytest.raises(ValueError):
        balanced_level_cut([], 1)


def test_num_shards_zero_raises() -> None:
    with pytest.raises(ValueError):
        balanced_level_cut([1.0, 2.0], 0)


def test_num_shards_exceeds_length_raises() -> None:
    with pytest.raises(ValueError):
        balanced_level_cut([1.0, 2.0], 3)


def test_non_1d_weights_raises() -> None:
    with pytest.raises(ValueError):
        balanced_level_cut(np.zeros((2, 2)), 1)


# --- helpers ----------------------------------------------------------------
