"""Host-side balanced level partition for Scheme-B (band-pipeline) sharding.

Pure Python/numpy -- no JAX. Chooses contiguous level-band boundaries that
minimize the maximum per-shard work, recomputed on resort.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def balanced_level_cut(
    weights: Sequence[float] | np.ndarray,
    num_shards: int,
    *,
    cross_band_edges: np.ndarray | None = None,
) -> tuple[int, ...]:
    """Cut a level sequence into contiguous bands minimizing the bottleneck.

    Classic min-max contiguous partition ("linear partition" / "painter's
    partition"), solved by DP: `dp[k][i] = min over j<i of
    max(dp[k-1][j], sum(weights[j:i]))`, with `dp[1][i] = sum(weights[0:i])`
    and boundaries reconstructed from the argmin. Among partitions tied on
    the bottleneck, a second DP pass restricted to bottleneck-optimal splits
    minimizes total cross-band edges.

    Args:
        weights: per-level work, length L, non-negative.
        num_shards: number of contiguous bands to split into; must satisfy
            1 <= num_shards <= L.
        cross_band_edges: optional (L+1,) array where entry p counts edges
            crossing a cut placed before level p (between level p-1 and p).
            Used only to break ties among bottleneck-optimal partitions; if
            None, ties are broken arbitrarily (by DP order).

    Returns:
        Band boundaries b with len(b) == num_shards + 1, 0 == b[0] < b[1]
        < ... < b[num_shards] == L; band d owns levels [b[d], b[d + 1]).

    Raises:
        ValueError: weights is empty, or num_shards is not in
            [1, len(weights)].
    """
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1:
        raise ValueError(f"balanced_level_cut: weights must be 1-D, got {w.ndim}-D")
    length = int(w.shape[0])
    if length == 0:
        raise ValueError("balanced_level_cut: weights is empty")
    if not (1 <= num_shards <= length):
        raise ValueError(
            f"balanced_level_cut: num_shards={num_shards} must satisfy "
            f"1 <= num_shards <= len(weights)={length}"
        )

    prefix = np.concatenate(([0.0], np.cumsum(w)))

    if num_shards == 1:
        return (0, length)
    if num_shards == length:
        return tuple(range(length + 1))

    # dp[i] = min achievable bottleneck partitioning levels [0, i) into the
    # current band count k; choice[k][i] = the argmin split point j.
    dp_prev = prefix[: length + 1].copy()  # k == 1
    choice = np.zeros((num_shards + 1, length + 1), dtype=np.int64)

    for k in range(2, num_shards + 1):
        dp_cur = np.full(length + 1, np.inf, dtype=np.float64)
        # i must leave >= (num_shards - k) levels for the remaining bands,
        # and >= k levels total to keep every band non-empty.
        for i in range(k, length - (num_shards - k) + 1):
            best_val = np.inf
            best_j = k - 1
            # j must leave a non-empty band [j, i) and enough head room.
            for j in range(k - 1, i):
                band_weight = prefix[i] - prefix[j]
                candidate = max(dp_prev[j], band_weight)
                if candidate < best_val:
                    best_val = candidate
                    best_j = j
            dp_cur[i] = best_val
            choice[k, i] = best_j
        dp_prev = dp_cur

    bottleneck = dp_prev[length]

    if cross_band_edges is not None:
        boundaries = _tie_break_min_comm(
            w, prefix, num_shards, bottleneck, cross_band_edges
        )
    else:
        boundaries = _reconstruct(choice, num_shards, length)

    return boundaries


def _reconstruct(choice: np.ndarray, num_shards: int, length: int) -> tuple[int, ...]:
    """Walk the DP choice table back from the final index to boundaries."""
    bounds = [length]
    i = length
    for k in range(num_shards, 1, -1):
        j = int(choice[k, i])
        bounds.append(j)
        i = j
    bounds.append(0)
    bounds.reverse()
    return tuple(bounds)


def _tie_break_min_comm(
    w: np.ndarray,
    prefix: np.ndarray,
    num_shards: int,
    bottleneck: float,
    cross_band_edges: np.ndarray,
) -> tuple[int, ...]:
    """Second DP pass: among bottleneck-optimal splits, minimize crossings.

    Restricts each transition to bands whose weight does not exceed the
    already-known optimal bottleneck (with float tolerance), then minimizes
    the summed `cross_band_edges` at interior boundaries via ordinary
    shortest-path DP.
    """
    length = w.shape[0]
    edges = np.asarray(cross_band_edges, dtype=np.float64)
    if edges.shape != (length + 1,):
        raise ValueError(
            "balanced_level_cut: cross_band_edges must have shape "
            f"({length + 1},), got {edges.shape}"
        )
    tol = 1e-9 * max(1.0, bottleneck)

    # k == 1: a single band [0, i) never crosses an interior boundary, so
    # its communication cost is 0 wherever that band is bottleneck-feasible.
    comm_prev = np.zeros(length + 1, dtype=np.float64)
    for i in range(length + 1):
        if prefix[i] - prefix[0] > bottleneck + tol:
            comm_prev[i] = np.inf
    choice = np.zeros((num_shards + 1, length + 1), dtype=np.int64)

    for k in range(2, num_shards + 1):
        comm_cur = np.full(length + 1, np.inf, dtype=np.float64)
        for i in range(k, length - (num_shards - k) + 1):
            best_cost = np.inf
            best_j = k - 1
            for j in range(k - 1, i):
                if comm_prev[j] == np.inf:
                    continue
                band_weight = prefix[i] - prefix[j]
                if band_weight > bottleneck + tol:
                    continue
                cost = comm_prev[j] + edges[j]
                if cost < best_cost:
                    best_cost = cost
                    best_j = j
            comm_cur[i] = best_cost
            choice[k, i] = best_j
        comm_prev = comm_cur

    return _reconstruct(choice, num_shards, length)


def level_to_shard(boundaries: Sequence[int]) -> np.ndarray:
    """Expand band boundaries into a per-level shard-id array.

    Args:
        boundaries: band boundaries as returned by balanced_level_cut.

    Returns:
        Per-level shard id, int array of length boundaries[-1].
    """
    b = list(boundaries)
    length = b[-1]
    out = np.zeros(length, dtype=np.int64)
    for shard_id in range(len(b) - 1):
        out[b[shard_id] : b[shard_id + 1]] = shard_id
    return out


def band_weights(
    weights: Sequence[float] | np.ndarray, boundaries: Sequence[int]
) -> list[float]:
    """Sum weights within each band.

    Args:
        weights: per-level work, length L, non-negative.
        boundaries: band boundaries as returned by balanced_level_cut.

    Returns:
        Per-band total weight, one entry per band.
    """
    w = np.asarray(weights, dtype=np.float64)
    b = list(boundaries)
    return [float(w[b[d] : b[d + 1]].sum()) for d in range(len(b) - 1)]
