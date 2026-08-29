"""topo.initial_levels: vectorized frontier-Kahn == scalar longest-path.

The frontier pass and its deep-graph fallback must both return the exact
longest-path levels initial_levels has always produced. Pinned against an
independent relaxation oracle over random DAGs, the extracted scalar
_kahn_levels, the deep-chain fallback, cycle rejection, and the allow_cycles
best-effort path.
"""

from __future__ import annotations

import numpy as np
import pytest

from plastax import topo
from plastax.topo import _MAX_VECTORIZED_LEVEL_ROUNDS, _kahn_levels


def _reference_levels(num_units: int, edges: np.ndarray) -> np.ndarray:
    """Independent longest-path oracle: relax every edge to a fixpoint.

    Neither Kahn nor frontier-based -- a plain Bellman-Ford-style relaxation,
    so agreement with initial_levels is real cross-checking, not a tautology.
    """
    levels = np.zeros(num_units, dtype=np.int32)
    if edges.shape[0] == 0:
        return levels
    srcs = edges[:, 0].tolist()
    dsts = edges[:, 1].tolist()
    for _ in range(num_units + 1):
        changed = False
        for u, v in zip(srcs, dsts, strict=True):
            if levels[u] + 1 > levels[v]:
                levels[v] = levels[u] + 1
                changed = True
        if not changed:
            break
    return levels


def _random_dag(rng: np.random.Generator, num_units: int, num_edges: int) -> np.ndarray:
    """Distinct random edges that only go forward in a random order -> a DAG."""
    order = rng.permutation(num_units)
    seen: set[tuple[int, int]] = set()
    for _ in range(num_edges * 3):
        if len(seen) >= num_edges:
            break
        a, b = (int(x) for x in rng.integers(0, num_units, size=2))
        if a == b:
            continue
        i, j = (a, b) if a < b else (b, a)
        seen.add((int(order[i]), int(order[j])))
    if not seen:
        return np.zeros((0, 2), dtype=np.int32)
    return np.array(sorted(seen), dtype=np.int32)


@pytest.mark.parametrize("seed", range(8))
def test_frontier_matches_reference_and_scalar_on_random_dags(seed: int) -> None:
    """On shallow random DAGs (vectorized path) all three agree exactly."""
    rng = np.random.default_rng(seed)
    for _ in range(6):
        n = int(rng.integers(2, 40))  # depth < cap -> exercises the frontier pass
        m = int(rng.integers(0, 2 * n))
        edges = _random_dag(rng, n, m)
        got = topo.initial_levels(n, edges)
        assert got.dtype == np.int32
        assert np.array_equal(got, _reference_levels(n, edges))
        assert np.array_equal(got, _kahn_levels(n, edges, allow_cycles=False))


def test_layered_net_multi_round_frontier_matches_reference() -> None:
    """A 4-layer net drives several frontier rounds; still matches the oracle."""
    rng = np.random.default_rng(1)
    sizes = [10, 20, 20, 5]
    offsets = np.cumsum([0, *sizes])
    n = int(offsets[-1])
    pairs: list[tuple[int, int]] = []
    for layer in range(len(sizes) - 1):
        for _ in range(40):
            u = int(rng.integers(offsets[layer], offsets[layer + 1]))
            v = int(rng.integers(offsets[layer + 1], offsets[layer + 2]))
            pairs.append((u, v))
    edges = np.array(sorted(set(pairs)), dtype=np.int32)
    got = topo.initial_levels(n, edges)
    assert np.array_equal(got, _reference_levels(n, edges))


def test_deep_chain_falls_back_to_scalar_and_is_correct() -> None:
    """A chain deeper than the round cap takes the fallback; levels stay exact."""
    n = _MAX_VECTORIZED_LEVEL_ROUNDS + 40  # depth n-1 > cap -> fallback path
    edges = np.array([[i, i + 1] for i in range(n - 1)], dtype=np.int32)
    got = topo.initial_levels(n, edges)
    assert np.array_equal(got, np.arange(n, dtype=np.int32))
    assert np.array_equal(got, _kahn_levels(n, edges, allow_cycles=False))


def test_empty_edges_all_zero() -> None:
    got = topo.initial_levels(5, np.zeros((0, 2), dtype=np.int32))
    assert np.array_equal(got, np.zeros(5, dtype=np.int32))


def test_raises_on_cycle_without_allow_cycles() -> None:
    edges = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int32)  # 3-cycle
    with pytest.raises(ValueError):
        topo.initial_levels(3, edges)


def test_allow_cycles_matches_scalar_best_effort() -> None:
    """allow_cycles keeps cycle units at their acyclic-predecessor level.

    Feeder 3->0 with a 0->1->2->0 cycle: unit 0 gets level 1 from the feeder,
    the rest stay 0 (their only predecessors are unsettled cycle units) --
    byte-identical to the scalar Kahn's best-effort partial.
    """
    edges = np.array([[3, 0], [0, 1], [1, 2], [2, 0]], dtype=np.int32)
    got = topo.initial_levels(4, edges, allow_cycles=True)
    assert np.array_equal(got, _kahn_levels(4, edges, allow_cycles=True))
    assert got.tolist() == [1, 0, 0, 0]


def test_duplicate_edges_are_handled_like_the_scalar_path() -> None:
    """A repeated edge must not break in-degree bookkeeping in either path."""
    edges = np.array([[0, 1], [0, 1], [1, 2]], dtype=np.int32)
    got = topo.initial_levels(3, edges)
    assert np.array_equal(got, _kahn_levels(3, edges, allow_cycles=False))
    assert got.tolist() == [0, 1, 2]
