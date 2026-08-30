"""The Driver's structural-event protocol works under Scheme-A sharding.

Checked in a clean subprocess (no jaxtyping instrumentation -- shard_map is
incompatible with it; see test_sharding.py). The Driver reads device scalars to
host and drives Python control flow to grow buckets on overflow and resort on
level reassignment. Under single-controller shard_map the state is a global
sharded jax.Array, so those reads reduce globally and the decisions match a
single-device Driver exactly. This pins that:

  * OVERFLOW: a grow-heavy churn driven to completion (add_conn overflows ->
    grow_bucket -> retrace, repeatedly) reaches the same net sharded vs single;
  * RESORT: topo.resort on a sharded state (recompute_levels + compacting
    scatter + sort) produces the same capacities, levels, and conn columns.

This is the single-controller (multi-device) case. True multi-controller
(jax.distributed, one process per node) additionally needs the state to be a
global array across processes; the host-side logic here is the same SPMD path,
validated separately on a multi-node run.

Run directly (`python tests/sharding_driver_equiv.py`) or via the wrapper in
test_sharding_driver.py.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import dataclasses
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

from dst_sparse import (
    _one_hot,
    _sample,
    build_sparse_mlp,
    make_net,
    teacher_task,
)
from mlp_xor import GradPreAct

import plastax as px
from plastax import topo

N_SHARDS = 4
_OPT = px.optim.adam(0.05, GradPreAct)
_LAYERS = (33, 128, 128, 10)
_BUDGETS = (512, 512, 128)


def _copy(state: px.NetworkState[None]) -> px.NetworkState[None]:
    return jax.tree_util.tree_map(lambda x: jnp.array(x), state)


def _edge_set(state: px.NetworkState[None]) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for bucket in state.conns:
        live = ~np.asarray(bucket["dead"])
        fr = np.asarray(bucket["from_id"])[live]
        to = np.asarray(bucket["to_id"])[live]
        pairs.update(zip(fr.tolist(), to.tolist(), strict=True))
    return pairs


def _build(sharded: bool) -> tuple[px.NetworkStatic, px.NetworkState[None]]:
    static, state = build_sparse_mlp(
        make_net(_OPT, method="set", mode="train"), _LAYERS, _BUDGETS, seed=0
    )
    if sharded:
        static = dataclasses.replace(static, sharding=px.ShardSpec("shard", N_SHARDS))
    return static, state


def _warm(
    static: px.NetworkStatic, state: px.NetworkState[None]
) -> px.NetworkState[None]:
    step = px.make_step(make_net(_OPT, method="set", mode="train"), static)
    teacher, rng = teacher_task(_LAYERS[0] - 1, _LAYERS[-1], 0)
    inp = None
    for _ in range(6):
        inp, label = _sample(teacher, rng)
        state = step(
            state, px.StepInputs(inputs=inp, targets=_one_hot(label, _LAYERS[-1]))
        ).state
    return state


def _run_growth_driver(
    static: px.NetworkStatic, state: px.NetworkState[None]
) -> tuple[int, tuple[int, ...], set[tuple[int, int]]]:
    """Grow-heavy churn via the Driver: overflow -> grow_bucket -> retrace."""
    # grow-heavy but bounded, so growth overflows into a couple of grow_bucket
    # rounds (enough to exercise the retrace path) without many recompiles.
    churn_net = make_net(_OPT, method="set", mode="churn", zeta=0.1, max_candidates=256)
    driver = px.Driver(churn_net, static, state)
    sp = px.StepInputs(inputs=jnp.zeros((_LAYERS[0],), jnp.float32), targets=None)
    for _ in range(3):
        driver.step(sp)
    return (
        int(px.state.live_conn_count(driver.state)),
        tuple(driver.static.level_capacities),
        _edge_set(driver.state),
    )


def _check_overflow_grow_shards() -> None:
    """A grow-heavy churn driven through grow_bucket matches single-device."""
    s_static, s_state = _build(False)
    s_state = _warm(s_static, s_state)
    initial_caps = tuple(s_static.level_capacities)
    s_live, s_caps, s_edges = _run_growth_driver(s_static, _copy(s_state))

    h_static, _ = _build(True)
    h_live, h_caps, h_edges = _run_growth_driver(h_static, _copy(s_state))

    if s_caps == initial_caps:
        raise AssertionError(
            f"overflow: no grow_bucket fired (caps stayed {initial_caps}); the "
            "scenario no longer exercises the overflow path -- retune it"
        )
    if s_live != h_live:
        raise AssertionError(f"overflow: live differs single={s_live} sharded={h_live}")
    if s_caps != h_caps:
        raise AssertionError(f"overflow: caps differ single={s_caps} sharded={h_caps}")
    if s_edges != h_edges:
        raise AssertionError("overflow: grown edge set differs sharded vs single")


def _check_resort_shards() -> None:
    """topo.resort on a sharded state matches single-device."""

    def resort_of(
        sharded: bool,
    ) -> tuple[tuple[int, ...], np.ndarray, list[dict[str, np.ndarray]]]:
        static, state = _build(sharded)
        step = px.make_step(make_net(_OPT, method="set", mode="train"), static)
        teacher, rng = teacher_task(_LAYERS[0] - 1, _LAYERS[-1], 0)
        inp, label = _sample(teacher, rng)
        state = step(
            state, px.StepInputs(inputs=inp, targets=_one_hot(label, _LAYERS[-1]))
        ).state
        new_static, new_state = topo.resort(static, state)
        conns = [{n: np.asarray(c) for n, c in b.items()} for b in new_state.conns]
        return (
            new_static.level_capacities,
            np.asarray(new_state.units[px.LEVEL.name]),
            conns,
        )

    s_caps, s_levels, s_conns = resort_of(False)
    h_caps, h_levels, h_conns = resort_of(True)

    if s_caps != h_caps:
        raise AssertionError(f"resort: caps differ single={s_caps} sharded={h_caps}")
    if not np.array_equal(s_levels, h_levels):
        raise AssertionError("resort: recomputed levels differ sharded vs single")
    if len(s_conns) != len(h_conns) or not all(
        set(a) == set(b)
        and all(np.allclose(a[n], b[n], atol=1e-5, equal_nan=True) for n in a)
        for a, b in zip(s_conns, h_conns, strict=True)
    ):
        raise AssertionError("resort: conn columns differ sharded vs single")


def main() -> None:
    """Run the Driver-sharding checks and print the pass sentinel."""
    if len(jax.devices()) < N_SHARDS:
        raise SystemExit(f"need >= {N_SHARDS} devices, got {len(jax.devices())}")
    _check_overflow_grow_shards()
    print("OK overflow -> grow_bucket -> retrace shards (== single-device)")
    _check_resort_shards()
    print("OK resort shards (== single-device)")
    print("DRIVER SHARDING CHECK PASS")


if __name__ == "__main__":
    main()
