"""Standalone check: a unit-capacity net steps identically under Scheme-A.

Run directly or via `test_unit_capacity.py`, outside pytest's jaxtyping
instrumentation (see `sharding_equiv.py` for why).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import plastax as px
from test_unit_capacity import _N, _assert_same_live_state, _build, _inputs, _net

N_SHARDS = 4


def main() -> None:
    """Step a sharded and a single-device copy and compare the live state."""
    net = _net(capacity=_N + 3, growth="grid")
    static, state = _build(net, capacity_align=N_SHARDS)
    s_static, s_state = _build(
        net, capacity_align=N_SHARDS, sharding=px.ShardSpec("conns", N_SHARDS)
    )
    step = px.make_step(net, static)
    s_step = px.make_step(net, s_static)
    grew = 0
    for t in range(4):
        state = step(state, _inputs()).state
        s_state = s_step(s_state, _inputs()).state
        grew += int(state.grown)
        assert int(s_state.grown) == int(state.grown), t
        _assert_same_live_state(s_state, state, t)
    assert grew > 0
    pruned = s_state.units[px.PRUNED.name].tolist()
    assert pruned == [False] * _N + [True] * 3, pruned
    print("UNIT CAPACITY SHARDING PASS")


if __name__ == "__main__":
    main()
