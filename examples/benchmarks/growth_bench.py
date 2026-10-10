"""Time one connection-growth phase per growth strategy (CPU or GPU).

The plastax counterpart of plastax-cpp's ``benchmarks/bench_growth.cpp``,
with the same network, rules and sweeps so the two libraries compare point
for point.

Network: four levels of N/4 units each, with C live connections split
evenly over the three adjacent level pairs (random endpoints, parallel edges
allowed). Levels are recomputed from the edges at build time, so a unit with
no incoming edge sits at level 0.

Strategies (every rule uses top_k selection of 16 per source level and a
``max_level_gap`` spanning all four levels, so nearly every candidate passes
the window):

- ``per_unit``: `ProposeAddConn`, ``proposer="per_unit"`` (the default).
  Every unit proposes P edges into a random unit of the next built level,
  each with a random score.
- ``per_connection``: the same proposal from each live connection's source.
- ``global``: P proposals from random sources.
- ``exhaustive``: `ScoreAddConn` over every unit pair, hashed scores.
- ``shortlist`` / ``shortlist_level``: `ScoreAddConn` with
  ``candidates="shortlist"`` / ``"shortlist_per_level"``, a hashed
  importance and ``shortlist_size`` M.

Each point builds the network with `NetworkBuilder.from_edges`, jits the
add_conn phase from `build_add_conn_phase` alone (state donated, the step
counter advanced so each call draws fresh proposals), compiles it ahead of
time (reported as ``compile_s``), warms up, then reports the median, min and
max of ``--reps`` calls, each timed with ``perf_counter`` around the call and
``jax.block_until_ready``. A fresh process needs about 15 calls before a GPU
call reaches its steady-state time (host-side warm-up), so the warm-up runs
``--warmup`` calls (20), cut short only once ``--warmup-seconds`` (10 s) have
passed, which only the slowest points (a second or more per call, where the
warm-up is negligible) reach. The calls warmed up are reported as
``warmup``.

Sweeps (``--sweep``, comma separated; default all), at fixed P = 4, M = 64,
N = 65536 and C = 65536 unless varied:

- ``np``: per_unit over N = 2^12..2^20 x P = 1..32.
- ``n``: every strategy vs N = 2^8..2^20 (exhaustive up to N = 4096).
- ``c``: every strategy but exhaustive vs C = 2^14..2^23.
- ``p``: per_unit and per_connection vs P = 1..32, global vs P = 64..2^20,
  the shortlists vs M = 16..256.

``--max-candidates`` skips points with more candidates per call (N x P,
C x P, N^2, ...) than it allows, a CPU time budget. Run with
``XLA_FLAGS=--xla_disable_hlo_passes=constant_folding``: with constant
folding on, XLA spends minutes folding the exhaustive grid on either backend
and about 15 s per point in XLA:CPU codegen of the slot claim at 32K-slot
buckets, while the run times are unchanged. The backend is JAX's default
device; on a GPU venv, e.g.:

    XLA_PYTHON_CLIENT_PREALLOCATE=false \\
    XLA_FLAGS=--xla_disable_hlo_passes=constant_folding \\
        .venv-gpu/bin/python examples/benchmarks/growth_bench.py --out growth_gpu.csv

``run_growth.sh`` runs both backends with these settings.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import plastax as px
from plastax.phases import build_add_conn_phase

Globals = dict[str, jax.Array]

LEVELS = 4
MAX_NEW_PER_LEVEL = 16
MAX_UNITS = 1 << 20
MAX_CONNS = 1 << 23
EXHAUSTIVE_MAX_N = 1 << 12
FIXED_N = 1 << 16
FIXED_C = 1 << 16
STRATEGIES = (
    "per_unit",
    "per_connection",
    "global",
    "exhaustive",
    "shortlist",
    "shortlist_level",
)
FIELDS = (
    "backend",
    "sweep",
    "strategy",
    "N",
    "C",
    "P",
    "levels",
    "candidates",
    "grown",
    "reps",
    "median_ms",
    "min_ms",
    "max_ms",
    "compile_s",
    "pool",
    "conn_capacity",
    "overflow",
    "warmup",
)


def hash01(a: jax.Array, b: jax.Array, salt: int) -> jax.Array:
    """Stateless integer hash of two int scalars and a salt to [0, 1).

    Args:
        a: first key.
        b: second key.
        salt: a static per-use salt.

    Returns:
        A float32 in [0, 1).
    """
    h = (a.astype(jnp.uint32) + jnp.uint32(salt)) * jnp.uint32(0x85EBCA77)
    h = (h ^ b.astype(jnp.uint32)) * jnp.uint32(0xC2B2AE3D)
    h = h ^ (h >> 15)
    h = h * jnp.uint32(0x2C1B3C6D)
    h = h ^ (h >> 13)
    return (h >> jnp.uint32(8)).astype(jnp.float32) / jnp.float32(1 << 24)


def next_level(src: jax.Array, width: int, rng: px.rng.Rng) -> px.Proposal:
    """A random unit of the next built level, or a veto from the last one.

    Edges only run toward higher ids, so growth never closes a cycle.

    Args:
        src: the proposing source unit id.
        width: units per level; unit u sits at built level u // width.
        rng: the proposal site's draw stream.

    Returns:
        The proposal, scored uniformly at random.
    """
    src = src.astype(jnp.int32)
    lvl = src // width
    dst = (lvl + 1) * width + rng.uniform_int(width).astype(jnp.int32)
    score = jnp.where(lvl + 1 < LEVELS, rng.uniform(), -jnp.inf)
    return px.Proposal(src, dst, score)


class LinearForward(px.ForwardPass):
    """A weighted-sum forward pass; only present because a net needs one."""

    combine = px.monoid.sum_

    def map(
        self,
        u: px.UnitView,
        dst: px.UnitIdx,
        src: px.UnitIdx,
        c: px.ConnView,
        cid: px.ConnIdx,
        g: Globals,
    ) -> jax.Array:
        del dst, g
        return c[px.WEIGHT, cid] * u[px.ACTIVATION, src]

    def apply(
        self, u: px.UnitView, i: px.UnitIdx, g: Globals, acc: jax.Array
    ) -> px.UnitWrite:
        del u, i, g
        return px.UnitWrite.of((px.ACTIVATION, acc))


class GrowthKnobs:
    """Selection and window knobs shared by every rule."""

    max_new_per_level = MAX_NEW_PER_LEVEL
    max_level_gap = LEVELS

    def init(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: Globals
    ) -> px.ConnWrite:
        del u, src, dst, g
        return px.ConnWrite.of((px.WEIGHT, jnp.float32(0.0)))


class PerUnitGrow(GrowthKnobs, px.ProposeAddConn):
    proposer = "per_unit"

    def __init__(self, p: int, width: int) -> None:
        self.proposals_per_proposer = p
        self.width = width

    def propose(
        self,
        u: px.UnitView,
        i: px.UnitIdx,
        j: jax.Array,
        g: Globals,
        rng: px.rng.Rng,
    ) -> px.Proposal:
        del u, j, g
        return next_level(jnp.asarray(i), self.width, rng)


class PerConnectionGrow(GrowthKnobs, px.ProposeAddConn):
    proposer = "per_connection"

    def __init__(self, p: int, width: int) -> None:
        self.proposals_per_proposer = p
        self.width = width

    def propose(
        self,
        u: px.UnitView,
        c: px.ConnView,
        cid: px.ConnIdx,
        j: jax.Array,
        g: Globals,
        rng: px.rng.Rng,
    ) -> px.Proposal:
        del u, j, g
        return next_level(c[px.FROM_ID, cid], self.width, rng)


class GlobalGrow(GrowthKnobs, px.ProposeAddConn):
    proposer = "global"

    def __init__(self, p: int, width: int) -> None:
        self.proposals_per_proposer = p
        self.width = width

    def propose(
        self, u: px.UnitView, j: jax.Array, g: Globals, rng: px.rng.Rng
    ) -> px.Proposal:
        del u, j, g
        src = rng.uniform_int(self.width * (LEVELS - 1))
        return next_level(src, self.width, rng)


class ScoreGrow(GrowthKnobs, px.ScoreAddConn):
    """Hashed scores over the exhaustive grid or an importance shortlist."""

    def __init__(self, candidates: str, m: int) -> None:
        self.candidates = candidates
        self.shortlist_size = m

    def score(
        self, u: px.UnitView, src: px.UnitIdx, dst: px.UnitIdx, g: Globals
    ) -> jax.Array:
        del u, g
        return hash01(jnp.asarray(src), jnp.asarray(dst), 0x5C0BE)

    def importance(self, u: px.UnitView, i: px.UnitIdx, g: Globals) -> jax.Array:
        del u, g
        return hash01(jnp.asarray(i), jnp.int32(0), 0x1A9017)


def make_rule(strategy: str, p: int, width: int) -> Any:
    """The growth rule of one strategy.

    Args:
        strategy: one of `STRATEGIES`.
        p: P for the proposers, M for the shortlists, unused for exhaustive.
        width: units per level.

    Returns:
        The add_conn policy instance.
    """
    if strategy == "per_unit":
        return PerUnitGrow(p, width)
    if strategy == "per_connection":
        return PerConnectionGrow(p, width)
    if strategy == "global":
        return GlobalGrow(p, width)
    if strategy == "exhaustive":
        return ScoreGrow("exhaustive", 0)
    if strategy == "shortlist":
        return ScoreGrow("shortlist", p)
    return ScoreGrow("shortlist_per_level", p)


def make_net(rule: Any) -> type[px.Network[Globals]]:
    """A topological net whose only structural trait is `rule`.

    Args:
        rule: the add_conn policy.

    Returns:
        The network class.
    """

    class Net(px.Network[Globals]):
        forward_pass = LinearForward()
        add_conn = rule
        propagation = px.Propagation.TOPOLOGICAL

    return Net


def layered_edges(n: int, c: int) -> tuple[np.ndarray, np.ndarray]:
    """C random edges split evenly over the adjacent pairs of the levels.

    Args:
        n: live units, LEVELS levels of n // LEVELS.
        c: live connections.

    Returns:
        ``(from_ids, to_ids)`` as int32 arrays.
    """
    width = n // LEVELS
    rng = np.random.default_rng(0x5EED)
    src, dst = [], []
    for lvl in range(1, LEVELS):
        e = c // (LEVELS - 1) + (lvl <= c % (LEVELS - 1))
        src.append(rng.integers(0, width, e, dtype=np.int32) + (lvl - 1) * width)
        dst.append(rng.integers(0, width, e, dtype=np.int32) + lvl * width)
    return np.concatenate(src), np.concatenate(dst)


def candidates_of(strategy: str, n: int, c: int, p: int) -> int:
    """Candidates per call, counted as plastax-cpp's bench counts them.

    Args:
        strategy: the strategy.
        n: live units.
        c: live connections.
        p: P or M.

    Returns:
        The candidate count.
    """
    return {
        "per_unit": n * p,
        "per_connection": c * p,
        "global": p,
        "exhaustive": n * n,
        "shortlist": p * p,
        "shortlist_level": LEVELS * p * p,
    }[strategy]


@dataclasses.dataclass(frozen=True)
class Point:
    sweep: str
    strategy: str
    n: int
    c: int
    p: int


def measure(
    pt: Point,
    *,
    reps: int,
    warmup: int,
    warmup_seconds: float,
    growth: str,
    backend: str,
) -> dict[str, Any]:
    """Build one network and time its growth phase.

    Args:
        pt: the sweep point.
        reps: timed calls.
        warmup: untimed calls after compiling, at most.
        warmup_seconds: stop warming up once this long has passed (after at
            least one call).
        growth: `build_add_conn_phase`'s claim engine.
        backend: the backend label for the row.

    Returns:
        One CSV row.
    """
    width = pt.n // LEVELS
    rule = make_rule(pt.strategy, pt.p, width)
    net = make_net(rule)
    frm, to = layered_edges(pt.n, pt.c)
    static, state = px.NetworkBuilder.from_edges(
        net,
        pt.n,
        frm,
        to,
        weights=np.full(frm.shape, 0.5, np.float32),
        input_ids=list(range(width)),
        output_ids=list(range(pt.n - width, pt.n)),
        globals_={},
        capacity_headroom=0.05,
    )
    del frm, to
    phase = build_add_conn_phase(net, static, growth=growth)
    inputs = px.StepInputs(inputs=jnp.zeros((width,), jnp.float32), targets=None)

    def call(st: px.NetworkState[Globals]) -> px.NetworkState[Globals]:
        new, _ = phase(st, inputs)
        return dataclasses.replace(new, step=new.step + 1)

    t0 = time.perf_counter()
    compiled: Callable[[Any], Any] = (
        jax.jit(call, donate_argnums=0).lower(state).compile()
    )
    compile_s = time.perf_counter() - t0

    warmed = 0
    t0 = time.perf_counter()
    while warmed < warmup and (
        warmed == 0 or time.perf_counter() - t0 < warmup_seconds
    ):
        state = jax.block_until_ready(compiled(state))
        warmed += 1
    ms = []
    for _ in range(reps):
        t0 = time.perf_counter()
        state = jax.block_until_ready(compiled(state))
        ms.append((time.perf_counter() - t0) * 1e3)
    return {
        "backend": backend,
        "sweep": pt.sweep,
        "strategy": pt.strategy,
        "N": pt.n,
        "C": pt.c,
        "P": pt.p,
        "levels": LEVELS,
        "candidates": candidates_of(pt.strategy, pt.n, pt.c, pt.p),
        "grown": int(state.grown),
        "reps": reps,
        "median_ms": f"{statistics.median(ms):.6f}",
        "min_ms": f"{min(ms):.6f}",
        "max_ms": f"{max(ms):.6f}",
        "compile_s": f"{compile_s:.3f}",
        "pool": pool_of(pt, static),
        "conn_capacity": sum(static.level_capacities),
        "overflow": int(bool(state.overflow)),
        "warmup": warmed,
    }


def pool_of(pt: Point, static: px.NetworkStatic) -> int:
    """Candidates plastax actually materializes per source-level bucket.

    Args:
        pt: the sweep point.
        static: the built network's static config.

    Returns:
        The per-bucket candidate pool (per_connection proposes from every
        connection slot, live or dead; shortlist_level counts one level).
    """
    if pt.strategy == "per_connection":
        return sum(static.level_capacities) * pt.p
    if pt.strategy == "shortlist_level":
        return pt.p * pt.p
    return candidates_of(pt.strategy, pt.n, pt.c, pt.p)


def pow2(lo: int, hi: int, step: int = 1) -> list[int]:
    """Powers of two 2^lo..2^hi.

    Args:
        lo: first exponent.
        hi: last exponent, inclusive.
        step: exponent stride.

    Returns:
        The powers.
    """
    return [1 << e for e in range(lo, hi + 1, step)]


def default_knob(strategy: str) -> int:
    """The P (or M) a strategy runs at when its sweep does not vary it.

    Args:
        strategy: the strategy.

    Returns:
        4 for the proposers, 64 for the shortlists, 0 for exhaustive.
    """
    if strategy == "exhaustive":
        return 0
    return 64 if strategy.startswith("shortlist") else 4


def build_points(sweeps: set[str], quick: bool, max_candidates: int) -> list[Point]:
    """The sweep grid, minus infeasible points.

    Args:
        sweeps: which sweeps to run.
        quick: a reduced smoke grid.
        max_candidates: skip points with more candidates than this
            (`candidates_of`).

    Returns:
        The points in run order.
    """
    ps = [1, 4, 16] if quick else [1, 2, 4, 8, 16, 32]
    ns = pow2(8, 16, 2) if quick else pow2(8, 20)
    cs = pow2(14, 18, 2) if quick else pow2(14, 23)
    pts: list[Point] = []
    if "np" in sweeps:
        for n in pow2(12, 16, 2) if quick else pow2(12, 20):
            pts += [Point("np", "per_unit", n, FIXED_C, p) for p in ps]
    if "n" in sweeps:
        for s in STRATEGIES:
            pts += [Point("n", s, n, FIXED_C, default_knob(s)) for n in ns]
    if "c" in sweeps:
        for s in STRATEGIES:
            if s != "exhaustive":
                pts += [Point("c", s, FIXED_N, c, default_knob(s)) for c in cs]
    if "p" in sweeps:
        for s in ("per_unit", "per_connection"):
            pts += [Point("p", s, FIXED_N, FIXED_C, p) for p in ps]
        gs = [64, 16384] if quick else [64, 1024, 16384, 262144, 1048576]
        pts += [Point("p", "global", FIXED_N, FIXED_C, p) for p in gs]
        ms = [16, 64, 256] if quick else [16, 32, 64, 128, 256]
        for s in ("shortlist", "shortlist_level"):
            pts += [Point("p", s, FIXED_N, FIXED_C, m) for m in ms]

    def feasible(pt: Point) -> bool:
        if pt.n // LEVELS == 0 or pt.n > MAX_UNITS or pt.c > MAX_CONNS:
            return False
        if pt.strategy == "exhaustive" and pt.n > EXHAUSTIVE_MAX_N:
            return False
        return candidates_of(pt.strategy, pt.n, pt.c, pt.p) <= max_candidates

    return [pt for pt in pts if feasible(pt)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sweep", default="np,n,c,p")
    ap.add_argument("--quick", action="store_true", help="a reduced smoke grid")
    ap.add_argument("--reps", type=int, default=None, help="timed calls (7)")
    ap.add_argument("--warmup", type=int, default=20, help="warm-up calls (20)")
    ap.add_argument(
        "--warmup-seconds",
        type=float,
        default=10.0,
        help="cut the warm-up short after this long (10 s)",
    )
    ap.add_argument(
        "--max-candidates",
        type=int,
        default=1 << 30,
        help="skip points with more candidates per call than this",
    )
    ap.add_argument("--growth", choices=("auto", "xla", "triton"), default="xla")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    reps = args.reps if args.reps is not None else (3 if args.quick else 7)
    backend = jax.default_backend()
    pts = build_points(set(args.sweep.split(",")), args.quick, args.max_candidates)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    with args.out.open("w", newline="") as f:
        out = csv.DictWriter(f, fieldnames=FIELDS)
        out.writeheader()
        for idx, pt in enumerate(pts, 1):
            row = measure(
                pt,
                reps=reps,
                warmup=args.warmup,
                warmup_seconds=args.warmup_seconds,
                growth=args.growth,
                backend=backend,
            )
            out.writerow(row)
            f.flush()
            print(
                f"[{idx}/{len(pts)}] {pt.sweep:3s} {pt.strategy:15s} N={pt.n:<8d} "
                f"C={pt.c:<8d} P={pt.p:<8d} {float(row['median_ms']):10.3f} ms "
                f"(compile {float(row['compile_s']):.1f} s)",
                flush=True,
            )
    print(f"wrote {args.out} in {time.perf_counter() - t_start:.0f} s", file=sys.stderr)


if __name__ == "__main__":
    main()
