# Growth benchmarks

Time of one connection-growth phase (the add_conn phase that
`plastax.phases.build_add_conn_phase` builds) for every growth strategy,
against live units N, live connections C and proposals per proposer P, on
CPU and GPU. The setup mirrors plastax-cpp's growth bench
(`benchmarks/bench_growth.cpp` in that repository) point for point, so the
last section compares the two libraries directly.

The numbers below are from after the single-sort selection change (each
bucket's winners are taken from one total-order sort of the shared candidate
list, instead of one full sort per bucket). The section "Single-sort
selection: before and after" profiles the change and compares both runs.

- Harness: `examples/benchmarks/growth_bench.py`.
- Driver: `examples/benchmarks/run_growth.sh`.
- Plots, fits and the cx comparison: `examples/benchmarks/plot_growth.py`.
- Raw data: `benchmarks/results/growth/`:
  - `growth_cpu.csv` and `growth_gpu.csv`, one row per point;
  - `growth_cpu_before.csv` and `growth_gpu_before.csv`, the same grid
    before the single-sort change;
  - `growth_before_after.csv`, every point of both runs, with the speedup;
  - `growth_fit.csv`, the per-unit fits;
  - `growth_vs_cx.csv`, every point both libraries ran, with the time ratio;
  - `device_clocks.csv`, the clock log of the GPU run.

## Setup

**Network.** Four levels of N/4 units each, built with
`NetworkBuilder.from_edges`. The C live connections are split evenly over the
three adjacent level pairs, with random endpoints (parallel edges allowed).
Levels are recomputed from the edges at build time, so a unit with no
incoming edge sits at level 0, as on the cx host. Each bucket gets 5 %
headroom, rounded up to a power of two (the `from_edges` default rounding).

**Rules.** Every rule uses `selection="top_k"` with `max_new_per_level = 16`
and `max_level_gap = 4`, which spans every level, so nearly every candidate
passes the window. The rules are:

- `per_unit`: `ProposeAddConn`, `proposer="per_unit"`, the default. Every
  unit proposes P edges into a random unit of the next built level, each with
  a random score drawn from the site's `Rng`.
- `per_connection`: `proposer="per_connection"`. Each connection's source
  proposes P edges in the same way.
- `global`: `proposer="global"`. P proposals from random sources.
- `exhaustive`: `ScoreAddConn` over every unit pair, with a hashed score.
- `shortlist`: `candidates="shortlist"`, with a hashed importance and
  `shortlist_size` M.
- `shortlist_level`: `candidates="shortlist_per_level"`, with a hashed
  importance and `shortlist_size` M.

Nothing is deduplicated (the default). A score rule's window also admits
backward and same-level pairs, so a scorer can fill the small bucket of the
last level. The `overflow` column records when a bucket ran out of free
slots. That drops commits but does not change the work of a call.

**Timing.** Each point jits the add_conn phase alone, with the state
donated as in a real step. The jitted function also advances the step
counter, so every call draws fresh proposals. It is compiled ahead of time
(`jit(...).lower(state).compile()`), and the compile time is reported
separately as `compile_s`. Then it runs 2 warm-up calls and reports the
median of 7 calls. Each call is timed with `time.perf_counter` around the
call and `jax.block_until_ready`, so GPU times include dispatch.

**Backends.**

- CPU: XLA:CPU on an i7-14700K (20 threads). Unlike the single-threaded cx
  host build, XLA uses every thread.
- GPU: an RTX 5000 Ada with `jax[cuda13]` 0.11.2. The SM clock was locked to
  2505 MHz and the memory clock to 8551 MHz; every sample of the clock log
  taken under load reads 2505/8551. Slots are claimed with the XLA engine
  (`growth="xla"`); the Triton claim was not installed.

Both runs set `XLA_FLAGS=--xla_disable_hlo_passes=constant_folding`.

- With constant folding on, XLA folds the exhaustive candidate grid at
  compile time: 188 s to compile N = 4096 on the GPU.
- On CPU, a fusion feeding the slot claim also takes about 15 s of LLVM code
  generation per point at 32K-slot buckets.
- Disabling the pass leaves run times unchanged, checked at five points on
  each backend.

With the flag set, a point compiles in a median of 0.9 s on CPU (max 1.7 s)
and 2.6 s on the GPU (max 4.4 s).
The per-point compile times are plotted in `growth_compile.png`.

**Sweeps.** These are cx's sweeps. Each holds the parameters it does not
vary fixed at P = 4, M = 64, N = 65536 and C = 65536.

| sweep | varies | strategies |
|---|---|---|
| `np` | N = 2^12..2^20 x P = 1..32 | per_unit (the full grid) |
| `n` | N = 2^8..2^20 | all; exhaustive only up to N = 4096 (N^2 candidates) |
| `c` | C = 2^14..2^23 | all but exhaustive |
| `p` | P = 1..32 (per_unit, per_connection), P = 64..2^20 (global), M = 16..256 (shortlists) | all but exhaustive |

The largest network holds about 8.4M connections, far below the
300M-edge limit. The GPU run covers all 201 points in about 9 minutes.

The CPU run skips points with more than 2^22 candidates per call
(`--max-candidates`), where one call took 6 s or more before the
single-sort change. It covers 191 points in 10 minutes. The 10 skipped
points are:

- `np`: the six points with N x P > 2^22;
- `n`: exhaustive at N = 4096;
- `c`: per_connection at C = 2^21..2^23.

Reproduce from the repository root (the GPU venv as in
`docs/development/tooling.md`):

```bash
UV_PROJECT_ENVIRONMENT=.venv-gpu uv sync --extra cuda13
examples/benchmarks/run_growth.sh .venv/bin/python .venv-gpu/bin/python   # add --quick for a smoke run
```

## Per-unit growth scales as N x P

![per-unit growth vs N, one line per P](results/growth/growth_np.png)

![per-unit growth vs N x P with the fitted power law](results/growth/growth_np_fit.png)

The fit is a least-squares fit of log t against log(N x P) over the `np`
grid. It starts above the fixed-cost floor. A second model fits
log t = a + alpha log N + beta log P on the same points.

| backend | fit range (N x P) | points | slope in N x P | R^2 | alpha (N) | beta (P) | time per candidate |
|---|---|---|---|---|---|---|---|
| CPU | >= 2^16 | 38 | **0.96** | 0.998 | 0.96 | 0.96 | 0.51 us |
| GPU | >= 2^20 | 21 | **1.40** | 0.958 | 1.40 | 1.41 | 1.7 ns |
| GPU | >= 2^23 | 6 | **1.19** | 0.99999 | 1.19 | 1.19 | 4.8 ns |

- **CPU.** Per-unit growth is linear in N x P: the slope is 0.96, and alpha
  equals beta, so the time depends on the product alone. Points with equal
  N x P coincide; for example, every point with N x P = 2^16 takes 38 to
  39 ms, from N = 4096 with P = 16 to N = 65536 with P = 1.
- **GPU.** The GPU also collapses onto N x P (alpha = beta).
  - Up to about 2^17 candidates a call sits at a floor of 0.6 to 0.8 ms.
  - Between 4M and 8M candidates there is a cliff: 6.3 ms jumps to 38 ms.
    This is at the same N x P as cx's device cliff, where the candidate and
    sort buffers outgrow the 64 MB L2. It inflates the slope from 2^20 up to
    1.40.
  - Past the cliff the slope is 1.19, about cx's 1.23.
- **Speedup.** At 4M candidates the GPU takes 6.3 ms against 2.2 s on CPU,
  about 350x.
- **Cost driver.** The selection's one sort of the N x P candidates on
  (-score, src, dst, index). On CPU it is about 94 % of a call (see the
  profile below).

## Strategies vs live units, live connections and P

![every strategy vs N](results/growth/growth_n.png)

![every strategy vs C](results/growth/growth_c.png)

![proposers vs P, shortlists vs M](results/growth/growth_p.png)

All times are the median growth-call time in ms.

| strategy | cost driver | CPU (N=C=65536) | GPU (N=C=65536) |
|---|---|---|---|
| per_unit, P=4 | N x P; flat in C | 134 | 0.91 |
| per_connection, P=4 | conn capacity x P (C log C) | 294 | 1.49 |
| global, P=4 | P, plus the slot claim over every bucket (grows with C on CPU) | 1.8 | 0.50 |
| exhaustive, N=2048 (CPU) / 4096 (GPU) | N^2 | 2477 | 97.6 |
| shortlist, M=64 | N ranking + M^2 | 5.2 | 0.54 |
| shortlist_level, M=64 | levels x (N ranking + M^2) | 7.9 | 1.06 |

### Per-connection

- Linear in C on both backends. On CPU it goes from 66 ms at C = 2^14 to
  4.2 s at C = 2^20, about 2x per doubling. On the GPU it goes from 0.9 ms
  at C = 2^14 to 399 ms at C = 2^23.
- The GPU has a cliff between C = 2^19 and 2^20 (6.9 ms to 30 ms). The
  per-connection pool is the conn capacity times P, so the cliff falls at
  the same 4M to 8M candidates as the per-unit one.
- The pool counts dead slots too: per_connection proposes from every slot of
  the arena, live or not, and vetoes the dead ones.

### Shortlists

- Flat in N up to about 2^18 on CPU and over the whole range on the GPU.
- Flat in C until the slot claim over the larger buckets shows on CPU,
  reaching 37 to 40 ms at C = 2^23.
- Growing in M on CPU (the M^2 grid: 43 ms for shortlist and 45 ms for
  shortlist_level at M = 256), and nearly flat in M on the GPU (0.5 to
  1.4 ms).
- shortlist_level draws a separate M x M grid per bucket, so it still sorts
  once per bucket; its times are unchanged by the single-sort change.

### Exhaustive

Exhaustive is N^2. At N = 2048 it takes 2.5 s on CPU. At N = 4096 it takes
98 ms on the GPU.

### Global

- Global costs only P plus a floor, the slot claim over every bucket.
- On CPU the claim's scan of every slot shows: the floor grows with the
  conn capacity, from 0.8 to 5.4 ms up to C = 2^20 to 35 ms at C = 2^23.
- On the GPU the floor is flat at 0.5 to 0.65 ms at every C and up to
  P = 1024, and P = 2^20 takes 2.1 ms.

### Per-unit (the default)

The per-unit default is flat in live connections:

- CPU: 133 to 136 ms from C = 2^14 to C = 2^20, rising to 171 ms at
  C = 2^23 as the slot claim over the larger buckets shows;
- GPU: 0.7 to 1.0 ms over the same range.

Its cost tracks N x P alone.

## Comparison with plastax-cpp

![per-unit growth vs N x P, px and cx](results/growth/growth_vs_cx.png)

`growth_vs_cx.csv` matches every point that both libraries ran: 392 points.
px CPU is matched against the cx host, and px GPU against the cx device. A
ratio above 1 means px is slower. The cx rows are plastax-cpp's current
results, including its rerun of the host per-connection sweeps (its host
proposer is now C log C, and runs to C = 2^20 here).

| strategy | point | px CPU | cx host | ratio | px GPU | cx device | ratio |
|---|---|---|---|---|---|---|---|
| per_unit | N=65536, P=4 | 134 | 27.3 | 4.9 | 0.91 | 0.33 | 2.8 |
| per_unit | N x P = 2^20 (N=2^18, P=4) | 525 | 133 | 3.9 | 1.83 | 0.57 | 3.2 |
| per_unit | N x P = 2^22 (N=2^20, P=4) | 2198 | 581 | 3.8 | 6.31 | 2.16 | 2.9 |
| per_unit | N x P = 2^25 (N=2^20, P=32) | skipped | | | 197 | 34.7 | 5.7 |
| per_connection | C=2^14, P=4 | 66.3 | 9.18 | 7.2 | 0.89 | 0.76 | 1.18 |
| per_connection | C=2^16, P=4 | 295 | 41.6 | 7.1 | 1.61 | 2.47 | **0.65** |
| per_connection | C=2^20, P=4 | 4188 | 929 | 4.5 | 30.4 | 13.3 | 2.3 |
| per_connection | C=2^23, P=4 | skipped | not run | | 399 | 116 | 3.5 |
| global | P=4 | 1.83 | 0.0005 | ~3500 | 0.50 | 0.15 | 3.2 |
| global | P=2^20 | 583 | 170 | 3.4 | 2.08 | 0.62 | 3.4 |
| exhaustive | N=2048 | 2477 | 632 | 3.9 | 7.10 | 2.13 | 3.3 |
| exhaustive | N=4096 | skipped | | | 97.6 | 16.0 | 6.1 |
| shortlist | M=64 | 5.2 | 22.5 | **0.23** | 0.54 | 5.89 | **0.09** |
| shortlist_level | M=64 | 7.9 | 104 | **0.08** | 1.06 | 28.8 | **0.04** |

All times are in ms. Over all matched points, the median ratio is 4.9 on
CPU and 2.3 on the GPU (11.8 and 3.2 before the single-sort change).

- **Per-unit, global and exhaustive.** cx is faster. Above the px floors
  the gap is about 4x on CPU. On the GPU it runs from about 3x at the floor
  to 6x at 32M candidates.
  - Both libraries scale the same way (per-unit slope about 1.0 on CPU,
    about 1.2 past the GPU's L2 cliff, alpha = beta). The gap is a
    constant factor.
  - What is left is px's one full sort of the candidate list on
    (-score, src, dst, index), where cx does not sort the whole list.
    On XLA:CPU that sort costs about 0.43 us per candidate (see the profile
    below).
  - XLA:CPU uses 20 threads against cx's one and is still about 4x slower.
- **Per-connection.** Both libraries are now C log C on CPU, and px is 4.5
  to 7x slower there. On the GPU px is 1.3 to 1.8x faster than cx from
  C = 2^16 to 2^18, within 20 % at 2^14, 2^15 and 2^19, and 2.3 to 3.5x
  slower past its cliff at C = 2^20.
- **Shortlists.** px is faster than cx on both backends.
  - On the GPU: 11x for shortlist and 27x for shortlist_level, because cx
    ranks units and enumerates the pairs on the host by design.
  - On CPU, both are faster than cx at M = 64 (4x and 13x).
- **Floors.** The px GPU floor is 0.5 to 0.65 ms per call, against 0.15 ms
  for cx. px pays a host dispatch, which `perf_counter` includes and CUDA
  events do not, plus the XLA slot claim of about 20 kernels per bucket.
  On CPU, global's px floor (about 2 ms) is the slot claim over every
  bucket, where cx's host loop touches only the few commits.

## Single-sort selection: before and after

![before/after speedup of every point vs its candidates](results/growth/growth_before_after.png)

**Profile.** Profiled on CPU at the per-unit point N = C = 65536, P = 4
(262144 candidates, four buckets) by timing the jitted phase with stages
replaced:

| variant | ms per call |
|---|---|
| before: one 4-key sort of the whole list per bucket | 337 |
| before, with the selection sort replaced by `lax.top_k` | 7.2 |
| one 4-key `lax.sort` of 262144 candidates, alone | 114 |
| after: one sort per call | 138 |
| after, with the sort stubbed out | 8.4 |

Before the change, the four per-bucket sorts were about 98 % of the call
(about 84 ms each), so the cx gap came mostly from them. The proposals, the
validity masks and the slot claim together cost about 7 ms. After the
change the one sort is still about 94 % of the call.

**Design.** Every strategy but `shortlist_per_level` draws all buckets from
one shared candidate list. A topological bucket only admits candidates
sourced at its own level, so the buckets' valid sets partition that list by
source level, and every other bucket scores a candidate -inf. The phase now
merges the per-bucket validity masks into one, scores the list once, runs
the within-step dedupe once (a pair's copies share a source, so a bucket),
and sorts it once into the total order (`total_order`). Each bucket's
winners are its own source level's members in that order, ranked by a
running count of them (`select_per_segment`). Restricted to one bucket, that
is the same order a per-bucket sort produces ahead of its first -inf
candidate, so the committed edges, their order, the overflow and resort
flags and every column are bit-identical. `shortlist_per_level` still
selects per bucket, since each bucket has its own grid.

**Equivalence.** `tests/test_growth_single_sort.py` keeps the per-bucket
formulation as a test-only reference and checks both against each other bit
for bit over 36 randomized layered nets. These cover every candidate source
and selection mode, the step cap, both dedupes, the window knobs, unit
capacities, PIPELINE, NaN and -inf scores, and buckets tight enough to
overflow. The reference also matches the previous phase exactly. The
growth goldens and the pinned claim digests pass unchanged. Across the two
bench runs, every point grows the same number of edges and raises the same
overflow flag.

**Speedup.** `growth_before_after.csv` holds every point of both runs. The
GPU "before" run is a rerun of the previous code in the same environment as
the "after" run (it matches the earlier published GPU run to a median ratio
of 1.00). The CPU "before" run is the earlier published one.

| strategy | CPU median speedup | GPU median speedup | at N = C = 65536 (CPU / GPU) |
|---|---|---|---|
| per_unit | 2.6x | 2.0x | 334 -> 134 ms / 1.67 -> 0.91 ms |
| per_connection | 2.8x | 2.1x | 826 -> 294 ms / 3.25 -> 1.49 ms |
| exhaustive | 2.4x | 2.1x | N = 2048: 6272 -> 2477 ms / 26.7 -> 7.1 ms |
| shortlist | 2.2x | 1.1x | 10.6 -> 5.2 ms / 0.81 -> 0.54 ms |
| global | 1.0x | 1.0x | 1.75 -> 1.84 ms / 0.51 -> 0.50 ms |
| shortlist_level | 1.0x | 1.0x | unchanged (per-bucket grids) |

- The speedup grows with the candidates, toward the bucket count (4x).
  Per-unit reaches 3.0x on CPU and 4.3x on the GPU (at 4M candidates on the
  GPU, 27.2 ms to 6.3 ms).
- global at small P is the slot-claim floor, which the change does not
  touch; its large-P points speed up like per-unit (3.8x on CPU and 2.8x
  on the GPU at P = 2^20).
