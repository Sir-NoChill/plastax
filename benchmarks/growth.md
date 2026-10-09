# Growth benchmarks

Time of one connection-growth phase (the add_conn phase that
`plastax.phases.build_add_conn_phase` builds) for every growth strategy,
against live units N, live connections C and proposals per proposer P, on
CPU and GPU. The setup mirrors plastax-cpp's growth bench
(`benchmarks/bench_growth.cpp` in that repository) point for point, so the
last section compares the two libraries directly.

- Harness: `examples/benchmarks/growth_bench.py`.
- Driver: `examples/benchmarks/run_growth.sh`.
- Plots, fits and the cx comparison: `examples/benchmarks/plot_growth.py`.
- Raw data: `benchmarks/results/growth/`:
  - `growth_cpu.csv` and `growth_gpu.csv`, one row per point;
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
and 2.7 s on the GPU (max 5.9 s; per_connection is the slowest at 4 to 6 s).
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
300M-edge limit. The GPU run covers all 201 points in about 10 minutes.

The CPU run skips points with more than 2^22 candidates per call
(`--max-candidates`), where one call takes 6 s or more. It covers 191
points in 24 minutes. The 10 skipped points are:

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
| CPU | >= 2^16 | 38 | **1.02** | 0.997 | 1.02 | 1.03 | 1.45 us |
| GPU | >= 2^20 | 21 | **1.51** | 0.981 | 1.51 | 1.50 | 5.0 ns |
| GPU | >= 2^23 | 6 | **1.19** | 0.99995 | 1.18 | 1.19 | 16.9 ns |

- **CPU.** Per-unit growth is linear in N x P: the slope is 1.02, and alpha
  equals beta, so the time depends on the product alone. Points with equal
  N x P coincide; for example, every point with N x P = 2^16 takes 100 to
  104 ms, from N = 4096 with P = 16 to N = 65536 with P = 1.
- **GPU.** The GPU also collapses onto N x P (alpha = beta).
  - Up to about 2^17 candidates a call sits at a floor of 0.6 to 1.2 ms.
  - Between 4M and 8M candidates there is a cliff: 27 ms jumps to 133 ms.
    This is at the same N x P as cx's device cliff, where the candidate and
    sort buffers outgrow the 64 MB L2. It inflates the slope from 2^20 up to
    1.51.
  - Past the cliff the slope is 1.19, about cx's 1.23.
- **Speedup.** At 4M candidates the GPU takes 27 ms against 6.4 s on CPU,
  about 240x.
- **Cost driver.** `build_add_conn_phase` computes the proposals once, then
  filters and sorts the whole global candidate list once per source-level
  bucket. Here there are four buckets: the network's three, plus the last
  level's bucket. The selection is a sort on (-score, src, dst, index), not
  a top-k. So a per-unit call costs about four full sorts of N x P
  candidates. This is the likely main contributor to the gap to cx below,
  but it has not been profiled separately.

## Strategies vs live units, live connections and P

![every strategy vs N](results/growth/growth_n.png)

![every strategy vs C](results/growth/growth_c.png)

![proposers vs P, shortlists vs M](results/growth/growth_p.png)

All times are the median growth-call time in ms.

| strategy | cost driver | CPU (N=C=65536) | GPU (N=C=65536) |
|---|---|---|---|
| per_unit, P=4 | N x P; flat in C | 333 | 1.73 |
| per_connection, P=4 | conn capacity x P (C log C) | 824 | 3.22 |
| global, P=4 | P, plus the slot claim over every bucket (grows with C on CPU) | 1.9 | 0.47 |
| exhaustive, N=2048 (CPU) / 4096 (GPU) | N^2 | 6272 | 290 |
| shortlist, M=64 | N ranking + M^2 | 9.9 | 0.71 |
| shortlist_level, M=64 | levels x (N ranking + M^2) | 10.3 | 1.14 |

### Per-connection

- Linear in C on both backends. On CPU it goes from 143 ms at C = 2^14 to
  12.0 s at C = 2^20, about 2x per doubling. On the GPU it goes from 1.5 ms
  at C = 2^14 to 1.25 s at C = 2^23.
- The GPU has a cliff between C = 2^19 and 2^20 (18 ms to 97 ms). The
  per-connection pool is the conn capacity times P, so the cliff falls at
  the same 4M to 8M candidates as the per-unit one.
- The pool counts dead slots too: per_connection proposes from every slot of
  the arena, live or not, and vetoes the dead ones.

### Shortlists

- Flat in N up to about 2^18 on CPU and over the whole range on the GPU.
  On CPU, shortlist_level rises to 33 ms at N = 2^20.
- Flat in C until the slot claim over the larger buckets shows on CPU,
  reaching 38 to 45 ms at C = 2^23.
- Growing in M on CPU (the M^2 grid, 118 ms at M = 256), and nearly flat in
  M on the GPU (0.7 to 1.4 ms).

### Exhaustive

Exhaustive is N^2. At N = 2048 it takes 6.3 s on CPU. At N = 4096 it takes
290 ms on the GPU.

### Global

- Global costs only P plus a floor, the slot claim over every bucket.
- On CPU the claim's scan of every slot shows: the floor grows with the
  conn capacity, from 0.8 to 3.6 ms up to C = 2^20 to 37 ms at C = 2^23.
- On the GPU the floor is flat at 0.45 to 0.6 ms at every C and up to
  P = 1024, and P = 2^20 takes 5.9 ms.

### Per-unit (the default)

The per-unit default is flat in live connections:

- CPU: 324 to 338 ms from C = 2^14 to C = 2^23;
- GPU: 1.4 to 1.9 ms over the same range.

Its cost tracks N x P alone.

## Comparison with plastax-cpp

![per-unit growth vs N x P, px and cx](results/growth/growth_vs_cx.png)

`growth_vs_cx.csv` matches every point that both libraries ran: 388 points.
px CPU is matched against the cx host, and px GPU against the cx device. A
ratio above 1 means px is slower.

| strategy | point | px CPU | cx host | ratio | px GPU | cx device | ratio |
|---|---|---|---|---|---|---|---|
| per_unit | N=65536, P=4 | 333 | 27.1 | 12.3 | 1.73 | 0.33 | 5.2 |
| per_unit | N x P = 2^20 (N=2^18, P=4) | 1551 | 133 | 11.7 | 5.00 | 0.57 | 8.8 |
| per_unit | N x P = 2^22 (N=2^20, P=4) | 6364 | 581 | 11.0 | 27.5 | 2.16 | 12.7 |
| per_unit | N x P = 2^25 (N=2^20, P=32) | skipped | | | 682 | 34.7 | 19.7 |
| per_connection | C=2^14, P=4 | 143 | 50.7 | 2.8 | 1.46 | 0.76 | 1.9 |
| per_connection | C=2^16, P=4 | 824 | 696 | 1.18 | 3.22 | 2.47 | 1.3 |
| per_connection | C=2^23, P=4 | skipped | not run | | 1248 | 116 | 10.8 |
| global | P=4 | 1.93 | 0.0005 | ~3700 | 0.47 | 0.15 | 3.1 |
| global | P=2^20 | 2212 | 170 | 13.0 | 5.91 | 0.62 | 9.6 |
| exhaustive | N=2048 | 6272 | 632 | 9.9 | 26.9 | 2.13 | 12.7 |
| exhaustive | N=4096 | skipped | | | 290 | 16.0 | 18.1 |
| shortlist | M=64 | 9.9 | 22.5 | **0.44** | 0.71 | 5.90 | **0.12** |
| shortlist_level | M=64 | 10.3 | 104 | **0.10** | 1.14 | 28.7 | **0.04** |

All times are in ms. Over all matched points, the median ratio is 11.8 on
CPU and 3.2 on the GPU.

- **Per-unit, global and exhaustive.** cx is faster. Above the px floors
  the gap is 10 to 16x on CPU. On the GPU it runs from 3x at the floor to
  20x at 32M candidates.
  - Both libraries scale the same way (per-unit slope about 1.0 on CPU,
    about 1.2 past the GPU's L2 cliff, alpha = beta). The gap is a
    constant factor.
  - The per-bucket full sort described above is the likely main cause (not
    profiled). cx does not sort the whole candidate list once per bucket.
  - XLA:CPU uses 20 threads against cx's one and is still about 12x slower.
- **Per-connection.** The two libraries have different complexity on CPU.
  - The cx host proposer is quadratic in C, because of its occurrence count;
    cx stops its host sweep at C = 2^16.
  - px is C log C. The ratio falls from 2.8 at C = 2^14 to 1.18 at
    C = 2^16, so px CPU overtakes the cx host just past C = 2^16.
  - On the GPU both are linear. px is 1.3 to 2.4x slower up to C = 2^19,
    and 7 to 11x slower from its cliff at C = 2^20 on.
- **Shortlists.** px is faster than cx on both backends.
  - On the GPU: 8x for shortlist and 25x for shortlist_level, because cx
    ranks units and enumerates the pairs on the host by design.
  - On CPU, shortlist_level is 3 to 29x faster at every M. shortlist is 2
    to 10x faster at M <= 64, but px is 1.3x slower at M = 128 and 4x
    slower at M = 256, where px's sort of the M^2 grid dominates.
- **Floors.** The px GPU floor is 0.45 to 0.6 ms per call, against 0.15 ms
  for cx. px pays a host dispatch, which `perf_counter` includes and CUDA
  events do not, plus the XLA slot claim of about 20 kernels per bucket.
  On CPU, global's px floor (about 2 ms) is the slot claim over every
  bucket, where cx's host loop touches only the few commits.
