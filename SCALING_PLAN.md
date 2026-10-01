# plastax scaling plan: step-time and memory laws, and the edge ceiling

2026-10-01 · Status: **planned**. Coding-agent handoff, structured like
`DISTRIBUTION_PLAN.md`: phases with acceptance criteria, HUMAN markers, and a
Deviations section this document owns. Commits follow the agent-commit
protocol (`AGENTS.md`, `TAGS.md`, `SCOPES.md`). This plan closes the open
`SCALE_PLAN.md` items 2.2 and 2.3 and the host-build limit. Results go to
Markdown with matplotlib PNGs under `docs/scaling/`.

## Safety rule (read first)

**Never build a network above ~300M edges on cdol01** (61 GB host RAM).
`NetworkBuilder.from_edges` at 600M edges exhausted host RAM and crashed the
session twice. Larger sizes are run only after S2 lands, and only on a host
with at least 128 GB of RAM (HUMAN: Narval or another big-memory node), or
via S2's chunked build with RSS measured to stay under budget on smaller
sizes first.

## Where we stand

- **Device state:** 13 B per slot:
  - `from` and `to` int32, 8 B;
  - weight f32, 4 B;
  - `dead` u8, 1 B;
  - plus extra columns per net (optimizer state, traces).
  - The slot count is `live × (1 + headroom)`, rounded up to `align`, so 300M
    live edges take 4.1 GB with 5% headroom and align 256.
- **Peak device memory**, light-churn probe at 300M: about 4.7 GB for plastax
  against 27.3 GB for C++ Plastix. C++ keeps about 97 B per slot resident,
  mostly scratch (radix buffers, keys, perm).
- **Host build:** about 45 B per edge with aligned capacities, after c184fd0
  and 35801f5. 600M would need about 30 GB. The two crashes predate the move
  from power-of-two padding to aligned capacities.
- **Step time** on the bench grid (k = 64 units churned, ~20K edges), RTX 5000
  Ada:

  | E | plastax | C++ in place |
  |---|---|---|
  | 5M | 0.49 ms | 0.19 ms |
  | 50M | 2.97 ms | 2.16 ms |
  | 200M | 11.2 ms | – |
  | 300M | 16.7 ms | 12.9 ms |

  The forward and the prune each run at the ~525 GB/s DRAM roofline. Two
  in-flight branches change this: `perf/fuse-prune-forward` and
  `perf/growth-kernel`.

## Phase S1 -- Close SCALE_PLAN 2.2 (forward profile)

- S1.1 After the fusion branches land, re-profile the forward at E50M and
  E300M with nsys `--cuda-graph-trace=node`. Report bytes per slot read
  against the minimum (13 B for the fused sweep) and the achieved GB/s.
- S1.2 Check for materialisation of the vmapped `map` output: the HLO should
  show no E-sized f32 temporary in the B = 1 forward.
- S1.3 Mark 2.2 closed in `SCALE_PLAN.md`, linking the profile, or list what
  remains.

Acceptance: the forward is within 10% of the roofline at E50M and E300M, or
the gap is explained.

## Phase S2 -- A host build under 20 B per edge (SCALE_PLAN 2.7)

- S2.1 **Measure first,** at 5M / 50M / 100M / 200M / 300M: peak host RSS of
  `from_edges`, with `/usr/bin/time -v`, or a `tracemalloc` + `psutil`
  sampler. Fit B per edge, and attribute it per stage: levels, per-bucket
  argsort, fancy-index copies, device transfer.
- S2.2 **Chunked build.**
  - Accept `from` / `to` / weights as memory-mapped arrays.
  - Compute levels on device (`recompute_levels`) or in chunks.
  - Sort each bucket on device. The device has the room: 13 B per slot
    against 32 GB, and a device sort of 300M int64 keys needs about 5 GB of
    scratch.
  - Stream into preallocated device buffers.
  - Keep `from_edges`'s signature, and add a `chunk_edges=` knob only if it is
    needed.
- S2.3 **Tests:** the chunked build is identical to the current build at small
  sizes (every column, every capacity); the RSS regression test runs at 50M as
  a slow test.

Acceptance: under 20 B per edge of host RSS at 300M, measured; a 600M
extrapolation under 15 GB.

## Phase S3 -- The edge ceiling on 32 GB (SCALE_PLAN 2.3)

- S3.1 Predict it from the S4 memory law. The light probe predicts about 1B+
  edges, the batched or optimizer-state nets fewer.
- S3.2 HUMAN gate: run 600M, then 1B, **only** after S2 shows the host budget
  holds, either on cdol01 with S2's measured RSS under 30 GB, or on a
  big-memory node. Use the light-churn probe and the bench churn (k = 64
  units).
- S3.3 Report the largest E that steps without OOM, its ms/step, and its peak
  device memory.

Acceptance: a measured ceiling, or a documented reason it could not be run
safely.

## Phase S4 -- Memory law

- S4.1 **Device state, analytically:** Σ over columns of bytes × slots, plus
  unit columns. Check this against `jax.Array.nbytes` over the state pytree
  for the bench cells and the optimizer nets (SGD, Adam), and as a function of
  `capacity_headroom` and `capacity_align`.
- S4.2 **Peak transients:** read `compiled.memory_analysis()` (temp size) for
  each step type (churn, training, batched per layout) across E. Then fit
  `peak = state + a·E + b·E·B + c`.
  - Flag any term that is not O(E): growth must be O(k), and no O(N²) term
    (the eager-grid bug class) may appear.
- S4.3 **Comparison:** plastax against C++ Plastix (in place and rebuild) and
  CSR+CUB, in bytes per live edge, from the plastix-synth-bench peak-memory
  columns.
- S4.4 HUMAN, out of plastax scope: the C++ scratch reduction (97 → ~30 B per
  slot) belongs to the C++ fork. Only reference it here.

Acceptance: a table and a plot of bytes per live edge against E for each step
type, with the fitted law and its residuals under 5%.

## Phase S5 -- Step-time law

- S5.1 **Fit** `t = t0 + α·E + β·k_edges (+ γ·E·B)` across the grid (E 5M-300M,
  sparsities 0.99-0.9999, B = 1), separately per phase: forward, prune,
  growth.
  - `t0` is the fixed per-step floor: the kernel count times launch cost, plus
    the host gap.
  - `α` should equal bytes per slot over DRAM bandwidth.
- S5.2 **Interpret the constants:**
  - `α` against roofline;
  - `t0` against the kernel count before and after the growth-kernel branch;
  - `β` against C++ (in-place growth is nearly free there).
- S5.3 **Batched:** extend with B for each layout, from the BATCHING_PLAN
  numbers.
- S5.4 **Figures:** step time against E (log-log) with the fitted law
  overlaid, per phase; the crossover E at which plastax overtakes C++, if the
  fusions bring it there.

Acceptance: a fitted law per phase whose prediction on held-out cells is
within 10%; `docs/scaling/RESULTS.md` with the figures.

## Deviations

(none yet)
