# plastax batching plan: batched training at edge-once cost

2026-10-01 · Status: **planned**. Coding-agent handoff, structured like
`DISTRIBUTION_PLAN.md`: phases with acceptance criteria, HUMAN markers, and a
Deviations section this document owns. Commits follow the agent-commit
protocol (`AGENTS.md`, `TAGS.md`, `SCOPES.md`). plastax stays
streaming-first; batching is a convenience for evaluation and mini-batch
training, and nothing here may slow the B = 1 path.

## Where we stand (`SCALE_PLAN.md` P4-P6, iterations 5-10)

- `make_step(net, static, batch_size=B, layout=...)` works for feed-forward
  (TOPOLOGICAL) nets.
  - Optimizer bundles reduce exactly, through `per_sample` /
    `incoming_batched`.
  - Other rules average the per-sample changes (mean of deltas).
- **The forward is edge-once for linear passes** (those declaring
  `linear_input`):
  - `"triton"` is jax_triton on NVIDIA;
  - `"csr"` is cuSPARSE;
  - the XLA edge-once product is used elsewhere;
  - `"auto"` on NVIDIA picks triton for 2 ≤ B ≤ 32 and CSR above that.
- Measured at 5.4M edges, ms per sample:

  | layout | B = 2 | B = 8 | B = 32 | B = 128 |
  |---|---|---|---|---|
  | triton | 0.386 | 0.143 | 0.090 | 0.081 |
  | csr | 1.208 | 0.353 | 0.130 | 0.077 |
  | edge list | 0.444 | 0.346 | 0.296 | 0.250 |

- **What is still per-sample, or rebuilt every step:**
  1. **The batched update.** The per-edge statistic Σ_b G[b,dst]·A[b,src],
     which is an SDDMM, is computed per sample: B passes over the edges. This
     is now the largest batched cost (P5.5 / P6.5).
  2. **Non-linear passes.** Any pass with a user `map` that is not declared
     linear falls back to the per-sample vmap edge list, which materialises an
     (E, B) intermediate.
  3. **The CSR view is rebuilt every step**, with one E-sized sort. At 200M
     edges and B = 128 this dominates: 2.67 ms per sample, against 3.49 for
     tuned CSR+CUB (plastix-synth-bench bigE).

## Phase B1 -- SDDMM for the batched update

- B1.1 **Structural declaration.** Add an optional, structural
  `linear_grad`, the mirror of `linear_input`, on `UpdateConn` / the optim
  bundles. It declares that `per_sample` is the outer-product statistic
  `G[dst] · A[src]`, where `G` and `A` are named unit fields. Without it, the
  current path is unchanged.
- B1.2 **Triton kernel** (jax_triton, NVIDIA): one edge-once pass.
  - Each program loads a block of (src, dst) and the B-length rows
    `G[:, dst]` and `A[:, src]`, and writes Σ_b of their product per edge.
  - There are no atomics, since there is one output per edge, so this should
    hit the DRAM roofline.
  - Feed the result to `incoming_batched` as the averaged statistic.
- B1.3 **XLA fallback, off NVIDIA:** a chunked gather of `(chunk, B)` rows and
  a reduction, with the chunk size bounded by a temporary-memory budget.
  Measure it against the per-sample loop; adopt it only if faster.
- B1.4 **Tests:** exactness against the per-sample path for all five optim
  bundles (SGD, momentum, Adam, AdamW, RMSprop), with tombstones and duplicate
  edges.
- B1.5 **Benchmark:** batched Adam training at 5.4M and 50M, B = 8 / 32 / 128,
  before and after, in ms per sample.

Acceptance: exact; the batched update is at least B/2× faster than the
per-sample loop at B ≥ 32; there is no B = 1 regression.

## Phase B2 -- Edge-once kernels for a generic `map` (SCALE_PLAN 6.4)

The obstacle: jax_triton kernels are written in Triton, so a user's JAX `map`
cannot be traced into them. Pallas could trace it, but Pallas Triton is
deprecated, and Mosaic GPU cannot express the scatter-add (`SCALE_PLAN.md`
iteration 10).

- B2.1 **Survey real maps.** Collect every `map` in `examples/`, `optim/` and
  `plastix-bench`, and classify them:
  - elementwise in a handful of edge and unit fields;
  - anything else (control flow, reductions, RNG).
  This bounds what a translator needs to cover.
- B2.2 **jaxpr → Triton translator, restricted.** Write a small code
  generator over the jaxpr of `map`:
  - supported: elementwise arithmetic, comparisons, `select`, and math
    functions (`exp`, `tanh`, `logistic`, `abs`, `max` / `min`);
  - inputs are gathered edge and unit fields, the output is combined by a
    named monoid (sum, max and min via atomics; others are not supported);
  - generated once per `NetworkStatic` (static at `make_step` time) and
    cached;
  - unsupported primitives fall back to the vmap edge list, with a debug
    reason exposed.
- B2.3 **Alternative, to measure before committing to B2.2:** an XLA
  edge-once map that computes the `map` per edge with B as the minor axis and
  reduces with `segment_sum` over a (capacity, B) array. It is edge-once in
  index reads but writes E × B. Fine at small B; it gives the bar B2.2 must
  beat.
- B2.4 **Tests:** for each surveyed map, exactness against the vmap edge list;
  a fallback test for an unsupported primitive.

Acceptance: the surveyed elementwise maps run edge-once on NVIDIA, at
least 2× faster than the vmap edge list at B = 8; no regression for linear
passes.

## Phase B3 -- Cache the CSR view across steps

- B3.1 **State.** The view (`perm`, `indices`, `indptr`) moves into
  `NetworkState`. Its shapes are static (capacity, and `num_units + 1`), and a
  structure-version counter is bumped by prune and growth. This is a pytree
  change, so it needs an `IMPLEMENTATION_PLAN.md` Deviation, and the pytree
  must stay donation-safe.
- B3.2 **Incremental update.**
  - Pruned edges stay in the view with value 0: `weight[perm]` already reads
    tombstones as 0 when `dead` masks them.
  - Grown edges go to a COO delta of static size `k_max`, applied after the
    SpMM by a small scatter.
  - The view is rebuilt when the delta overflows, or every R steps.
- B3.3 **Static nets** (no add or prune phase): build once and never rebuild.
  This is the evaluation case.
- B3.4 **Re-tune `auto`.** Re-measure the crossover between triton and CSR at
  5.4M, 50M and 200M with the cached view, and replace the fixed B > 32 rule
  with a rule depending on E and B if the data supports one.
- B3.5 **Tests:** the cached view matches a fresh rebuild after every step,
  over many churn steps; delta overflow forces a rebuild.

Acceptance: at 200M and B = 128, the per-sample time falls clearly below
tuned CSR+CUB's 3.49 ms; the view rebuild no longer appears in steady-state
profiles with low churn.

## Phase B4 -- Scope extensions (decide after B1-B3)

- B4.1 Batched PIPELINE (recurrent) nets need per-sample unit state that
  persists across steps, as `(B, num_units)` columns in state. This was
  deferred in P4.1. HUMAN: decide whether it is wanted.
- B4.2 Data-parallel sharding: shard the batch over devices and replicate the
  arena. It needs only an all-reduce of the B1 statistic, unlike Scheme-A.
  Useful on multi-GPU nodes (Narval).
- B4.3 Prune and growth currently see batch-mean unit statistics. Decide
  whether any heuristic (RigL's gradient magnitude) needs per-sample or
  squared statistics, and expose them through the B1 kernel.

## Deviations

(none yet)
