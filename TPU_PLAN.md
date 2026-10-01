# plastax TPU plan: validation on hardware, then a SparseCore edge kernel

2026-10-01 · Status: **planned, blocked on TPU access** (HUMAN: Google Cloud
verification in progress). Coding-agent handoff, structured like
`DISTRIBUTION_PLAN.md`: phases with acceptance criteria, HUMAN markers, and a
Deviations section this document owns. Commits follow the agent-commit
protocol (`AGENTS.md`, `TAGS.md`, `SCOPES.md`).

## Where we stand (verified 2026-10-01, `SCALE_PLAN.md` iteration 11)

- **Every step type compiles for TPU, ahead of time, on any host.** The `tpu`
  extra (libtpu 0.0.48 with jax 0.11.2) compiles against a topology
  description via `jax.experimental.topologies.get_topology_desc`. These
  topologies work: v5e:2x2, v6e:2x2, v5p:2x2x1, v4:2x2x1.
  `examples/benchmarks/tpu_aot_check.py` covers:
  - churn with proposal growth, dedupe and grid growth;
  - streaming Adam;
  - batched Adam in each layout.
- **What runs on TPU today.**
  - The edge-list gather and scatter stay on the TensorCore; XLA does not
    offload them to SparseCore by default on v4, v5p or v6e.
  - Batched `layout="auto"` off NVIDIA uses the XLA edge-once product.
  - `layout="csr"` compiles to generic XLA, since the cuSPARSE lowering is CUDA
    only.
- **The XLA cost model predicts the TensorCore edge list will be slow.** It
  charges random gather and scatter at tile granularity: the 5.4M-edge forward
  is estimated at 70-94 GB of traffic, about 4 KB per index, against 4 B on
  GPU. This is an estimate, not a measurement.
- **The SparseCore route exists in jax 0.11.2.**
  - `jax.experimental.pallas.tpu_sc` provides `load_gather`, `store_scatter`
    and `addupdate_scatter` on vector subcores (`VectorSubcoreMesh`). Those
    are the edge-once primitive: gather `act[src]`, then scatter-add into
    `dst`.
  - They operate on refs in **subcore VMEM, not HBM**, and have no interpret
    mode, so they compile without hardware but only run on hardware.
- **No sparse MXU mode exists on TPU.** NVIDIA-style 2:4 sparsity does not
  apply either: our nets are 99-99.99% unstructured.

## Phase T0 -- Access and environment (HUMAN)

- T0.1 HUMAN: obtain a TPU VM. Preferred order:
  - v5p or v6e, which have SparseCore (the gather/scatter hardware);
  - then v5e, which is TensorCore only and is the baseline.
  - Record the topology, the HBM size, and the libtpu version in this
    document.
- T0.2 Install: `uv sync --extra tpu`, then
  `python -c "import jax; print(jax.devices())"`.
- T0.3 Data: bench cells are generated on the TPU VM by
  `plastix-synth-bench/gen`, or copied up. `E50M_s0.999` is about 600 MB.
  Respect the host-RAM rule: the host build needs about 45 B/edge (see
  `SCALING_PLAN.md`).

Acceptance: `jax.devices()` lists TPU devices; `tpu_aot_check.py` runs
unchanged on the VM.

## Phase T1 -- Correctness and baseline on hardware

- T1.1 Run the full fast suite with `JAX_PLATFORMS=tpu`. Expected
  differences, each to be checked and either fixed or recorded as a Deviation:
  - default matmul precision (bf16 passes on TPU unless
    `jax_default_matmul_precision` is set);
  - uint64 support for the dedupe sort, which needs `jax_enable_x64`
    semantics on TPU;
  - scatter determinism.
- T1.2 Measure the B = 1 churn step on bench cells `E5M_s0.999` and
  `E50M_s0.999`, broken down by phase (forward, prune, growth) with
  `jax.profiler` traces.
  - Compare with the cost-model estimate and with the GPU numbers (0.49 and
    2.97 ms/step on the RTX 5000 Ada).
- T1.3 Measure batched SGD at 5.4M edges (B = 2 / 8 / 32 / 128) for
  `edge_list`, the XLA edge-once product, and `csr` (generic). Re-tune the
  off-NVIDIA branch of `layout="auto"` from these numbers.
- T1.4 Probe XLA's own SparseCore offload. Search libtpu's flag list for
  sparse-core offload options for gather, scatter and segment ops, and try any
  found on the edge-list forward. If XLA offloads by itself, T2 may shrink to
  a layout change.

Acceptance: the suite is green on TPU, or every failure is listed with a
Deviation; there is a per-phase table for E5M and E50M; "auto" is re-tuned
for TPU.

## Phase T2 -- SparseCore edge kernel (design and AOT now, run with T0)

This can be written and AOT-compiled for v5p and v6e before hardware arrives.
Correctness and speed need T0.

- T2.1 **Spike.** Write a minimal `tpu_sc` kernel: one vector subcore gathers
  `x[src]` for a block of edges and `addupdate_scatter`s into a VMEM
  accumulator. Then:
  - AOT-compile it for v5p:2x2x1 and inspect the lowered module.
  - Record from the compile result and the docs the subcore count, the VMEM
    per subcore, and the vector width.
- T2.2 **Destination-blocked view.** Because the accumulator must live in
  VMEM, each subcore owns a destination range sized to its VMEM.
  - The edges of a bucket are partitioned by `dst` range (one tile per range),
    and within a tile by `src` range, so the source activations can be DMA'd
    in by range.
  - This is a new derived view of the arena, built the same way as the CSR
    view (`build_csr_forward`): a per-step device sort keyed on
    `(dst_block, src_block)`, or a cached view (see `BATCHING_PLAN.md` B3).
  - The arena stays source-major; prune and growth are unchanged.
- T2.3 **Kernel.** For each tile: DMA the edge block (src, dst, weight) and
  the source-range activations into VMEM, gather, multiply, then
  `addupdate_scatter` into the accumulator. Write the accumulator back once
  per destination range.
  - Open question for T0: how `addupdate_scatter` resolves duplicate
    destinations within one vector (are they summed, or last-wins?). If they
    are not summed, the view must spread duplicate `dst` across lanes, for
    example by sorting `dst` within a tile and doing a segmented reduce.
- T2.4 **Plumbing.** Add `layout="sparsecore"` for linear passes (`linear_input`
  declared), matching how `"csr"` and `"triton"` are wired.
  - Make "auto" pick it on SparseCore TPUs, and fall back to the edge list on
    v5e.
  - Scheme-A: one view per shard, with partial sums all-reduced as in the CSR
    path.
- T2.5 **Tests.** Under `@pytest.mark.tpu`, skipped without TPU devices:
  - exactness against the edge-list forward on random nets, including
    duplicate edges and tombstones;
  - the AOT compile check runs in normal CI via the `tpu` extra, if CI time
    allows; otherwise it is a slow test.

Acceptance: an AOT compile check is in CI or the slow tests; on hardware, the
kernel is exact against the edge list and the forward is faster than the
TensorCore edge list at E50M. If it is not faster, record why and stop at T1.

## Phase T3 -- Beyond the forward (after T2 pays off)

- T3.1 The backward (reduce into sources) through the same view, keyed on
  `(src_block, dst_block)`.
- T3.2 The prune predicate and the free-block counts as a streaming
  TensorCore pass, which is plain elementwise work. Check that XLA's TPU
  fusion reads each column once; this mirrors the GPU fusion work (prune
  fused into the forward).
- T3.3 Growth on TPU: the many small kernels of the GPU path become many small
  HLO ops. Measure first. The single-kernel growth work on GPU
  (`perf/growth-kernel`) may have a TPU analogue, a TensorCore Pallas kernel,
  if launch-like overhead shows up.
- T3.4 Multi-chip: Scheme-A over a v5p slice, measuring ICI all-reduce cost
  against compute.

## Open questions (resolve on hardware)

- VMEM per vector subcore, and the subcore count per chip (v5p, v6e).
- Duplicate-destination semantics of `addupdate_scatter`.
- Whether XLA has any flag-gated automatic SparseCore offload for scatter.
- Whether bf16 activations are acceptable for the forward. The arena stays
  f32; this would be an opt-in.

## Deviations

(none yet)
