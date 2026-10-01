# Changelog

All notable changes to plastax are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/) -- before 1.0, a minor release may
change the API. The first release is iterated as `0.1.0rcN` candidates. Versions come from git tags (hatch-vcs); see RELEASING.md.

## [0.1.0rc2] - Unreleased

Scale work: in-place structural churn at a cost that follows the churn, a
leaner and faster arena, batched steps with CSR / Triton backends, and TPU
readiness. Measurements and the full plan: `SCALE_PLAN.md`,
`docs/scale_plan/RESULTS.md`.

### Added

- `ProposeAddConn`: growth from policy-emitted proposals
  (`num_proposals`, `propose(u, j, g) -> (src, dst, score)`) instead of a
  candidate grid, so growth costs O(k) per step (plastix's sampled
  `GrowFanout`). Proposals may grow **parallel edges** unless the policy sets
  `dedupe = True` (an exact per-step check).
- Batched steps: `make_step(net, static, batch_size=B)` for feed-forward
  (TOPOLOGICAL) nets. plastax remains streaming-first; batching is a
  convenience. Optimizer bundles reduce exactly (one step on the batch-mean
  gradient, via new `per_sample` / `incoming_batched` hooks); other update
  rules average per-sample changes.
- `make_step(..., layout=)` for batched linear passes (a pass declaring
  `linear_input`): `"edge_list"`, `"csr"` (per-step CSR view + cuSPARSE on
  NVIDIA), `"triton"` (an edge-once Triton kernel via jax_triton on NVIDIA,
  an XLA edge-once product elsewhere), or `"auto"`.
- Aligned bucket capacities: `capacity_policy(align=)`, `capacity_align` on
  the builders, and `NetworkStatic.capacity_headroom` / `capacity_align`, so
  build, `grow_bucket` and `resort` size buckets with one policy.
- `Driver(check_every=N)`: read the overflow / resort flags every N steps
  instead of after every step.
- Optional extras: `cuda13`, `triton` (jax-triton), `tpu` (libtpu, also for
  ahead-of-time TPU compilation on any host).
- `examples/benchmarks/`: churn / layout / sort probes, `triton_check.py`,
  and `tpu_aot_check.py` (compile every step type for a TPU topology without
  a TPU).

### Changed

- Buckets are laid out source-major `(dead, from_id, to_id)` instead of
  destination-sorted; scatter-adds no longer serialise on one destination
  (GPU forward up to 3.3x faster). Results change only in float summation
  order.
- `Network.add_conn` is typed `AddConn | ProposeAddConn | None`.
- `topo.resort` keeps the build's headroom and never shrinks a carried-over
  bucket.
- No sweep passes a sorted-segment hint any more (`indices_are_sorted`).

### Fixed

- The add_conn duplicate check overflowed int32 once `num_units > 46340`
  (distinct pairs collided).
- The forward passed `indices_are_sorted=True` on buckets that in-place churn
  had un-sorted (undefined behaviour in XLA).
- `build_add_conn_phase` materialised the `num_units^2` candidate grid for
  every policy, even proposal and shortlist ones.
- Batched mean-of-writes no longer drifts columns a rule does not write.
- Scheme-A: batched CSR products are all-reduced across shards.

### Performance

- Synthetic churn at 300M edges (k = 64 edges per level): 144 ms -> 13.7 ms
  per step on an RTX 5000 Ada; state 4.1 GB. On plastix-synth-bench's own
  grid (64 units churned per update) plastax is 3.7-3.9x faster than a
  tuned CSR + CUB rebuild from 50M edges up and 1.3-1.6x behind hand-written
  C++ in place.
- Free-slot claim by a two-level search; unrolled binary searches; no top_k
  when every proposal fits; faster host builds (packed-key sort).

## [0.1.0rc1]

First release candidate: the v1 ("rung 0") core -- declarative traits over a
struct-of-arrays edge arena; pipeline and topological propagation; forward,
loss, backward, update_conn, prune_conn, add_conn and reset_global phases;
named-monoid combines; donation-based in-place state; the host driver's
grow / resort retrace protocol; Scheme-A multi-device sharding; optimizer
bundles (SGD, momentum, Adam, AdamW, RMSprop). See `IMPLEMENTATION_PLAN.md`.

[0.1.0rc2]: https://github.com/Sir-NoChill/plastax/compare/v0.1.0rc1...HEAD
[0.1.0rc1]: https://github.com/Sir-NoChill/plastax/releases/tag/v0.1.0rc1
