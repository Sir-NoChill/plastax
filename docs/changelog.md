# Changelog

All notable changes to plastax are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/) -- before 1.0, a minor release may
change the API. The first release is iterated as `0.1.0rcN` candidates. Versions come from git tags (hatch-vcs); see {doc}`development/releasing`.

## [0.1.0rc3] - Unreleased

Fused Triton kernels for the streaming churn step, plus release preparation.

### Added

- `make_step(fuse_prune=...)` ("auto", "triton", "xla", "off"): a streaming
  step can evaluate the prune_conn predicate inside the forward's edge sweep,
  so each bucket's edge columns are read once instead of twice. On an NVIDIA
  GPU with `plastax[triton]`, an unsharded linear forward runs as one Triton
  kernel per bucket. "auto" fuses only then; the decision is recorded on the
  returned step as `step.prune_fusion`. Tombstones and free-slot counts match
  the two-pass step exactly.
- `make_step(growth=...)` ("auto", "xla", "triton"): selects the add_conn
  free-slot claim. "triton" claims and writes every growing bucket in one
  jax_triton kernel instead of about 20 XLA kernels per bucket; both engines
  pick the same slots. "auto" uses Triton where it can run.
- Conformance-vector scripts (`scripts/parity_vectors.py`,
  `scripts/emit_parity_goldens.py`) that emit plastax results as goldens for
  the plastax-cpp C++ library (the five `mlp_optim_*` optimizer vectors), and
  a bit-exact NumPy port of plastax-cpp's `UniformReal` pinned to the C++
  golden.
- Benchmarks: `examples/benchmarks/fused_prune_check.py`;
  `churn_probe.py --growth`; `triton_check.py` times the backward product
  and a dead-slot fraction.
- `examples/benchmarks/growth_bench.py`, `run_growth.sh` and `plot_growth.py`:
  the time of one growth phase per strategy against live units, live
  connections and P on CPU and GPU, with a per-unit N x P fit and a
  point-by-point comparison against plastax-cpp (`benchmarks/growth.md`).
- `CITATION.cff`, contributing guide, CODEOWNERS and Dependabot config; CI
  now checks docstrings with pydoclint and builds the docs with warnings as
  errors.

- Growth-rule knobs shared by `ScoreAddConn` and `ProposeAddConn`:
  `selection` ("top_k" default, "threshold" with a per-step `threshold(g)`,
  "all"), `max_new_per_step` (a cap across source levels, level-ascending),
  `direction` ("any" default, "deeper", "same_or_deeper"),
  `allow_self_loops` (default off), `trigger` ("every_step" default,
  `("every", n)`, "on_units_added", or "when" with a `when(g)` method) and
  `on_overflow` ("flag" default, or "error" to raise). Combinations are
  validated when the `Network` subclass is defined.
- `ScoreAddConn.candidates`: "exhaustive" (default), "shortlist" (the M x M
  grid of the top-`shortlist_size` units by `importance`) or
  "shortlist_per_level" (per source level: its top-M sources x the top-M
  destinations its validity window admits; topological only).
- `predicate_add_conn(should_add, init, **knobs)` adapts a boolean predicate
  to a `ScoreAddConn` (True grows, False vetoes; `selection = "all"`,
  `dedupe_step = True`).
- `NetworkState.grown` (connections the step's growth committed),
  `NetworkState.overflow` (whether it dropped any for lack of capacity) and
  `NetworkState.units_added` (units the step's unit addition placed).
- `Network.structural_interval` (default 1) runs unit addition and growth
  (and so the resort growth triggers) only on every n-th step. Connection and
  unit pruning run every step, as in plastax-cpp.
- `Network.unit_capacity` (default None): the number of unit slots. The built
  units are live and the slots above them are free, marked in the new
  built-in `PRUNED` unit column. A slot holding no live unit is skipped by
  every pass's apply and takes no part in growth. Input and output units are
  always built units and are never pruned, so the loss is unchanged. A net
  without a capacity has no `PRUNED` column and steps exactly as before.
  `Network.max_levels` (default 1024) bounds unit levels.
- `UpdateUnit` and the `Network.update_unit` slot: `update(u, i, g)` writes
  every live unit, inputs and outputs included, after the backward pass and
  before the connection update. A batched step runs it on each sample's
  units. The unit-update conformance goldens are enforced.
- `PruneUnit` and the `Network.prune_unit` slot (requires `unit_capacity`):
  `predicate(u, i, g)` is evaluated on the pre-phase state of every live unit
  except the inputs and outputs, which are never pruned. Pruning is
  permanent: a selected unit is marked `PRUNED`, its columns reset to their
  declared defaults (it keeps its level), and every connection incident to it
  is tombstoned in the same phase. It runs every step after the connection
  update and before connection pruning, and once on the batch-mean state of a
  batched step. Declaring it under Scheme-A sharding raises
  `NotImplementedError`. The unit-prune conformance goldens are enforced.
- `AddUnit` and the `Network.add_unit` slot (requires `unit_capacity`):
  `spawn(u, parent, g)` returns whether a unit live at the start of the
  phase spawns a child and the child's level offset; `init(u, child, parent,
  g)` writes the child's fields. The i-th spawning parent by ascending id
  takes the i-th lowest free slot, ids pruned earlier in the same step
  included; a spawn with no free slot left is dropped and sets the new
  `NetworkState.unit_overflow`. A child starts at its column defaults with
  the level `clamp(level(parent) + offset, 1, max_levels - 1)`, has no
  connections, and is not a parent in its own step. The phase runs after
  connection pruning and before growth, gated by `structural_interval`, once
  on the batch-mean state of a batched step; it sets
  `NetworkState.units_added`, which the "on_units_added" growth trigger reads
  in the same step. Declaring it under Scheme-A sharding raises
  `NotImplementedError`. All unit-lifecycle conformance goldens are enforced.
- All growth_v2 conformance goldens are enforced: scoring, selection, the
  validity window, triggers and growth on the batch-mean state, including
  the per-level shortlist.
- `SoftmaxCrossEntropyLoss(seed_field)`: softmax over the output activations
  with cross-entropy against a target distribution, in the max-subtracted
  log-sum-exp form, so it stays finite for logits whose exponentials
  overflow. It matches plastax-cpp's loss bit for bit on the `loss_v1`
  conformance goldens, which both libraries enforce.
- `UnitView.gather(spec, ids)` reads one field at several units.
- Per-connection proposers (`proposer = "per_connection"`) run under Scheme-A
  sharding. Each shard proposes for its own connections; their
  `(src, dst, occurrence)` ranks and rng keys are global across shards, so
  the sharded step commits exactly what the single-device step does.
  Previously this combination raised `NotImplementedError`.

### Changed

- **Breaking:** an overflowing growth is finished inside the same step. When
  a step's growth claim drops selected candidates for lack of room, the
  `Driver` grows the full buckets and claims exactly those candidates, in the
  total order, at the same step; forward, backward and the updates are not
  re-run, the step counter advances once, and the result equals a step whose
  buckets were large enough from the start. Previously the whole step re-ran
  as a new step, so an overflowing step applied its learning update twice.
  `StepResult.growth_remainder` carries the dropped candidates
  (`phases.GrowthRemainder`) and `make_growth_retry(net, static)` is the
  jitted claim; `state.overflow` reads False after a `Driver.step`. The
  `Driver` now takes `batch_size`, `layout`, `fuse_prune` and `growth` and
  passes them to `make_step`, so batched steps recover the same way. The
  `grow_claim_topological_regrow` golden's `retry` block is now that claim at
  the same step, plus a `step_growth` block for the whole step's growth.
- **Breaking:** a batched step no longer averages the unit state implicitly.
  How the per-sample unit columns combine before the once-per-batch phases is
  a new policy slot, `Network.batch_reduction` (`BatchReduction`), mapping
  each unit column to a `Reduction`: `MEAN`, `SUM`, `FIRST` (sample 0) or
  `NOT_BATCHED`. `make_step(..., batch_size=B)` requires one. Every column a
  per-sample phase writes (the input scatter's `ACTIVATION`, the loss's seed
  field and each column a forward, backward or unit-update rule writes) must
  be declared `MEAN`, `SUM` or `FIRST`: an undeclared or `NOT_BATCHED` one is
  rejected when the `Network` subclass is defined (`ACTIVATION`, the seed
  field) or when the batched step is first traced (rule writes). Unwritten
  columns need no declaration and keep their value. `FieldReductions`
  declares columns one by one; `MeanFloatFirstRest()` reproduces the old
  behaviour (the mean of every floating column, sample 0 of the rest) and
  gives bit-identical results. To migrate, add
  `batch_reduction = px.MeanFloatFirstRest()` to every batched net.
  `phases.batch_mean_units` is replaced by `phases.reduce_batch_units`.
- PIPELINE growth claims per source level: each level's candidates first take
  the dead slots its own former connections left, then spill to the bucket's
  never-used tail (levels ascending); a level never takes another level's
  dead slot, and `overflow` now means the tail ran out. The new
  `NetworkState.tail_start` is the bucket's high-water mark; resort compacts
  and resets it. The `Driver` grows a PIPELINE bucket whose tail ran out (it
  previously regrew only full buckets, and looped forever when other levels'
  dead slots remained). TOPOLOGICAL growth keeps its strict per-bucket claim.
  plastax-cpp claims identically (ADR-010).
- **Breaking:** the loss is whole-output. A `Loss` declares `seed_field` (the
  float unit column its gradient seed goes to) and implements
  `calculate_loss(u, outputs, targets, g) -> (loss, seed)`, called once over
  every output unit; the framework writes `seed` into `seed_field` at the
  output ids. This replaces `per_output(u, i, target, g)`, which saw one
  output at a time and so could not express losses that couple the outputs.
  Port a per-output loss by computing over `u.gather(px.ACTIVATION, outputs)`
  and returning the summed loss with the seed vector; a loss that still
  defines `per_output` fails validation with a pointer to the new shape.
  MSE-style losses ported this way produce bit-identical results.
- **Breaking:** `AddConn` is renamed `ScoreAddConn`, and growth rules
  declare `max_new_per_level` instead of `max_candidates` (including the
  examples' `make_net(max_new_per_level=...)`). The shortlist attributes are
  now `candidates = "shortlist"` / `"shortlist_per_level"` with
  `shortlist_size`, replacing `max_candidate_units` and
  `shortlist_per_level = True`. Setting a removed name raises a `TypeError`
  naming its replacement.
- **Breaking:** score rules no longer deduplicate by default. The single
  `dedupe` flag (default on) is replaced by `dedupe_live` and `dedupe_step`,
  both defaulting off as on the propose path, so a score rule regrows live
  edges as parallel edges unless it sets `dedupe_live = True`.
- **Breaking:** the per-level shortlist's destination pool is the rule's
  validity window (gap and `direction`) rather than strictly deeper units; a
  rule that relied on the old pool declares `direction = "deeper"`.
- Growth selection commits in the total order even when the budget equals
  the candidate pool. Previously such a step committed in candidate order,
  which decided which candidates an overflowing bucket dropped.
- **Breaking:** `ProposeAddConn` declares who proposes. Rules carry a
  `proposer` ("per_unit" — the default — "per_connection", or "global"), and
  `propose` takes the proposer-specific signature with a counter-based `rng`
  argument (`plastax.rng`) keyed by the network seed, the step counter and the
  proposing site, so proposal streams replay exactly and vary per step. The
  `num_proposals` attribute is now `proposals_per_proposer`, and the single
  `dedupe` flag is two opt-in stages, `dedupe_live` and `dedupe_step`, both
  defaulting off (a duplicate proposal grows a parallel edge by design).
  Existing rules: declare `proposer = "global"`, accept (and ignore) `rng`,
  and spell out the dedupe stages. A new `Proposal` NamedTuple names the
  `(src, dst, score)` triple.
- Topological networks with a growth rule now allocate a connection bucket
  for the deepest unit level, so growth can source edges there (they commit
  backward and trigger a resort), matching the C++ implementation and the
  conformance goldens. Candidates sourced at the deepest level were
  previously dropped without trace; score rules that relied on that silent
  window should veto with `-inf` (a merely-low finite score is still a
  candidate).
- Growth selection follows a deterministic total candidate order:
  `(-score, src, dst, candidate index)` ascending. Distinct scores select
  exactly as before; score ties now resolve by the lower source id, then the
  lower destination id, then the earlier candidate, instead of by candidate
  position alone — identical across backends and matching the C++
  implementation. A NaN score now sorts last (it was already vetoed at
  commit).
- **Breaking:** the growth window moved off the network onto the growth rule.
  `Network.neighbourhood` is removed; set `max_level_gap` (int, default 1, read
  structurally) on the `AddConn`/`ProposeAddConn` policy instead. A subclass
  that still sets `neighbourhood` fails validation with a pointer to the new
  name. Window semantics are unchanged.
- The add_conn phase is assembled from named, individually testable stage
  functions (`candidates_grid`/`candidates_shortlist`/`candidates_per_level`/
  `candidates_propose`, `apply_validity`, `dedupe_live`, `dedupe_step`,
  `select`); the computation is unchanged.
- `layout="auto"` keeps the Triton batched product through batch 64.
- The fused prune feeds its free-slot counts straight to the Triton claim.
- The CIFAR dynamic-sparse baseline rewires on device.
- Contributor documentation moved into the docs site (Development section).

### Fixed

- The Triton batched kernel masks dead slots and pre-reduces runs of equal
  targets in the backward product.
- jaxtyping's import hook no longer instruments the Triton edge kernel.
- A topological `topo.resort` raises `ValueError` when the live connections
  contain a cycle (growth can commit one, through an input or between
  same-level units), as plastax-cpp's level recompute does; it used to
  level the cycle units at the relaxation bound. The new `topo.has_cycle`
  makes the check. `topo.initial_levels` takes `input_ids` and keeps them at
  level 0 when an acyclic edge feeds them, as `recompute_levels` does; a
  cycle through an input still raises.

### Performance

- Growth on the GPU sorts the total candidate order with three stable
  one-key radix passes from 2^17 candidates (CUB, where the four-key sort
  ran as XLA's merge network), ranks the source levels in one pass, and
  claims every bucket's slots in one batched XLA claim. The committed edges
  are unchanged. Steady-state growth-call times on an RTX 5000 Ada: per-unit
  0.56 -> 0.36 ms at 262K candidates and 197 -> 24 ms at 32M, exhaustive
  N = 4096 98 -> 16 ms; the per-unit phase runs 72 kernels instead of 123.
- The XLA claim counts `grown` from its commits instead of reducing every
  dead mask twice: a global proposer's growth call takes 2.3x less on CPU.
- `growth_bench.py` warms up for 20 calls before timing (the GPU floor was
  overstated by up to 0.3 ms) and records the calls warmed up.

## [0.1.0rc2] - 2026-10-01

Scale work: in-place structural churn at a cost that follows the churn, a
leaner and faster arena, batched steps with CSR / Triton backends, and TPU
readiness.

### Added

- `ProposeAddConn`: growth from policy-emitted proposals
  (`num_proposals`, `propose(u, j, g) -> (src, dst, score)`) instead of a
  candidate grid, so growth costs O(k) per step (plastax-cpp's sampled
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
- `Network.add_conn` is typed `AddConn | ProposeAddConn | None` (since renamed `ScoreAddConn`).
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
  per step on an RTX 5000 Ada; state 4.1 GB. On the plastax-cpp synthetic
  benchmark's grid (64 units churned per update) plastax is 3.7-3.9x faster than a
  tuned CSR + CUB rebuild from 50M edges up and 1.3-1.6x behind hand-written
  C++ in place.
- Free-slot claim by a two-level search; unrolled binary searches; no top_k
  when every proposal fits; faster host builds (packed-key sort).

## [0.1.0rc1]

First release candidate: the v1 core -- declarative traits over a
struct-of-arrays edge arena; pipeline and topological propagation; forward,
loss, backward, update_conn, prune_conn, add_conn and reset_global phases;
named-monoid combines; donation-based in-place state; the host driver's
grow / resort retrace protocol; Scheme-A multi-device sharding; optimizer
bundles (SGD, momentum, Adam, AdamW, RMSprop).

[0.1.0rc3]: https://github.com/Sir-NoChill/plastax/compare/v0.1.0rc2...HEAD
[0.1.0rc2]: https://github.com/Sir-NoChill/plastax/compare/v0.1.0rc1...v0.1.0rc2
[0.1.0rc1]: https://github.com/Sir-NoChill/plastax/releases/tag/v0.1.0rc1
