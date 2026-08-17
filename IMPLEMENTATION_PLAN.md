# plastax v1 implementation plan (coding-agent handoff)

Audience: a coding agent implementing this package. Read this whole file,
then the two design documents, before writing code. The stubs in
`src/plastax/` are the agreed API surface: implement them in place; do not
rename or restructure without recording the deviation (see "Deviations").

## Required reading, in order

1. `../plastix-jax-rung0-design.md` — the design this package implements.
   Sections are cited below as [D:n].
2. `../plastix-jax-lowering-analysis.md` — wider context (rung ladder);
   v1 is "rung 0" only, but phase bodies must remain separately traced
   functions so rung 1 composite wrapping stays a local change [D:7].
3. C++ semantics oracle, local clone at `../plastix`:
   - `include/plastix/traits.hpp` — trait slots and policy concepts.
   - `include/plastix/dispatch_cpu.hpp` — authoritative phase semantics:
     forward `:41-67`, pipeline `:202-223`, backward `:232-258`,
     update-conn two-pass `:450-469`, grow-fanout `:740-762`.
   - `examples/mlp-xor/`, `examples/ipc-multilayer/` — the two v1 examples.
4. JAX internals (local clone `../jax`, v0.11.1) — only when a mechanism is
   in doubt; key line references appear in [D] and in module docstrings.

## Scope contract

In scope: Propagation.PIPELINE and Propagation.TOPOLOGICAL; phases forward,
loss, backward, update_conn, prune_conn, add_conn, reset_global; named
monoids (sum/prod/max/min) including pytree-of-monoids struct accumulators;
donation-based in-place state; host driver with resort/grow retrace
protocol; CPU and GPU via plain jit (no Pallas, no FFI, no composites yet).

Out of scope (do NOT implement, even partially): AddUnit/PruneUnit,
generic `Monoid(op, identity)` lowering (raise UnsupportedMonoidError),
jax.Ref arenas, hijax, multi-device sharding, any MLIR emission.

## Non-negotiable invariants

These are the design decisions already made; violating them is a failed
milestone regardless of tests passing.

1. Phase elision is Python-level: absent trait slots contribute zero
   equations to the jaxpr. Never use lax.cond for presence [D:2].
2. All shapes static. Leaf mutation never retraces; only NetworkStatic
   changes do. The only retrace events are bucket growth and level
   reassignment [D:1, D:4].
3. Per-level conn arenas in topological mode; pipeline is the 1-bucket
   degenerate case. Deletion never resorts; level-preserving adds never
   resort [D:4, D:5].
4. Live counts derived from tombstone masks, never stored [D:1].
5. The step function is shape-preserving on the state pytree so every leaf
   donates; the donation warning is a CI error (pyproject filterwarnings).
6. Dead-slot handling via null-slot scatter (index = num_units or
   level_capacities[i]), never boolean-masked gathers that change shape.
7. Policy calls are vmapped per-element functions; policies never see raw
   columns, only views and write records [D:2].
8. Type discipline: mypy --strict must pass (ty check as the fast
   pre-commit pass; see TOOLING.md). Keep FieldSpec generic
   typing intact end to end (views return arrays typed by the spec's DT).
   Minimal comments; where a line needs explanation, cite the design doc
   section or jax/C++ file:line inline at that line.

## Milestones

Each milestone ends with: tests listed in `tests/README.md` for it passing,
mypy --strict clean, ruff clean, and a commit following the repository
commit protocol (see "Handoff conventions").

### M1: state, builder, topology, validation (est. 4-5 days)

Implement `_types.py` helpers as needed, `state.py` (make_empty_state,
live_conn_count, grow_bucket), `views.py`, `traits._validate_traits`,
`builder.py` (incl. from_topology), `topology.py`, `topo.initial_levels`,
`topo.capacity_policy`.
Acceptance: test_pytree, test_builder, test_topology green. Key subtleties:
- topology generators are host-side numpy edge enumeration; NO lowering
  work, no jax compute besides initializer calls (decision 2026-08-17).
  Conv kernels are unrolled per-edge — see topology.py docstring. dense/
  conv2d edge counts and conv receptive fields must match lax conv shape
  semantics exactly (test against lax.conv_general_dilated output shapes).
- Framing note for docs/comments: static-network matmul parity is a NON
  GOAL. The framework targets continual/streaming settings with runtime
  connection change; explicit edges are the point, not an inefficiency to
  apologize for. Densification (dense-bucket -> dot_general rewrite) is a
  rung 1 curiosity, not a v1 objective.
- register_dataclass meta fields must be hashable tuples; FieldSpec is
  frozen and hashable by construction — keep it that way.
- Builder finalize sorts each bucket by (dead, to_id) so
  indices_are_sorted=True holds from the start [D:3].
- make_empty_state fills columns with FieldSpec.default; dead defaults True.

### M2: pipeline sweeps + step assembly (est. 4-5 days)

Implement `monoid.segment_reduce`, `sweep.py`, `phases.build_phases` for
forward/loss/reset_global in pipeline mode, `step.make_step` with donation
and the weakref_lru_cache pattern (jax/_src/pjit.py:612 as template).
Acceptance: test_forward_pipeline, test_phases_elision green.
- Forward sweep: gather src activation + conn fields -> vmapped map ->
  segment_reduce into num_units accumulators -> vmapped apply over units ->
  write records merged into unit columns. Accumulator identity reset in the
  epilogue [D:3].
- Loss phase: clamp targets to output units, per_output over the static
  output-id tuple, loss value reduced into globals_.
- StepInputs scatters onto the static input-id tuple (from builder marks).

### M3: topological mode + backward (est. 4-5 days)

Implement the level-walk composition (Python loop over buckets, forward
1..L, backward L..1), `sweep.build_backward_sweep` (accumulate into
FROM_ID, apply on sources — direction reversal per dispatch_cpu.hpp:232).
Acceptance: test_forward_topo, test_backward green, mlp_xor example
trains XOR to convergence (matches C++ behavior qualitatively; exact
oracle in M5).

### M4: dynamics + resort protocol (est. 5-6 days)

Implement `phases.build_add_conn_phase`, prune/update phases,
`topo.recompute_levels`, `topo.resort`, `driver.Driver`.
Acceptance: test_update_prune, test_add_conn, test_resort green, and the
retrace-count assertions hold: pure add/prune (level-preserving) workload
compiles exactly once; one resort => exactly one recompilation
(jax.test_util.assert_num_jit_and_pmap_compilations, test_util.py:342).
- AddConn candidate window: neighbourhood levels ahead of src, as in
  native Neighbourhood=1 semantics [D:5]; candidates against dead slots of
  the SOURCE level bucket.
- Driver.step ordering: run step; on overflow grow_bucket and retry same
  inputs; on needs_resort resort between steps [driver.py docstring].

### M5: examples, oracle parity, hardening (est. 4-5 days)

Implement both examples end to end; build the oracle harness; donation and
retrace CI tests.
Acceptance: all tests green including test_oracle_cpp and test_donation.

Oracle harness: the C++ repo's examples print per-step scalars. Add a tiny
dump target there ONLY if one already exists to extend (do not refactor the
C++ build); otherwise generate golden CSVs once, check them into
`tests/golden/`, and document the generation command in the test file.
Tolerances: rtol=1e-5 pipeline (same reduction order), rtol=1e-4
topological (segment reduction order differs from the C++ sweep).

## Deviations

If a stub signature proves wrong during implementation, record the change
in this file under this heading (date, module, old -> new, one-line reason)
and update the design doc if the change is semantic. Do not silently drift.

## Handoff conventions

- Commits: follow the repository's agent-commit protocol (the
  `agent-commit` skill: `git agent-commit`, Conventional Commits with
  mandatory scope, hooks must run and pass). One commit per milestone
  minimum; more is fine at coherent boundaries.
- GPG identity, if signing is configured: "Claude (agent)"
  <ai@blobfish.icu>.
- Keep this plan, `tests/README.md`, and the README current as part of
  each milestone's definition of done.
- Do not add dependencies beyond pyproject without recording a Deviation.
- Performance work (fusion inspection, benchmarks vs plastix-bench) is
  explicitly NOT part of v1 milestones; correctness and the retrace
  contract are.
