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

### Recorded

Tooling / infrastructure (2026-08-17, scaffolding handoff):

- pyproject (build): `requires-python` `>=3.11` -> `>=3.12`. jax 0.11 requires
  >=3.12, so uv's universal resolver has no solution at 3.11. Interpreter
  pinned to 3.13 via `.python-version`.
- pyproject (build): dev dependencies moved from the
  `[project.optional-dependencies].dev` extra to a PEP 735
  `[dependency-groups].dev` table so the bare `uv run` hook entries resolve
  them by default; setup is now `uv sync`. Added PyPI metadata (MIT license,
  author, classifiers, keywords, `py.typed`).
- pyproject (build): dropped the `follow_imports="skip"` override for
  jax.*/jaxtyping.* — both ship py.typed, so mypy resolves `jax.Array` and
  erases jaxtyping's `dtype[array, shape]` to `array`; F722/F821 ignored
  globally (jaxtyping shape strings read as forward refs to ruff only).
- repo: no repo-local commit wrapper — contributors attribute and sign with
  their own agents (per user, review 2026-08-17).

Scaffold type-cleanliness (2026-08-17, `src/plastax/`, no bodies implemented):

- All modules: converted to PEP 695 generics (`class Foo[T]`, `def f[T]`)
  under the new 3.12 floor, as ruff UP046/UP047 require. Behaviour-preserving
  (generics erase at runtime); the module-level `TypeVar`/`Generic` forms are
  gone. Revert by ignoring UP046/UP047 in ruff if the explicit-TypeVar style
  is preferred.
- _types.py: `UnitIdx`/`ConnIdx` are `NewType(_, Int32[Array, ""])` — scalar
  int32, distinct index types. (The interim `object` base, used while
  follow_imports=skip made jax.Array resolve to Any, is reverted now that the
  skip is gone and jaxtyping erases the shape to a real subclassable `array`.)
- traits.py: private `_validate_traits` param `type[Network[object]]` ->
  `type[Network[Any]]`; protocol variance is inferred per-class under PEP 695.
- topology.py: `dense`/`conv2d` `init=` default is now a module-level
  initializer singleton (ruff B008), the same initializer object as before.
- 18 `# noqa: F722` (3 also `# type: ignore[name-defined]` in monoid.py) for
  jaxtyping shape-string annotations misparsed as forward refs — the friction
  TOOLING.md sanctions silencing with rule-scoped ignores.

M1 (2026-08-17, host-side construction + state + topology + validation):

- state.py / NetworkStatic: added `input_ids: tuple[int, ...]` and
  `output_ids: tuple[int, ...]` static meta fields. The rung0 sketch had no
  field for the builder-recorded input/output units that M2's loss (clamp to
  outputs) and StepInputs (scatter onto inputs) consume.
- topology.py: `Block` Protocol gains `@runtime_checkable`. Without it,
  pytest's jaxtyping+beartype instrumentation cannot build a checker for the
  bare Protocol where it appears in `input_units`/`dense`/`conv2d`/`sequential`
  signatures, and the package fails to import under pytest. (Scaffold bug.)
- topology.py: the concrete `Block` impl `_EdgeBlock` is a non-frozen
  dataclass — a frozen field is read-only and does not structurally satisfy
  `Block.num_units: int`.
- state.grow_bucket imports `topo.capacity_policy` function-locally (topo
  imports state at module level; a top-level back-import would cycle).
- register_dataclass: verified NO change needed — the bare
  `@jax.tree_util.register_dataclass` correctly reads
  `field(metadata=dict(static=True))` to split meta vs data fields in jax 0.11.

M2 (2026-08-17, pipeline forward sweep + step assembly):

- step.py / StepResult: registered as a pytree dataclass (jit cannot return
  an unregistered dataclass) and gains `loss: Float[Array, ""]`. "Loss reduced
  into globals_" is not implementable with an opaque user `GS` (`Network[None]`
  is exercised), so the summed `per_output` loss lands in `StepResult.loss`,
  sibling to `overflow` (0.0 when `net.loss is None`).
- phases.py / Phase: return type `state` -> `(state, loss_contribution)` so
  make_step folds loss into StepResult.loss without any phase touching
  globals_; non-loss phases return 0.0. `StepInputs` is now a registered
  pytree dataclass (inputs leaf; targets leaf or None).
- sweep.build_forward_sweep: added `input_ids` param. dispatch_cpu.hpp:217-222
  Applies only over [NumInput, NumUnits); plastax input ids are an arbitrary
  tuple, so input units are skipped via a static boolean mask at merge (else
  each step overwrites the scattered inputs with their identity accumulator).
- step.py: cache via `functools.cache`, not `jax.util.weakref_lru_cache` (not
  importable off jax==0.11.0). Same hash/eq reuse on (net, static); strong
  refs, benign for v1.
- __init__.py: export `StepInputs` (was omitted; needed to call any StepFn).
- monoid.py: added `Monoid.identity_for(dtype)`; `materialize_acc_columns` is
  a correct utility but not on the forward hot path (jax.ops.segment_* supplies
  identity-at-rest for named monoids).
- step.py: one `# type: ignore[arg-type]` for a mypy Hashable false positive on
  `type[Network[GS]]` (parameterized generic base); `net` is hashable.

M3b (2026-08-17, update_conn phase + mlp_xor XOR training):

- sweep.py: new primitives `_build_conn_update` (private core) plus
  `build_incoming_conn_update`/`build_outgoing_conn_update` (public
  directional wrappers), mirroring `_accumulate_into` plus
  `build_forward_accumulate`/`build_backward_accumulate`'s shape. No
  segment_reduce: a `ConnWrite` targets only the writing edge's own row
  (never a cross-edge aggregation), so the dead-conn "drop" is a plain
  `jnp.where(dead, old, written)` keep-old merge rather than an
  out-of-bounds null-slot scatter -- the degenerate case of the same
  discipline.
- phases.py: `build_update_conn_phase` (public, matching
  `build_add_conn_phase`'s naming -- not the private `_build_*_phase`
  convention forward/backward/loss/reset_global use -- so it stays
  independently unit-testable, per tests/test_update_conn.py). Two full
  passes over EVERY bucket in `state.conns` (PIPELINE's 1-tuple or
  TOPOLOGICAL's per-level tuple): all buckets' incoming pass completes and
  is merged before any bucket's outgoing pass runs, matching
  dispatch_cpu.hpp:450-469's two flat loops over its single (unbucketed)
  conn arena. `build_phases`' update_conn guard-raise is removed;
  prune_conn/add_conn guards stay (M4).
- examples/mlp_xor.py: implemented per mlp_xor.cpp's traits
  (SigmoidForwardPass, SigmoidBackwardPass, MSELoss, GradientDescentConn).
  New extra unit field `loss_grad`, beyond the stub's `grad_pre_act`:
  dispatch_cpu.hpp stages Loss's dL/dActivation into BackwardAcc, a
  framework-internal per-unit column that is always fresh (zeroed right
  after the Apply that consumes it); plastax's backward accumulator is
  instead a value local to `backward_phase`'s own trace closure
  (phases.py), with no channel for an earlier, separate phase function to
  write into it. Reusing `grad_pre_act` itself for the seed (additive
  apply, `u[GradPreAct, i] + acc`) is unsafe across more than one training
  step: `grad_pre_act` is written every step for every non-input unit, so a
  hidden unit's stale PREVIOUS-step value would be added into its NEW
  gradient. `loss_grad` is written only for `output_ids`, every step (never
  stale), and permanently 0.0 for every other unit (never written at all),
  which is what makes `(acc + u[LossGrad, i]) * a * (1-a)` safe as a
  REPLACE (matching the oracle's Apply exactly) across arbitrarily many
  steps. Topology: 3 input-slot units (x1, x2, a constant-1.0 "bias", fed
  as `StepInputs.inputs[2]` every call, exactly mirroring mlp_xor.cpp's
  `Inputs[.][2]`) -> 4 hidden -> 1 output, via topology.sequential/dense
  (not the oracle's exact RandomUniformWeight seeds/API -- exact oracle
  parity is M5). `XorNetEval`: a forward-only sibling `Network` sharing
  XorNet's exact field set, used for final inference against the SAME
  (static, state) with no side effects (mirrors mlp_xor.cpp's separate
  `Net.DoForwardPass` vs. `Net.DoStep`, and doubles as a phase-elision
  example). Trains to loss ~9e-4 and classifies all 4 patterns correctly in
  ~2.7s wall-clock (`uv run python examples/mlp_xor.py`).
- .pre-commit-config.yaml / .github/workflows/ci.yml: ruff's `examples/`
  exclude narrowed to `^examples/ipc_multilayer\.py$`; CI's ruff
  check/format invocations gained `examples/mlp_xor.py` explicitly (mypy's
  `files=["src"]` is unchanged -- examples/ correctness is enforced by
  tests/test_mlp_xor.py actually training and asserting XOR, not by mypy).

## Handoff conventions

- Commits: Conventional Commits with a mandatory scope (TAGS.md / SCOPES.md),
  one scope per commit, hooks must run and pass, never `--no-verify`. One
  commit per milestone minimum; more is fine at coherent boundaries.
- Contributors and their agents commit under their own identity and sign as
  they choose; the repo ships no signing wrapper or keys.
- Keep this plan, `tests/README.md`, and the README current as part of
  each milestone's definition of done.
- Do not add dependencies beyond pyproject without recording a Deviation.
- Performance work (fusion inspection, benchmarks vs plastix-bench) is
  explicitly NOT part of v1 milestones; correctness and the retrace
  contract are.
