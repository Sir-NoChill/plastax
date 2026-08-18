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

M4a (2026-08-17, PruneConn tombstoning + K-bounded AddConn; M4b's
topo.recompute_levels/resort/driver.py stay untouched NotImplementedError
stubs):

- phases.py: new `build_prune_conn_phase` (vmaps `prune_conn.predicate`
  over every row of every bucket -- including already-dead rows, since
  vmap cannot skip them the way dispatch_cpu.hpp:541-542's `continue` does;
  harmless, as `dead | predicate(...)` is `True` regardless of a
  meaningless predicate result on an already-dead row -- and ORs the result
  into that bucket's `dead` column; never touches `needs_resort`).
  `build_phases`' prune_conn/add_conn guard-raises are removed and both are
  wired in at their documented phase-order position (forward, loss,
  backward, update_conn, prune_conn, add_conn, reset_global).
- phases.py: `build_add_conn_phase(net, static) -> Phase[GS]` ->
  `build_add_conn_phase(net, static, *, overflow_sink: list[Bool[Array,
  ""]] | None = None) -> Phase[GS]`; `build_phases` gained the same keyword
  -only `overflow_sink` parameter (default `None`, so every existing caller
  -- test_phases_elision.py, test_update_conn.py -- is unaffected). This is
  the overflow-surfacing mechanism (task deviation slot 2): `Phase[GS]`'s
  2-tuple return `(state, loss_contribution)` has no channel for a THIRD,
  add_conn-only signal, and widening it to a 3-tuple would have rippled
  into every phase builder plus both of those existing test files' direct
  `phase(...)` call sites. A length-1 mutable list threaded only to
  `build_add_conn_phase`, which overwrites its single element with the
  computed overflow flag on every call, keeps `Phase` and `build_phases`'
  own return type unchanged for every other caller. `step.py`'s
  `_cached_make_step` creates the sink once (alongside `phases`), passes it
  through, and reads `overflow_sink[0]` into `StepResult.overflow` in place
  of the old hardcoded `jnp.bool_(False)`; the mutation happens once, at
  `jax.jit`'s single trace of `step`'s body, so it is an ordinary jaxpr
  data dependency, not a stale Python-side read. Considered and rejected: a
  new `NetworkState.overflow` field (breaks test_pytree.py's explicit
  leaf-count formula) and a 3-tuple `Phase` (ripples into the two test
  files above); both are viable but strictly more invasive than the sink.
- phases.py `build_add_conn_phase`'s candidate window is narrower than
  dispatch_cpu.hpp:811-822's full rolling window: only dst strictly AHEAD
  of src, `0 < level[dst] - level[src] <= net.neighbourhood` (excludes the
  oracle's same-source-level La==Lb pairs). A same-level pair is the only
  way an accepted edge could ever force a Kahn relevel (an edge into a unit
  at or behind its own source's level needs that destination bumped to
  source_level + 1); since M4b's `topo.recompute_levels`/`resort` are not
  implemented yet, there is no host mechanism to act on a relevel request
  this milestone. Restricting the window to strictly-ahead pairs makes
  every accepted candidate provably level-preserving by construction (dst's
  level already exceeds src's, so Kahn's `level(dst) = max(incoming src
  levels) + 1` cannot increase) -- `needs_resort` is computed for real
  (`unit_level[dst] > unit_level[src]` per accepted candidate, ORed across
  buckets), not hardcoded, so a future wider window is correct without
  touching this mechanism; it is simply never observed to fire given this
  window. Flagging for review per the task's ambiguity callout: this is a
  deliberate narrowing of the oracle's window, not a bug, but it does mean
  M4a's AddConn can never itself be the trigger for M4b's resort path --
  only a later, wider window (or PruneConn-then-AddConn level churn, still
  excluded here) would exercise it.
- phases.py `build_add_conn_phase`: per source-level bucket, candidates are
  the full (num_units, num_units) unit-id grid (masked by the level window
  and, in TOPOLOGICAL mode, by `level[src] == bucket_idx`; in PIPELINE mode
  every unit is a source-level candidate, since the single bucket holds
  every live conn regardless of source level), scored via `ac.score`
  (invalid pairs forced to -inf), reduced via `lax.top_k(..., k=min(
  ac.max_candidates, num_units**2))`. Slot claim: a prefix-sum
  (`cumsum(dead) - 1`) over the bucket's OWN dead mask gives each dead
  position's 0-based rank among free slots; scattering `arange(capacity)`
  by that rank (dropping live positions via an always-out-of-bounds
  `sink_len = max(capacity, k)` index) inverts it into "the position of the
  j-th free slot" for j in `[0, k)`, with `capacity` itself standing in for
  "no such free slot" -- a top-k'd candidate commits only if it is both a
  real (in-window) candidate and has room; the rest -- invalid, or valid
  but out of room -- scatter to `capacity` (one past the bucket's own valid
  range) and vanish under `.at[...].set(..., mode="drop")`'s default
  FILL_OR_DROP. `overflow` is "a real, top-k-selected candidate had no
  room"; distinct from `needs_resort` (see above), matching the driver.py
  docstring's two separate escalation paths (grow-and-retry vs.
  resort-between-steps) even though M4b's Driver does not exist yet to
  consume either.
- phases.py `build_add_conn_phase`: a newly-committed slot's conn fields
  NOT covered by `ac.init`'s `ConnWrite` (e.g. an unset extra field) are
  reset to that field's `FieldSpec.default` rather than inheriting
  whatever a previous tenant left there -- a real, not hypothetical,
  concern given prune_conn runs immediately before add_conn in the same
  step, so a slot freed by THIS step's own pruning can be reclaimed by
  add_conn before its stale weight would ever be read. Mirrors native
  `SOAAllocator::Allocate()`'s placement-new default-construction on claim
  (alloc.hpp:171-188), applied to plastax's dead-slot recycling in place of
  native's bump allocation (plastax intentionally diverges from native's
  allocator here per rung0 design section 5 -- recycling dead slots via a
  prefix-sum claim rather than bump-allocating and periodically
  compacting -- so this is carrying over the SAFETY property of Allocate(),
  not its mechanism).
- IMPLEMENTATION_PLAN.md M4 section's `test_add_conn`/`test_update_prune`
  acceptance is split: this dispatch (M4a) implements and tests
  PruneConn/AddConn; `test_resort` (recompute_levels/resort/retrace-count)
  stays skipped pending M4b, per the task's explicit scope boundary.

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
