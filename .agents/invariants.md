# plastax invariants

The design decisions that are already made. **Violating one is a failed change
even if every test passes** (IMPLEMENTATION_PLAN.md, "Non-negotiable
invariants"). If a task appears to require breaking one, stop and surface it —
that is almost always a sign the approach is wrong, not that the invariant is.

Each entry: the rule, why it exists, and how it is enforced (so you know what
will catch you).

---

### 1. Phase elision is Python-level

An absent trait slot (`None` on the `Network` subclass) contributes **zero**
equations to the jaxpr. Presence is decided in Python when `build_phases`
assembles the phase tuple — **never** with `lax.cond` at runtime.

- *Why:* a `cond` on a static condition still emits both branches into the
  jaxpr, costs compile time, and defeats the whole "you pay only for the traits
  you declare" model.
- *Enforced by:* `tests/test_phases_elision.py` compares `build_phases` output
  structure directly (absent phase ⇒ absent function). `phases.build_phases`
  (`phases.py:64`) appends each phase iff `getattr(net, slot) is not None`.

### 2. All shapes static; retrace only on structural change

Mutating leaf *values* never retraces. The **only** events that mint a new
`NetworkStatic` (and thus a new `jax.jit` cache key) are **bucket growth**
(`state.grow_bucket`) and **level reassignment** (`topo.resort`) — both
host-side.

- *Why:* the arena is fixed-capacity SoA precisely so the step is one stable
  jitted program reused across steps; data-dependent shapes would retrace every
  step.
- *Enforced by:* `NetworkStatic` is a frozen `register_dataclass` with every
  field `static=True` (`state.py:22`), so it is hashable and is the cache key
  (`step._cached_make_step`, `functools.cache`). `tests/test_resort.py` pins the
  compile count with `assert_num_jit_and_pmap_compilations` (1 per fresh net, 1
  per resort). `tests/test_pytree.py` confirms leaf changes don't change the
  PyTreeDef.

### 3. Live counts are derived, never stored

The number of live connections is always computed from the `DEAD` tombstone
mask; there is no maintained counter.

- *Why:* a stored counter is a second source of truth that drifts under
  grow/prune and would have to be donated and kept consistent.
- *Enforced by:* `state.live_conn_count` (`state.py:119`) counts `~dead`; fresh
  slots default `DEAD=True` (`_types.py:120`). `tests/test_update_prune.py`
  checks derived counts after pruning.

### 4. Dead slots via the null-slot trick

Dead connections are dropped by redirecting their target index to `num_units`
(or the bucket capacity) — out of range for `segment_reduce` with
`mode=FILL_OR_DROP`. **Never** a boolean-masked gather that changes shape.

- *Why:* a shape-changing gather would make the step non-jit-stable and break
  donation; the null-slot keeps everything static-shaped.
- *Enforced by:* `sweep._accumulate_into` (`sweep.py:113`) and
  `monoid.segment_reduce` (`monoid.py:84`). Conn-update sweeps instead keep-old
  the dead row (`where(dead, old, written)`) since they don't aggregate.
  `tests/test_forward_pipeline.py` covers dead-slot null-scatter.

### 5. The step is shape-preserving on state so every leaf donates

`make_step` uses `jax.jit(..., donate_argnums=0)` on the whole `NetworkState`
pytree. Every phase must return a state with identical pytree structure,
shapes, and dtypes, so XLA can reuse the input buffers in place.

- *Why:* in-place state update is the memory model; a non-donatable leaf silently
  doubles memory and breaks the O(E) claim.
- *Enforced by:* `pyproject.toml` promotes JAX's "Some donated buffers were not
  usable" warning to an **error** (`filterwarnings`), globally. `step.py:57`
  sets `donate_argnums=0`. `tests/test_donation.py` verifies buffers are freed
  and donation is never wasted.

### 6. Policies are pure, vmapped, per-element

Policy functions (`map`/`apply`, `predicate`, `score`/`init`, `incoming`/
`outgoing`, …) see only `UnitView`/`ConnView` for one element and return
`UnitWrite`/`ConnWrite`. They never touch raw columns, never see other
elements, never carry state.

- *Why:* purity + per-element shape is what lets the sweep `vmap` them and what
  makes optimizer state shard for free under Scheme-A.
- *Enforced by:* the sweep only ever hands policies views and unwraps write
  records to plain dicts before `vmap` (`sweep.py:209`). `UnitWrite`/`ConnWrite`
  are intentionally **not** pytree-registered — do not register them. Runtime
  jaxtyping/beartype checks during tests catch signature violations.

### 7. Type discipline: mypy --strict is the gate

`mypy --strict src` must pass; it is the authoritative CI / pre-push gate. `ty`
is the fast pre-commit pass only. Keep `FieldSpec` generic typing intact end to
end (views return arrays typed by the spec's `DT`).

- *Why:* the generic `FieldSpec[DT]` chain is what gives type-safe column access;
  weakening it erodes the safety the whole SoA API is built on.
- *Enforced by:* `.pre-commit-config.yaml` (`ty` on pre-commit, `mypy --strict`
  on pre-push) and CI. If `ty` false-positives on a load-bearing jaxtyping
  annotation, use a rule-scoped ignore and record it in IMPLEMENTATION_PLAN.md
  Deviations — **never** weaken the mypy gate.

### 8. Scope contract: what is out of v1

**Do not implement, even partially:** AddUnit / PruneUnit; generic
`Monoid(op, identity)` lowering (the three `Monoid` methods must keep raising
`UnsupportedMonoidError` for non-named monoids); `jax.Ref` arenas; hijax; any
MLIR emission; densification (dense-bucket → `dot_general` rewrite). Static
dense-matmul parity is a **non-goal**, not a missing feature.

In scope and *implemented*: `Propagation.PIPELINE` and `TOPOLOGICAL`;
AddConn/PruneConn dynamics; named monoids (incl. pytree-of-monoids struct
accumulators); donation-based in-place state; host driver retrace protocol;
Scheme-A multi-device sharding.

- *Why:* v1 is "rung 0" — the trace-time metaprogramming rung. Later rungs
  (composites, Pallas, FFI) are deliberately deferred; premature scope creep
  breaks the clean lowering ladder (see the design docs).
- *Enforced by:* code review, the plan docs, and `monoid.UnsupportedMonoidError`
  guards (`monoid.py`).

---

## Supporting structural invariants

These follow from the above but are worth stating for anyone touching `topo`,
`phases`, or `builder`:

- **Topological leveling:** no edge's source level is `>=` its destination's
  level. Forward/backward bucket walks and `add_conn`'s `needs_resort` decision
  all depend on it. PIPELINE mode drops this (cycles allowed; levels cosmetic,
  1 bucket).
- **Bucket ordering:** each bucket is sorted by `(dead, to_id)` so
  `indices_are_sorted=True` holds for the segment reductions. `builder.finalize`
  establishes it; `topo.resort` restores it after redistribution.
- **Deletion never resorts. Level-preserving adds never resort.** `resort` runs
  only when `add_conn` set `needs_resort` (a non-level-preserving commit).
- **`-inf` AddConn score is a hard veto** — never committed even with free
  slots — distinct from a merely-low finite score.
- **Reserved field names** (`from_id`, `to_id`, `dead`, `weight`, `activation`,
  `level`) cannot be reused by `extra_unit_fields`/`extra_conn_fields`; enforced
  at subclass definition (`traits._validate_field_names`).
