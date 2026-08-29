# AGENTS.md — plastax

Instructions for coding agents working in this repository. Read this file
first, then load the reference docs and skills it points to **before** you
write code. Everything here is binding: a rule in this file beats the style of
the surrounding code.

> This file is agent-agnostic (the `AGENTS.md` convention). Claude Code also
> reads the user's global `CLAUDE.md`; nothing here overrides the safety rules
> in the system prompt.

---

## What plastax is

plastax expresses a **dynamically structured, highly sparse** neural network
(90–99 %+ of weights absent) as a set of *declarative traits* over a
**struct-of-arrays (SoA) edge arena**. A `plastax.Network` subclass declares —
as class attributes, not overridden methods — its forward/backward passes,
loss, connection-update rule, prune/grow policies, per-unit and per-edge
fields, and propagation model. At **trace time** the library assembles and
`jax.jit`-specializes the corresponding step function; the step donates the
whole state pytree and runs in place.

The central claim (see `SPARSE_PLAN.md`): mask-based dynamic sparse training
(SET, RigL) keeps weights, gradients, **and optimizer state** dense at
`O(N²)`; plastax runs the *same algorithms* on a live-edge arena at `O(E)`.
Static dense-matmul parity is an explicit **non-goal** — explicit edges are the
point, not an inefficiency to apologize for.

The mental model in one line:

```
Network subclass (traits.py)  ──build_phases──▶  ordered pure phases (phases.py)
                                    │
                              make_step (step.py): scatter inputs, run phases,
                              wrap in shard_map? then jax.jit(donate_argnums=0)
                                    │
                              Driver (driver.py) host loop: run step; on overflow
                              grow_bucket + retrace; on needs_resort → topo.resort + retrace
```

---

## Golden rules (non-negotiable invariants)

These are design decisions already made. Violating one is a failed change
**even if tests pass**. The full list with rationale is in
[`.agents/invariants.md`](.agents/invariants.md); the short form:

1. **Phase elision is Python-level.** An absent trait slot (`None`) contributes
   *zero* equations to the jaxpr. Never gate phase presence with `lax.cond`.
2. **All shapes static.** Leaf mutation never retraces. The *only* retrace
   events are bucket growth (`state.grow_bucket`) and level reassignment
   (`topo.resort`) — both host-side, both producing a new `NetworkStatic`.
3. **Live counts are derived** from tombstone (`DEAD`) masks, never stored.
4. **Dead slots via the null-slot trick** (redirect index to `num_units` /
   bucket capacity so `segment_reduce` with `FILL_OR_DROP` drops it). Never a
   boolean-masked gather that changes shape.
5. **The step is shape-preserving on the state pytree** so every leaf donates;
   the "donated buffers were not usable" warning is promoted to a **hard error**
   in CI (`pyproject.toml` filterwarnings).
6. **Policies are pure, vmapped, per-element.** They see `UnitView`/`ConnView`
   and return `UnitWrite`/`ConnWrite` records — never raw columns, never
   cross-element state.
7. **Type discipline.** `mypy --strict` must pass (it is the authoritative CI
   gate; `ty` is the fast pre-commit pass). Keep `FieldSpec` generic typing
   intact end to end.
8. **Out of v1 scope — do not implement, even partially:** AddUnit/PruneUnit,
   generic `Monoid(op, identity)` lowering (raise `UnsupportedMonoidError`),
   `jax.Ref` arenas, hijax, MLIR emission. (Multi-device *Scheme-A* sharding
   *is* implemented; `shard.py` Scheme-B is host-side partitioning math only.)

---

## Module map

`src/plastax/` — each module's commit **scope** is in parentheses (see
`SCOPES.md`). Detailed per-module contracts: [`.agents/architecture.md`](.agents/architecture.md).

| Module | Scope | Responsibility |
|---|---|---|
| `_types.py` | `types` | Index NewTypes (`UnitIdx`/`ConnIdx`), `FieldSpec` SoA-column descriptor, built-in columns (`FROM_ID`/`TO_ID`/`DEAD`/`WEIGHT`/`ACTIVATION`/`LEVEL`), `Propagation`, `ShardSpec`. Dependency-graph **leaf** — no plastax imports, no JAX compute. |
| `monoid.py` | `monoid` | `Monoid[Acc]` + `MonoidTree`; named `sum/prod/max/min` lowered to `segment_*`, `combine_pairwise`, `collective` (all-reduce). Arena-agnostic pure algebra. |
| `state.py` | `state` | Two-tier state: `NetworkStatic` (hashable jit cache key) + `NetworkState[GS]` (mutable SoA pytree). `make_empty_state`, `live_conn_count`, `grow_bucket`. |
| `views.py` | `views` | `UnitView`/`ConnView` (read, indexed by `(FieldSpec, Idx)`) and `UnitWrite`/`ConnWrite` (write records). Deliberately **not** pytree-registered. |
| `traits.py` | `traits` | `Network` base class + policy `Protocol`s (`ForwardPass`, `BackwardPass`, `Loss`, `UpdateConn`, `PruneConn`, `AddConn`, `ResetGlobal`); `__init_subclass__` validation. The **declarative surface**. |
| `sweep.py` | `sweep` | Primitive gather → vmapped map → `segment_reduce` → masked apply, one bucket at a time; conn-update sweeps. The low-level engine. |
| `phases.py` | `phases` | `build_phases`: compiles declared traits into the ordered phase tuple (forward → loss → backward → update_conn → prune_conn → add_conn → reset_global), eliding absent slots. `StepInputs`. |
| `topo.py` | `topo` | Level assignment (`initial_levels`, `recompute_levels`), `resort` (rebucket after structural change), `capacity_policy`. |
| `step.py` | `step` | `make_step`: cached assembly, input scatter, jit, donation, Scheme-A `shard_map` wrap. `StepResult`. |
| `builder.py` | `builder` | Host-side eager construction: `NetworkBuilder` (`add_unit`/`add_conn`/`finalize`/`from_topology`). |
| `driver.py` | `driver` | Host control loop: retrace/overflow/resort protocol around one jitted step. |
| `topology.py` | `topology` | Host-side topology DSL: `dense`, `conv2d` (unrolled per-edge), `input_units`, `sequential`, `Block`/`EdgeSet`/`Topology`. |
| `shard.py` | `shard`* | Scheme-B band-partition math (`balanced_level_cut`). Pure numpy. (*commit under `topo`/`step` per SCOPES.md — `shard` has no dedicated scope; ask if unsure.) |
| `optim/` | `optim` | Optimizer *bundles*: `sgd`, `momentum`, `adam`, `adamw`, `rmsprop`. Each = an `UpdateConn` policy + per-connection `state_fields` (`opt/…` columns). |

**Public API stability boundary** = `src/plastax/__init__.py`'s `__all__` (33
names). Breaking any of them is a `type(scope)!:` change with a
`BREAKING CHANGE:` footer and an IMPLEMENTATION_PLAN.md Deviations entry.
Names reachable only via submodule import (`plastax.topo.*`, `plastax.shard.*`,
`plastax.state.live_conn_count`, `plastax.phases.build_phases`) are
semi-internal but tests reach into them — renaming still has blast radius.

---

## How to make a change

1. **Route the change.** Decide which module owns it — use the map above,
   [`.agents/architecture.md`](.agents/architecture.md), or invoke the
   **`plastax-architecture`** skill. Most new algorithms are a *new policy class
   implementing an existing Protocol* and touch **no framework module** — see
   the **`plastax-algorithm-scaffold`** skill.
2. **Respect the invariants** (above / `.agents/invariants.md`). If a change
   seems to require breaking one, stop and surface it — it is almost always the
   wrong approach.
3. **Write Google-style docstrings** on the public surface (see below).
4. **Add tests** in `tests/test_<topic>.py` with the right oracle tolerance and
   markers (see [`.agents/architecture.md`](.agents/architecture.md) §Testing).
   If the code broke, the test must fail.
5. **Update docs** that the change touches (`docs/`, README, the plan docs).
   Docs-in-sync is a review condition.
6. **Review** with the **`plastax-review`** skill before committing.
7. **Commit** under the agent-commit protocol (below). Let the hooks run.

---

## Docstrings, linting, types

- **Docstrings: Google style**, enforced on `src/plastax` by ruff `D`
  (`convention=google`) **and** pydoclint (`--style=google`). Types live in the
  signature, never duplicated in the docstring. Public dataclass fields go in
  `Attributes:`; PEP 695 type params in `Type Args:`. `__init__` docstrings are
  intentionally omitted (documented on the class; `D107` ignored, `DOC301`
  enforced). Tests and examples are exempt.
  > **Direction note:** the maintainer is considering migrating to Doxygen-style
  > docblocks as a single source of truth for generated docs (`prompt.md`).
  > Until that migration lands and the hooks change, **write Google-style** — do
  > not pre-emptively introduce Doxygen `@brief`/`@param` syntax; it will fail
  > pydoclint today.
- **Lint/format: ruff** (`E/F/I/UP/B/ANN/D`), autofix + `ruff-format`. `F722`
  is disabled repo-wide — do not add `# noqa: F722` on jaxtyping string
  annotations.
- **Types: `ty` (fast, pre-commit) + `mypy --strict` (authoritative, CI /
  pre-push).** If `ty` false-positives on a load-bearing jaxtyping annotation,
  silence it with a rule-scoped ignore and record it in IMPLEMENTATION_PLAN.md
  Deviations — never weaken the mypy gate.
- Full toolchain reference: `TOOLING.md`.

---

## Commit & hook protocol (mandatory)

- **Every commit is `type(scope): subject`** with a **mandatory scope**, one
  scope per commit. `type` ∈ `TAGS.md`, `scope` ∈ `SCOPES.md`. A diff spanning
  many scopes is a signal to split the commit.
- Commit via the global **`git agent-commit`** wrapper (invoke the
  **`agent-commit`** skill; there is no repo-local commit wrapper). Breaking
  public-API changes use `type(scope)!:` + `BREAKING CHANGE:` footer + a
  Deviations entry in the same commit.
- **The hooks are the gate. Never use `--no-verify`.**
  - **pre-commit:** trailing-whitespace/EOF/toml/merge-conflict, ruff (autofix),
    ruff-format, pydoclint, `ty check`.
  - **commit-msg:** `scripts/check-commit-msg.sh` enforces the type(scope)
    grammar against TAGS.md/SCOPES.md (single source of truth).
  - **pre-push:** `mypy --strict src`, then `pytest -m "not slow"`.
- Install (once): `uv run pre-commit install --hook-type pre-commit --hook-type commit-msg --hook-type pre-push`.

---

## Environment

All commands from `plastax/`. Interpreter pinned to 3.13 (`.python-version`);
`requires-python >= 3.12`.

```bash
uv sync                     # .venv + editable install + dev group
uv run pytest -m "not slow" # fast suite (what pre-push runs)
uv run pytest               # full suite (includes optax/C++ oracle parity)
```

Tests force `JAX_PLATFORMS=cpu` and fake 4 CPU devices (`tests/conftest.py`) for
determinism and to run sharding tests without accelerators. GPU-scaled sparse
work uses a separate venv (see the sparse example docstrings).

---

## Agent tooling in this repo

Skill playbooks live in [`.agents/skills/`](.agents/skills/) (versioned in-repo,
not under `.claude/`, so they are shared). Read a skill's `SKILL.md` and follow
it; see [`.agents/README.md`](.agents/README.md) for how to make them invocable
as slash-skills on your machine.

| Skill | Use it when |
|---|---|
| **`plastax-architecture`** | You need the project layout and where a contribution goes. Start here on any unfamiliar change. |
| **`plastax-algorithm-scaffold`** | You are adding a plasticity algorithm, optimizer, phase, topology generator, or monoid. Gives the exact contract, touchpoints, template, and test/docs/commit checklist. |
| **`plastax-review`** | Before committing/pushing, or on any "review this" request. Multi-role review tailored to plastax invariants, JAX/trace contracts, and oracle parity. Powers a review pass on top of the deterministic hooks. |

Reference docs (agent-facing, deeper than this file):

- [`.agents/architecture.md`](.agents/architecture.md) — full layout, per-module
  contracts, data flow, contribution routing, testing conventions.
- [`.agents/invariants.md`](.agents/invariants.md) — the non-negotiables, with
  rationale and how each is enforced.
- [`.agents/glossary.md`](.agents/glossary.md) — arena, tombstone, monoid,
  level, bucket, phase elision, null-slot trick, retrace, Scheme-A/B, and the
  rest of the vocabulary.
- [`.agents/README.md`](.agents/README.md) — index of the above.

Planning/spec docs (human-facing, authoritative on scope and history):
`IMPLEMENTATION_PLAN.md` (v1 core), `SPARSE_PLAN.md` (dynamic sparse training),
`ECOSYSTEM_ROADMAP.md` (optim/heuristics/tools tracks), `DISTRIBUTION_PLAN.md`
(packaging), `TOOLING.md`, `TAGS.md`, `SCOPES.md`.
