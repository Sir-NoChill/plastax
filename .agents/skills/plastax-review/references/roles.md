# plastax review roles

Pick the N roles that fit what changed (Step 3). **Senior Engineer** is always
on; the **Invariant & JAX-Contract Auditor** is always on for any `src/plastax/`
change. Each role is an **independent pass** — review the diff through that lens
only. With subagents, each brief is a subagent prompt; without, run them one at
a time and reset framing between each.

For every finding, give: `file.py:line`, the problem, why it matters, a proposed
fix, and a suggested severity (synthesis reconciles severities). A role that
reports everything reports nothing — stay in your lane.

## Selecting roles

- Any change → **Senior Engineer** (always).
- Any `src/plastax/` change → **Invariant & JAX-Contract Auditor** (always).
- Forward/backward/loss/update numerics, a new optimizer, monoid, or anything
  compared to the C++/optax oracle → **Oracle-Parity Reviewer**.
- Prune/grow policies, SET/RigL, churn, candidate shortlisting, regrow-init,
  sparsity/edge-count behaviour → **Sparse-Dynamics Reviewer**.
- Changes to `src/plastax/__init__.py`'s `__all__`, a Protocol signature, a
  public dataclass, or a factory signature → **API-Design Reviewer**.
- Tests added/changed, or behaviour that should have tests → **Test-Quality
  Reviewer**.
- Comments, docstrings, `docs/`, `.agents/`, README, plan docs → **Documentation
  & Comments Reviewer**.
- Hot-path / memory changes (sweep inner loop, arena sizing, extra columns,
  sharding) → **Performance & Memory Reviewer**.

Two to four roles plus the two always-on is typical. More is not better.

## Role briefs

### Senior Engineer (maintainability) — always on
You review for the human who maintains this six months on. Targets: duplicated
logic, **diverged copies** of a rule (the delta rule, a monoid fold, a
prefix-sum slot-claim, a masked merge implemented slightly differently in a new
place), over-engineering, oversized functions, dead or misleading comments,
logic in the wrong module (orchestration leaking into a policy, numerics into
the driver — check against `architecture.md` §6 and the module's own docstring
"add here / not here"). Would a careful maintainer accept this or groan? Prefer a
named shared helper over a repeated idiom; prefer code that needs no comment.

### Invariant & JAX-Contract Auditor — always on for src/plastax
You review only for the eight invariants (`.agents/invariants.md`) and the JAX
trace contract. Hunt, concretely:
- a `lax.cond`/`jnp.where` used to gate a phase/branch where **Python-level
  elision** was required (inv #1);
- a **data-dependent shape**, or a `NetworkStatic` change on a non-structural
  event → a per-step **retrace** (inv #2);
- a **stored count** duplicating the `DEAD`-derived one (inv #3);
- a **shape-changing masked gather** where the null-slot trick was required (inv
  #4);
- a state leaf whose **shape/dtype changes across a step** → silent **donation
  loss** (inv #5);
- a policy reading a **raw column** / another element / instance state, or
  pytree-registering a write record (inv #6);
- a broken `FieldSpec[DT]` generic or a change that fails **mypy --strict** (inv
  #7);
- **out-of-scope** work or an un-guarded generic monoid (inv #8).
For each, name the invariant and the concrete failure it causes (retrace every
step, doubled memory, wrong reduction). These are correctness, not style — grade
them `high`/`blocker`.

### Oracle-Parity Reviewer
You review only numerical correctness against the reference oracles. Check that
forward/backward/update/loss semantics still match the C++ plastix oracle
(`../plastix` dispatch_cpu.hpp) and that optimizers still match optax; that the
**delta rule** and **monoid identities** are correct and not re-derived
divergently; that any tolerance change is justified; and that the parity test
covering this path would still pass. Give the concrete divergence scenario
(which input, which term, expected vs actual), not just "looks off".

### Sparse-Dynamics Reviewer
You review only dynamic-sparse behaviour. Check prune/grow correctness: SET vs
RigL differ only in `score`; `-inf` is a hard veto; growth writes `WEIGHT` (+ its
own fields) only, never `opt/…`; regrown edges zero their optimizer state via
`FieldSpec.default` (no special-casing); candidate shortlisting stays
`O(num_units + M²)` and holds the target sparsity/edge-count; churn keeps the
live count ~constant. Flag anything that would drift sparsity, orphan a unit, or
leak stale state into a regrown slot.

### API-Design Reviewer
You review only the public surface. Check for breaking changes to `__all__`, a
Protocol signature, a public dataclass, or a factory; consistency with existing
API conventions (factory naming, `opt/…` field prefixes, keyword-only
hyperparameters); sensible defaults. A break must be a `type(scope)!:` commit
with a `BREAKING CHANGE:` footer and an IMPLEMENTATION_PLAN.md Deviations entry,
and the symbol must stay documented (`__all__` → `docs/api.md`). Flag silent
breaks to existing callers/examples/tests.

### Test-Quality Reviewer
You review only the tests. For each: if the code under test broke, would this
fail? Flag tests that assert nothing, mock away the thing under test, or only
cover the happy path. Check the **oracle tolerance** is right and justified; that
heavy/oracle tests are `slow` + `importorskip`; that a stateful optimizer has a
regrow-zeroing test and a structural change has a retrace-count assertion; that
RNGs are seeded and `conftest.py`'s CPU/device setup isn't overridden; that an
example used as acceptance asserts its own success criteria.

### Documentation & Comments Reviewer
You review only comments and docs. Enforce: comments explain **why** (cite a
design-doc section or a jax/C++ `file:line`), not what; no progress/edit-history
comments; no encapsulation leaks; **Google-style** docstrings (no Doxygen
`@brief` syntax — it fails pydoclint today); `docs/` (esp. `optimizers.md`),
README, and the `.agents/` docs still match the code after the change; a new
public symbol is in `__all__`. Stale or inaccurate docs/comments are findings.

### Performance & Memory Reviewer
You review only performance and memory, and only on likely-hot or memory-shaping
paths (the sweep inner map/reduce, arena/bucket sizing, added columns, sharding
collectives). Flag work hoistable out of a vmapped body, an added per-edge column
that meaningfully grows `O(E)` state without need, an accidental host↔device
transfer per step, or a collective that doesn't match the sharding scheme. Don't
micro-optimise cold host-side build code. The library's whole thesis is `O(E)`
memory — a change that quietly reintroduces `O(N²)` state is a `high` finding.

## Adversarial pass

Run once, after the role passes, to find what they were too charitable to flag.
Assume the change hides a defect. Prioritise:
- **A disguised invariant break** — a `where`/`cond` that reintroduces a runtime
  branch for something static; a leaf that changes shape only on some code path.
- **Diverged copies** — grep for the other implementations of any rule this
  change touches (delta rule, monoid identity, slot-claim, masked merge) and
  compare them literally; report any that disagree.
- **Silent donation/retrace regression** — trace one step mentally: does every
  leaf come back same-shape/dtype? does `NetworkStatic` stay put?
- **Hollow tests** — a test green regardless of correctness; a tolerance loose
  enough to hide the bug.
- **Meant-vs-did comments / stale docs** — a docstring or `.agents/` claim the
  code no longer satisfies.
- **Silent assumptions** — an invariant the change relies on that nothing
  enforces (e.g. phase-order read that isn't actually guaranteed).

Report these in the same finding format; synthesis merges and ranks them.
