# plastax review checklist (baseline)

Run over the change before the role passes. Each item is a question; a "no" or
"unsure" is a candidate finding. This is the generic maintainability checklist
plus the plastax-specific contracts the deterministic hooks cannot check. The
project's own docs win over any generic default here: `.agents/invariants.md`,
`.agents/architecture.md`, `AGENTS.md`, `TOOLING.md`.

## Invariants (the plastax core — check every `src/plastax/` change)

- **Phase elision (Python-level):** does any new code use `lax.cond`/`jnp.where`
  to *turn a phase or a whole branch on/off* based on a static condition, where
  Python-level presence/elision was required? An absent trait must contribute
  zero jaxpr equations. (invariant #1)
- **Static shapes / retrace:** does any change introduce a data-dependent shape,
  or make `NetworkStatic` change on a non-structural event? The only legitimate
  retrace triggers are `grow_bucket` and `resort`. A change that retraces per
  step is a serious defect. (invariant #2)
- **Derived counts:** does the change store a live-connection count (or any
  quantity derivable from the `DEAD` mask) instead of deriving it? (invariant #3)
- **Null-slot discipline:** does aggregation over edges still drop dead slots via
  the null-slot redirect (index → `num_units`/capacity) rather than a
  shape-changing masked gather? (invariant #4)
- **Donation / shape-preservation:** does every phase return a state whose
  pytree structure, shapes, and dtypes are identical to the input? A new or
  reshaped/retyped leaf silently loses donation (and the "donated buffers were
  not usable" warning is a CI error). (invariant #5)
- **Pure per-element policies:** do policy functions touch only views for a
  single element and return write records? Any raw-column access, cross-element
  read, or instance state beyond hyperparameters is a break. Are `UnitWrite`/
  `ConnWrite` still *not* pytree-registered? (invariant #6)
- **Type discipline:** does the change keep `FieldSpec[DT]` generics intact end
  to end, and would `mypy --strict` pass? Any new `ty`-ignore recorded in
  IMPLEMENTATION_PLAN.md Deviations? (invariant #7)
- **v1 scope:** does the change implement (even partially) something out of
  scope — AddUnit/PruneUnit, generic `Monoid(op, identity)` lowering, `jax.Ref`
  arenas, hijax, MLIR, densification? Do the `Monoid` methods still raise
  `UnsupportedMonoidError` for non-named monoids? (invariant #8)

## Structural contracts

- **Topological ordering:** for TOPOLOGICAL nets, does the change preserve
  "source level < destination level", the `(dead, to_id)` bucket sort
  (`indices_are_sorted=True`), and "deletion/level-preserving-add never resort"?
- **Phase order assumptions:** does new code rely on a read that the fixed phase
  order (forward → loss → backward → update_conn → prune_conn → add_conn →
  reset_global) actually guarantees? (e.g. an update reading a grad the backward
  pass writes.)
- **AddConn veto:** is `-inf` still treated as a hard veto (never grown), distinct
  from a low finite score?
- **Reserved names:** do new `extra_*_fields` avoid the reserved column names?

## Oracle parity (numerics)

- If the change touches forward/backward/update/loss numerics, is it still
  consistent with the semantics oracle (the C++ plastix repo for core passes,
  optax for optimizers)? Would the parity test still hold, and at the right
  tolerance?
- Is the **delta rule** (`grad_field[dst] * ACTIVATION[src]`) implemented once and
  reused, not re-derived slightly differently in a new optimizer/grow policy?
- Are monoid identities correct for the dtype (0 / 1 / -inf / +inf)?

## Duplication and divergence

- Does the change duplicate logic that already exists (a sweep idiom, a
  prefix-sum slot-claim, a masked merge, the delta rule, a monoid fold)? Grep
  before answering — copies hide across `sweep.py`/`phases.py`/`topo.py`.
- Worse: are copies of the "same" rule subtly **diverged** (a different tolerance,
  a different identity, a different direction)? Diverged copies are latent bugs.
- Is a symptom patched with a stacked `if` instead of fixing the root cause?

## Comments and docs

- Does every comment explain **why** (a design-doc citation, a jax/C++ file:line,
  a gotcha) rather than restate the code? plastax's convention is minimal
  comments that cite the design decision — flag what-comments and edit-history/
  progress comments.
- Are docstrings **Google style** (types in the signature only; `Attributes:` for
  public fields; no redundant `__init__` docstring)? Do not accept Doxygen
  `@brief`/`@param` syntax — it fails pydoclint today.
- Is external documentation still accurate: `docs/` (esp. `docs/optimizers.md`),
  `README.md`, and the `.agents/` docs if a contract/routing changed? A new
  public symbol must be in `__init__.py`'s `__all__`.

## Tests

- If the code broke, would a test fail? Is the **oracle tolerance** appropriate
  and justified (external `1e-4/1e-5`; internal same-order `1e-6/1e-6`;
  cross-mode `1e-5/1e-5`; exact invariant `atol=0.0`)?
- Is a heavy/oracle test marked `slow` and does it `importorskip` optional deps?
- Does a stateful-optimizer change include a regrow-zeroing test? Does a
  structural change include a retrace-count assertion?
- Does the test avoid overriding `conftest.py`'s CPU/device setup and seed its
  RNGs?

## Structure, bloat, style

- Separation of concerns: is logic in the module that owns it (per
  `architecture.md` §6 and each module's docstring), not leaking orchestration
  into a policy or numerics into the driver?
- Over-engineering: any abstraction/indirection the change doesn't need?
- Leftover experiment code, dead branches, commented-out blocks?
- Commit hygiene: does the change fit **one** `SCOPES.md` scope? A multi-scope
  diff is a signal to split the commit.
