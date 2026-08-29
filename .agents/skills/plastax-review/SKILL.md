---
name: plastax-review
description: >-
  Review a change to the plastax codebase the way a careful maintainer of a
  jit-traced, donation-based, SoA-arena JAX library would — catching what the
  deterministic hooks (ruff, mypy, pydoclint, pytest) cannot: broken invariants
  (phase elision, static shapes, donation, null-slot, derived counts, pure
  policies), JAX/trace-contract hazards (retrace regressions, lost donation,
  lax.cond misuse, data-dependent shapes), oracle-parity risks vs the C++/optax
  references, dynamic-sparse semantics, plus the usual duplicated/diverged logic,
  dead comments, stale docs, and hollow tests. Use whenever the user asks to
  review a diff, branch, PR, or "the changes"; says review, code review, check my
  code, is this ready to commit/push, final check, or look for smells; or when a
  change is about to be committed or pushed. Can also power a pre-push/CI pass on
  top of the repo's existing hooks. Prefer this over an ad-hoc review even when
  the user doesn't say "skill".
---

# plastax review

Review a plastax change as a demanding maintainer would: run the deterministic
hooks first, then a checklist pass, then N role passes chosen by what changed,
then an adversarial pass, then one deduplicated severity-ranked report. This
skill is the *judgement* layer on top of the repo's *mechanical* layer (the
pre-commit/pre-push hooks) — it does not re-run linters by eye, it hunts the
plastax-specific failure modes a linter can't see.

The guiding principle (as in the generic human-review skill this is modeled on):
**an LLM reads your codebase and repeats it back.** In plastax the highest-
leverage version of that is the *invariants*: a single `lax.cond` where elision
was required, or one non-shape-preserving leaf, teaches the next change the wrong
house style and quietly breaks the donation/retrace model. Stopping that is the
job.

## When running headless (CI or a hook)

If invoked by a wrapper with a `REVIEW_CONTEXT` (diff range, fail threshold,
output path): do the full review, **write the report to the given path and
nothing to chat**, make reasonable assumptions instead of asking, and end with a
single machine-readable verdict line (see `references/report-format.md` →
"Verdict contract"). Otherwise you are interactive: talk normally and offer to
fix findings.

## Step 1 — Scope the change

Determine what you are reviewing, in order of preference:
1. An explicit range/PR the user or `REVIEW_CONTEXT` gave you.
2. Else the branch diff vs. base: `git diff --merge-base main...HEAD`.
3. Else working-tree changes: `git diff HEAD`.

Read the diff **and** enough surrounding code to judge it — for plastax that
means: the module's own docstring (each states its "add here / not here"
contract), the Protocol in `traits.py` if a policy changed, and the matching
test file. Load the project's conventions: [`AGENTS.md`](../../../AGENTS.md),
[`.agents/invariants.md`](../../invariants.md),
[`.agents/architecture.md`](../../architecture.md), `TOOLING.md`,
`SCOPES.md`, `TAGS.md`. **These win over generic defaults and over surrounding
code.**

## Step 2 — Run the deterministic layer, then the checklist

The repo already ships the mechanical checks — run them and fold real failures
into the report rather than re-deriving them by eye:

```bash
uv run ruff check .            # lint (E/F/I/UP/B/ANN/D)
uv run ruff format --check .   # format
uv run pydoclint --style=google --arg-type-hints-in-docstring=False \
  --check-return-types=False --check-class-attributes=True \
  --skip-checking-private-functions=True src/plastax   # Google-docstring contract
uv run ty check src            # fast type pass
uv run mypy --strict src       # authoritative type gate (pre-push)
uv run pytest -m "not slow" -q # fast suite (pre-push); add the full suite if oracles are in scope
```

(If a command's tool is missing, note it and skip — don't fail the review on
infrastructure.) Treat their output as pre-computed findings. Then run the
plastax checklist in [`references/checklist.md`](references/checklist.md)
yourself — that is the judgement pass the tools can't do (is this a real
invariant break? does this comment earn its place? will this retrace?). Note
candidate findings; don't write the report yet.

## Step 3 — Pick the roles that fit the change

Do **not** run every role on every diff — it dilutes signal. Choose the N roles
from [`references/roles.md`](references/roles.md) that match the surface area.
Always include **Senior Engineer (maintainability)** and, for any change to
`src/plastax/`, the **Invariant & JAX-Contract Auditor** — those are the failure
modes this skill exists to catch. Add Oracle-Parity, Sparse-Semantics,
API-Design, Test-Quality, or Docs roles as the diff warrants (the file gives the
selection rules).

Run each selected role as an **independent pass** — a fresh read through that
role's lens only.
- **With subagents:** spawn the selected roles in parallel, one each, using the
  per-role brief in `references/roles.md`.
- **Without subagents:** run them sequentially, resetting your framing between
  each; keep each role's findings in a separate list for Step 5.

## Step 4 — Adversarial pass

One deliberately skeptical pass to find what the role passes were too charitable
to flag. For plastax, prioritise: a `lax.cond`/`jnp.where` that should have been
Python-level elision; a state leaf whose shape or dtype changes across a step
(silent donation loss); a stored count that duplicates the derived one; a policy
that reaches for a raw column or another element; a diverged copy of the delta
rule or a monoid identity; a test that would pass even if the numerics were
wrong; a comment describing intended behaviour the code doesn't implement. See
`references/roles.md` → "Adversarial".

## Step 5 — Synthesise

Merge into one report using [`references/report-format.md`](references/report-format.md):
- **Deduplicate** — one finding per issue, listing the roles that raised it.
- **Assign a severity** (`blocker`/`high`/`medium`/`low`/`nit`) by what a
  maintainer would actually do. An **invariant break, a donation/retrace
  regression, or an oracle-parity break is at least `high`, usually `blocker`** —
  those are correctness for this library, not style.
- **Make each finding actionable:** `file.py:line`, what's wrong, why it matters
  (tie it to a concrete failure — a retrace every step, a doubled memory
  footprint, a silent numeric divergence — not abstract cleanliness), and a
  concrete fix.
- Lead with the highest severity. Keep it tight.

## Step 6 — Deliver

- **Interactive:** present the report, then offer to fix findings at or above a
  severity the user picks. Remember the commit gate: the change also needs the
  correct `type(scope)` and a Deviations entry if it breaks `__all__`.
- **Headless:** write the report to the given path and emit the verdict line;
  do not attempt fixes in a hook.

## Reference files

- [`references/checklist.md`](references/checklist.md) — the baseline checklist,
  plastax-specific (Step 2).
- [`references/roles.md`](references/roles.md) — the role roster, per-role
  briefs, selection rules, and the adversarial brief (Steps 3–4).
- [`references/report-format.md`](references/report-format.md) — required report
  structure, the severity scale, and the headless verdict contract (Step 5–6).

## Relationship to the hooks

This skill does **not** replace the pre-commit/pre-push hooks — it complements
them. The hooks are the non-bypassable mechanical gate (ruff, pydoclint, ty,
mypy --strict, fast pytest, commit-scope grammar) and **must always run and
pass; never `--no-verify`**. This review catches the semantic and design issues
those tools structurally cannot: invariant violations, trace-contract
regressions, oracle-parity risk, diverged logic, and stale docs. Run it before
you commit, so the commit that reaches the hooks is already clean.
