# plastax review — report format and severity

## Severity scale

Grade by what a plastax maintainer would actually do, not how strongly you feel.
For this library, **correctness includes the invariants, the trace contract, and
oracle parity** — those are not style.

- `blocker` — a maintainer would refuse to merge. A broken invariant that
  corrupts state or silently loses donation; a per-step retrace regression; an
  oracle-parity break (wrong numerics vs C++/optax); a sparsity/regrow bug that
  leaks stale state; a public-API break shipped without the `!:`/Deviations
  protocol; a test that gives false confidence about any of these.
- `high` — fix before merge. Real duplication/**divergence** of a rule (delta
  rule, monoid identity, slot-claim); a change that reintroduces `O(N²)` state; a
  missing test for a new numeric/structural path; a misleading comment or stale
  doc on load-bearing logic; a Google-docstring/pydoclint contract violation on
  the public surface.
- `medium` — fix soon; not a lone merge-blocker. Localised smell, moderate
  over-engineering, a function grown too big, thin coverage on a minor path, a
  tolerance that's loose but not hiding a known bug.
- `low` — worth doing, low urgency. Minor naming/structure, small doc drift.
- `nit` — cosmetic/preference. Never gates a pipeline.

When roles disagree on severity for one finding, take the highest justified
level and say why in one clause.

## Report structure

Use this exact template. Omit a section only if genuinely empty.

```
# plastax review — <change description or diff range>

## Verdict
<one line, e.g. "2 findings at/above `high` (1 invariant break) — would not merge as-is"
 or "No findings above `low` — good to commit">

## Findings
<ordered highest severity first; one block each>

### [<severity>] <short title>
- Where: `path/to/file.py:LINE` (add more locations if the issue recurs)
- Raised by: <roles, e.g. Invariant & JAX-Contract Auditor, Adversarial>
- Invariant / contract: <if applicable — e.g. "inv #5 donation / shape-preservation">
- Problem: <what is wrong>
- Why it matters: <the concrete failure — retrace every step, doubled memory,
  numeric divergence vs optax, stale-state leak — not abstract cleanliness>
- Fix: <concrete change, or the shape of one>

## Commit-readiness
<the type(scope) this change should use (SCOPES.md); whether it touches __all__
 and thus needs `type(scope)!:` + BREAKING CHANGE + a Deviations entry; whether
 docs (docs/, .agents/) and tests are in sync>

## Summary counts
blocker: N | high: N | medium: N | low: N | nit: N

## Notes / assumptions
<anything assumed because you could not ask — especially headless — and any
 convention conflicts you resolved (project docs win over surrounding code)>
```

Keep prose tight; this is read under time pressure, often right after a failed
push. Lead with what would stop the merge.

## Headless verdict contract

After writing the report to the given path, print exactly one final line:

```
PLASTAX_REVIEW_VERDICT max_severity=<level> blocker=<n> high=<n> medium=<n> low=<n> nit=<n>
```

`<level>` is the highest severity found, or `none`. Severity ordering for the
gate: `none < nit < low < medium < high < blocker`. The wrapper compares
`max_severity` against its configured `fail_on` and sets the exit code — the
review only reports counts, it never decides pass/fail itself. If the review
cannot complete (no diff, tooling error), print
`PLASTAX_REVIEW_VERDICT max_severity=error` and let the wrapper treat it as a
non-blocking skip.

This mirrors the repo's philosophy: the deterministic hooks (ruff, pydoclint,
mypy, pytest, commit-scope) are the non-bypassable gate; this review is the
judgement layer whose gating policy lives in config and stays auditable.
