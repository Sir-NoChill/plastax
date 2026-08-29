# .agents/ — agent-facing reference docs

Deeper reference material for coding agents, pointed to from
[`../AGENTS.md`](../AGENTS.md). Start with `AGENTS.md`; come here for detail.

| Doc | Read it when |
|---|---|
| [`architecture.md`](architecture.md) | You need the full project layout, per-module contracts, the trace-time assembly path, **where a contribution goes**, and testing/example conventions. The `plastax-architecture` skill loads this. |
| [`invariants.md`](invariants.md) | Before any change that touches the framework core — the non-negotiable design rules, each with rationale and how it's enforced. |
| [`glossary.md`](glossary.md) | Any term you don't recognize: arena, tombstone, monoid, bucket, phase elision, null-slot trick, retrace, churn step, Scheme-A/B, delta rule, … |

## Skills (in [`skills/`](skills/))

Task playbooks for agents, versioned in-repo alongside these docs. Each is a
`SKILL.md` (with `references/` where needed). To use one, read its `SKILL.md`
and follow it. To make them invocable as Claude Code slash-skills on a given
machine, symlink or copy `skills/<name>/` into that machine's discovery path
(`~/.claude/skills/` or a project `.claude/skills/`) — they are kept here, not
under `.claude/`, so they are committed and shared with the repo.

| Skill | Purpose |
|---|---|
| `plastax-architecture` | Loads `architecture.md` and routes a contribution to the right module. |
| `plastax-algorithm-scaffold` | Guided workflow + templates to add a plasticity algorithm, optimizer, phase, topology generator, or monoid, with the exact contract and the test/docs/commit checklist. |
| `plastax-review` | Multi-role code review tailored to plastax (invariants, JAX/trace contracts, oracle parity), on top of the deterministic pre-commit/pre-push hooks. |

## Relationship to the other docs

- **`../AGENTS.md`** — the entry point and the binding rules. Everything here
  elaborates it.
- **Planning/spec docs at the repo root** (`IMPLEMENTATION_PLAN.md`,
  `SPARSE_PLAN.md`, `ECOSYSTEM_ROADMAP.md`, `DISTRIBUTION_PLAN.md`) — authoritative
  on scope, milestones, and history. These `.agents/` docs summarize the
  *stable* architecture; the plan docs own *what's next*.
- **`TOOLING.md`, `TAGS.md`, `SCOPES.md`** — toolchain and commit-metadata
  contracts. The commit-msg hook parses TAGS.md/SCOPES.md directly, so they are
  the single source of truth for commit types and scopes.

## Keeping these current

When a change alters a module's contract, an invariant, or the public surface,
update the matching doc here in the **same commit** (docs-in-sync is a review
condition — see the `plastax-review` skill). Cite `file.py:line` and keep the
routing table in `architecture.md` §6 accurate — it is the first thing an agent
consults.
