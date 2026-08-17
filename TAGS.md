# Commit types (TAGS.md)

Consumed by the agent-commit protocol: every commit message is
`type(scope): subject` where `type` is one of the tags below and `scope`
is defined in SCOPES.md. Scope is mandatory.

- `feat` — new user-visible capability (new phase, generator, API surface)
- `fix` — bug fix, including semantic divergence from the C++ oracle
- `docs` — README, TOOLING, plan, design-doc pointers, docstrings-only
- `refactor` — behavior-preserving restructuring
- `perf` — performance change with no semantic effect (must cite evidence)
- `test` — tests/golden files only
- `build` — pyproject, dependencies, packaging, uv
- `ci` — pre-commit/pre-push hooks, CI workflow
- `style` — formatting only (rare; ruff-format should make these moot)
- `chore` — repo mechanics fitting nothing above (use sparingly, never as
  an escape hatch for "couldn't pick a scope")
- `revert` — reverts a prior commit; reference it in the body

Breaking changes to the public API (`plastax.*` exports): `type(scope)!:`
plus a `BREAKING CHANGE:` footer, and a Deviations entry in
IMPLEMENTATION_PLAN.md in the same commit.
