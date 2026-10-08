# Commit scopes

Scope is the second axis of `type(scope): subject` and is REQUIRED.
Scopes map to the package structure; prefer the most specific one that
covers the whole diff. A diff spanning many scopes is a signal to split
the commit. The commit-msg hook (`scripts/check-commit-msg.sh`) parses this
list directly, so keep the bullet format.

Module scopes (src/plastax/):

- `types` — _types.py (FieldSpec, index NewTypes, Propagation, builtins)
- `monoid` — monoid.py (combine contract, segment reductions)
- `state` — state.py (NetworkStatic/NetworkState, arenas, grow_bucket)
- `views` — views.py (Unit/ConnView, write records)
- `traits` — traits.py (Network base, policy Protocols, validation)
- `sweep` — sweep.py (gather/map/reduce/apply core)
- `phases` — phases.py (phase builders, elision, StepInputs)
- `topo` — topo.py (levels, resort, capacity policy)
- `step` — step.py (assembly, jit, donation, caching)
- `builder` — builder.py (host construction, finalize, from_topology)
- `driver` — driver.py (host loop, retrace/overflow protocol)
- `distributed` — distributed.py (multi-controller state placement, Scheme-A mesh)
- `topology` — topology.py (dense/conv2d/sequential generators)
- `optim` — optim/ (Optimizer bundles: sgd, momentum, adam; state_fields contract)
- `rng` — rng.py (counter-based Philox RNG, the cross-library draw contract)

Cross-cutting scopes:

- `api` — __init__.py exports / public-surface changes spanning modules
- `examples` — examples/
- `tests` — tests/ when the diff is test infrastructure rather than one
  module's tests (a test for sweep.py alone is `test(sweep): ...`)
- `tooling` — dev-tool config: ruff/mypy/ty/pytest tables in pyproject,
  docs/development/tooling.md, uv, and the code-quality hooks in .pre-commit-config.yaml
  (ruff, ruff-format, pydoclint, mypy, pytest). Distribution metadata is
  `packaging`; commit-governance hooks are `repo`.
- `packaging` — distribution + dependency metadata in pyproject: `[project]`,
  `[build-system]`, `[dependency-groups]`, `[tool.hatch.*]` (runtime/dev/docs
  deps, version, classifiers, URLs, wheel/sdist contents)
- `release` — release automation: .github/workflows/publish.yml,
  docs/development/releasing.md
- `docs-site` — user-facing Sphinx / Read the Docs documentation: docs/ and
  .readthedocs.yaml (its `docs` dependency group lives in pyproject, `packaging`)
- `plan` — project governance and contributor guidance: README.md,
  AGENTS.md, .agents/, docs/development/scopes.md and tags.md
- `repo` — git plumbing and commit-governance mechanics: scripts/, the
  commit-msg type/scope gate wiring, .git hooks, agent identity

If no scope fits, do not invent one: split the change or ask.
