# Commit scopes (SCOPES.md)

Scope is the second axis of `type(scope): subject` and is REQUIRED.
Scopes map to the package structure; prefer the most specific one that
covers the whole diff. A diff spanning many scopes is a signal to split
the commit.

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
- `topology` — topology.py (dense/conv2d/sequential generators)

Cross-cutting scopes:

- `api` — __init__.py exports / public-surface changes spanning modules
- `examples` — examples/
- `tests` — tests/ when the diff is test infrastructure rather than one
  module's tests (a test for sweep.py alone is `test(sweep): ...`)
- `tooling` — pyproject, .pre-commit-config.yaml, TOOLING.md, uv
- `plan` — IMPLEMENTATION_PLAN.md, SCOPES.md, TAGS.md, README.md
- `repo` — git/hook/scripts mechanics (scripts/git-agent-commit etc.)

If no scope fits, do not invent one: split the change or ask.
