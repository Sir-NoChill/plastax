# plastax tooling: environment, pre-commit, and JAX-interop testing

Short reference for the toolchain. All commands run from `plastax/`.

## Environment: uv

```
uv sync                      # .venv + editable install + dev group (default)
uv run pytest                # run anything inside the venv
```

Interpreter is pinned by `.python-version` (3.13); `requires-python` is
`>=3.12` (jax 0.11's floor). Dev tools live in the PEP 735
`[dependency-groups].dev` table, which `uv sync`/`uv run` install by default —
so the bare `uv run ty|mypy|pytest` hook entries resolve them without extra
flags. `uv.lock` is gitignored (library convention: CI resolves against the
current dependency floor).

## Lint + format: ruff

One tool for both. `ruff check` (rules pinned in pyproject: E/F/I/UP/B/ANN)
and `ruff format`. Formatting is not a style debate; it is a hook.

## Types: ty first, mypy strict as fallback

Primary checker is `ty` (Astral, experimental). Because it is pre-1.0, the
contract is: `ty check` runs in pre-commit as the fast checker, and
`mypy --strict` runs in CI as the authoritative gate. If ty false-positives
on something load-bearing (jaxtyping annotations are the likely friction),
silence it locally with a rule-scoped ignore and note it in
IMPLEMENTATION_PLAN.md Deviations; do not weaken the mypy strict gate.
jaxtyping erases to `jax.Array` for static checkers, so neither checker
needs a plugin.

## JAX-interop testing

The JAX-specific test infrastructure, beyond plain pytest:

1. Runtime shape/dtype checking: jaxtyping's pytest hook with beartype —
   `pytest --jaxtyping-packages=plastax,beartype.beartype` (wired into
   `addopts`). Every annotated signature in the package becomes a runtime
   contract during tests, at zero cost outside them.
2. Determinism: tests run on CPU (`JAX_PLATFORMS=cpu` in conftest) so CI
   needs no accelerator and float reductions are reproducible; the oracle
   tolerances in tests/README.md assume this.
3. Retrace contract: `jax.test_util.assert_num_jit_and_pmap_compilations`
   for the "exactly N compilations" tests; debug misses locally with
   `JAX_EXPLAIN_CACHE_MISSES=1` (config.py:1303).
4. Donation contract: the "Some donated buffers were not usable" warning is
   promoted to an error via pytest filterwarnings (pyproject).
5. NaN hygiene: `JAX_DEBUG_NANS=1` is opt-in for local debugging, not CI
   default (it disables some fusion and would mask performance-shape bugs).
6. Version pin: CI tests against the pinned floor `jax==0.11.*`; the design
   docs cite 0.11.1 internals (Ref API, donation platform list), so a jax
   upgrade is a deliberate change with a Deviations entry, not a routine
   bump.

## Pre-commit / pre-push

`.pre-commit-config.yaml` defines: ruff check (autofix), ruff format,
ty check, and hygiene basics on pre-commit; mypy --strict and the fast
pytest suite (`-m "not slow"`) on pre-push. Install with
`uv run pre-commit install --hook-type pre-commit --hook-type pre-push`.
The agent-commit protocol requires these hooks to run and pass — never
commit with `--no-verify` (the wrapper rejects it outright).

## Commit workflow (agent-commit protocol)

Commit metadata contracts live at the repo root: TAGS.md (types) and
SCOPES.md (mandatory scopes). Agent commits go through the wrapper:

```
scripts/git-agent-commit            # or symlink onto PATH as git-agent-commit
git agent-commit -m "feat(sweep): add named-monoid segment reduce"
```

The wrapper injects the agent identity (Agent (claude) <ai@blobfish.icu>)
and signs the commit. If $AGENT_SIGNING_KEY is unset it auto-resolves the
agent's secret key by email ($AGENT_GIT_EMAIL), so signing works with zero
per-shell setup (env vars do not persist across agent tool calls); set
AGENT_REQUIRE_SIGNING=1 to hard-fail when no key is found. It refuses
--no-verify and forwards to `git commit` so hooks run normally. Invoke it by
path (`scripts/git-agent-commit ...`) to use this repo's identity and guards
rather than any `git-agent-commit` earlier on PATH. Humans use plain
`git commit`; the wrapper keeps agent work auditable.
