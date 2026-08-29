# plastax tooling: environment, pre-commit, and JAX-interop testing

Short reference for the toolchain. All commands run from `plastax/`.

## Environment: uv

```
uv sync                      # .venv + editable install + dev group (default)
uv run pytest                # run anything inside the venv
```

Interpreter is pinned by `.python-version` (3.13); `requires-python` is
`>=3.12` (uv's universal resolver has no solution at 3.11). Dev tools live in the PEP 735
`[dependency-groups].dev` table, which `uv sync`/`uv run` install by default —
so the bare `uv run ty|mypy|pytest` hook entries resolve them without extra
flags. `uv.lock` is gitignored (library convention: CI resolves against the
current dependency floor).

### GPU (optional)

The default `uv sync` installs the **CPU** jax wheel. For an NVIDIA GPU, add the
`cuda12` extra (declared in `[project.optional-dependencies]`; `gpu` is an
alias) so a CUDA-enabled jaxlib + plugin resolves instead:

```
uv sync --extra cuda12                 # or: pip install "plastax[cuda12]"
```

plastax itself is backend-agnostic pure Python — the extra only swaps the jax
wheel. On a **shared** GPU, set `XLA_PYTHON_CLIENT_PREALLOCATE=false` so jax
grabs only what it needs rather than pre-reserving ~75 % of VRAM. The
dynamic-sparse CIFAR example (`examples/cifar_dst.py`) is the main GPU workload;
validated with `jax[cuda12]==0.11.0` on an RTX 3060 Ti.

## Lint + format: ruff

One tool for both. `ruff check` (rules pinned in pyproject: E/F/I/UP/B/ANN/D)
and `ruff format`. Formatting is not a style debate; it is a hook.

## Docstrings: Google style, gated by ruff D + pydoclint

The library surface (`src/plastax`) ships **Google-style docstrings**. Two
tools enforce this, both wired into pre-commit:

- **ruff `D`** (pydocstyle, `convention = "google"`) — checks presence and
  shape. Scoped to `src/plastax` via `per-file-ignores` (tests and examples
  are exempt). `D107` is ignored: `__init__` docstrings are intentionally
  omitted — constructors are documented on the class.
- **pydoclint** (`--style=google`) — checks that `Args:`/`Returns:`/`Raises:`
  match the actual signature. Types stay in the signature, never duplicated in
  the docstring, so it runs with `--arg-type-hints-in-docstring=False
  --check-return-types=False`; `--check-class-attributes=True` guards
  `Attributes:` completeness; `--skip-checking-private-functions=True` limits
  the contract to the public surface (so a private validator may document a
  delegated exception without tripping DOC503). pydoclint's default `DOC301`
  is kept, which forbids a redundant `__init__` docstring — the other half of
  the D107 decision above.

Conventions in the docstrings themselves: PEP 695 type parameters go in a
`Type Args:` section; public dataclass fields go in `Attributes:` (description
only — the generator, e.g. mkdocstrings/griffe, sources types from the
signature). Run the pydoclint check directly with:

```
uv run pydoclint --style=google --arg-type-hints-in-docstring=False \
  --check-return-types=False --check-class-attributes=True \
  --skip-checking-private-functions=True src/plastax
```

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
6. Version floor: the declared runtime floor is `jax>=0.10.2`. plastax is
   validated on 0.10.2 (the Alliance/Narval wheelhouse's GPU-capable set:
   jax/jaxlib/jax-cuda12-plugin/jax-cuda12-pjrt all 0.10.2; full fast suite
   green) through 0.11.x (local CUDA). `uv.lock` is gitignored so CI resolves
   the latest jax satisfying the floor; a floor change is a deliberate change
   with a Deviations entry, not a routine bump.

## Pre-commit / pre-push

`.pre-commit-config.yaml` defines: ruff check (autofix), ruff format,
ty check, and hygiene basics on pre-commit; mypy --strict and the fast
pytest suite (`-m "not slow"`) on pre-push. Install with
`uv run pre-commit install --hook-type pre-commit --hook-type pre-push`.
These hooks must run and pass — never commit or push with `--no-verify`.

## Commit conventions

Commit metadata contracts live at the repo root: TAGS.md (types) and
SCOPES.md (mandatory scopes). Every commit is `type(scope): subject` with a
mandatory scope, one scope per commit; the hooks above are the gate (never
`--no-verify`).

The repository ships no signing wrapper or keys. Contributors — and their
coding agents — commit under their own identity and attribute or sign their
work as they see fit; configure your agent's author identity and optional GPG
key in your own environment.
