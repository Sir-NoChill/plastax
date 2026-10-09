# Contributing to plastax

Thanks for your interest in plastax. This guide covers how to set up a
development environment, where a change belongs, the rules every change must
respect, and what a pull request needs before it can be merged. The full
toolchain reference lives in the documentation under *Development*
(`docs/development/`).

## Setting up

plastax uses [uv](https://docs.astral.sh/uv/). From a clone of the repository:

```bash
uv sync                                    # .venv + editable install + dev tools (CPU jax)
uv run pre-commit install --hook-type pre-commit --hook-type commit-msg --hook-type pre-push
uv run pytest -m "not slow"                # fast suite (what pre-push runs)
uv run pytest                              # full suite, including optax / C++ oracle parity
```

The interpreter is pinned to 3.13 (`.python-version`); plastax supports 3.12+.
Tests always run on CPU with four fake devices (`tests/conftest.py`), so no
accelerator is needed. For GPU or TPU work, use a separate virtualenv (for
example `UV_PROJECT_ENVIRONMENT=.venv-gpu uv sync --extra cuda13`); see
`docs/development/tooling.md`.

## Where does my change go?

Most contributions do **not** touch the framework. A new plasticity algorithm
is almost always a new policy class implementing an existing Protocol and
assigned as a class attribute on a `plastax.Network` subclass.

| You are adding… | Files you touch | Start from |
|---|---|---|
| A forward/backward pass, loss, connection-update, prune or grow policy | a new class in your module or example | `examples/mlp_xor.py`, `examples/dst_sparse.py` |
| An optimizer | `src/plastax/optim/_<name>.py` + `optim/__init__.py` | `optim/_momentum.py`, `optim/_adam.py` |
| A layer / connectivity generator | `src/plastax/topology.py` | `dense`, `conv2d` |
| A new reduction for accumulators | `src/plastax/monoid.py` | the `_Named` tables |
| A new phase category (rare) | `traits.py` → `phases.py` → maybe `sweep.py` | open an issue first |

A few facts that are easy to get wrong:

- Policies are pure per-element functions. They read through
  `UnitView`/`ConnView` and return `UnitWrite`/`ConnWrite` records. They never
  index raw columns, read another element's state, or keep state beyond
  hyperparameters.
- The phase order is fixed: forward → loss → backward → update_unit →
  update_conn → prune_unit → prune_conn → add_unit → add_conn → reset_global.
  An `UpdateConn` can read what the backward pass wrote; `prune_conn` sees
  `update_conn`'s fresh weights; growth sees the units `add_unit` spawned.
- Forward `map` accumulates into the destination unit, backward `map` into the
  source unit.
- A growth score of `-inf` (any non-finite score, from a `ScoreAddConn` or a
  `ProposeAddConn` proposal) is a hard veto, distinct from a low score.
- Growth deduplicates nothing by default: a rule that must never grow a
  parallel edge sets `dedupe_live = True` (and `dedupe_step = True` against
  repeats within a step).
- Extra per-unit / per-connection fields are declared with
  `extra_unit_fields` / `extra_conn_fields` and must not collide with the
  reserved columns (`from_id`, `to_id`, `dead`, `weight`, `activation`,
  `level`, `pruned`). A regrown edge's fields start at their `FieldSpec` default, so
  optimizer state should default to 0.0 rather than special-casing regrowth.
- Optimizers form the gradient with the delta rule
  `grad_field[dst] * ACTIVATION[src]`, and never read the network globals.

## Design invariants

These are settled design decisions. A change that breaks one will not be
merged even if the tests pass; if your change seems to need it, open an issue
to discuss the approach first.

1. **Phase elision is Python-level.** An absent trait contributes zero
   equations to the jaxpr; never gate a phase with `lax.cond`.
2. **All shapes are static.** The only retrace events are bucket growth
   (`state.grow_bucket`) and level reassignment (`topo.resort`).
3. **Live counts are derived** from the `DEAD` mask, never stored.
4. **Dead slots use the null-slot trick** (redirect the index out of range so
   the segment reduction drops it), never a shape-changing masked gather.
5. **Steps preserve the state pytree's shapes and dtypes** so every leaf is
   donated. Unusable donated buffers are a test failure.
6. **Policies are pure, vmapped and per-element** (see above).
7. **Type discipline:** `mypy --strict` must pass, with `FieldSpec` generics
   intact end to end.
8. **Out of scope for now:** adding/pruning units under Scheme-A sharding
   (both run on a single device and raise `NotImplementedError` when
   sharded), generic `Monoid(op, identity)` lowering (it must keep raising
   `UnsupportedMonoidError`), `jax.Ref` arenas, and MLIR emission.

## Tests

Add tests in `tests/test_<topic>.py`, one file per trait or mechanism, and ask
of each one: *if the code broke, would this fail?* Seed every RNG and don't
override the CPU/device setup in `conftest.py`.

Pick the comparison tolerance by what you compare against, and say why in a
comment:

| Comparison | Tolerance |
|---|---|
| External oracle (optax, the plastax-cpp library) | `rtol=1e-4, atol=1e-5` |
| Internal numpy reference, same reduction order | `rtol=1e-6, atol=1e-6` |
| Cross-mode equivalence (pipeline vs topological) | `rtol=1e-5, atol=1e-5` |
| Exact invariants (e.g. regrown state is zeroed) | `atol=0.0` |

Mark heavy or optional-dependency tests `@pytest.mark.slow` and use
`pytest.importorskip`. A new optimizer needs an optax parity test and, if it
carries state, a regrowth-zeroing test (see `test_optim.py` and
`test_optim_sparse.py`). Retrace-count tests use
`assert_num_jit_and_pmap_compilations` (see `test_resort.py`).

Examples in `examples/<name>.py` are welcome: each should demonstrate one
point, use only the public API, and have a `main()` that asserts its own
success criteria so it doubles as an acceptance test.

## Style, docs and types

- **Formatting and lint:** `ruff format` and `ruff check` (run by pre-commit).
- **Docstrings:** Google style on everything under `src/plastax`, checked by
  ruff `D` and pydoclint. Types live in the signature only; public dataclass
  fields go under `Attributes:`; PEP 695 type parameters under `Type Args:`;
  constructors are documented on the class, not in `__init__`.
- **Types:** `ty` runs on pre-commit, `mypy --strict src` on pre-push and in
  CI. Silence a checker only with a rule-scoped ignore and a comment saying
  why; never weaken the mypy gate.
- **Comments** explain *why* (a gotcha, a reference to the C++ oracle or a
  paper), not *what* the code does or how it got there.
- **Docs:** keep `docs/` in sync. A new public symbol goes in
  `src/plastax/__init__.py`'s `__all__` (and therefore the API reference); a
  new optimizer gets an entry in `docs/optimizers.md`; user-visible changes get
  a line in `docs/changelog.md`.

## Commits and pull requests

Every commit message is `type(scope): subject` with a mandatory scope, one
scope per commit. The allowed types and scopes are listed in
`docs/development/tags.md` and `docs/development/scopes.md`, and the
commit-msg hook enforces them. A diff that spans several scopes is a signal to
split the commit.

Breaking changes to the public API (`plastax.*` exports) use
`type(scope)!: subject`, a `BREAKING CHANGE:` footer, and a changelog entry in
the same commit.

The hooks are the gate: please don't bypass them with `--no-verify`. Before
opening a pull request, make sure that:

- [ ] `uv run pytest` passes (or at least the fast suite, if you can't run the
      slow one; say so in the PR);
- [ ] `uv run mypy --strict src` and `uv run ruff check src tests examples`
      are clean;
- [ ] the docs build: `uv run --group docs sphinx-build -W -b html docs docs/_build`;
- [ ] tests, docs and the changelog cover the change.

Coding agents are welcome. `AGENTS.md` and `.agents/` hold agent-facing
guidance, including review and scaffolding playbooks. Commit under your own
identity and take responsibility for what your agent produces.

By contributing, you agree that your contributions are licensed under the
project's MIT license.
