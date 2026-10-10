# Tooling

Short reference for the toolchain: environment, pre-commit, and JAX-interop
testing. All commands run from the repository root.

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
alias) so a CUDA-enabled jaxlib + plugin resolves instead, or `cuda13` for a
CUDA-13 driver stack (the two extras are mutually exclusive):

```
uv sync --extra cuda12                 # or: pip install "plastax[cuda12]"
uv sync --extra cuda13                 # or: pip install "plastax[cuda13]"
```

The batched `layout="triton"` kernel (NVIDIA only) needs the `triton` extra:
`uv sync --extra cuda13 --extra triton` (or `pip install "plastax[cuda13,triton]"`).
Without it, batched steps use cuSPARSE CSR or the XLA edge list.

### TPU without a TPU (ahead-of-time compilation)

The `tpu` extra installs libtpu, which compiles against a TPU *topology
description* on any host -- no TPU attached. `examples/benchmarks/tpu_aot_check.py`
lowers and compiles every step type (churn with each growth path, streaming
and batched training in each layout) for a target generation and prints XLA's
memory / cost analysis and the scatter, gather and sort ops in the optimized
HLO. It validates that each path lowers on TPU; timings need hardware.

```
UV_PROJECT_ENVIRONMENT=.venv-tpu uv sync --extra tpu
JAX_PLATFORMS=cpu .venv-tpu/bin/python examples/benchmarks/tpu_aot_check.py --topology v5p:2x2x1
```

Topology names must cover a whole host (v5e:2x2, v6e:2x2, v5p:2x2x1,
v4:2x2x1). Pallas TPU (TensorCore) kernels can additionally run on CPU in
interpret mode (`interpret=pltpu.InterpretParams()`); SparseCore (`tpu_sc`)
kernels have no interpret support and can only be compiled, not run, off TPU.

plastax itself is backend-agnostic pure Python — the extra only swaps the jax
wheel. On a **shared** GPU, set `XLA_PYTHON_CLIENT_PREALLOCATE=false` so jax
grabs only what it needs rather than pre-reserving ~75 % of VRAM. The
dynamic-sparse CIFAR example (`examples/cifar_dst.py`) is the main GPU workload;
validated with `jax[cuda12]==0.11.0` on an RTX 3060 Ti and with
`jax[cuda13]==0.11.0` on an RTX 5000 Ada.

### GPU benchmark probes

`examples/benchmarks/` holds standalone scripts for measuring plastax on a GPU.
They are not collected by pytest (`testpaths = ["tests"]`) and need a CUDA
jaxlib: build a separate venv so the CPU-pinned dev venv stays untouched.

```bash
UV_PROJECT_ENVIRONMENT=.venv-gpu uv sync --extra cuda13   # or --extra cuda12
export XLA_PYTHON_CLIENT_PREALLOCATE=false                # shared GPU
.venv-gpu/bin/python examples/benchmarks/churn_probe.py --width 158114 --edges 50000000
```

| Script | Measures |
|---|---|
| `churn_probe.py` | Per-phase cost of a churn step (forward, prune, add) on a three-layer synthetic net; `--json` appends a result line. |
| `layouts_probe.py` | One sparse layer as COO `segment_sum`, BCOO, BCSR (cuSPARSE) and dense: forward at batch 1 and B, plus the CSR rebuild. |
| `sort_probe.py` | Which sort formulation XLA lowers to a radix sort. |
| `fused_prune_check.py` | The fused forward + prune Triton kernel (`make_step(fuse_prune=)`) against the two-pass step over churn steps, then both timed. |
| `triton_check.py` | The jax_triton edge kernel against the XLA product, checked and timed. |
| `tpu_aot_check.py` | Ahead-of-time TPU compilation of every step type (see above). |
| `growth_bench.py` | One add_conn phase per growth strategy against live units, live connections and P (CPU or GPU); `run_growth.sh` runs both backends with locked GPU clocks and `plot_growth.py` renders the plots and fits. Results and the write-up are in `benchmarks/growth.md`. |

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
silence it locally with a rule-scoped ignore and explain why in a comment;
do not weaken the mypy strict gate.
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
   tolerances assume this.
3. Retrace contract: `jax.test_util.assert_num_jit_and_pmap_compilations`
   for the "exactly N compilations" tests; debug misses locally with
   `JAX_EXPLAIN_CACHE_MISSES=1` (config.py:1303).
4. Donation contract: the "Some donated buffers were not usable" warning is
   promoted to an error via pytest filterwarnings (pyproject).
5. NaN hygiene: `JAX_DEBUG_NANS=1` is opt-in for local debugging, not CI
   default (it disables some fusion and would mask performance-shape bugs).
6. Version floor: the declared runtime floor is `jax>=0.10.2`. plastax is
   validated on jax 0.10.2 through 0.11.x. `uv.lock` is gitignored so CI resolves
   the latest jax satisfying the floor; a floor change is a deliberate change
   with a changelog entry, not a routine bump.

## Pre-commit / pre-push

`.pre-commit-config.yaml` defines: ruff check (autofix), ruff format,
pydoclint, ty check, and hygiene basics on pre-commit; mypy --strict and the
fast pytest suite on pre-push. Install with
`uv run pre-commit install --hook-type pre-commit --hook-type commit-msg --hook-type pre-push`.
These hooks must run and pass — never commit or push with `--no-verify`.

`default_stages: [pre-commit]` keeps the lint hooks from running again on
commit-msg and pre-push over files they already checked. mypy is incremental
through `.mypy_cache/` (a few seconds cold, well under a second warm), so the
pre-push cost is the test suite.

### Test tiers

| Tier | Selection | Runs in |
|---|---|---|
| fast | `-m "not slow"` | pre-push hook (in parallel), and as part of the full suite |
| slow | `@pytest.mark.slow` | the full suite only: CI and the full local gate |
| full | everything (`uv run pytest`) | CI (both Python versions) and the full local gate |

The `slow` tier holds the tests whose cost is not compilation but work: the
optax oracle parity (`test_optim.py`, whose optax dependency is test-only), the
multi-controller fan-outs that spawn several JAX processes
(`test_mc_sharding.py`, `test_mc_construct.py`, `test_mc_driver.py`), training
every showcase optimizer to convergence
(`test_mlp_xor.py::test_every_showcase_optimizer_learns_xor`), and the CBP
replacement-rate oracle (`test_cbp.py::test_v1_local_threshold_agrees_with_the_oracle_on_rate`).
Mark a new test `slow` only for the same reason; a test that is merely slow to
compile belongs in the fast tier, where the compilation cache absorbs it.

The pre-push hook runs the fast tier as

```
uv run pytest -m "not slow" -q -n auto --maxprocesses=12
```

and skips it when the pushed commits touch only prose (`docs/`,
`benchmarks/`, `.agents/`, `.github/`, `*.md` and similar; the hook's
`exclude` pattern is the list). On a 20-thread host it takes about 30-40 s
with a warm cache and 70-90 s cold.

Two things make it fast:

- **pytest-xdist.** Every test runs in a worker process with its own JAX
  runtime (the conftest environment is set per worker), so tests share no
  device state. More than about 12 workers oversubscribes a 20-thread host,
  because each worker's XLA also uses a thread pool.
- **JAX's persistent compilation cache.** Most of the suite's time is XLA
  compilation, not execution. `tests/conftest.py` points
  `JAX_COMPILATION_CACHE_DIR` at `.cache/jax` (git-ignored) and caches every
  executable, which cuts a run to about a third. The cache key is a hash of
  the HLO module, the compile options, `XLA_FLAGS`, the backend and the
  jax/jaxlib versions, so any change to the code being compiled is a miss,
  never a stale hit; tracing, lowering (including the donation check) and
  execution still run on every test. The subprocess-based tests inherit the
  setting. The cache is dropped at session start once it outgrows 1 GiB, and
  `rm -rf .cache/jax` or `JAX_ENABLE_COMPILATION_CACHE=false` gives a cold run.
  jax's `jax_compilation_cache_check_contents` cannot be used to audit it on
  CPU: XLA:CPU compiles kernels in parallel and packs them into object files
  in a run-dependent order, so two fresh compiles of the same HLO can differ
  byte-for-byte in their machine-code section while their HLO is identical.

The full local gate, matching CI, runs every tier serially:

```
uv sync && uv run ruff check src tests examples \
  && uv run ruff format --check src tests examples \
  && uv run pydoclint --style=google --arg-type-hints-in-docstring=False \
       --check-return-types=False --check-class-attributes=True \
       --skip-checking-private-functions=True src/plastax \
  && uv run ty check src && uv run mypy --strict src \
  && uv run pytest \
  && uv run python scripts/parity/emit.py --check \
  && uv run --group docs sphinx-build -W --keep-going -b html docs docs/_build
```

The C++ oracle and golden tests look for a plastix checkout at
`$PLASTAX_CPP_DIR` (default `../plastax-cpp`) and skip without one.

## Commit conventions

Commit metadata contracts: {doc}`tags` (types) and {doc}`scopes`
(mandatory scopes). Every commit is `type(scope): subject` with a
mandatory scope, one scope per commit; the hooks above are the gate (never
`--no-verify`).

The repository ships no signing wrapper or keys. Contributors — and their
coding agents — commit under their own identity and attribute or sign their
work as they see fit; configure your agent's author identity and optional GPG
key in your own environment.
