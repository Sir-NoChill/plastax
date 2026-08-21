# plastax distribution plan: PyPI publication, docs, and release automation

Coding-agent handoff, structured like `IMPLEMENTATION_PLAN.md`: phases with
acceptance criteria, HUMAN markers for steps Drew must perform, and a
Deviations section this document owns. Commits follow the agent-commit
protocol (Conventional Commits, mandatory scope, hooks must pass). Sources
are consolidated in `distribution-plan.bib`.

Decisions locked for this plan (2026-08-21):

- Docs: Sphinx + napoleon on Read the Docs, furo theme.
- Versioning/release: hatch-vcs tag-driven versions; GitHub Actions publishes
  to PyPI via OIDC trusted publishing [pypi-trusted].
- First-party algorithms ship in-tree (`plastax.optim`, later
  `plastax.heuristics`) -- see `ECOSYSTEM_ROADMAP.md`.

Current state this plan builds on (verified 2026-08-21): hatchling backend,
`name = "plastax"`, `version = "0.1.0.dev0"`, runtime deps `jax>=0.11.0` +
`jaxtyping>=0.2.36` only, `requires-python >= 3.12`, PEP 735 dev group, CI
matrix py3.12/3.13 (ruff, ty, mypy --strict, pytest), pre-commit + pre-push
hooks, MIT license, Google docstrings gated by ruff D + pydoclint. No path
dependencies; the sibling `jax`/`xla`/`stablehlo` clones are reference-only.

---

## Phase 0 -- Preflight

- P0.1 Confirm the name `plastax` is unregistered:
  `https://pypi.org/pypi/plastax/json` must return 404 (it appeared free on
  2026-08-21; re-check at execution time). If taken, STOP and ask Drew.
- P0.2 `src/plastax/py.typed` exists (verified 2026-08-21); confirm
  hatchling includes it in the wheel.
- P0.3 The repo currently has NO tags (verified 2026-08-21), so hatch-vcs
  must be configured with `raw-options.fallback_version = "0.1.0.dev0"` or
  an initial `v0.1.0rc1` tag must precede the first build (P1.2).
- P0.4 Fix an undeclared runtime dependency: `numpy` is imported throughout
  `src/plastax` (`_types`, `builder`, `topology`, `step`, ...) but appears
  only in the dev group; it currently installs transitively via jax. Add
  `numpy>=2` (match jax 0.11's floor) to `[project].dependencies` -- a
  library must declare what it imports, not rely on transitive resolution.

Acceptance: name free; `py.typed` present in a locally built wheel;
tag-derivable version confirmed.

## Phase 1 -- Packaging metadata hardening

- P1.1 Extend `[project]` in `pyproject.toml`:
  - `readme = "README.md"`, `keywords` (jax, plasticity, sparse,
    dynamic-networks, continual-learning), `classifiers`: Development Status
    3 - Alpha, Intended Audience :: Science/Research, Programming Language ::
    Python :: 3.12 / 3.13, Topic :: Scientific/Engineering :: Artificial
    Intelligence, Typing :: Typed.
  - `[project.urls]` gains `Documentation = "https://plastax.readthedocs.io"`.
- P1.2 Switch to dynamic versioning:
  - `dynamic = ["version"]`; add `hatch-vcs` to `[build-system].requires`;
    `[tool.hatch.version] source = "vcs"`; set
    `raw-options.version_scheme = "post-release"` only if Drew wants
    post-release counting -- default scheme is fine otherwise [hatch-vcs].
  - Tag format `vX.Y.Z`; record the SemVer intent (pre-1.0: minor bumps may
    break API) in README [semver].
- P1.3 Distribution contents:
  - Wheel: `src/plastax` only (hatchling default with src layout), plus
    LICENSE and `py.typed`. Verify tests/examples/docs are excluded.
  - Sdist: additionally include `tests/`, `examples/`, `README.md`,
    `CHANGELOG.md` via `[tool.hatch.build.targets.sdist]`.
- P1.4 Create `CHANGELOG.md` in Keep a Changelog format [keepachangelog],
  seeded with an `Unreleased` section summarizing v1 scope (pull from
  `IMPLEMENTATION_PLAN.md` milestones, not from git log noise).
- P1.5 GPU story stays documentation-only: plastax depends on `jax` (CPU
  wheel); users targeting CUDA/TPU install `jax[cuda12]` / `jax[tpu]`
  themselves per the JAX install matrix [jax-install]. Do NOT add
  passthrough extras -- jax extras names churn; a stale extra is worse than
  an install doc. Record this rationale in the docs installation page.

Acceptance: `uv build` produces sdist+wheel; `twine check dist/*` passes;
wheel is pure (`py3-none-any`); wheel contains `py.typed`, no tests;
version derives from a `v0.1.0rc1` test tag on a throwaway branch.

## Phase 2 -- Build verification

- P2.1 Add a `just`/script target `verify-dist`:
  fresh venv -> `pip install dist/*.whl` -> `python -c "import plastax"` ->
  run `examples/mlp_xor.py` under `JAX_PLATFORMS=cpu` as a smoke test.
- P2.2 Run `check-wheel-contents` and add it to the dev group.
- P2.3 Repeat P2.1 on Python 3.12 and 3.13 (mirror the CI matrix).

Acceptance: smoke passes on both interpreters from the wheel alone, in a
directory outside the repo (catches accidental reliance on repo files).

## Phase 3 -- Developer documentation (Sphinx + Read the Docs)

- P3.1 Add a `docs` PEP 735 group: `sphinx>=8`, `furo`,
  `sphinx-autodoc-typehints`, `myst-parser`, `sphinx-copybutton`.
  Note: the repo's existing `docs/parallel_mnist_plan.md` is an experiment
  plan, not user docs; move it to `docs/internal/` untouched.
- P3.2 Scaffold `docs/conf.py`: extensions `sphinx.ext.autodoc`,
  `sphinx.ext.napoleon`, `sphinx.ext.intersphinx` (python, jax, numpy),
  `sphinx_autodoc_typehints`, `myst_parser`, `sphinx_copybutton`; furo
  theme. Napoleon config: `napoleon_google_docstring = True`,
  `napoleon_custom_sections = [("Type Args", "params_style")]` -- the
  codebase documents PEP 695 type parameters under a custom `Type Args:`
  section (see `TOOLING.md`). Types render from signatures
  (`autodoc_typehints = "signature"`), matching the pydoclint contract that
  docstrings never duplicate types [sphinx-napoleon].
- P3.3 Page structure (narrative pages in MyST markdown so prose can be
  reused from existing design docs):
  - `index` -- what plastax is, one runnable snippet.
  - `installation` -- uv/pip, CPU default, GPU via jax extras (P1.5),
    Python >=3.12 requirement and why (jax 0.11 floor).
  - `quickstart` -- walk `examples/mlp_xor.py`.
  - `concepts/` -- traits and the Network contract; SoA arenas and views;
    named monoids; propagation (PIPELINE vs TOPOLOGICAL, recurrent cycles);
    the retrace/overflow protocol and Driver; donation contract. Distill
    from `../plastix-jax-rung0-design.md`; link, do not duplicate, the
    rung-ladder analysis.
  - `examples` -- gallery page linking `examples/` with one-paragraph
    orientation each (mlp_xor, ipc_multilayer, echo_state_network,
    sharded_reservoir, parallel_mnist).
  - `api/` -- autodoc pages per public module (`builder`, `state`, `step`,
    `driver`, `traits`, `topology`, `monoid`, `views`, `shard`, `optim`),
    driven by `__init__.py` exports; private modules excluded.
  - `contributing` -- toolchain summary (condensed `TOOLING.md`), commit
    conventions (`SCOPES.md`/`TAGS.md`), how to run the fast vs slow suites.
  - `changelog` -- include `CHANGELOG.md` via myst include.
- P3.4 `.readthedocs.yml`: ubuntu-24.04, python 3.13, install via the
  `docs` dependency group, `fail_on_warning: true` [rtd-config].
- P3.5 CI: add a docs job running `sphinx-build -W -b html docs docs/_build`
  on the same trigger as `check`.
- P3.6 Documentation debt created by this phase:
  - `TOOLING.md` names "mkdocstrings/griffe" as the assumed generator --
    update that sentence to napoleon/autodoc.
  - README: add Installation (pip) and Documentation (RTD link) sections;
    keep the design-doc reading order.
- P3.7 HUMAN: create the Read the Docs project, connect the GitHub repo,
  enable PR build previews.

Acceptance: `sphinx-build -W` clean locally and in CI; every `__init__`
export appears in the API reference; RTD build green after P3.7.

## Phase 4 -- Release automation

- P4.1 `.github/workflows/publish.yml`, triggered on tag push `v*`:
  1. `build`: `uv build`, `twine check`, upload dist as artifact.
  2. `verify`: matrix py3.12/3.13, install wheel from artifact, run the
     P2.1 smoke test.
  3. `publish`: `pypa/gh-action-pypi-publish@release/v1`, permissions
     `id-token: write`, GitHub environment `pypi` (required reviewers =
     Drew), no stored tokens; PEP 740 attestations are on by default
     [pypi-trusted][gh-pypi-action].
  - Tags matching `v*rc*` publish to TestPyPI instead (separate environment
    `testpypi`, `repository-url: https://test.pypi.org/legacy/`).
- P4.2 `RELEASING.md` checklist: CHANGELOG section cut; `uv run pytest` full
  suite (including `slow`); docs build clean; tag `vX.Y.Z` created and
  signed by Drew (release tags are human-signed; the agent GPG identity
  signs commits, not release tags); push tag; verify
  `pip install plastax==X.Y.Z` from a clean venv after the workflow lands.
- P4.3 HUMAN: PyPI trusted publisher DONE 2026-08-21 (workflow
  `publish.yml`, environment `pypi`). Remaining: TestPyPI pending publisher
  (same repo/workflow, environment `testpypi`) and the two GitHub
  environments with protection rules -- see the infrastructure table in
  `RELEASING.md` [pypi-trusted].
- P4.4 First release sequence: `v0.1.0rc1` -> TestPyPI -> install-check ->
  `v0.1.0` -> PyPI. The paper (mid-Oct target) can then cite a pinned,
  installable artifact.

Acceptance: rc lands on TestPyPI via CI only; `v0.1.0` on PyPI; project
page renders README, classifiers, and Documentation link correctly.

## Phase 5 -- End-user tooling requirements (documentation deliverable)

Captured in `installation` + `contributing` docs pages rather than code:

- Runtime: Python >=3.12, `pip install plastax` (pulls jax CPU + jaxtyping).
- Accelerators: user-installed `jax[cuda12]`/`jax[tpu]` per [jax-install];
  plastax is backend-agnostic pure-Python.
- Downstream trait authors (users writing their own Network subclasses)
  should be pointed at: jaxtyping runtime checking with beartype in their
  test suites (mirroring `addopts = --jaxtyping-packages=...`), the Driver
  retrace counters for detecting recompile storms, and uv for env
  management. These are recommendations, not dependencies.

## Commit protocol for this plan

Use the agent-commit skill for every commit. Check `SCOPES.md` first: add
`packaging`, `release`, and `docs-site` scopes if not present (via the
process SCOPES.md itself specifies). Suggested sequencing: one commit per
phase-item cluster, e.g. `build(packaging): dynamic version via hatch-vcs`,
`ci(release): tag-driven trusted publishing workflow`,
`docs(docs-site): sphinx scaffold with napoleon + furo`.

## Deviations

- 2026-08-21 (workflow name): the PyPI trusted publisher was registered
  against `publish.yml`, not the planned `release.yml`; the workflow file
  is named accordingly. The filename and the `pypi` environment name are
  part of the OIDC publisher identity and must not change.
- 2026-08-21 (P1.1 mostly pre-existing): readme/keywords/classifiers were
  already present in pyproject; only `[project.urls]` (incl. Documentation)
  was added. P1.1 reduces to a verification step.
- 2026-08-21 (scaffolding executed early): `.readthedocs.yaml`, the `docs`
  PEP 735 group, a minimal Sphinx scaffold (`docs/conf.py`, `index.md`,
  `api.md`), `RELEASING.md`, and the `docs/internal/` move (P3.1) were
  scaffolded ahead of phase order so RTD and the publish pipeline are
  exercisable now. `fail_on_warning` is temporarily `false` until Phase 3
  replaces the placeholder API page; P3.4 flips it. The CI docs job (P3.5)
  and `sphinx-build -W` acceptance remain outstanding.
- 2026-08-21 (RTD install method): docs deps install via
  `pip install --group docs .` (pip >= 25.1 PEP 735 support) rather than a
  uv-on-RTD setup -- fewer moving parts on the builder image. Revisit if
  RTD's pip lags. Post-P1.2 note: hatch-vcs on RTD's shallow clone may need
  a `git fetch --unshallow --tags` pre_install job (comment left in
  `.readthedocs.yaml`).
