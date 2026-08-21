# Releasing plastax

Companion to `DISTRIBUTION_PLAN.md` Phase 4. Publishing is tag-driven and
tokenless; nothing is released from a developer machine.

## Infrastructure (state as of 2026-08-21)

| Piece | Status | Notes |
|---|---|---|
| PyPI trusted publisher | CONFIGURED | repo `Sir-NoChill/plastax`, workflow `publish.yml`, environment `pypi`. The workflow filename and environment name are part of the OIDC identity -- renaming either breaks publishing. |
| TestPyPI trusted publisher | PENDING (HUMAN) | register the same publisher on test.pypi.org with environment `testpypi` to enable the rc flow. |
| GitHub environment `pypi` | PENDING (HUMAN) | Settings -> Environments -> New: `pypi`; add Drew as required reviewer so publishes pause for approval. Referencing it from the workflow alone creates it without protection rules. |
| GitHub environment `testpypi` | PENDING (HUMAN) | same, no reviewer needed. |
| Read the Docs | CONFIGURED | https://app.readthedocs.org/projects/plastax/ ; builds from `.readthedocs.yaml` via the RTD GitHub App webhook. Enable "build pull requests" in RTD settings for PR previews. |

No secrets or environment variables are required anywhere: PyPI auth is
OIDC (`id-token: write` in `publish.yml`), RTD authenticates via its GitHub
App. The only workflow env var is `JAX_PLATFORMS=cpu` on the verify job,
which forces the smoke test onto the CPU backend of the runners.

## Release checklist

1. Full suite green: `uv run pytest` (including `slow`), `uv run mypy
   --strict src`, `uv run ruff check src tests examples`.
2. Docs build clean: `uv run sphinx-build -b html docs docs/_build`
   (add `-W` once P3.4 flips fail_on_warning).
3. Cut a CHANGELOG section for the version (P1.4).
4. Tag: `git tag -s vX.Y.Z -m "plastax X.Y.Z"` -- release tags are signed
   by Drew, not the agent identity; hatch-vcs derives the version from this
   tag (P1.2), so the pyproject version is never edited by hand.
5. `git push origin vX.Y.Z`. The `Publish` workflow builds, twine-checks,
   wheel-smoke-tests on 3.12/3.13, then waits on the `pypi` environment
   approval before uploading.
6. Post-release: from a clean venv outside the repo,
   `pip install plastax==X.Y.Z && python -c "import plastax"`; check the
   PyPI page renders README/classifiers and RTD built the tag.

Release candidates: tag `vX.Y.ZrcN`; the workflow routes any tag containing
`rc` to TestPyPI instead of PyPI (requires the pending TestPyPI publisher).

First release sequence (P4.4): `v0.1.0rc1` -> TestPyPI -> install check ->
`v0.1.0` -> PyPI.
