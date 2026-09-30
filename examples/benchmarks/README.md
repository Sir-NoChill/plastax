# GPU benchmark probes

Standalone scripts for measuring plastax on a GPU. They are not collected by
pytest (`testpaths = ["tests"]`) and need a CUDA jaxlib: build a separate venv
so the CPU-pinned dev venv stays untouched.

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
| `plot_step_300M.py` | Regenerates `docs/scale_plan/step_300M.png` for `SCALE_PLAN.md`. |

At large sizes XLA's sort autotuner logs failed allocations of many GB (up to
TiB) while it probes workspace sizes; it falls back and the runs are correct.
