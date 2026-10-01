# Installation

Requires Python >= 3.12 (the jax 0.10 floor).

```
pip install plastax            # or: uv add plastax
```

The default install pulls the **CPU** wheel of JAX. plastax itself is pure
Python and backend-agnostic: the backend is whichever jaxlib is installed.

## Accelerators: optional extras

On a GPU or TPU machine, install the matching extra. Without it, JAX falls
back to the CPU **silently** (it logs a warning, then runs), so a plain
`pip install plastax` on a GPU box runs at CPU speed with no error. Each extra
is a thin passthrough to JAX's own extra, pinned to the version validated
here:

| extra | installs | for | validated with |
|---|---|---|---|
| `plastax[cuda12]` | `jax[cuda12]` (CUDA 12 jaxlib + plugin) | NVIDIA GPUs, CUDA 12 driver stack | jax 0.11.0, RTX 3060 Ti |
| `plastax[gpu]` | alias of `cuda12` | the broadest-compatibility NVIDIA install | as `cuda12` |
| `plastax[cuda13]` | `jax[cuda13]` (CUDA 13 jaxlib + plugin) | NVIDIA GPUs with a CUDA 13 driver | jax 0.11.2, RTX 5000 Ada (driver 615) |
| `plastax[tpu]` | `jax[tpu]` (libtpu) | Cloud TPU hosts; also ahead-of-time compilation for TPU on any host | jax 0.11.2, libtpu 0.0.48 (AOT for v4 / v5e / v5p / v6e) |
| `plastax[triton]` | `jax-triton` (and `triton`) | the batched edge-once kernel, `make_step(..., layout="triton")`, on NVIDIA GPUs | jax-triton 0.4.1, triton 3.8.0 |

`cuda12`, `cuda13` and `tpu` are mutually exclusive (each swaps in a
different jaxlib). `triton` combines with a CUDA extra:
`pip install "plastax[cuda13,triton]"`. On a shared GPU set
`XLA_PYTHON_CLIENT_PREALLOCATE=false` so JAX does not reserve most of the
device memory up front.

### Why passthrough extras

An earlier plan (`DISTRIBUTION_PLAN.md` P1.5) kept the GPU story
documentation-only, because JAX's extra names change between releases (the
move from `cuda12` to `cuda13` is one such change) and a stale extra is worse
than an install note. The extras were kept anyway, for two reasons:

- the silent CPU fallback above, which makes "forgot the CUDA jaxlib" an
  invisible order-of-magnitude slowdown rather than an error; and
- the accelerator paths the library now has. The batched CSR (cuSPARSE) and
  Triton layouts exist only on NVIDIA, and `examples/benchmarks/tpu_aot_check.py`
  needs libtpu.

The cost is the churn risk. Each extra above lists the JAX version it was
validated with. Before every release, re-check that each extra still resolves
(`uv lock` resolves all of them; see `RELEASING.md`). An extra whose JAX name
disappears is removed, not left stale.

## From source

```
git clone https://github.com/Sir-NoChill/plastax && cd plastax
uv sync                                   # dev environment (CPU)
UV_PROJECT_ENVIRONMENT=.venv-gpu uv sync --extra cuda13 --extra triton
```

See `TOOLING.md` for the GPU and TPU development setups.
