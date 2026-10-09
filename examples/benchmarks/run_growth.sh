#!/usr/bin/env bash
# Run the growth bench on CPU and GPU and render the plots.
#
#   examples/benchmarks/run_growth.sh CPU_PYTHON GPU_PYTHON [growth_bench args...]
#
# CPU_PYTHON is a Python with plastax and the CPU jaxlib (e.g. .venv/bin/python)
# and GPU_PYTHON one with a CUDA jaxlib (e.g. .venv-gpu/bin/python, see
# docs/development/tooling.md). Either may be "-" to skip it. Extra args
# (e.g. --quick) go to both runs. GROWTH_CPU_MAX_CANDIDATES (default 2^22)
# caps the candidates per call on CPU, where the largest points take minutes.
#
# Both runs disable XLA's constant folding, which otherwise spends seconds
# to minutes compiling a point (the exhaustive grid, the claim fusions on
# CPU) without changing its run time. The GPU run locks the clocks (SM 2505 /
# memory 8551 MHz, the sustained throttle-free point of an RTX 5000 Ada;
# override with GROWTH_LOCK_GR / GROWTH_LOCK_MEM) through `sudo -n nvidia-smi`
# and always resets them on exit.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="$ROOT/benchmarks/results/growth"
CPU_PY="${1:?usage: run_growth.sh CPU_PYTHON GPU_PYTHON [args...]}"
GPU_PY="${2:?usage: run_growth.sh CPU_PYTHON GPU_PYTHON [args...]}"
shift 2
mkdir -p "$OUT"
NO_FOLD=--xla_disable_hlo_passes=constant_folding

if [ "$CPU_PY" != "-" ]; then
  JAX_PLATFORMS=cpu XLA_FLAGS="$NO_FOLD" \
    "$CPU_PY" "$HERE/growth_bench.py" --out "$OUT/growth_cpu.csv" \
    --max-candidates "${GROWTH_CPU_MAX_CANDIDATES:-4194304}" "$@"
fi

if [ "$GPU_PY" != "-" ]; then
  GR="${GROWTH_LOCK_GR:-2505}"
  MEM="${GROWTH_LOCK_MEM:-8551}"
  SMI=""
  reset_clocks() {
    if [ -n "$SMI" ]; then kill "$SMI" 2>/dev/null || true; fi
    sudo -n nvidia-smi -rgc >/dev/null 2>&1 || true
    sudo -n nvidia-smi -rmc >/dev/null 2>&1 || true
  }
  trap reset_clocks EXIT
  trap 'exit 130' INT TERM
  sudo -n nvidia-smi -lgc "$GR,$GR" >/dev/null
  sudo -n nvidia-smi -lmc "$MEM,$MEM" >/dev/null
  # Clock log for the whole run (idle samples between points read low).
  nvidia-smi --query-gpu=timestamp,name,clocks.sm,clocks.mem,utilization.gpu \
    --format=csv -lms 500 > "$OUT/device_clocks.csv" &
  SMI=$!
  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS="$NO_FOLD" \
    "$GPU_PY" "$HERE/growth_bench.py" --out "$OUT/growth_gpu.csv" "$@"
  reset_clocks
  trap - EXIT
fi

uv run --with matplotlib --with pandas \
  python "$HERE/plot_growth.py" "$OUT" \
  --cx "${GROWTH_CX_RESULTS:-$ROOT/../plastix/benchmarks/results/growth}"
