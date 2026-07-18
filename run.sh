#!/usr/bin/env bash
#
# Run diffusion-bench as a background-friendly, low-priority job.
#
# It gives the process most of the CPU but yields to interactive work (via
# `nice`), and caps the thread pool so a couple of cores stay free for the rest
# of the system. All arguments are passed straight through to `python3 -m src`.
#
# Usage:
#   ./run.sh --prompt "a cute, tabby cat" --models "dreamlike-art/dreamlike-photoreal-2.0" --steps "10"
#
# Tunables (override via environment):
#   DIFFUSION_BENCH_LEAVE_FREE   cores to leave free for the system (default: 2)
#   DIFFUSION_BENCH_NICE         nice level; higher = yields more (default: 15)
#   PYTORCH_MPS_HIGH_WATERMARK_RATIO
#                                Apple Silicon only. Caps how much unified memory
#                                MPS may use so the GPU/UI stays responsive and the
#                                machine doesn't swap. Unset by default because too
#                                low will OOM large models (e.g. HiDream). Try 0.7
#                                for the smaller SD / dreamlike models.

set -euo pipefail

# run from the repo root regardless of where the script is invoked from
cd "$(dirname "$0")"

# --- detect total CPU cores (portable: Linux + macOS) ---
if command -v nproc >/dev/null 2>&1; then
  TOTAL_CORES="$(nproc)"
elif command -v sysctl >/dev/null 2>&1; then
  TOTAL_CORES="$(sysctl -n hw.ncpu)"
else
  TOTAL_CORES=4
fi

# leave some cores for the rest of the system, but always keep at least 1 thread
LEAVE_FREE="${DIFFUSION_BENCH_LEAVE_FREE:-2}"
THREADS=$(( TOTAL_CORES - LEAVE_FREE ))
if [ "$THREADS" -lt 1 ]; then
  THREADS=1
fi

NICE_LEVEL="${DIFFUSION_BENCH_NICE:-15}"

export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export VECLIB_MAXIMUM_THREADS="$THREADS"

# Apple Silicon only: pass through an MPS memory cap if the caller set one.
# (Left unset by default so large models don't OOM.)
if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
  if [ -n "${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-}" ]; then
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO
  fi
fi

echo "Running with ${THREADS} thread(s) of ${TOTAL_CORES} cores, nice level ${NICE_LEVEL}"

exec nice -n "$NICE_LEVEL" python3 -m src "$@"
