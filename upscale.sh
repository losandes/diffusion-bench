#!/usr/bin/env bash
#
# Run the standalone upscaler as a background-friendly, low-priority job.
#
# Mirrors run.sh (nice + capped thread pool) but drives `python3 -m src.upscale`.
# All arguments are passed straight through.
#
# Usage:
#   ./upscale.sh -i images/clip-animatediff.mp4 --to 1920x1080
#   ./upscale.sh -i frame.png --upscaler realesrgan --scale 4
#
# Tunables (override via environment):
#   DIFFUSION_BENCH_LEAVE_FREE   cores to leave free for the system (default: 2)
#   DIFFUSION_BENCH_NICE         nice level; higher = yields more (default: 15)

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

echo "Upscaling with ${THREADS} thread(s) of ${TOTAL_CORES} cores, nice level ${NICE_LEVEL}"

exec nice -n "$NICE_LEVEL" python3 -m src.upscale "$@"
