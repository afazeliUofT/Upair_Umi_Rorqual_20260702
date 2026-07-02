#!/usr/bin/env bash
#SBATCH --job-name=u3-baselines
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --output=logs/final_u3/%x_%j.out

set -euo pipefail
ROOT="${UPAIR_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
[[ -n "$ROOT" && -f "$ROOT/upair_portable_env.sh" ]] || {
    echo "[FATAL] Invalid repository root: $ROOT" >&2
    exit 2
}
ROOT="$(cd "$ROOT" && pwd)"
SCRIPT_DIR="$ROOT/final_u3_eval"
cd "$ROOT"
source "$ROOT/upair_portable_env.sh"
upair_activate

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TF_CPP_VMODULE="${TF_CPP_VMODULE:-bfc_allocator=0}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

last_count=-1
stagnant=0
while true; do
  set +e
  bash "$SCRIPT_DIR/baseline_worker.sh"
  rc=$?
  set -e
  if [[ $rc -eq 0 ]]; then
    exit 0
  fi
  count="$(find "$ROOT/_final_u3_baseline_chunks" -name chunk_result.csv -type f | wc -l)"
  echo "[24H-WRAPPER] baseline worker rc=$rc chunk_count=$count"
  if [[ "$count" == "$last_count" ]]; then
    stagnant=$((stagnant + 1))
  else
    stagnant=0
  fi
  last_count="$count"
  if (( stagnant >= 2 )); then
    echo "[FATAL] No baseline progress across three attempts." >&2
    exit "$rc"
  fi
  sleep 15
done
