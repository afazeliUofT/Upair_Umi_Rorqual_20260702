#!/usr/bin/env bash
#SBATCH --job-name=umi-baselines
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:h100:1
#SBATCH --output=logs/umi_sensitivity/%x_%j.out

set -euo pipefail
ROOT="${UPAIR_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
[[ -n "$ROOT" && -f "$ROOT/upair_portable_env.sh" ]] || {
  echo "[FATAL] Invalid repository root: $ROOT" >&2
  exit 2
}
ROOT="$(cd "$ROOT" && pwd)"
cd "$ROOT"
source "$ROOT/upair_portable_env.sh"
upair_activate

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TF_CPP_VMODULE="${TF_CPP_VMODULE:-bfc_allocator=0}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

last="-1"
stagnant=0
while true; do
  set +e
  python "$ROOT/umi_sensitivity/driver.py" eval-baselines
  rc=$?
  set -e
  [[ $rc -eq 0 ]] && exit 0

  now="$(python "$ROOT/umi_sensitivity/driver.py" signature --kind baseline)"
  echo "[24H] baselines rc=$rc chunks=$now"
  if [[ "$now" == "$last" ]]; then
    stagnant=$((stagnant + 1))
  else
    stagnant=0
  fi
  last="$now"
  if (( stagnant >= 2 )); then
    echo "[FATAL] No UMi baseline progress across three attempts." >&2
    exit "$rc"
  fi
  sleep 15
done
