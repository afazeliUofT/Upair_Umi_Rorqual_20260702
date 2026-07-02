#!/usr/bin/env bash
#SBATCH --job-name=umi-probe
#SBATCH --time=02:00:00
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

rm -rf "$ROOT/_umi_smoke_end_to_end"
python "$ROOT/umi_sensitivity/driver.py" static
python "$ROOT/umi_sensitivity/driver.py" probe-runtime
