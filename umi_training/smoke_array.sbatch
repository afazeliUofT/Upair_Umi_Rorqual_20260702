#!/usr/bin/env bash
#SBATCH --job-name=umi-smoke
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:h100:1
#SBATCH --signal=B:TERM@300
#SBATCH --output=logs/umi_training/%x_%A_%a.out

set -euo pipefail
ROOT="${UPAIR_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
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

variant="$(python umi_training/driver.py variant "$SLURM_ARRAY_TASK_ID")"
python umi_training/driver.py smoke-train "$variant"
