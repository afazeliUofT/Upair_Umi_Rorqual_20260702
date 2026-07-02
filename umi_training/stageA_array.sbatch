#!/usr/bin/env bash
#SBATCH --job-name=umi-optA
#SBATCH --time=24:00:00
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
prefix="umiNorm_v1_b32_prb8_u34610_1dmrs_stageA"
study="${prefix}_${variant}"
db="$ROOT/UMI_training/optuna_db/${study}.db"

python umi_training/driver.py prepare-study A "$variant"

python -u scripts/run_optuna_1dmrs_structure.py \
  --config "$ROOT/configs/twc_comprehensive_mu32_umi_training.yaml" \
  --variant "$variant" \
  --study-name "$study" \
  --storage "sqlite:///$db" \
  --stage A \
  --source-top-k 0 \
  --steps 4000 \
  --n-trials 20 \
  --target-total-trials 20 \
  --max-attempts 20 \
  --eval-every 1000 \
  --checkpoint-every 1000 \
  --log-every 100 \
  --val-steps 96 \
  --val-ebno-db=-4,-2,0,2,4 \
  --val-user-counts=1,2,3,4 \
  --val-user-count-weights=1,3,6,10 \
  --train-user-count-weights=1,3,6,10 \
  --train-ebno-min -6 \
  --train-ebno-max 5 \
  --train-batch-size 32 \
  --validation-batch-size 32 \
  --validation-microbatch-size 16 \
  --objective-metric hybrid_nmse_ratio \
  --objective-aggregation recent_mean \
  --objective-recent-k 3 \
  --objective-min-step 2000 \
  --tpe-startup-trials 8 \
  --pruner percentile \
  --pruner-percentile 25 \
  --pruner-startup-trials 6 \
  --pruner-min-trials 4 \
  --prune-warmup-steps 1000 \
  --prune-interval-steps 1000 \
  --memory-cleanup-every-steps 100 \
  --val-memory-cleanup-every-batch
