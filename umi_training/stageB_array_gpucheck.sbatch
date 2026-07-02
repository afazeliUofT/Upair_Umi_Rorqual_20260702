#!/usr/bin/env bash
#SBATCH --job-name=umi-optB
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

echo "[GPUCHECK] host=$(hostname)"
echo "[GPUCHECK] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L || true
nvidia-smi || true

python - <<'PY_GPUCHECK'
import os
import tensorflow as tf

print("[GPUCHECK] CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
gpus = tf.config.list_physical_devices("GPU")
print("[GPUCHECK] tf_gpus =", gpus)

if not gpus:
    raise SystemExit("[GPUCHECK-FATAL] no TensorFlow GPU visible")

with tf.device("/GPU:0"):
    x = tf.random.uniform([512, 512], dtype=tf.float32)
    y = tf.matmul(x, x)
    print("[GPUCHECK] matmul_sum =", float(tf.reduce_sum(y).numpy()))

from sionna.phy import config as sionna_config
sionna_config.seed = 12345
print("[GPUCHECK] sionna_seed_ok =", float(sionna_config.tf_rng.uniform([1]).numpy()[0]))
print("[GPUCHECK] PASS")
PY_GPUCHECK

variant="$(python umi_training/driver.py variant "$SLURM_ARRAY_TASK_ID")"
prefixA="umiNorm_v1_b32_prb8_u34610_1dmrs_stageA"
prefixB="umiNorm_v1_b32_prb8_u34610_1dmrs_stageB"
studyA="${prefixA}_${variant}"
studyB="${prefixB}_${variant}"
dbA="$ROOT/UMI_training/optuna_db/${studyA}.db"
dbB="$ROOT/UMI_training/optuna_db/${studyB}.db"

python umi_training/driver.py prepare-study B "$variant"

python -u scripts/run_optuna_1dmrs_structure.py \
  --config "$ROOT/configs/twc_comprehensive_mu32_umi_training.yaml" \
  --variant "$variant" \
  --study-name "$studyB" \
  --storage "sqlite:///$dbB" \
  --stage B \
  --source-study-name "$studyA" \
  --source-storage "sqlite:///$dbA" \
  --source-top-k 6 \
  --steps 12000 \
  --n-trials 6 \
  --target-total-trials 6 \
  --max-attempts 6 \
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
  --disable-pruning \
  --memory-cleanup-every-steps 100 \
  --val-memory-cleanup-every-batch
