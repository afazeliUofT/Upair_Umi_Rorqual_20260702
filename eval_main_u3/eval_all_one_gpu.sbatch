#!/usr/bin/env bash
#SBATCH --job-name=main-u3-eval1gpu
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:h100:1
#SBATCH --signal=B:TERM@300
#SBATCH --output=logs/eval_main_u3/%x_%j.out

set -euo pipefail

ROOT="${UPAIR_REPO_ROOT:-/home/rsadve1/links/scratch/Extended_UPAIR_Narval_b32m16_portable_underUMI}"
cd "$ROOT"

source eval_main_u3/common.sh
activate_env

echo "[GPU] host=$(hostname)"
echo "[GPU] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L || true

python - <<'PY'
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
print("[GPU] tf_gpus =", gpus)
if not gpus:
    raise SystemExit("[FATAL] no TensorFlow GPU visible")
PY

num_tasks="$(wc -l < eval_main_u3/tasks.tsv)"
echo "[INFO] Sequential evaluation on one GPU"
echo "[INFO] num_tasks=$num_tasks"

for task_id in $(seq 0 $((num_tasks - 1))); do
    echo
    echo "============================================================"
    echo "[ONE-GPU] Starting task_id=$task_id / $((num_tasks - 1))"
    echo "============================================================"

    bash eval_main_u3/eval_point_worker.sh "$task_id"

    echo "[ONE-GPU] Finished task_id=$task_id"
done

echo "[ONE-GPU] All tasks finished"
