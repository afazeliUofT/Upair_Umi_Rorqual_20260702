#!/usr/bin/env bash
#SBATCH --job-name=umi-train-lm
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

export UPAIR_FINAL_TRAIN_BATCH_SIZE="${UPAIR_FINAL_TRAIN_BATCH_SIZE:-16}"
export UPAIR_FINAL_EVAL_BATCH_SIZE="${UPAIR_FINAL_EVAL_BATCH_SIZE:-16}"
export UPAIR_FINAL_VAL_MICROBATCH_SIZE="${UPAIR_FINAL_VAL_MICROBATCH_SIZE:-8}"
export UPAIR_FINAL_MEMORY_CLEANUP_EVERY_STEPS="${UPAIR_FINAL_MEMORY_CLEANUP_EVERY_STEPS:-25}"

variant="$(python umi_training/driver.py variant "$SLURM_ARRAY_TASK_ID")"

last="-1"
stagnant=0

while true; do
  set +e
  python -u umi_training/train_final_lowmem.py "$variant"
  rc=$?
  set -e

  [[ "$rc" -eq 0 ]] && exit 0

  state="$ROOT/UMI_training/runs_rx16/seed7/1dmrs/$variant/metrics/train_state.json"

  now="$(python - "$state" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.exists():
    print(-1)
else:
    try:
        print(int(json.loads(p.read_text()).get("latest_step", -1)))
    except Exception:
        print(-1)
PY
)"

  echo "[FINAL-LM-WRAPPER] variant=$variant rc=$rc step=$now"

  if [[ "$now" == "$last" ]]; then
    stagnant=$((stagnant + 1))
  else
    stagnant=0
  fi

  last="$now"

  if (( stagnant >= 2 )); then
    echo "[FATAL] No final-training progress across three attempts." >&2
    exit "$rc"
  fi

  sleep 15
done
