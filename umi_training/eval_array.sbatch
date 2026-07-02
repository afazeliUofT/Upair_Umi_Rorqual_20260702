#!/usr/bin/env bash
#SBATCH --job-name=umi-eval
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:h100:1
#SBATCH --signal=B:TERM@300
#SBATCH --output=logs/umi_training/%x_%A_%a.out

set -euo pipefail
domain="${1:?Usage: sbatch eval_array.sbatch <umi|cdlc>}"
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

last="-1"
stagnant=0
while true; do
  set +e
  bash umi_training/eval_variant_worker.sh "$domain" "$variant"
  rc=$?
  set -e
  [[ "$rc" -eq 0 ]] && exit 0

  if [[ "$domain" == "umi" ]]; then
    root="_umi_trained_umi_eval_chunks"
  else
    root="_umi_trained_cdlc_eval_chunks"
  fi
  now="$(find "$root" -name chunk_result.csv -type f 2>/dev/null | wc -l)"
  echo "[EVAL-WRAPPER] domain=$domain variant=$variant rc=$rc chunks=$now"
  if [[ "$now" == "$last" ]]; then
    stagnant=$((stagnant + 1))
  else
    stagnant=0
  fi
  last="$now"
  if (( stagnant >= 2 )); then
    echo "[FATAL] No evaluation progress across three attempts." >&2
    exit "$rc"
  fi
  sleep 15
done
