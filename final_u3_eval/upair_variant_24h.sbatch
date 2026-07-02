#!/usr/bin/env bash
#SBATCH --job-name=u3-upair
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --output=logs/final_u3/%x_%j.out

set -euo pipefail
VARIANT="${1:?Usage: sbatch final_u3_eval/upair_variant_24h.sbatch <variant>}"
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

export UPAIR_DMRS_CASE="1dmrs"
export UPAIR_SEED="7"
export UPAIR_OPTUNA_STAGEB_PREFIX="clean_b32_prb8_d256_40k_smart_trueDMRS_u34610_1dmrs_stageB"
export UPAIR_EVAL_CHUNK_ROOT="$ROOT/_isolated_eval_chunks"
export UPAIR_PIPELINE_RECEIVERS="upair5g_lmmse"
export UPAIR_PIPELINE_USERS="3"
export UPAIR_PIPELINE_EBNOS="-4,-3,-2,-1,0,1"
export UPAIR_PIPELINE_CHUNK_BATCHES="20"
export UPAIR_PIPELINE_MICRO="8"
export UPAIR_PIPELINE_TARGET_BLOCK_ERRORS="100"
export UPAIR_PIPELINE_MAX_BATCHES="2000"
export UPAIR_PIPELINE_MIN_BATCHES="20"

signature() {
  python - "$VARIANT" <<'PY'
import json, sys
from pathlib import Path
import pandas as pd
v = sys.argv[1]
state = Path(f"TWC_plots_comprehensive/runs_rx16/seed7/1dmrs/{v}/metrics/train_state.json")
step = -1
complete = False
if state.exists():
    try:
        d = json.loads(state.read_text())
        step = int(d.get("latest_step", -1))
        complete = bool(d.get("training_complete", False))
    except Exception:
        pass
chunks = 0
root = Path("_isolated_eval_chunks")
for p in root.rglob("chunk_result.csv"):
    try:
        df = pd.read_csv(p, nrows=1)
        if df.empty:
            continue
        r = df.iloc[0]
        if (str(r.get("variant", "")) == v and
            str(r.get("receiver", "")) == "upair5g_lmmse" and
            int(r.get("num_users", -1)) == 3 and
            -4.0 <= float(r.get("ebno_db")) <= 1.0):
            chunks += 1
    except Exception:
        pass
print(f"step={step};complete={int(complete)};chunks={chunks}")
PY
}

last=""
stagnant=0
attempt=0
while true; do
  attempt=$((attempt + 1))
  echo "[24H-WRAPPER] variant=$VARIANT attempt=$attempt"
  set +e
  bash "$ROOT/upair_variant_pipeline_worker.sh" "$VARIANT"
  rc=$?
  set -e
  if [[ $rc -eq 0 ]]; then
    echo "[24H-WRAPPER] COMPLETE variant=$VARIANT"
    exit 0
  fi

  now="$(signature)"
  echo "[24H-WRAPPER] worker_rc=$rc progress=$now"
  if [[ "$now" == "$last" ]]; then
    stagnant=$((stagnant + 1))
  else
    stagnant=0
  fi
  last="$now"

  if (( stagnant >= 2 )); then
    echo "[FATAL] No progress across three attempts; inspect this job log." >&2
    exit "$rc"
  fi
  sleep 15
done
