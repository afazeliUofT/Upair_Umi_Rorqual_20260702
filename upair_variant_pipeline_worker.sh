#!/usr/bin/env bash
# Worker for exactly one architecture variant.
# Resubmittable: resumes training and resumes isolated eval chunks.
set -euo pipefail

VARIANT="${1:?Usage: bash upair_variant_pipeline_worker.sh <variant>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

source "${ROOT}/upair_portable_env.sh"
upair_activate

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TF_CPP_VMODULE="${TF_CPP_VMODULE:-bfc_allocator=0}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

CONFIG="${UPAIR_CONFIG:-${ROOT}/configs/twc_comprehensive_mu32_base.yaml}"
DMRS_CASE="${UPAIR_DMRS_CASE:-1dmrs}"
SEED="${UPAIR_SEED:-7}"
STAGEB_PREFIX="${UPAIR_OPTUNA_STAGEB_PREFIX:-clean_b32_prb8_d256_40k_smart_trueDMRS_u34610_1dmrs_stageB}"
OUT_ROOT="${UPAIR_EVAL_CHUNK_ROOT:-${ROOT}/_isolated_eval_chunks}"

# Defaults mimic the original evaluation stopping policy through isolated chunks.
# Override these if you want a shorter exploratory evaluation.
RECEIVERS_RAW="${UPAIR_PIPELINE_RECEIVERS:-${UPAIR_EVAL_RECEIVERS:-baseline_ls_lmmse,baseline_ls_2dlmmse_lmmse,upair5g_lmmse,perfect_csi_lmmse}}"
USERS_RAW="${UPAIR_PIPELINE_USERS:-${UPAIR_EVAL_USERS:-1,2,3,4}}"
EBNOS_RAW="${UPAIR_PIPELINE_EBNOS:-${UPAIR_EVAL_EBNOS:--4,-3,-2,-1,0,1,2,3,4}}"
CHUNK_BATCHES="${UPAIR_PIPELINE_CHUNK_BATCHES:-${UPAIR_EVAL_CHUNK_BATCHES:-20}}"
MICRO="${UPAIR_PIPELINE_MICRO:-${UPAIR_EVAL_MICRO:-8}}"

# Empty => read from configs/twc_comprehensive_mu32_base.yaml.
TARGET_BLOCK_ERRORS="${UPAIR_PIPELINE_TARGET_BLOCK_ERRORS:-}"
MAX_BATCHES="${UPAIR_PIPELINE_MAX_BATCHES:-}"
MIN_BATCHES="${UPAIR_PIPELINE_MIN_BATCHES:-}"

mkdir -p "${OUT_ROOT}" logs/pipeline

split_csv() {
  local raw="${1//,/ }"
  # shellcheck disable=SC2206
  local arr=( ${raw} )
  printf '%s\n' "${arr[@]}"
}

training_complete() {
  python - "$VARIANT" <<'PY'
import json, sys
from pathlib import Path
v = sys.argv[1]
p = Path(f"TWC_plots_comprehensive/runs_rx16/seed7/1dmrs/{v}/metrics/train_state.json")
if not p.exists():
    sys.exit(1)
try:
    d = json.loads(p.read_text())
except Exception:
    sys.exit(1)
sys.exit(0 if bool(d.get("training_complete", False)) else 1)
PY
}

echo "================================================================================"
echo "[PIPELINE] variant=${VARIANT}"
echo "[PIPELINE] root=${ROOT}"
echo "[PIPELINE] stageB_prefix=${STAGEB_PREFIX}"
echo "[PIPELINE] receivers=${RECEIVERS_RAW}"
echo "[PIPELINE] users=${USERS_RAW}"
echo "[PIPELINE] ebnos=${EBNOS_RAW}"
echo "[PIPELINE] chunk_batches=${CHUNK_BATCHES} micro=${MICRO}"
echo "================================================================================"

best_json="${ROOT}/optuna/${STAGEB_PREFIX}_${VARIANT}_best_params.json"
best_db="${ROOT}/optuna/${STAGEB_PREFIX}_${VARIANT}.db"
if [[ ! -s "${best_json}" && ! -s "${best_db}" ]]; then
  echo "[PIPELINE] Missing Stage-B best for ${VARIANT}" >&2
  echo "  ${best_json}" >&2
  echo "  ${best_db}" >&2
  exit 2
fi

if training_complete; then
  echo "[PIPELINE] training already complete for ${VARIANT}; skipping training."
else
  echo "[PIPELINE] training missing/incomplete for ${VARIANT}; running training-only resume."
  export UPAIR_COMPREHENSIVE_SKIP_FINAL_EVAL=1
  python -u "${ROOT}/scripts/run_comprehensive_mu32_ablation.py" \
    --config "${CONFIG}" \
    --variants "${VARIANT}" \
    --dmrs-cases "${DMRS_CASE}" \
    --seeds "${SEED}" \
    --eval-users "${USERS_RAW}" \
    --use-optuna-best-1dmrs \
    --optuna-best-storage-dir "${ROOT}/optuna" \
    --optuna-best-study-prefix "${STAGEB_PREFIX}" \
    --require-optuna-best \
    --no-global-summary

  if ! training_complete; then
    echo "[PIPELINE] Training is still incomplete for ${VARIANT}; resubmit this same pipeline later." >&2
    exit 20
  fi
fi

echo "[PIPELINE] starting/resuming isolated evaluation for ${VARIANT}."

status_args_base=(--input-root "${OUT_ROOT}" --config "${CONFIG}" --variant "${VARIANT}" --chunk-batches "${CHUNK_BATCHES}")
if [[ -n "${TARGET_BLOCK_ERRORS}" ]]; then
  status_args_base+=(--target-block-errors "${TARGET_BLOCK_ERRORS}")
fi
if [[ -n "${MAX_BATCHES}" ]]; then
  status_args_base+=(--max-batches "${MAX_BATCHES}")
fi
if [[ -n "${MIN_BATCHES}" ]]; then
  status_args_base+=(--min-batches "${MIN_BATCHES}")
fi

while IFS= read -r receiver; do
  [[ -n "${receiver}" ]] || continue
  while IFS= read -r users; do
    [[ -n "${users}" ]] || continue
    while IFS= read -r ebno; do
      [[ -n "${ebno}" ]] || continue

      echo
      echo "--------------------------------------------------------------------------------"
      echo "[PIPELINE] eval point variant=${VARIANT} receiver=${receiver} U=${users} Eb/N0=${ebno}"
      echo "--------------------------------------------------------------------------------"

      while true; do
        status_file="$(mktemp)"
        python "${ROOT}/scripts/isolated_eval_status.py" \
          "${status_args_base[@]}" \
          --receiver "${receiver}" \
          --num-users "${users}" \
          --ebno-db "${ebno}" \
          --shell > "${status_file}"
        # shellcheck disable=SC1090
        source "${status_file}"
        rm -f "${status_file}"

        echo "[PIPELINE] status done=${DONE} reason=${REASON} chunks=${NUM_CHUNKS_DONE} batches=${NUM_BATCHES} block_errors=${BLOCK_ERRORS}/${TARGET_BLOCK_ERRORS} next_chunk=${NEXT_CHUNK}"

        if [[ "${DONE}" == "1" ]]; then
          break
        fi

        python -u "${ROOT}/scripts/run_isolated_eval_chunk.py" \
          --config "${CONFIG}" \
          --variant "${VARIANT}" \
          --dmrs-case "${DMRS_CASE}" \
          --seed "${SEED}" \
          --num-users "${users}" \
          --receiver "${receiver}" \
          --ebno-db "${ebno}" \
          --chunk-idx "${NEXT_CHUNK}" \
          --chunk-batches "${CHUNK_BATCHES}" \
          --receiver-microbatch-size "${MICRO}" \
          --stageb-prefix "${STAGEB_PREFIX}" \
          --optuna-dir "${ROOT}/optuna" \
          --output-root "${OUT_ROOT}"
      done

      safe_ebno="${ebno//-/m}"
      safe_ebno="${safe_ebno//./p}"
      merged_csv="${OUT_ROOT}/merged_${VARIANT}_u${users}_${receiver}_e${safe_ebno}.csv"
      python "${ROOT}/scripts/merge_isolated_eval_chunks.py" \
        --input-root "${OUT_ROOT}" \
        --output-csv "${merged_csv}" \
        --variant "${VARIANT}" \
        --receiver "${receiver}" \
        --num-users "${users}" \
        --ebno-db "${ebno}"

    done < <(split_csv "${EBNOS_RAW}")
  done < <(split_csv "${USERS_RAW}")
done < <(split_csv "${RECEIVERS_RAW}")

echo "[PIPELINE] COMPLETE variant=${VARIANT}"
