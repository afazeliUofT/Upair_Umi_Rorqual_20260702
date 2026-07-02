#!/usr/bin/env bash
# Submit one job per variant. Each job requests exactly one GPU and runs chunks sequentially.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

source "${ROOT}/upair_submit_lib.sh"
upair_ensure_venv

mkdir -p "${UPAIR_REPO_ROOT}/logs/pipeline" "${UPAIR_REPO_ROOT}/logs/submit"

TIME_LIMIT="${UPAIR_TIME_PIPELINE:-30:00:00}"
# Nibi needs a GPU type. Override on another cluster if needed.
export UPAIR_GRES="${UPAIR_GRES:-gpu:h100:1}"
export UPAIR_MEM="${UPAIR_MEM:-32G}"
export UPAIR_CPUS="${UPAIR_CPUS:-8}"

echo "[PIPELINE-SUBMIT] ROOT=${UPAIR_REPO_ROOT}"
echo "[PIPELINE-SUBMIT] VENV=${UPAIR_VENV_PATH}"
echo "[PIPELINE-SUBMIT] GRES=${UPAIR_GRES} TIME=${TIME_LIMIT}"
echo "[PIPELINE-SUBMIT] variants:"
upair_variants | sed 's/^/  - /'

while IFS= read -r variant; do
  [[ -n "${variant}" ]] || continue

  job="upairP-$(upair_first_n_chars "${variant}" 13)"
  log="${UPAIR_REPO_ROOT}/logs/pipeline/pipeline_${variant}_%j.out"
  jobfile="${UPAIR_REPO_ROOT}/logs/submit/pipeline_${variant}.sbatch"
  upair_write_sbatch_header "${jobfile}" "${job}" "${TIME_LIMIT}" "${log}"

  cat >> "${jobfile}" <<SBATCH
set -euo pipefail
cd "${UPAIR_REPO_ROOT}"
source "${UPAIR_REPO_ROOT}/upair_portable_env.sh"
upair_activate

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TF_CPP_MIN_LOG_LEVEL="\${TF_CPP_MIN_LOG_LEVEL:-2}"
export TF_CPP_VMODULE="\${TF_CPP_VMODULE:-bfc_allocator=0}"
export TF_FORCE_GPU_ALLOW_GROWTH="\${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export TF_GPU_ALLOCATOR="\${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

# Carry pipeline scope if the submitter set these once before calling this wrapper.
export UPAIR_CONFIG="${UPAIR_CONFIG:-}"
export UPAIR_DMRS_CASE="${UPAIR_DMRS_CASE:-1dmrs}"
export UPAIR_SEED="${UPAIR_SEED:-7}"
export UPAIR_OPTUNA_STAGEB_PREFIX="${UPAIR_OPTUNA_STAGEB_PREFIX:-clean_b32_prb8_d256_40k_smart_trueDMRS_u34610_1dmrs_stageB}"
export UPAIR_PIPELINE_RECEIVERS="${UPAIR_PIPELINE_RECEIVERS:-}"
export UPAIR_PIPELINE_USERS="${UPAIR_PIPELINE_USERS:-}"
export UPAIR_PIPELINE_EBNOS="${UPAIR_PIPELINE_EBNOS:-}"
export UPAIR_PIPELINE_CHUNK_BATCHES="${UPAIR_PIPELINE_CHUNK_BATCHES:-20}"
export UPAIR_PIPELINE_MICRO="${UPAIR_PIPELINE_MICRO:-8}"
export UPAIR_PIPELINE_TARGET_BLOCK_ERRORS="${UPAIR_PIPELINE_TARGET_BLOCK_ERRORS:-}"
export UPAIR_PIPELINE_MAX_BATCHES="${UPAIR_PIPELINE_MAX_BATCHES:-}"
export UPAIR_PIPELINE_MIN_BATCHES="${UPAIR_PIPELINE_MIN_BATCHES:-}"

bash "${UPAIR_REPO_ROOT}/upair_variant_pipeline_worker.sh" "${variant}"
SBATCH

  echo "[PIPELINE-SUBMIT] submitting ${variant}"
  upair_submit_job_script "${jobfile}"
done < <(upair_variants)
