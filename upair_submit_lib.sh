#!/usr/bin/env bash
# Shared helpers for portable UPAIR Slurm wrappers.
set -euo pipefail
# Explicit TensorFlow evaluation-memory environment.
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TF_CPP_VMODULE="${TF_CPP_VMODULE:-bfc_allocator=0}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

# Keep TensorFlow allocator dumps from flooding Slurm logs.
# Real errors still surface as Python exceptions/Tracebacks.
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TF_CPP_VMODULE="${TF_CPP_VMODULE:-bfc_allocator=0}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"

UPAIR_LIB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${UPAIR_LIB_ROOT}/upair_portable_env.sh"

UPAIR_VARIANTS_DEFAULT=(
  main_d256_b4_r2
  shallow_d256_b2_r2
  deep_d256_b6_r2
  narrow_d192_b4_r2
  wide_d320_b4_r2
  wide_deep_d320_b6_r2
  mlpwide_d256_b4_r4
)

upair_variants() {
  if [[ -n "${UPAIR_VARIANTS:-}" ]]; then
    local raw="${UPAIR_VARIANTS//,/ }"
    # shellcheck disable=SC2206
    local arr=( ${raw} )
    printf '%s\n' "${arr[@]}"
  else
    printf '%s\n' "${UPAIR_VARIANTS_DEFAULT[@]}"
  fi
}

upair_first_n_chars() {
  local value="$1"
  local n="${2:-18}"
  printf '%s' "${value:0:${n}}"
}

upair_write_sbatch_header() {
  local file="$1"
  local job_name="$2"
  local time_limit="$3"
  local log_file="$4"
  local cpus="${UPAIR_CPUS:-8}"
  local mem="${UPAIR_MEM:-32G}"
  local gpu_directive="${UPAIR_GPU_DIRECTIVE:-}"
  if [[ -z "${gpu_directive}" ]]; then
    gpu_directive="--gres=${UPAIR_GRES:-gpu:1}"
  fi

  cat > "${file}" <<SBATCH
#!/usr/bin/env bash
#SBATCH --job-name=${job_name}
#SBATCH --time=${time_limit}
#SBATCH --cpus-per-task=${cpus}
#SBATCH --mem=${mem}
#SBATCH --output=${log_file}
SBATCH
  if [[ -n "${UPAIR_SLURM_ACCOUNT:-}" ]]; then
    echo "#SBATCH --account=${UPAIR_SLURM_ACCOUNT}" >> "${file}"
  fi
  if [[ -n "${UPAIR_SLURM_PARTITION:-}" ]]; then
    echo "#SBATCH --partition=${UPAIR_SLURM_PARTITION}" >> "${file}"
  fi
  if [[ -n "${gpu_directive}" && "${gpu_directive}" != "none" && "${gpu_directive}" != "NONE" ]]; then
    echo "#SBATCH ${gpu_directive}" >> "${file}"
  fi
}

upair_submit_job_script() {
  local file="$1"
  chmod +x "${file}"
  if [[ "${UPAIR_RUN_LOCAL:-0}" == "1" ]]; then
    echo "[SUBMIT] UPAIR_RUN_LOCAL=1 -> running locally: ${file}"
    bash "${file}"
  else
    command -v sbatch >/dev/null 2>&1 || { echo "[SUBMIT] sbatch not found. Set UPAIR_RUN_LOCAL=1 for local/smoke testing." >&2; return 1; }
    sbatch "${file}"
  fi
}
