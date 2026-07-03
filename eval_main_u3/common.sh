#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/rsadve1/links/scratch/Extended_UPAIR_Narval_b32m16_portable_underUMI"
VENV="${UPAIR_VENV_PATH:-/home/rsadve1/links/scratch/.vevn_upair_potable}"

VARIANT="main_d256_b4_r2"
CONFIG="$ROOT/configs/twc_comprehensive_mu32_umi_training.yaml"
PREFIX="umiNorm_v1_b32_prb8_u34610_1dmrs_stageB"
OPTUNA_DIR="$ROOT/optuna"
CHECKPOINT="$ROOT/UMI_training/runs_rx16/seed7/1dmrs/main_d256_b4_r2/checkpoints/best.weights.h5"

OUT_ROOT="$ROOT/_main_umitrained_u3_eval_chunks"
SHARED_COV="$ROOT/_main_umitrained_u3_shared_cov/u3_umi/artifacts/empirical_covariances.npz"

CHUNK_BATCHES=20
MICRO=4
TARGET_ERRORS=100
MIN_BATCHES=20
MAX_BATCHES=2000

activate_env() {
    module load StdEnv/2023 >/dev/null 2>&1 || true
    module load python/3.11 >/dev/null 2>&1 || true
    source "$VENV/bin/activate"
    export PYTHONPATH="$ROOT/src:$ROOT/scripts:${PYTHONPATH:-}"
    export PYTHONUNBUFFERED=1
    export PYTHONDONTWRITEBYTECODE=1
    export PYTHONNOUSERSITE=1
    export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
    export TF_CPP_VMODULE="${TF_CPP_VMODULE:-bfc_allocator=0}"
    export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
    export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
}

safe_tag() {
    python - "$1" <<'PY'
import sys
s = str(sys.argv[1])
print(s.replace("-", "m").replace("+", "p").replace(".", "p").replace(",", "_"))
PY
}
