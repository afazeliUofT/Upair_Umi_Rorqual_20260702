#!/usr/bin/env bash
#SBATCH --job-name=upairP-shallow_d256_
#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/rsadve1/scratch/Extended_UPAIR_Narval_b32m16_portable/logs/pipeline/pipeline_shallow_d256_b2_r2_%j.out
#SBATCH --gres=gpu:h100:1
set -euo pipefail
cd "/home/rsadve1/scratch/Extended_UPAIR_Narval_b32m16_portable"
source "/home/rsadve1/scratch/Extended_UPAIR_Narval_b32m16_portable/upair_portable_env.sh"
upair_activate

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TF_CPP_VMODULE="${TF_CPP_VMODULE:-bfc_allocator=0}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

# Carry pipeline scope if the submitter set these once before calling this wrapper.
export UPAIR_CONFIG=""
export UPAIR_DMRS_CASE="1dmrs"
export UPAIR_SEED="7"
export UPAIR_OPTUNA_STAGEB_PREFIX="clean_b32_prb8_d256_40k_smart_trueDMRS_u34610_1dmrs_stageB"
export UPAIR_PIPELINE_RECEIVERS="upair5g_lmmse,baseline_ls_lmmse,baseline_ls_2dlmmse_lmmse,perfect_csi_lmmse"
export UPAIR_PIPELINE_USERS="3"
export UPAIR_PIPELINE_EBNOS="-4,-3,-2,-1,0,1,2,3,4"
export UPAIR_PIPELINE_CHUNK_BATCHES="20"
export UPAIR_PIPELINE_MICRO="8"
export UPAIR_PIPELINE_TARGET_BLOCK_ERRORS=""
export UPAIR_PIPELINE_MAX_BATCHES=""
export UPAIR_PIPELINE_MIN_BATCHES=""

bash "/home/rsadve1/scratch/Extended_UPAIR_Narval_b32m16_portable/upair_variant_pipeline_worker.sh" "shallow_d256_b2_r2"
