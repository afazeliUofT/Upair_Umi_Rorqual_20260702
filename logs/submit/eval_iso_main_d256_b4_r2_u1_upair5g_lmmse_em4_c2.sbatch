#!/usr/bin/env bash
#SBATCH --job-name=uIso-main_d25-u1-upair5g_-em4-c2
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/rsadve1/scratch/Extended_UPAIR_Narval_b32m16_portable/logs/eval_iso/eval_main_d256_b4_r2_u1_upair5g_lmmse_em4_c2_%j.out
#SBATCH --gres=gpu:h100:1
set -euo pipefail
cd "/home/rsadve1/scratch/Extended_UPAIR_Narval_b32m16_portable"
source "/home/rsadve1/scratch/Extended_UPAIR_Narval_b32m16_portable/upair_portable_env.sh"
upair_activate
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TF_CPP_MIN_LOG_LEVEL="2"
export TF_CPP_VMODULE="bfc_allocator=0"
export TF_FORCE_GPU_ALLOW_GROWTH="true"
export TF_GPU_ALLOCATOR="cuda_malloc_async"
python -u "/home/rsadve1/scratch/Extended_UPAIR_Narval_b32m16_portable/scripts/run_isolated_eval_chunk.py"   --config "/home/rsadve1/scratch/Extended_UPAIR_Narval_b32m16_portable/configs/twc_comprehensive_mu32_base.yaml"   --variant "main_d256_b4_r2"   --dmrs-case "1dmrs"   --seed "7"   --num-users "1"   --receiver "upair5g_lmmse"   --ebno-db "-4"   --chunk-idx "2"   --chunk-batches "20"   --receiver-microbatch-size "8"   --stageb-prefix "clean_b32_prb8_d256_40k_smart_trueDMRS_u34610_1dmrs_stageB"   --optuna-dir "/home/rsadve1/scratch/Extended_UPAIR_Narval_b32m16_portable/optuna"   --output-root "/home/rsadve1/scratch/Extended_UPAIR_Narval_b32m16_portable/_isolated_eval_chunks"
