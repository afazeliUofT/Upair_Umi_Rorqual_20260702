#!/usr/bin/env bash
#SBATCH --job-name=upairTE-narrow_d192_b
#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/rsadve1/scratch/Extended_UPAIR_Narval_b32m16_portable/logs/train_eval/train_eval_narrow_d192_b4_r2_%j.out
#SBATCH --gres=gpu:h100:1
set -euo pipefail
cd "/home/rsadve1/scratch/Extended_UPAIR_Narval_b32m16_portable"
source "/home/rsadve1/scratch/Extended_UPAIR_Narval_b32m16_portable/upair_portable_env.sh"
upair_activate
python -u "/home/rsadve1/scratch/Extended_UPAIR_Narval_b32m16_portable/scripts/run_comprehensive_mu32_ablation.py"   --config "/home/rsadve1/scratch/Extended_UPAIR_Narval_b32m16_portable/configs/twc_comprehensive_mu32_base.yaml"   --variants "narrow_d192_b4_r2"   --dmrs-cases "1dmrs"   --seeds "7"   --eval-users "1,2,3,4"   --use-optuna-best-1dmrs   --optuna-best-storage-dir "/home/rsadve1/scratch/Extended_UPAIR_Narval_b32m16_portable/optuna"   --optuna-best-study-prefix "clean_b32_prb8_d256_40k_smart_trueDMRS_u34610_1dmrs_stageB"   --require-optuna-best   --no-global-summary   
