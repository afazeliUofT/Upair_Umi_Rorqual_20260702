#!/usr/bin/env bash
#SBATCH --job-name=upairA-shallow_d256_b
#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/rsadve1/scratch/Extended_UPAIR_Narval_b32m16_portable/logs/optuna/clean_b32_prb8_d256_40k_smart_u34610_1dmrs_stageA_shallow_d256_b2_r2_%j.out
#SBATCH --gres=gpu:h100:1
set -euo pipefail
cd "/home/rsadve1/scratch/Extended_UPAIR_Narval_b32m16_portable"
source "/home/rsadve1/scratch/Extended_UPAIR_Narval_b32m16_portable/upair_portable_env.sh"
upair_activate
python -u "/home/rsadve1/scratch/Extended_UPAIR_Narval_b32m16_portable/scripts/run_optuna_1dmrs_structure_isolated.py"   --config "/home/rsadve1/scratch/Extended_UPAIR_Narval_b32m16_portable/configs/twc_comprehensive_mu32_base.yaml"   --variant "shallow_d256_b2_r2"   --study-name "clean_b32_prb8_d256_40k_smart_u34610_1dmrs_stageA_shallow_d256_b2_r2"   --storage "sqlite:////home/rsadve1/scratch/Extended_UPAIR_Narval_b32m16_portable/optuna/clean_b32_prb8_d256_40k_smart_u34610_1dmrs_stageA_shallow_d256_b2_r2.db"   --stage A   --n-trials "20"   --target-total-trials "20"   --max-attempts "30"   --steps "4000"   --train-batch-size "32"   --validation-batch-size "32"   --validation-microbatch-size "16"   --seed "7"   --tpe-startup-trials "8"   --pruner-startup-trials "6"   --pruner-min-trials "4"   --objective-recent-k "2"   --objective-min-step "1000"
