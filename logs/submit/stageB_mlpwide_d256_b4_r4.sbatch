#!/usr/bin/env bash
#SBATCH --job-name=upairB-mlpwide_d256_b
#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/rsadve1/scratch/Extended_UPAIR_Narval_b32m16_portable/logs/optuna/clean_b32_prb8_d256_40k_smart_trueDMRS_u34610_1dmrs_stageB_mlpwide_d256_b4_r4_%j.out
#SBATCH --gres=gpu:h100:1
set -euo pipefail
cd "/home/rsadve1/scratch/Extended_UPAIR_Narval_b32m16_portable"
source "/home/rsadve1/scratch/Extended_UPAIR_Narval_b32m16_portable/upair_portable_env.sh"
upair_activate
python -u "/home/rsadve1/scratch/Extended_UPAIR_Narval_b32m16_portable/scripts/run_optuna_1dmrs_structure_isolated.py"   --config "/home/rsadve1/scratch/Extended_UPAIR_Narval_b32m16_portable/configs/twc_comprehensive_mu32_base.yaml"   --variant "mlpwide_d256_b4_r4"   --study-name "clean_b32_prb8_d256_40k_smart_trueDMRS_u34610_1dmrs_stageB_mlpwide_d256_b4_r4"   --storage "sqlite:////home/rsadve1/scratch/Extended_UPAIR_Narval_b32m16_portable/optuna/clean_b32_prb8_d256_40k_smart_trueDMRS_u34610_1dmrs_stageB_mlpwide_d256_b4_r4.db"   --stage B   --source-study-name "clean_b32_prb8_d256_40k_smart_u34610_1dmrs_stageA_mlpwide_d256_b4_r4"   --source-storage "sqlite:////home/rsadve1/scratch/Extended_UPAIR_Narval_b32m16_portable/optuna/clean_b32_prb8_d256_40k_smart_u34610_1dmrs_stageA_mlpwide_d256_b4_r4.db"   --source-top-k "6"   --n-trials "8"   --target-total-trials "8"   --max-attempts "12"   --steps "12000"   --train-batch-size "32"   --validation-batch-size "32"   --validation-microbatch-size "16"   --seed "7"   --tpe-startup-trials "4"   --pruner-startup-trials "4"   --pruner-min-trials "3"   --objective-recent-k "3"   --objective-min-step "2000"
