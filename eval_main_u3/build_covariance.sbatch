#!/usr/bin/env bash
#SBATCH --job-name=main-u3-cov
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:h100:1
#SBATCH --output=logs/eval_main_u3/%x_%j.out

set -euo pipefail
ROOT="${UPAIR_REPO_ROOT:-/home/rsadve1/links/scratch/Extended_UPAIR_Narval_b32m16_portable_underUMI}"
cd "$ROOT"

source eval_main_u3/common.sh
activate_env

nvidia-smi -L || true

python eval_main_u3/build_shared_covariance.py
