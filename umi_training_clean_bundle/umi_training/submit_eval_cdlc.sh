#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/upair_portable_env.sh"
upair_activate
python umi_training/driver.py training-status
python umi_training/driver.py final-guard
pending="$(python umi_training/driver.py pending-eval cdlc)"
if [[ -z "$pending" ]]; then
  echo "[CDLC-EVAL] already complete"
  exit 0
fi
mkdir -p logs/umi_training
sbatch --export=ALL,UPAIR_REPO_ROOT="$ROOT" --array="${pending}%7" --job-name=umi2cdlc umi_training/eval_array.sbatch cdlc
