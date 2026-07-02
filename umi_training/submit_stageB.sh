#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/upair_portable_env.sh"
upair_activate
python umi_training/driver.py study-status A
pending="$(python umi_training/driver.py pending-stage B)"
if [[ -z "$pending" ]]; then
  echo "[STAGE-B] already complete"
  exit 0
fi
mkdir -p logs/umi_training
sbatch --export=ALL,UPAIR_REPO_ROOT="$ROOT" --array="${pending}%7" umi_training/stageB_array.sbatch
