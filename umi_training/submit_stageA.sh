#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/upair_portable_env.sh"
upair_activate
python umi_training/driver.py smoke-status
pending="$(python umi_training/driver.py pending-stage A)"
if [[ -z "$pending" ]]; then
  echo "[STAGE-A] already complete"
  exit 0
fi
mkdir -p logs/umi_training
sbatch --export=ALL,UPAIR_REPO_ROOT="$ROOT" --array="${pending}%7" umi_training/stageA_array.sbatch
